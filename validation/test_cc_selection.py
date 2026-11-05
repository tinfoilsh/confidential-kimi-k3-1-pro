"""Check the real CC buffer and multicast-path selectors without a GPU context."""
import os
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from vllm.utils import platform_utils
from vllm.models.kimi_k3.nvidia import latent_moe_runner

DISABLE_UVA = 'VLLM_DISABLE_UVA'
DISABLE_TAIL = 'VLLM_KIMI_K3_LATENT_MOE_TAIL_FUSION'
FLASHINFER_ALLREDUCE = 'VLLM_ALLREDUCE_USE_FLASHINFER'
TENSOR_PARALLEL_SIZE = 8
NORMALIZATION_EPSILON = 1e-6


class CCSelectionTest(TestCase):
    def tearDown(self):
        platform_utils.is_uva_available.cache_clear()

    def test_uva_override_wins_over_pinned_memory_availability(self):
        with mock.patch.dict(os.environ, {DISABLE_UVA: '1'}):
            platform_utils.is_uva_available.cache_clear()
            with mock.patch.object(platform_utils, 'is_pin_memory_available', return_value=True):
                self.assertFalse(platform_utils.is_uva_available())

    def test_native_uva_selection_is_preserved_when_override_is_off(self):
        with mock.patch.dict(os.environ, {DISABLE_UVA: '0'}):
            platform_utils.is_uva_available.cache_clear()
            with mock.patch.object(platform_utils, 'is_pin_memory_available', return_value=True):
                self.assertTrue(platform_utils.is_uva_available())

    def test_cc_constructor_does_not_initialize_multicast_tail(self):
        with mock.patch.dict(os.environ, {DISABLE_TAIL: '0'}), \
             mock.patch.object(latent_moe_runner.MoERunner, '__init__', return_value=None), \
             mock.patch.object(latent_moe_runner.torch.cuda, 'Event', return_value=None), \
             mock.patch.object(latent_moe_runner.current_platform, 'is_cuda', return_value=True), \
             mock.patch.object(latent_moe_runner.current_platform, 'is_device_capability_family', return_value=True), \
             mock.patch.object(latent_moe_runner.LatentMoERunner, '_use_fused_path', return_value=True):
            runner = latent_moe_runner.LatentMoERunner()
        self.assertFalse(runner.enable_k3_latent_moe_tail_fusion)
        self.assertFalse(hasattr(runner, '_k3_latent_moe_tail_op'))

    def test_disabled_flashinfer_uses_ordinary_reduction_and_normalization(self):
        from vllm.model_executor.layers import fused_allreduce_gemma_rms_norm as fusion

        hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        runner = SimpleNamespace(moe_config=SimpleNamespace(tp_size=TENSOR_PARALLEL_SIZE))
        norm = torch.nn.RMSNorm(hidden.shape[-1], eps=NORMALIZATION_EPSILON)
        reduced = hidden * TENSOR_PARALLEL_SIZE
        with mock.patch.dict(os.environ, {FLASHINFER_ALLREDUCE: '0'}), \
             mock.patch.object(fusion, '_can_use_flashinfer', side_effect=AssertionError('FlashInfer workspace requested')), \
             mock.patch.object(fusion, 'flashinfer_trtllm_fused_allreduce_norm', side_effect=AssertionError('FlashInfer collective requested')), \
             mock.patch.object(latent_moe_runner, 'tensor_model_parallel_all_reduce', return_value=reduced):
            actual = latent_moe_runner.LatentMoERunner.allreduce_norm_latent_out(runner, hidden, norm)
        expected = reduced * torch.rsqrt(reduced.square().mean(dim=-1, keepdim=True) + NORMALIZATION_EPSILON)
        torch.testing.assert_close(actual, expected)
