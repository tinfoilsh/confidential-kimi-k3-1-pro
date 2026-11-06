"""Keep attention warmup output finite before it reaches the MoE router."""
from unittest import TestCase

import torch

from vllm.config import CompilationConfig, DeviceConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

TOKEN_COUNT = 4
HEAD_COUNT = 2
HEAD_DIMENSION = 128


class DummyKDATest(TestCase):
    def test_missing_metadata_initializes_the_whole_output(self):
        layer = KimiK3DeltaAttention.__new__(KimiK3DeltaAttention)
        torch.nn.Module.__init__(layer)
        layer.prefix = 'model.layers.0.self_attn'
        config = VllmConfig(device_config=DeviceConfig(device='cpu'),
                             compilation_config=CompilationConfig(custom_ops=['none']))
        for metadata in (None, {}):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(metadata=metadata, dtype=dtype):
                    output = torch.full((1, TOKEN_COUNT, HEAD_COUNT, HEAD_DIMENSION),
                                        float('nan'), dtype=dtype)
                    unused = torch.empty(0, dtype=dtype)
                    with set_forward_context(metadata, config):
                        layer._forward(unused, unused, unused, unused, output)
                    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
