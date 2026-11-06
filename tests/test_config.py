"""Validate the serving configuration and model identities."""
import json
from pathlib import Path
import unittest
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
CACHE_KEYS = {'VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
              'TORCH_EXTENSIONS_DIR','FLASHINFER_WORKSPACE_BASE','NUMBA_CACHE_DIR'}
CONTEXT_LIMIT = 1048576
ALLOCATOR_SETTINGS = 'expandable_segments:False'


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT/'tinfoil-config.yml').read_text())
        self.container, = self.config['containers']
        self.command = self.container['command']
        self.env = {key:str(value) for row in self.container['env'] for key,value in row.items()}

    def option(self,name):
        return self.command[self.command.index(name)+1]

    def test_model_paths_reference_pinned_verified_packs(self):
        roots = []
        for model in self.config['models']:
            self.assertRegex(model['repo'],r'^[^@]+@[0-9a-f]{40}$')
            root,offset,verity_uuid = model['mpk'].split('_')
            self.assertRegex(root,r'^[0-9a-f]{64}$')
            self.assertGreater(int(offset),0)
            uuid.UUID(verity_uuid)
            roots.append('/tinfoil/mpk/mpk-'+root)
        self.assertEqual(self.option('--model'),roots[0])
        draft = json.loads(self.option('--speculative-config'))
        self.assertEqual(draft['model'],roots[1])
        self.assertEqual(draft['method'],'dspark')
        self.assertEqual(draft['rejection_sample_method'],'standard')

    def test_image_reference_is_quoted_for_the_release_action(self):
        source = (ROOT/'tinfoil-config.yml').read_text()
        self.assertRegex(source,r'(?m)^\s*image: "ghcr\.io/[^"\n]+@sha256:[0-9a-f]{64}"')

    def test_uniform_target_and_draft_sizes_have_graph_captures(self):
        depth = json.loads(self.option('--speculative-config'))['num_speculative_tokens']
        slots = int(self.option('--max-num-seqs'))
        graphs = json.loads(self.option('--compilation-config'))
        sizes = set(graphs['cudagraph_capture_sizes'])
        for count in range(1,slots+1):
            self.assertIn(count*depth,sizes)
            self.assertIn(count*(depth+1),sizes)
        self.assertGreaterEqual(graphs['max_cudagraph_capture_size'],max(sizes))

    def test_served_name_matches_the_configured_model_and_container(self):
        served_name = self.option('--served-model-name')
        self.assertEqual(served_name,'kimi-k3-1-pro')
        self.assertEqual(self.container['name'],served_name)
        self.assertEqual(self.config['models'][0]['name'],served_name)

    def test_cache_writes_are_supported_by_the_read_only_container(self):
        self.assertTrue(self.container['read_only'])
        self.assertEqual(self.container['restart'],'always')
        self.assertIn('/root',self.container['tmpfs'])
        for key in CACHE_KEYS:
            self.assertTrue(self.env[key].startswith('/root/'),key)
        self.assertEqual(self.env['FLASHINFER_NO_DOWNLOAD'],'1')
        self.assertNotEqual(self.config['shim'].get('tls-mode'),'self-signed')

    def test_collective_and_recurrent_state_settings_are_explicit(self):
        self.assertEqual(self.env['VLLM_FLASHINFER_ALLREDUCE_BACKEND'],'trtllm')
        self.assertEqual(self.env['VLLM_ALLREDUCE_USE_SYMM_MEM'],'0')
        self.assertEqual(self.env['NCCL_NVLS_ENABLE'],'0')
        self.assertIn('--disable-custom-all-reduce',self.command)
        self.assertEqual(self.option('--mamba-ssm-cache-dtype'),'float32')
        self.assertEqual(self.option('--kv-cache-dtype'),'fp8')
        self.assertEqual(int(self.option('--tensor-parallel-size')),self.config['gpus'])
        self.assertGreater(self.config['cpus'],0)
        self.assertGreater(self.config['memory'],0)

    def test_context_parallelism_keeps_compatible_kernels_and_allocations(self):
        tensor_parallel = int(self.option('--tensor-parallel-size'))
        context_parallel = int(self.option('--decode-context-parallel-size'))
        self.assertEqual(context_parallel,tensor_parallel)
        self.assertEqual(self.option('--dcp-comm-backend'),'a2a')
        kernel = json.loads(self.option('--kernel-config'))
        self.assertEqual(kernel['flashinfer_sparse_mla_kernel'],'trtllm')
        self.assertEqual(int(self.option('--max-model-len')),CONTEXT_LIMIT)
        for name in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'):
            self.assertEqual(self.env[name],ALLOCATOR_SETTINGS)


if __name__ == '__main__':
    unittest.main()
