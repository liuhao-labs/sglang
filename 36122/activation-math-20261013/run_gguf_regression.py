"""Run the unchanged AOT GGUF tests with freshly compiled source kernels.

Test harness redirects GGUF Python imports into gguf_probe test namespace and
uses pinned, LFS-hash-verified files. It does not replace installed torch ops.
"""
import os
from pathlib import Path
from unittest.mock import patch
import torch
import pytest
import sgl_kernel
import huggingface_hub

torch.cuda.set_per_process_memory_fraction(0.025)
library=Path('/probe/gguf-build-integrated/gguf_probe_integrated.so')
torch.ops.load_library(str(library))
native=torch.ops.gguf_probe
bindings={
    'ggml_dequantize':native.dequantize.default,
    'ggml_mul_mat_a8':native.mul_mat.default,
    'ggml_mul_mat_vec_a8':native.mul_mat_vec.default,
    'ggml_moe_a8':native.moe.default,
    'ggml_moe_a8_vec':native.moe_vec.default,
    'ggml_moe_get_block_size':native.block_size.default,
    'ggml_supports_iq_mmq':native.supports_iq_mmq.default,
}
def snapshot(repo,revision,allow_patterns):
    expected={'Isotr0py/test-gguf-sample':('dense','d82b8773934ef260d8d8a896a7c197bc69a0fac1'),
              'SzymonOzog/test-gguf-moe-sample':('moe','2b77eb27ae2ea4b3f68cf3042961509a9f5847b8')}
    kind,pin=expected[repo]
    assert revision==pin
    return str(Path('/probe/fixtures')/kind/pin)
with patch.multiple(sgl_kernel,create=True,**bindings), \
     patch.object(huggingface_hub,'snapshot_download',snapshot):
    code=pytest.main(['-vv','-ra','--tb=short','-p','no:cacheprovider',
                     '/src/python/sglang/kernels/aot/tests/test_gguf.py'])
print('GPU_PEAK_RESERVED_BYTES',torch.cuda.max_memory_reserved(),flush=True)
raise SystemExit(code)
