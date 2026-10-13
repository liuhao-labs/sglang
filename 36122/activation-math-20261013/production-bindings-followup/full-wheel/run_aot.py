import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

os.environ['HF_HUB_CACHE'] = '/probe/full-wheel/hf'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/probe/full-wheel/hf'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
torch.cuda.set_per_process_memory_fraction(.035)
import sgl_kernel
import pytest

binary = Path(sgl_kernel.common_ops.__file__).resolve()
assert binary.is_relative_to(Path('/probe/full-wheel/site')), binary
assert sgl_kernel.ggml_supports_iq_mmq() is True
print(json.dumps({'kind': 'provenance', 'torch': torch.__version__,
                  'kernel': metadata.version('sglang-kernel'), 'common_ops': str(binary),
                  'common_ops_sha256': hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
                  'capability': True, 'namespace': 'sgl_kernel (full PR wheel)',
                  'operator_replacements': False, 'fixture_download_replacement': False}), flush=True)
code = pytest.main(['-x', '-vv', '-ra', '--tb=short', '-p', 'no:cacheprovider',
                    '/src/python/sglang/kernels/aot/tests/test_gguf.py'])
print(json.dumps({'kind': 'result', 'pytest_exit_code': code,
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()}), flush=True)
raise SystemExit(code)
