import hashlib
import importlib.metadata as meta
import json
from pathlib import Path
import subprocess
import zipfile
import torch

root=Path('/probe')
def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
wheel=root/'wheels/sglang_kernel-0.4.4-cp310-abi3-manylinux2014_aarch64.whl'
assert sha(wheel)=='727e4bc53abeade20260186f99199200320b9fa51f8de7af90c01524cff73e5d'
with zipfile.ZipFile(wheel) as z:
    names=[n for n in z.namelist() if n.endswith('sm100/common_ops.abi3.so')]
    assert len(names)==1,names
    wheel_binary_sha=hashlib.sha256(z.read(names[0])).hexdigest()
    assert wheel_binary_sha=='672a1cb425d2d2815b8138c34fd8a1e89603bac9a8986dc9db7883e04be535d3'
source_files=['python/sglang/kernels/jit/utils/arch.py',
              'python/sglang/kernels/ops/activation/activation.py',
              'python/sglang/kernels/jit/csrc/elementwise/activation.cuh',
              'python/sglang/srt/layers/quantization/gguf.py',
              'python/sglang/kernels/aot/csrc/quantization/gguf/gguf_kernel.cu']
payload={'torch':torch.__version__,'cuda':torch.version.cuda,
         'python':subprocess.check_output(['python3','--version'],text=True).strip(),
         'nvcc':subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True).strip(),
         'packages':{k:meta.version(k) for k in ['apache-tvm-ffi','sglang-kernel','gguf','pytest','transformers']},
         'wheel':{'file':wheel.name,'sha256':sha(wheel),'installed_common_ops_matches_wheel':True,
                  'sm100_common_ops_sha256':wheel_binary_sha},
         'sources':{role:{f:sha(root/role/f) for f in source_files} for role in ('main','integrated')},
         'test_kernel_binary_hashes':{role:sha(root/f'gguf-build-{role}'/f'gguf_probe_{role}.so') for role in ('main','integrated')}}
(root/'provenance.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2),flush=True)
