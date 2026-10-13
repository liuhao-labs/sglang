import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

role=os.environ['PROBE_ROLE']
build_dir=Path('/probe')/f'gguf-build-{role}'
build_dir.mkdir(exist_ok=True)
cuda_flags=['-O3','-std=c++20','--threads=1','-gencode=arch=compute_121a,code=sm_121a']
if role=='integrated':
    cflags=['-O0','-std=c++20','-DPROBE_PR=1']
else:
    cflags=['-O0','-std=c++20']
path=load(name=f'gguf_probe_{role}',
          sources=['/probe/gguf_bindings.cc','/src/python/sglang/kernels/aot/csrc/quantization/gguf/gguf_kernel.cu'],
          build_directory=str(build_dir),
          extra_include_paths=['/src/python/sglang/kernels/aot/include'],
          extra_cflags=cflags,extra_cuda_cflags=cuda_flags,
          is_python_module=False,verbose=True)
print(json.dumps({'role':role,'torch':torch.__version__,'cuda':torch.version.cuda,
                  'library':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  'flags':cuda_flags,'kind':'focused_source_build'},sort_keys=True),flush=True)
