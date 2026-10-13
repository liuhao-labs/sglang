import hashlib
import importlib.metadata as meta
import json
import os
from pathlib import Path

import torch
assert torch.__version__.split('+')[0]=='2.14.1',torch.__version__
import sgl_kernel
from sglang.kernels.jit.utils.arch import get_activation_cuda_cflags
from sglang.srt.layers.quantization import gguf as gguf_layer

root=Path('/probe/runtime-049')
binary=Path(sgl_kernel.common_ops.__file__).resolve()
assert binary.is_relative_to(root/'probe-site'),binary
assert Path(torch.__file__).resolve().is_relative_to(root/'probe-site'),torch.__file__
assert meta.version('sglang-kernel')=='0.4.9'
assert meta.version('transformers')=='5.19.0'
payload={'role':os.environ['PROBE_ROLE'],'torch':torch.__version__,'torch_file':torch.__file__,
         'cuda':torch.version.cuda,'kernel_version':meta.version('sglang-kernel'),
         'common_ops':str(binary),'common_ops_sha256':hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),
         'sglang_source':__import__('sglang').__file__,'gguf_source':gguf_layer.__file__,
         'transformers_version':meta.version('transformers'),
         'packages':{p:meta.version(p) for p in ['torch','torchvision','sglang-kernel','flashinfer-python','transformers',
                                              'xgrammar','tokenizers','safetensors','huggingface-hub','triton',
                                              'compressed-tensors','nvidia-nccl-cu13','nvidia-cudnn-cu13',
                                              'nvidia-cutlass-dsl','nvidia-cutlass-dsl-libs-base','nvidia-cutlass-dsl-libs-cu13']},
         'version_check_disable_env':{p:os.environ.get(p) for p in ['FLASHINFER_DISABLE_VERSION_CHECK','SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK']},
         'capability':getattr(gguf_layer,'_supports_iq_mmq',False),
         'activation_policy':get_activation_cuda_cflags(),
         'bindings':'unmodified published wheel and production SRT (no test namespace)',
         'status':'IMPORT_PASS'}
print(json.dumps(payload,sort_keys=True),flush=True)
root.joinpath('import-'+os.environ['PROBE_ROLE']+'.json').write_text(json.dumps(payload,indent=2)+'\n')
