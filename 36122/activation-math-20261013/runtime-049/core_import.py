from pathlib import Path
import hashlib
import importlib.metadata as meta
import torch
print('Torch',torch.__version__,torch.version.cuda,torch.__file__,flush=True)
assert torch.__version__.split('+')[0]=='2.14.1'
assert Path(torch.__file__).is_relative_to(Path('/probe/runtime-049/probe-site'))
import sgl_kernel
assert meta.version('sglang-kernel')=='0.4.9'
binary=Path(sgl_kernel.common_ops.__file__)
assert binary.is_relative_to(Path('/probe/runtime-049/probe-site'))
print('Kernel',meta.version('sglang-kernel'),binary,
      hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),flush=True)
print('PUBLISHED_KERNEL_IMPORT_PASS',flush=True)
