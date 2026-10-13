"""Verify official 0.4.9 binary compatibility against the cached Torch 2.11.

Imports only, no fake metadata/version bypass and no CUDA tensor allocation.
"""
from pathlib import Path
import hashlib
import torch
print('Torch',torch.__version__,torch.version.cuda,flush=True)
import sgl_kernel
print('sgl_kernel',sgl_kernel.__file__,flush=True)
print('common_ops',sgl_kernel.common_ops.__file__,flush=True)
binary=Path(sgl_kernel.common_ops.__file__)
print('SHA256',hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),flush=True)
