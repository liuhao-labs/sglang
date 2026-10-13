import sys
import torch
import pytest
torch.cuda.set_per_process_memory_fraction(0.02)
raise SystemExit(pytest.main([
    '-vv','-ra','--tb=short','-p','no:cacheprovider',
    '/src/test/registered/kernels/ops/activation/test_activation.py',
    '-k','test_activation_correctness or test_activation_out_param',
]))
