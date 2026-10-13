import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.015)
torch.manual_seed(35019)
from sglang.kernels.jit.utils.arch import get_activation_cuda_cflags
from sglang.kernels.ops.activation import activation as act

ROOT = Path(os.environ['PROBE_RESULT_DIR'])
ROOT.mkdir(parents=True, exist_ok=True)


def emit(row):
    print(json.dumps(row, sort_keys=True), flush=True)


def compare(a, b, label, atol=0.01, rtol=0.01):
    af, bf = a.double(), b.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    assert finite, label
    delta = (af - bf).abs()
    bound = atol + rtol * bf.abs()
    ratio = float((delta / bound).max()) if a.numel() and (atol or rtol) else 0.0
    record = dict(label=label, finite=finite, count=a.numel(),
                  mismatches=int((a != b).sum()), max_abs=float(delta.max()) if a.numel() else 0.,
                  rmse=float(delta.square().mean().sqrt()) if a.numel() else 0.,
                  max_tolerance_ratio=ratio, atol=atol, rtol=rtol)
    emit(record)
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return record


emit({'kind': 'provenance', 'torch': torch.__version__, 'cuda': torch.version.cuda,
      'tvm_ffi': importlib.metadata.version('apache-tvm-ffi'),
      'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
      'activation_source': act.__file__, 'policy': get_activation_cuda_cflags(),
      'cache': os.environ['SGLANG_JIT_CACHE_DIR'], 'seed':35019})
assert get_activation_cuda_cflags() == ['--use_fast_math']
assert not list(Path(os.environ['SGLANG_JIT_CACHE_DIR']).rglob('*.so'))
records = []
for dtype in (torch.float16, torch.bfloat16):
    dt = str(dtype).split('.')[-1]
    # Enumerate every finite representable input, including subnormals, signed
    # zero, exponent extremes, and values near activation rounding boundaries.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(dtype)
    values = values[torch.isfinite(values)].float()
    padding = (-values.numel()) % 256
    gates = F.pad(values, (0, padding)).reshape(-1, 256)
    enum_x = torch.cat([gates, torch.ones_like(gates)], dim=-1).to(dtype).cuda()
    families = [('all_finite_representable', enum_x)]
    for rows, hidden, scale in [(1,16,1.),(127,256,1.),(128,2048,3.),(129,512,10.)]:
        families.append((f'random_{rows}x{hidden}_s{scale:g}',
                         torch.randn(rows, hidden*2, device='cuda', dtype=dtype)*scale))
    precise = act.activation_module(dtype, fast_math=False)
    fast = act.activation_module(dtype)
    for op in ('silu','gelu'):
        for family, x in families:
            width=x.shape[-1]//2
            out_precise=torch.empty(x.shape[0],width,device='cuda',dtype=dtype)
            precise.run_activation(x,out_precise,op)
            out_fast=act.run_activation(op,x,None)
            # Float64 evaluation avoids inherited half/bfloat16 rounding.
            lhs=x[:,:width].double()
            rhs=x[:,width:].double()
            reference=(F.silu(lhs) if op=='silu' else F.gelu(lhs,approximate='none'))*rhs
            reference=reference.to(dtype)
            key=f'{dt}:{op}:{family}'
            records.append(compare(out_fast,out_precise,key+':fast_vs_precise'))
            records.append(compare(out_fast,reference,key+':fast_vs_f64'))
            records.append(compare(out_precise,reference,key+':precise_vs_f64'))
            torch.save({'x':x.cpu(),'precise':out_precise.cpu(),'fast':out_fast.cpu()},ROOT/(key.replace(':','_')+'.pt'))
    # Graph compatibility on the real default activation wrapper.
    x=families[3][1]
    for op in ('silu','gelu'):
        expected=act.run_activation(op,x,None)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured=act.run_activation(op,x,None)
        graph.replay()
        compare(captured,expected,f'{dt}:{op}:cuda_graph',atol=0.,rtol=0.)
    del families,enum_x,precise,fast,x
    torch.cuda.empty_cache()
torch.cuda.synchronize()
libraries=[]
for p in sorted(Path(os.environ['SGLANG_JIT_CACHE_DIR']).rglob('*.so')):
    libraries.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
emit({'kind':'binaries','libraries':libraries})
emit({'kind':'result','status':'PASS','comparisons':len(records),
      'peak_allocated':torch.cuda.max_memory_allocated(),
      'peak_reserved':torch.cuda.max_memory_reserved()})
(ROOT/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
