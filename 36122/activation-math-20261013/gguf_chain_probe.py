"""Source-level integration probe with fresh GGUF kernels in a test namespace.

The production fused_moe_gguf function and shared default activation are used.
Only GGUF operator bindings are redirected to freshly built isolated binaries;
no installed operator is replaced, and no numerical kernel/reference is mocked.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from unittest.mock import patch

import gguf
import numpy as np
import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.025)
torch.manual_seed(35019)
from sglang.kernels.ops.activation import activation as act
from sglang.srt.layers.quantization import gguf as g
import sgl_kernel

role=os.environ['PROBE_ROLE']
result=Path('/probe')/f'results-{role}'
library=Path('/probe')/f'gguf-build-{role}'/f'gguf_probe_{role}.so'
torch.ops.load_library(str(library))
native=torch.ops.gguf_probe
for name,target in [('ggml_dequantize','dequantize'),('ggml_mul_mat_a8','mul_mat'),
                    ('ggml_mul_mat_vec_a8','mul_mat_vec'),('ggml_moe_a8','moe'),
                    ('ggml_moe_a8_vec','moe_vec'),('ggml_moe_get_block_size','block_size')]:
    setattr(g,name,getattr(native,target).default)
if role=='integrated':
    g._supports_iq_mmq=bool(native.supports_iq_mmq.default())
    g.MMQ_QUANT_TYPES=g._build_mmq_quant_types(g._supports_iq_mmq)

def emit(row):
    print(json.dumps(row,sort_keys=True),flush=True)

emit({'kind':'provenance','role':role,'gguf_source':g.__file__,
      'native_library':str(library),'native_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
      'torch':torch.__version__,'cuda':torch.version.cuda,
      'installed_kernel_version':importlib.metadata.version('sglang-kernel'),
      'installed_common_ops':sgl_kernel.common_ops.__file__,
      'installed_common_ops_sha256':hashlib.sha256(Path(sgl_kernel.common_ops.__file__).read_bytes()).hexdigest(),
      'binding':'fresh native test namespace; production wrapper and default shared JIT activation',
      'iq3_s_block_size':int(native.block_size.default(int(gguf.GGMLQuantizationType.IQ3_S)))})

QT=gguf.GGMLQuantizationType
types=[QT.IQ1_S,QT.IQ2_XXS,QT.IQ2_XS,QT.IQ2_S,QT.IQ3_XXS,QT.IQ3_S,QT.IQ4_NL,QT.IQ4_XS]
cached={}
def weights(q):
    if q in cached:
        return cached[q]
    file=Path('/probe/fixtures/moe/2b77eb27ae2ea4b3f68cf3042961509a9f5847b8')/f'Quant_{q.name}_512.gguf'
    tensors=gguf.GGUFReader(file).tensors
    block,packed=gguf.GGML_QUANT_SIZES[q]
    # Fixed submatrix of the real pinned fixture: E=8, H=256, I=256.
    w1_np=np.array(tensors[0].data[:8,:512,:256//block*packed],copy=True)
    w2_np=np.array(tensors[1].data[:8,:256,:256//block*packed],copy=True)
    cached[q]=(w1_np,w2_np,gguf.dequantize(w1_np,q),gguf.dequantize(w2_np,q))
    return cached[q]

def close(a,b,label,atol,rtol):
    assert torch.isfinite(a).all() and torch.isfinite(b).all(),label
    delta=(a.float()-b.float()).abs()
    ratio=float((delta/(atol+rtol*b.float().abs())).max()) if (atol or rtol) else 0.
    record={'comparison':label,'max_abs':float(delta.max()),'rmse':float(delta.square().mean().sqrt()),
            'max_tolerance_ratio':ratio,'atol':atol,'rtol':rtol,'finite':True}
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    return record

pairs=[(q,q) for q in types]+[(QT.IQ3_S,QT.Q5_0),(QT.Q5_0,QT.IQ3_S)]
cases=0
for q1,q2 in pairs:
    w1_np,_,w1_ref,_=weights(q1)
    _,w2_np,_,w2_ref=weights(q2)
    w1=torch.tensor(w1_np,device='cuda')
    w2=torch.tensor(w2_np,device='cuda')
    for dtype in (torch.float16,torch.bfloat16):
        dw1=torch.tensor(w1_ref,device='cuda',dtype=dtype)
        dw2=torch.tensor(w2_ref,device='cuda',dtype=dtype)
        precise=act.activation_module(dtype,fast_math=False)
        def precise_activation(input,op):
            width=input.shape[-1]//2
            out=torch.empty((*input.shape[:-1],width),dtype=dtype,device='cuda')
            precise.run_activation(input.view(-1,2*width),out.view(-1,width),op)
            return out
        for op in ('silu','gelu'):
            for tokens in (1,127,128,129):
                torch.manual_seed(35019+int(q1)*100+int(q2)+tokens)
                x=torch.randn(tokens,256,dtype=dtype,device='cuda')*0.25
                ids=torch.randint(0,8,(tokens,2),dtype=torch.int32,device='cuda')
                routing=torch.softmax(torch.randn(tokens,2,device='cuda'),dim=-1).to(dtype)
                args=(x,w1,w2,routing,ids,int(q1),int(q2),op)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    fast=g.fused_moe_gguf(*args)
                events={e.key:e.count for e in prof.key_averages() if 'gguf_probe::' in e.key or 'activation' in e.key}
                wanted='gguf_probe::moe' if role=='integrated' and tokens>=128 else 'gguf_probe::moe_vec'
                assert events.get(wanted)==2,(role,op,tokens,events)
                with patch.object(g,'silu_and_mul',lambda a:precise_activation(a,'silu')), \
                     patch.object(g,'gelu_and_mul',lambda a:precise_activation(a,'gelu')):
                    exact=g.fused_moe_gguf(*args)
                gate=torch.einsum('mh,mknh->mkn',x,dw1[ids.long()])
                left,right=gate.chunk(2,dim=-1)
                active=(F.silu(left.float()) if op=='silu' else F.gelu(left.float(),approximate='none'))*right.float()
                reference=torch.einsum('mki,mkoi->mko',active.to(dtype),dw2[ids.long()])
                reference=(reference*routing[:,:,None]).sum(dim=1)
                label=f'{q1.name}_{q2.name}_{dtype}_{op}_M{tokens}'
                comparison=[close(fast,exact,label+':fast_vs_precise',0.01,0.01),
                            close(fast,reference,label+':fast_vs_dequant_reference',1.,0.1)]
                emit({'kind':'moe_case','label':label,'events':events,'comparisons':comparison})
                torch.save({'fast':fast.cpu(),'precise':exact.cpu()},result/(label.replace('torch.','')+'.pt'))
                cases+=1
        del dw1,dw2,precise
    del w1,w2
    torch.cuda.empty_cache()
emit({'kind':'result','status':'PASS','cases':cases,
      'peak_reserved':torch.cuda.max_memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated()})
