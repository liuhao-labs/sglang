"""Published 0.4.9 production bindings, without any test operator replacement."""
import json
import os
from pathlib import Path
import numpy as np
import torch
import gguf

torch.cuda.set_per_process_memory_fraction(.025)
from sglang.srt.layers.quantization import gguf as g

assert not getattr(g,'_supports_iq_mmq',False)
role=os.environ['PROBE_ROLE']
qt=gguf.GGMLQuantizationType
names=('IQ1_S','IQ2_XXS','IQ2_XS','IQ2_S','IQ3_XXS','IQ3_S','IQ4_NL','IQ4_XS')
outputs=[]
for name in names:
    q=getattr(qt,name)
    tensor=gguf.GGUFReader(Path('/probe/fixtures/moe/2b77eb27ae2ea4b3f68cf3042961509a9f5847b8')/f'Quant_{name}_512.gguf').tensors
    block,packed=gguf.GGML_QUANT_SIZES[q]
    a=np.array(tensor[0].data[:8,:512,:256//block*packed],copy=True)
    b=np.array(tensor[1].data[:8,:256,:256//block*packed],copy=True)
    w1=torch.tensor(a,device='cuda'); w2=torch.tensor(b,device='cuda')
    for dtype in (torch.float16,torch.bfloat16):
        dw1=torch.tensor(gguf.dequantize(a,q),device='cuda',dtype=dtype)
        dw2=torch.tensor(gguf.dequantize(b,q),device='cuda',dtype=dtype)
        for op in ('silu','gelu'):
            torch.manual_seed(35019+int(q))
            x=torch.randn(128,256,device='cuda',dtype=dtype)*.25
            ids=torch.randint(0,8,(128,2),device='cuda',dtype=torch.int32)
            weights=torch.full((128,2),.5,device='cuda',dtype=dtype)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                out=g.fused_moe_gguf(x,w1,w2,weights,ids,int(q),int(q),op)
            events={e.key:e.count for e in prof.key_averages() if 'sgl_kernel::ggml_moe' in e.key or '_run_activation' in e.key}
            assert events.get('sgl_kernel::ggml_moe_a8_vec')==2,events
            assert events.get('sgl_kernel::ggml_moe_a8',0)==0,events
            gate=torch.einsum('mh,mknh->mkn',x,dw1[ids.long()])
            l,r=gate.chunk(2,dim=-1)
            activated=(torch.nn.functional.silu(l.float()) if op=='silu' else torch.nn.functional.gelu(l.float(),approximate='none'))*r.float()
            ref=(torch.einsum('mki,mkoi->mko',activated.to(dtype),dw2[ids.long()])*weights[:,:,None]).sum(1)
            assert torch.isfinite(out).all()
            torch.testing.assert_close(out,ref,atol=1.,rtol=.1)
            key=f'{name}_{dtype}_{op}'
            print(json.dumps({'role':role,'case':key,'status':'PASS','events':events,
                              'max_abs':float((out-ref).float().abs().max())}),flush=True)
            outputs.append({'case':key,'out':out.cpu()})
    del w1,w2,dw1,dw2
    torch.cuda.empty_cache()
torch.save(outputs,Path('/probe/runtime-049')/f'published-moe-{role}.pt')
print(json.dumps({'kind':'result','status':'PASS','cases':len(outputs),
                  'peak_reserved':torch.cuda.max_memory_reserved()}),flush=True)
