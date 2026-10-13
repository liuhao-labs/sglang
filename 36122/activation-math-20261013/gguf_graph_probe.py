import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import gguf

torch.cuda.set_per_process_memory_fraction(0.025)
torch.manual_seed(35019)
from sglang.srt.layers.quantization import gguf as g

role=os.environ['PROBE_ROLE']
native_path=Path('/probe')/f'gguf-build-{role}'/f'gguf_probe_{role}.so'
torch.ops.load_library(str(native_path))
native=torch.ops.gguf_probe
for name,target in [('ggml_moe_a8','moe'),('ggml_moe_a8_vec','moe_vec'),('ggml_moe_get_block_size','block_size')]:
    setattr(g,name,getattr(native,target).default)
if role=='integrated':
    g._supports_iq_mmq=bool(native.supports_iq_mmq.default())
    g.MMQ_QUANT_TYPES=g._build_mmq_quant_types(g._supports_iq_mmq)
q=gguf.GGMLQuantizationType.IQ3_S
file=Path('/probe/fixtures/moe/2b77eb27ae2ea4b3f68cf3042961509a9f5847b8/Quant_IQ3_S_512.gguf')
tensors=gguf.GGUFReader(file).tensors
block,packed=gguf.GGML_QUANT_SIZES[q]
w1=torch.tensor(np.array(tensors[0].data[:8,:512,:256//block*packed],copy=True),device='cuda')
w2=torch.tensor(np.array(tensors[1].data[:8,:256,:256//block*packed],copy=True),device='cuda')
for dtype in (torch.float16,torch.bfloat16):
    x=torch.randn(128,256,dtype=dtype,device='cuda')*.25
    ids=torch.randint(0,8,(128,2),dtype=torch.int32,device='cuda')
    routing=torch.full((128,2),.5,dtype=dtype,device='cuda')
    for op in ('silu','gelu'):
        fn=lambda:g.fused_moe_gguf(x,w1,w2,routing,ids,int(q),int(q),op)
        for _ in range(2):
            expected=fn()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured=fn()
        graph.replay()
        assert torch.isfinite(captured).all()
        torch.testing.assert_close(captured,expected,atol=0.,rtol=0.)
        print(json.dumps({'role':role,'op':op,'dtype':str(dtype),
                          'route':'mmq' if role=='integrated' else 'vector',
                          'status':'PASS','bit_equal':torch.equal(captured,expected)}),flush=True)
print(json.dumps({'kind':'graph_result','status':'PASS','cases':4,
                  'peak_reserved':torch.cuda.max_memory_reserved()}),flush=True)
