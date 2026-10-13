import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
repo = '/home/liuhao/sglang-worktrees/issue-35019'
main = '27b68303d043ddbddf9861c55c26bc8617ea7c35'
paths = {'srt': 'python/sglang/srt/layers/quantization/gguf.py',
         'cuda': 'python/sglang/kernels/aot/csrc/quantization/gguf/gguf_kernel.cu',
         'mmq': 'python/sglang/kernels/aot/csrc/quantization/gguf/mmq.cuh',
         'moe': 'python/sglang/kernels/aot/csrc/quantization/gguf/moe.cuh'}
data = {key: subprocess.check_output(['git','-C',repo,'show',main+':'+path],text=True)
        for key,path in paths.items()}
names = ('IQ1_S','IQ2_XXS','IQ2_XS','IQ2_S','IQ3_XXS','IQ3_S','IQ4_NL','IQ4_XS')
assert 'MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES' in data['srt']
assert 'ggml_supports_iq_mmq' not in data['cuda']
for key in ('mmq','moe'):
    assert all('mul_mat_'+name.lower() not in data[key] for name in names)
payload = {'status':'PASS','main_sha':main,
           'main_mmq_type_set':'STANDARD_QUANT_TYPES | KQUANT_TYPES',
           'main_has_iq_mmq_capability':False,
           'iq_formats':list(names),
           'source_blobs':{key:subprocess.check_output(['git','-C',repo,'rev-parse',main+':'+path],text=True).strip()
                           for key,path in paths.items()},
           'runtime_evidence':'published-moe-summary.json: capability=false; actual vector operator twice per case',
           'interpretation':'main and official 0.4.9 still lack the proposed IQ-MMQ capability; current numeric compatibility is not new performance evidence'}
root.joinpath('full-wheel/necessity-summary.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload),flush=True)
