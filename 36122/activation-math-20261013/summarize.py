import hashlib
import json
from pathlib import Path
import torch

root=Path('/probe')
def entries(file):
    return [json.loads(l) for l in file.read_text().splitlines() if l.startswith('{')]
summary={}
for role in ('main','integrated'):
    a=entries(root/f'results-{role}/activation.log')
    c=entries(root/f'results-{role}/gguf-chain.log')
    summary[role]={'activation':{},'moe':{}}
    for dtype in ('float16','bfloat16'):
        for op in ('silu','gelu'):
            rows=[v for v in a if v.get('label','').startswith(dtype+':'+op+':') and v['label'].endswith('fast_vs_precise')]
            summary[role]['activation'][dtype+':'+op]={'max_abs':max(v['max_abs'] for v in rows),
                'max_tolerance_ratio':max(v['max_tolerance_ratio'] for v in rows),
                'total_mismatches':sum(v['mismatches'] for v in rows),'comparisons':len(rows)}
    case=[v for v in c if v.get('kind')=='moe_case']
    for target in ('fast_vs_precise','fast_vs_dequant_reference'):
        rows=[d for v in case for d in v['comparisons'] if d['comparison'].endswith(target)]
        summary[role]['moe'][target]={'max_abs':max(v['max_abs'] for v in rows),
            'max_tolerance_ratio':max(v['max_tolerance_ratio'] for v in rows),'cases':len(rows)}
    summary[role]['activation_result']=[v for v in a if v.get('kind')=='result'][-1]
    summary[role]['moe_result']=[v for v in c if v.get('kind')=='result'][-1]
cross=[]
for file in sorted((root/'results-main').glob('*.pt')):
    counterpart=root/'results-integrated'/file.name
    a=torch.load(file,map_location='cpu',weights_only=True)
    b=torch.load(counterpart,map_location='cpu',weights_only=True)
    if 'x' in a:
        assert torch.equal(a['x'],b['x']),file.name
        assert torch.equal(a['fast'],b['fast']),file.name
        assert torch.equal(a['precise'],b['precise']),file.name
    else:
        torch.testing.assert_close(a['fast'],b['fast'],atol=.01,rtol=.01)
    delta=(a['fast'].float()-b['fast'].float()).abs()
    cross.append({'file':file.name,'bit_equal':torch.equal(a['fast'],b['fast']),
                  'max_abs':float(delta.max())})
summary['cross_main_pr']={'cases':len(cross),'bit_equal':sum(v['bit_equal'] for v in cross),
                          'max_abs':max(v['max_abs'] for v in cross)}
(root/'numeric-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'cross-main-pr.json').write_text(json.dumps(cross,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
