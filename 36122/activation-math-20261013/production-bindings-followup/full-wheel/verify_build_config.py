import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
commands = json.loads(root.joinpath('build-gb10/compile_commands.json').read_text())
common = [row for row in commands if 'common_ops_' in row['output']]
assert len(common) == 74, len(common)
cuda = [row for row in common if row['file'].endswith('.cu')]
assert len(cuda) == 70
for row in cuda:
    assert re.findall(r'-gencode=arch=([^ ]+)', row['command']) == ['compute_121a,code=sm_121a']
    assert '/probe/runtime-049/probe-site/torch/include' in row['command']
    assert '/usr/local/lib/python3.12/dist-packages/torch' not in row['command']
    if 'common_ops_sm90_build' in row['output']:
        assert '-use_fast_math' in row['command']
    else:
        assert '-use_fast_math' not in row['command']
gguf = [row for row in cuda if row['file'].endswith('/gguf_kernel.cu')]
assert len(gguf) == 2
payload = {'status':'PASS', 'common_translation_units':len(common),
           'common_cuda_translation_units':len(cuda), 'common_architecture':'SM121a only',
           'all_targets_translation_units':len(commands), 'torch':'2.14.1+cu130',
           'gguf_compile_commands':gguf,
           'scope':'all default targets; architecture restriction only on common_ops'}
root.joinpath('config-summary.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({k:v for k,v in payload.items() if k!='gguf_compile_commands'}),flush=True)
