import json
from pathlib import Path

import torch

root = Path('/probe/runtime-049')
for suffix in ('', '-latest'):
    roles = ('main'+suffix, 'integrated'+suffix)
    if not all(root.joinpath(f'published-moe-{role}.pt').exists() for role in roles):
        continue
    main, integrated = [torch.load(root/f'published-moe-{role}.pt', weights_only=True)
                        for role in roles]
    assert len(main) == len(integrated) == 32
    assert all(a['case'] == b['case'] and torch.equal(a['out'], b['out'])
               for a, b in zip(main, integrated))
    logs = [json.loads(line) for role in roles
            for line in root.joinpath(f'published-moe-{role}.log').read_text().splitlines()
            if line.startswith('{')]
    cases = [row for row in logs if 'case' in row]
    assert len(cases) == 64 and all(row['status'] == 'PASS' for row in cases)
    assert all(row['events'].get('sgl_kernel::ggml_moe_a8_vec') == 2 for row in cases)
    payload = {'status': 'PASS', 'roles': roles, 'paired_cases': 32, 'all_bit_identical': True,
               'max_reference_abs': max(row['max_abs'] for row in cases),
               'max_reserved_bytes': max(row['peak_reserved'] for row in logs
                                         if row.get('kind') == 'result')}
    root.joinpath(f'published-moe-summary{suffix}.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload), flush=True)
