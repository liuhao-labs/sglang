import json
import subprocess
from pathlib import Path

repo = '/home/liuhao/sglang-worktrees/issue-35019'
upstream = '7ff3a112846a3fcc12a27bba09d644885a226e4d'
latest = '27b68303d043ddbddf9861c55c26bc8617ea7c35'
head = '0a39ec3a8754e3c4426950a5a93046c048a03f07'
base = '6e7beace143964386ce2e4418401b7369f0dabd0'
def git(*args):
    return subprocess.check_output(['git', '-C', repo, *args], text=True).strip()

pr_paths = git('diff', '--name-only', base, head).splitlines()
activation_paths = git('ls-tree', '-r', '--name-only', upstream,
                      'python/sglang/kernels/ops/activation',
                      'python/sglang/kernels/jit/utils').splitlines()
activation_paths += git('ls-tree', '-r', '--name-only', upstream,
                        'python/sglang/kernels/jit/include').splitlines()
paths = sorted(set(pr_paths + activation_paths + [
    'python/pyproject.toml', 'python/sglang/kernels/aot/CMakeLists.txt',
    'python/sglang/kernels/aot/pyproject.toml',
    'python/sglang/kernels/aot/cmake/flashmla.cmake']))
rows = []
for path in paths:
    def blob(sha):
        p = subprocess.run(['git', '-C', repo, 'rev-parse', f'{sha}:{path}'],
                           capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    a, b = blob(upstream), blob(latest)
    rows.append({'path': path, 'frozen_blob': a, 'latest_blob': b, 'unchanged': a == b})
changes = git('diff', '--name-only', upstream, latest).splitlines()
payload = {'head': head, 'base': base, 'frozen_main': upstream,
           'latest_main': latest, 'changed_paths': changes,
           'latest_integration_sha': '449c32e493e80e67190fcf1f61976dc6045159e9',
           'latest_integration_tree': 'bb5c83cc334e80c51e6a0eb44045a5358c189162',
           'latest_integration_aot_tree': git('rev-parse', '449c32e493e80e67190fcf1f61976dc6045159e9:python/sglang/kernels/aot'),
           'pr_path_intersection': sorted(set(pr_paths) & set(changes)),
           'core_changed': [row for row in rows if not row['unchanged']],
           'blob_comparison': rows,
           'upstream_commits': git('log', '--oneline', f'{upstream}..{latest}').splitlines(),
           'status': git('status', '--porcelain')}
Path(__file__).with_name('latest-drift.json').write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps({k: v for k, v in payload.items() if k not in
                  ('blob_comparison', 'changed_paths')}, indent=2))
