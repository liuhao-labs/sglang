import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root/'full-wheel/source'
prefix = 'python/sglang/kernels/aot/'
latest = root/'integrated-latest'
paths = subprocess.check_output(['git','-C',str(latest),'ls-tree','-r','--name-only','HEAD',prefix],text=True).splitlines()
rows = []
for path in paths:
    if not (latest/path).is_file():
        continue
    relative = path.removeprefix(prefix)
    staged = source/relative
    assert staged.is_file(), relative
    actual = hashlib.file_digest(staged.open('rb'), 'sha256').hexdigest()
    expected = hashlib.file_digest((latest/path).open('rb'), 'sha256').hexdigest()
    assert actual == expected, relative
    rows.append({'path': relative, 'sha256': actual})
payload = {'status':'PASS','tracked_files':len(rows),
           'latest_integration_sha':subprocess.check_output(['git','-C',str(latest),'rev-parse','HEAD'],text=True).strip(),
           'aot_tree':subprocess.check_output(['git','-C',str(latest),'rev-parse','HEAD:python/sglang/kernels/aot'],text=True).strip(),
           'files':rows}
root.joinpath('full-wheel/source-equivalence.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({k:v for k,v in payload.items() if k!='files'}),flush=True)
