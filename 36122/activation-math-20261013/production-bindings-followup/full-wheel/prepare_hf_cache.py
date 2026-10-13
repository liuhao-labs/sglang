"""Create an offline HF snapshot cache from the already hash-verified fixtures."""
import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pairs = [('dense', 'Isotr0py/test-gguf-sample', 'd82b8773934ef260d8d8a896a7c197bc69a0fac1', 34),
         ('moe', 'SzymonOzog/test-gguf-moe-sample', '2b77eb27ae2ea4b3f68cf3042961509a9f5847b8', 9)]
rows = []
for kind, repo, revision, count in pairs:
    fixtures = root/'fixtures'/kind/revision
    metadata = json.loads(fixtures.joinpath('repo.json').read_text())
    indexed = {s['rfilename']: s for s in metadata['siblings']}
    destination = root/'full-wheel/hf'/('models--'+repo.replace('/', '--'))/'snapshots'/revision
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(fixtures.glob('*.gguf'))
    assert len(files) == count
    for file in files:
        expected = indexed[file.name]['lfs']['sha256']
        actual = hashlib.file_digest(file.open('rb'), 'sha256').hexdigest()
        assert actual == expected, file
        output = destination/file.name
        if not output.exists():
            os.link(file, output)
        rows.append({'repo': repo, 'revision': revision, 'file': file.name,
                     'sha256': actual, 'size': file.stat().st_size})
root.joinpath('full-wheel/hf-manifest.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps({'verified_fixture_count': len(rows), 'status': 'PASS'}), flush=True)
