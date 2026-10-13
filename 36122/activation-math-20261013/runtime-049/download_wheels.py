"""Download exact PyPI wheels to disk, including failed-transfer retries.

Resumes are deliberately avoided for TLS EOF: some CDN/proxy paths returned
repeatedly stalled Range responses. Hash validation prevents partial use.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path('/probe/wheels')
def run(url,path):
    subprocess.run(['curl','--proxy','http://127.0.0.1:17890','--fail','--silent','--show-error',
                    '--location','--retry','3','--retry-all-errors','--retry-delay','2',
                    '--max-time','180','--output',str(path),url],check=True)
manifest=[]
for spec in sys.argv[1:]:
    name,version=spec.split('==')
    metadata=root/(name+'-'+version+'-pypi.json')
    run(f'https://pypi.org/pypi/{name}/{version}/json',metadata)
    data=json.loads(metadata.read_text())
    candidates=[f for f in data['urls'] if f['filename'].endswith('.whl') and 'musllinux' not in f['filename'] and
                ('aarch64' in f['filename'] or 'py3-none-any' in f['filename']) and
                ('-cp312-' in f['filename'] or '-cp310-abi3-' in f['filename'] or '-py3-none-' in f['filename'])]
    assert len(candidates)==1,(spec,[f['filename'] for f in candidates])
    row=candidates[0]
    wheel=root/row['filename']
    valid=lambda:wheel.is_file() and wheel.stat().st_size==row['size'] and hashlib.file_digest(wheel.open('rb'),'sha256').hexdigest()==row['digests']['sha256']
    if not valid():
        run(row['url'],wheel)
    assert valid(),wheel
    record={'name':name,'version':version,'file':row['filename'],'url':row['url'],
            'bytes':row['size'],'sha256':row['digests']['sha256']}
    print(json.dumps(record),flush=True)
    manifest.append(record)
