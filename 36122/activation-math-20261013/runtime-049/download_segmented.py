"""Bounded segmented exact-wheel downloader using explicit local proxy."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path('/probe/wheels')
def fetch(url,path,extra=()):
    subprocess.run(['curl','--proxy','http://127.0.0.1:17890','--fail','--silent','--show-error',
                    '--location','--retry','3','--retry-all-errors','--retry-delay','2',
                    '--max-time','180',*extra,'--output',str(path),url],check=True)
for spec in sys.argv[1:]:
    name,version=spec.split('==')
    metadata=root/(name+'-'+version+'-pypi.json')
    fetch(f'https://pypi.org/pypi/{name}/{version}/json',metadata)
    data=json.loads(metadata.read_text())
    rows=[f for f in data['urls'] if f['filename'].endswith('.whl') and ('aarch64' in f['filename'] or 'py3-none-any' in f['filename']) and
          ('-py3-none-' in f['filename'] or '-cp312-' in f['filename'] or '-cp310-abi3-' in f['filename'])]
    assert len(rows)==1,(spec,[f['filename'] for f in rows])
    row=rows[0]
    final=root/row['filename']
    chunks=root/(row['filename']+'.parts')
    chunks.mkdir(exist_ok=True)
    total=row['size']; stride=8*1024**2
    def one(start):
        stop=min(total,start+stride)-1
        part=chunks/f'{start:012d}'
        if not part.exists() or part.stat().st_size!=stop-start+1:
            fetch(row['url'],part,('--range',f'{start}-{stop}'))
        assert part.stat().st_size==stop-start+1,(part,part.stat().st_size,stop-start+1)
        print(json.dumps({'spec':spec,'start':start,'end':stop,'status':'chunk_verified'}),flush=True)
        return part
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts=list(pool.map(one,range(0,total,stride)))
    with final.open('wb') as stream:
        for p in parts:
            with p.open('rb') as source:
                while block:=source.read(1024**2):
                    stream.write(block)
    actual=hashlib.file_digest(final.open('rb'),'sha256').hexdigest()
    assert actual==row['digests']['sha256'],(actual,row['digests']['sha256'])
    print(json.dumps({'spec':spec,'file':str(final),'sha256':actual,'bytes':total,'status':'verified'}),flush=True)
