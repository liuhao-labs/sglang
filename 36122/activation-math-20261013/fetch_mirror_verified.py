import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

root=Path('/probe/fixtures')
root.mkdir(exist_ok=True)
def curl(url,path):
    subprocess.run(['curl','--proxy','http://127.0.0.1:17890','--http1.1',
                    '--fail','--silent','--show-error','--location',
                    '--retry','5','--retry-all-errors','--retry-delay','2','--max-time','180',
                    '--output',str(path),url],check=True)

types=('IQ1_M','IQ1_S','IQ2_XXS','IQ2_S','IQ2_XS','IQ3_S','IQ3_XXS','IQ4_NL','IQ4_XS',
       'Q2_K','Q3_K','Q4_K','Q5_K','Q6_K','Q4_0','Q5_0','Q8_0')
moe=('IQ1_S','IQ2_XXS','IQ2_XS','IQ2_S','IQ3_XXS','IQ3_S','IQ4_NL','IQ4_XS','Q5_0')
for kind,repo,revision,names in [
    ('moe','SzymonOzog/test-gguf-moe-sample','2b77eb27ae2ea4b3f68cf3042961509a9f5847b8',
     [f'Quant_{q}_512.gguf' for q in moe]),
    ('dense','Isotr0py/test-gguf-sample','d82b8773934ef260d8d8a896a7c197bc69a0fac1',
     [f'Quant_{q}_{h}.gguf' for q in types for h in (256,1024)]),
]:
    d=root/kind/revision
    d.mkdir(parents=True,exist_ok=True)
    metadata=d/'repo.json'
    curl(f'https://hf-mirror.com/api/models/{repo}/revision/{revision}?blobs=true',metadata)
    data=json.loads(metadata.read_text())
    assert data['sha']==revision,(data['sha'],revision)
    expected={f['rfilename']:f for f in data['siblings']}
    def one(name):
        f=expected[name]; file=d/name
        sha=f['lfs']['sha256']; size=f['lfs']['size']
        if not file.exists() or file.stat().st_size!=size or hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=sha:
            curl(f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}',file)
        actual=hashlib.file_digest(file.open('rb'),'sha256').hexdigest()
        assert actual==sha,(name,sha,actual)
        assert file.stat().st_size==size,(name,size,file.stat().st_size)
        print(json.dumps({'kind':kind,'repo':repo,'revision':revision,'filename':name,
                          'sha256':actual,'size':size,'verified_against_lfs':True}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one,names))
print('PINNED_FIXTURES_VERIFIED',flush=True)
