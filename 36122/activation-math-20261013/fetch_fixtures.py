import hashlib
import json
import time
from pathlib import Path
from huggingface_hub import snapshot_download

cache='/probe/hf'
iq=('IQ1_S','IQ2_XXS','IQ2_XS','IQ2_S','IQ3_XXS','IQ3_S','IQ4_NL','IQ4_XS','Q5_0')
for repo,revision,patterns in [
    ('Isotr0py/test-gguf-sample','d82b8773934ef260d8d8a896a7c197bc69a0fac1','*.gguf'),
    ('SzymonOzog/test-gguf-moe-sample','2b77eb27ae2ea4b3f68cf3042961509a9f5847b8',[f'Quant_{q}_512.gguf' for q in iq]),
]:
    for attempt in range(6):
        try:
            path=Path(snapshot_download(repo,revision=revision,allow_patterns=patterns,cache_dir=cache,max_workers=2))
            break
        except Exception as error:
            print(json.dumps({'repo':repo,'attempt':attempt+1,'error':repr(error)}),flush=True)
            if attempt==5:
                raise
            time.sleep(3)
    for file in sorted(path.glob('*.gguf')):
        h=hashlib.file_digest(file.open('rb'),'sha256').hexdigest()
        print(json.dumps({'repo':repo,'revision':revision,'file':file.name,'bytes':file.stat().st_size,'sha256':h}),flush=True)
