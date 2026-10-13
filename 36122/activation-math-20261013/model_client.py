import json
import os
from pathlib import Path
import time
import requests

result=Path('/probe')/f'results-{os.environ["PROBE_ROLE"]}'
for _ in range(120):
    try:
        r=requests.get('http://127.0.0.1:18080/health',timeout=2)
        if r.status_code==200:
            break
    except requests.RequestException:
        pass
    log=(result/'model-server.log').read_text()
    if 'Traceback (most recent call last)' in log:
        raise RuntimeError('server failed during startup; see model-server.log')
    time.sleep(2)
else:
    raise RuntimeError('isolated model did not become healthy in 240s')
r=requests.post('http://127.0.0.1:18080/generate',json={
    'text':'Tell me one reason to work outdoors.',
    'sampling_params':{'temperature':0,'max_new_tokens':16},
},timeout=60)
r.raise_for_status()
payload=r.json()
(result/'model-client.json').write_text(json.dumps(payload,indent=2)+'\n')
assert payload['meta_info']['completion_tokens']>0,payload
print(json.dumps({'status':'PASS','health':200,'generate':r.status_code,'response':payload}),flush=True)
