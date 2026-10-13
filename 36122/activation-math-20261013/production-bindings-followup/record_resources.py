import datetime
import json
import subprocess
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
mem = {line.split(':')[0]: int(line.split()[1])*1024
       for line in Path('/proc/meminfo').read_text().splitlines()
       if line.split(':')[0] in ('MemAvailable', 'SwapFree', 'SwapTotal')}
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu,utilization.memory',
                               '--format=csv,noheader'], text=True).strip()
stats = subprocess.check_output(['docker', 'stats', '--no-stream', '--format',
                                 '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'], text=True).strip()
direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    health = direct.open('http://127.0.0.1:8101/health', timeout=10).status
except Exception as error:
    health = repr(error)
payload = {'time': datetime.datetime.now().astimezone().isoformat(),
           'memory': mem, 'gpu': gpu, 'containers': stats, 'existing_service_health': health}
with root.joinpath('resource-samples.jsonl').open('a') as handle:
    handle.write(json.dumps(payload)+'\n')
print(json.dumps(payload), flush=True)
