import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
experiments = root/'full-wheel'
repo = '/home/liuhao/sglang-worktrees/issue-35019'
main = '27b68303d043ddbddf9861c55c26bc8617ea7c35'
integration = '449c32e493e80e67190fcf1f61976dc6045159e9'
prefix = 'python/sglang/kernels/aot/'
units = ['csrc/expert_specialization/es_fp8_blockwise.cu', 'csrc/gemm/fp8_gemm_kernel.cu']
blobs = []
for unit in units:
    a = subprocess.check_output(['git','-C',repo,'rev-parse',main+':'+prefix+unit],text=True).strip()
    b = subprocess.check_output(['git','-C',repo,'rev-parse',integration+':'+prefix+unit],text=True).strip()
    assert a == b
    blobs.append({'path': prefix+unit, 'main_blob': a, 'integration_blob': b})
attempts = [
    {'log':'build-attempt-1.log', 'exit':'build-exit-1.json', 'result':'configuration failure: unextracted CUTLASS submodule; corrected'},
    {'log':'build-attempt-2.log', 'exit':'build-exit-2.json', 'result':'stopped experiment: stale Torch library CMake cache; corrected with clean build'},
    {'log':'build-attempt-3-memory10g.log', 'exit':'build-exit-3.json', 'result':'default full build, 10 GiB cgroup OOM, es_fp8_blockwise.cu'},
    {'log':'build-attempt-4-memory14g.log', 'exit':'build-exit-4.json', 'result':'default full build, 14 GiB cgroup OOM, fp8_gemm_kernel.cu'},
    {'log':'build-attempt.log', 'exit':'build-exit.json', 'result':'all-source/all-default-target build with SM121a-only common_ops, 14 GiB cgroup OOM, fp8_gemm_kernel.cu'}]
for attempt in attempts:
    attempt['exit_codes'] = json.loads(experiments.joinpath(attempt['exit']).read_text())
    assert experiments.joinpath(attempt['log']).is_file()
oom = [json.loads(line) for line in experiments.joinpath('oom-events.jsonl').read_text().splitlines()]
assert len(oom) == 3 and all(row['Action']=='oom' for row in oom)
assert not list(experiments.joinpath('dist').glob('*.whl'))
payload = {'status':'RESOURCE_BLOCKED', 'torch':'2.14.1+cu130', 'cuda_toolkit':'13.0.88',
           'full_wheel_produced':False, 'capability_true_full_wheel_validation':False,
           'planned_80_enabled_production_cases_executed':False,
           'planned_153_full_wheel_aot_cases_executed':False,
           'build_workers':1, 'nvcc_threads':1, 'max_container_memory_gib':14,
           'attempts':attempts, 'oom_event_count':len(oom), 'unchanged_failing_source_units':blobs,
           'reason_no_further_memory_increase':'protect live service on shared unified-memory device; initial available memory about 27.8 GiB',
           'scope':'build resource failures; no reproduced GGUF numerical/source regression'}
experiments.joinpath('failure-summary.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({k:v for k,v in payload.items() if k not in ('attempts','unchanged_failing_source_units')},indent=2),flush=True)
