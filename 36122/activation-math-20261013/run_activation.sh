#!/bin/bash
set -euo pipefail
role=${1:?main or integrated}
probe_dir=/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj
mkdir -p "$probe_dir/results-$role" "$probe_dir/cache-$role"
awk '/MemAvailable/{if ($2<24*1024*1024) exit 1}' /proc/meminfo
docker run --rm --name "pr36122-math-$role" --network none --read-only --gpus device=0 \
  --user 1004:1004 --cpus 2 --memory 8g --memory-swap 8g --pids-limit 256 \
  --cap-drop ALL --security-opt no-new-privileges --shm-size 512m \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,uid=1004,gid=1004 \
  --mount "type=bind,src=$probe_dir,dst=/probe" \
  --mount "type=bind,src=$probe_dir/$role,dst=/src,readonly" \
  -e PYTHONPATH=/src/python -e PYTHONDONTWRITEBYTECODE=1 \
  -e "SGLANG_CACHE_DIR=/probe/cache-$role" \
  -e "SGLANG_JIT_CACHE_DIR=/probe/cache-$role/jit" \
  -e "TORCH_EXTENSIONS_DIR=/probe/cache-$role/torch" \
  -e HOME=/tmp -e CUDA_HOME=/probe/toolkit -e MAX_JOBS=1 \
  -e "NVCC_RECORD=/probe/results-$role/nvcc.log" \
  -e "PROBE_RESULT_DIR=/probe/results-$role" \
  --entrypoint /bin/bash quay.io/gpustack/runner:cuda13.0-sglang0.5.15.post1 \
  -c 'exec nice -n 10 timeout --signal=TERM --kill-after=20s 20m python3 -u /probe/activation_probe.py' \
  2>&1 | tee "$probe_dir/results-$role/activation.log"
