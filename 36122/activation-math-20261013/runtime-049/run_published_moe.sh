#!/bin/bash
set -euo pipefail
role=${1:?main or integrated}
probe_dir=/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj
awk '/MemAvailable/{if ($2<24*1024*1024) exit 1}' /proc/meminfo
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if (( util>10 )); then exit 20; fi
mkdir -p "$probe_dir/runtime-049/published-cache-$role"
docker run --rm --name "pr36122-published-moe-$role" --network none --read-only --gpus device=0 \
  --user 1004:1004 --cpus 1 --memory 8g --memory-swap 8g --pids-limit 256 \
  --cap-drop ALL --security-opt no-new-privileges --shm-size 512m \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,uid=1004,gid=1004 \
  --mount "type=bind,src=$probe_dir,dst=/probe" \
  --mount "type=bind,src=$probe_dir/$role,dst=/src,readonly" \
  -e PYTHONPATH=/probe/runtime-049/triton-site:/probe/runtime-049/cutlass-site/nvidia_cutlass_dsl/dsl_packages:/probe/runtime-049/cutlass-site:/probe/runtime-049/fi-site:/probe/runtime-049/cudnn-site:/probe/runtime-049/native-site:/probe/runtime-049/probe-site:/src/python \
  -e PYTHONDONTWRITEBYTECODE=1 -e USER=pr36122 -e LOGNAME=pr36122 -e HOME=/tmp \
  -e LD_LIBRARY_PATH=/probe/runtime-049/native-site/nvidia/nccl/lib:/probe/runtime-049/cudnn-site/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  -e "SGLANG_CACHE_DIR=/probe/runtime-049/published-cache-$role" \
  -e "SGLANG_JIT_CACHE_DIR=/probe/runtime-049/published-cache-$role/jit" \
  -e CUDA_HOME=/probe/toolkit -e MAX_JOBS=1 -e OMP_NUM_THREADS=1 \
  -e "NVCC_RECORD=/probe/runtime-049/published-nvcc-$role.log" -e CUDA_VERSION=13 \
  -e "PROBE_ROLE=$role" --entrypoint /bin/bash \
  local/pr36122-runtime-no-old-fi-cache:20261013 \
  -c 'exec nice -n 10 timeout --signal=TERM --kill-after=20s 10m python3 -u /probe/runtime-049/published_moe_probe.py' \
  2>&1 | tee "$probe_dir/runtime-049/published-moe-$role.log"
