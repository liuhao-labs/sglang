#!/bin/bash
set -euo pipefail
probe_dir=/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj
mode=${1:?matrix or aot}
case "$mode" in matrix|aot) ;; *) exit 2;; esac
awk '/MemAvailable/{if ($2<24*1024*1024) exit 1}' /proc/meminfo
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if ((util>10)); then exit 20; fi
mapfile -t wheels < <(find "$probe_dir/full-wheel/dist" -maxdepth 1 -name '*.whl' -type f)
((${#wheels[@]}==1)) || exit 2
wheel_name=${wheels[0]##*/}
mkdir -p "$probe_dir/full-wheel/site" "$probe_dir/full-wheel/cache-$mode"
set +e
docker run --rm --name "pr36122-full-wheel-$mode" --network none --read-only --gpus device=0 \
  --user 1004:1004 --cpus 1 --memory 8g --memory-swap 8g --pids-limit 512 \
  --cap-drop ALL --security-opt no-new-privileges --shm-size 512m \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,uid=1004,gid=1004 \
  --mount "type=bind,src=$probe_dir,dst=/probe" \
  --mount "type=bind,src=$probe_dir/integrated-latest,dst=/src,readonly" \
  -e PYTHONPATH=/probe/full-wheel/site:/probe/runtime-049/triton-site:/probe/runtime-049/cutlass-site/nvidia_cutlass_dsl/dsl_packages:/probe/runtime-049/cutlass-site:/probe/runtime-049/fi-site:/probe/runtime-049/cudnn-site:/probe/runtime-049/native-site:/probe/runtime-049/probe-site:/src/python \
  -e LD_LIBRARY_PATH=/probe/runtime-049/native-site/nvidia/nccl/lib:/probe/runtime-049/cudnn-site/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  -e PYTHONDONTWRITEBYTECODE=1 -e USER=pr36122 -e LOGNAME=pr36122 -e HOME=/tmp \
  -e "SGLANG_CACHE_DIR=/probe/full-wheel/cache-$mode" \
  -e "SGLANG_JIT_CACHE_DIR=/probe/full-wheel/cache-$mode/jit" \
  -e CUDA_HOME=/probe/toolkit -e MAX_JOBS=1 -e OMP_NUM_THREADS=1 -e CUDA_VERSION=13 \
  -e "NVCC_RECORD=/probe/full-wheel/nvcc-$mode.log" \
  -e "PROBE_MODE=$mode" -e "PROBE_WHEEL=$wheel_name" \
  --entrypoint /bin/bash local/pr36122-runtime-no-old-fi-cache:20261013 \
  -c 'set -euo pipefail; python3 -m pip install --no-index --no-deps --no-compile --target /probe/full-wheel/site "/probe/full-wheel/dist/$PROBE_WHEEL"; if [[ "$PROBE_MODE" == matrix ]]; then exec nice -n 10 timeout --signal=TERM --kill-after=20s 15m python3 -u /probe/full-wheel/production_probe.py; else exec nice -n 10 timeout --signal=TERM --kill-after=20s 15m python3 -u /probe/full-wheel/run_aot.py; fi' \
  2>&1 | tee "$probe_dir/full-wheel/production-$mode.log"
exit_codes=("${PIPESTATUS[@]}")
printf '{"docker_exit_code":%d,"tee_exit_code":%d}\n' "${exit_codes[0]}" "${exit_codes[1]}" | tee "$probe_dir/full-wheel/production-$mode-exit.json"
exit "${exit_codes[0]}"
