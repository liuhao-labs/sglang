#!/bin/bash
set -euo pipefail
probe_dir=/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj
awk '/MemAvailable/{if ($2<24*1024*1024) exit 1}' /proc/meminfo
mkdir -p "$probe_dir/full-wheel/dist" "$probe_dir/full-wheel/build"
set +e
docker run --rm --name pr36122-full-wheel-build --network none --read-only \
  --user 1004:1004 --cpus 2 --memory 14g --memory-swap 14g --pids-limit 512 \
  --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g,uid=1004,gid=1004 \
  --mount "type=bind,src=$probe_dir,dst=/probe" \
  -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/probe/runtime-049/cudnn-site:/probe/runtime-049/native-site:/probe/runtime-049/probe-site \
  -e LD_LIBRARY_PATH=/probe/runtime-049/native-site/nvidia/nccl/lib:/probe/runtime-049/cudnn-site/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs \
  -e CUDA_HOME=/usr/local/cuda -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CMAKE_PREFIX_PATH=/probe/runtime-049/probe-site/torch/share/cmake \
  -e CMAKE_BUILD_PARALLEL_LEVEL=1 -e MAX_JOBS=1 -e OMP_NUM_THREADS=1 \
  -e "PROBE_GB10_ONLY=${PROBE_GB10_ONLY:-0}" \
  --entrypoint /bin/bash local/pr36122-runtime-no-old-fi-cache:20261013 \
  -c 'exec nice -n 15 timeout --signal=TERM --kill-after=30s 180m python3 -u /probe/build_full_wheel.py' \
  2>&1 | tee "$probe_dir/full-wheel/build-attempt.log"
exit_codes=("${PIPESTATUS[@]}")
printf '{"docker_exit_code":%d,"tee_exit_code":%d}\n' "${exit_codes[0]}" "${exit_codes[1]}" | tee "$probe_dir/full-wheel/build-exit.json"
exit "${exit_codes[0]}"
