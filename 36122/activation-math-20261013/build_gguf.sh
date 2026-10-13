#!/bin/bash
set -euo pipefail
role=${1:?main or integrated}
probe_dir=/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj
mkdir -p "$probe_dir/results-$role"
docker run --rm --name "pr36122-gguf-build-$role" --network none --read-only \
  --user 1004:1004 --cpus 1 --memory 6g --memory-swap 6g --pids-limit 256 \
  --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,uid=1004,gid=1004 \
  --mount "type=bind,src=$probe_dir,dst=/probe" \
  --mount "type=bind,src=$probe_dir/$role,dst=/src,readonly" \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e MAX_JOBS=1 \
  -e CUDA_HOME=/usr/local/cuda -e "PROBE_ROLE=$role" \
  -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint /bin/bash quay.io/gpustack/runner:cuda13.0-sglang0.5.15.post1 \
  -c 'exec nice -n 10 timeout --signal=TERM --kill-after=20s 30m python3 -u /probe/build_gguf.py' \
  2>&1 | tee "$probe_dir/results-$role/gguf-build.log"
