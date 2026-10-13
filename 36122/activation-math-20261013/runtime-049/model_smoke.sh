#!/bin/bash
set -euo pipefail
python3 -m sglang.launch_server \
  --model-path /probe/model/Qwen2.5-0.5B-Instruct.IQ3_S.gguf \
  --load-format gguf --host 127.0.0.1 --port 18080 \
  --mem-fraction-static 0.05 --max-total-tokens 1024 --max-running-requests 1 \
  --chunked-prefill-size -1 --attention-backend triton --sampling-backend pytorch \
  --disable-cuda-graph --skip-server-warmup --random-seed 35019 \
  >"/probe/runtime-049/results-${PROBE_ROLE}/server.log" 2>&1 &
probe_server_pid=$!
trap 'kill "$probe_server_pid" 2>/dev/null || true; wait "$probe_server_pid" 2>/dev/null || true' EXIT
python3 /probe/runtime-049/model_client.py
