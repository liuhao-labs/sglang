#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import hashlib
from pathlib import Path
p=Path('/probe/model/Qwen2.5-0.5B-Instruct.IQ3_S.gguf')
assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()=='b7685f565fc6e49a8f9bc0fe09b9bb22ca0d5e4b10635cd4b181597f113794b6'
PY
python3 -m sglang.launch_server \
  --model-path /probe/model/Qwen2.5-0.5B-Instruct.IQ3_S.gguf \
  --load-format gguf --host 127.0.0.1 --port 18080 \
  --mem-fraction-static 0.03 --max-total-tokens 1024 --max-running-requests 1 \
  --chunked-prefill-size -1 --attention-backend triton --sampling-backend pytorch \
  --disable-cuda-graph --skip-server-warmup --random-seed 35019 \
  >"/probe/results-${PROBE_ROLE}/model-server.log" 2>&1 &
probe_server_pid=$!
trap 'kill "$probe_server_pid" 2>/dev/null || true; wait "$probe_server_pid" 2>/dev/null || true' EXIT
python3 /probe/model_client.py
