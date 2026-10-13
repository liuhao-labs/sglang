# Recorded container commands

Local artifact root:
`/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj`.
Commands operate in detached `main/` and `integrated/` verification trees. The
original PR worktree was only read. Runner image ID and package hashes are
recorded in `provenance.json` and `state-final.log`.

## Compile and numeric controls

```bash
bash run_activation.sh main
bash run_activation.sh integrated
bash build_gguf.sh integrated
bash build_gguf.sh main
bash run_chain.sh integrated
bash run_chain.sh main
```

The initial `run_activation.sh main` launch failed because an nvcc wrapper
symlink used a host absolute path inside a container; replacing it with
`../../nvcc-record.sh` fixed the launch. Both roles then started with empty
caches and compiled precise and fast-math binaries. Launch flags are in the
supplied scripts; nvcc invocations are in `results-*/nvcc.log` and
`results-*/gguf-build.log`.

## Registered CPU regression

```bash
probe_dir=/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj
docker run --rm --network none --read-only --memory 4g --memory-swap 4g \
  --cpus 1 --user 1004:1004 \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m,uid=1004,gid=1004 \
  -e NVIDIA_VISIBLE_DEVICES=void -e USER=pr36122 -e LOGNAME=pr36122 \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e SGLANG_CACHE_DIR=/tmp/cache \
  --mount "type=bind,src=$probe_dir/integrated,dst=/src,readonly" \
  -e PYTHONPATH=/src/python --entrypoint /bin/bash \
  quay.io/gpustack/runner:cuda13.0-sglang0.5.15.post1 \
  -c 'cd /src; exec nice -n 10 timeout 10m python3 -m pytest -vv -ra --tb=short -p no:cacheprovider test/registered/unit/layers/quantization/test_gguf_mmq_types.py'
```

The first CPU launch lacked `USER/LOGNAME` for uid 1004, causing a getpwuid
collection error. The passing rerun uses the environment above. No project
code was changed to work around that container setup error.

## Registered activation regression

```bash
docker run --rm --name pr36122-registered-activation --network none --read-only \
  --gpus device=0 --user 1004:1004 --cpus 1 --memory 8g --memory-swap 8g \
  --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges --shm-size 512m \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,uid=1004,gid=1004 \
  --mount "type=bind,src=$probe_dir,dst=/probe" \
  --mount "type=bind,src=$probe_dir/integrated,dst=/src,readonly" \
  -e PYTHONPATH=/src/python -e PYTHONDONTWRITEBYTECODE=1 \
  -e USER=pr36122 -e LOGNAME=pr36122 -e SGLANG_IS_IN_CI=true \
  -e SGLANG_JIT_KERNEL_RUN_FULL_TESTS=0 \
  -e SGLANG_CACHE_DIR=/probe/cache-integrated \
  -e SGLANG_JIT_CACHE_DIR=/probe/cache-integrated/jit \
  -e HOME=/tmp -e CUDA_HOME=/probe/toolkit -e MAX_JOBS=1 \
  -e NVCC_RECORD=/probe/results-integrated/nvcc.log --entrypoint /bin/bash \
  quay.io/gpustack/runner:cuda13.0-sglang0.5.15.post1 \
  -c 'exec nice -n 10 timeout --signal=TERM --kill-after=20s 15m python3 -u /probe/run_registered.py'
```

## Existing GGUF AOT test module

```bash
docker run --rm --name pr36122-gguf-regression --network none --read-only \
  --gpus device=0 --user 1004:1004 --cpus 2 --memory 8g --memory-swap 8g \
  --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges --shm-size 512m \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,uid=1004,gid=1004 \
  --mount "type=bind,src=$probe_dir,dst=/probe" \
  --mount "type=bind,src=$probe_dir/integrated,dst=/src,readonly" \
  -e PYTHONPATH=/src/python -e PYTHONDONTWRITEBYTECODE=1 \
  -e USER=pr36122 -e LOGNAME=pr36122 \
  -e SGLANG_CACHE_DIR=/probe/cache-integrated \
  -e SGLANG_JIT_CACHE_DIR=/probe/cache-integrated/jit -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e CUDA_HOME=/probe/toolkit -e MAX_JOBS=1 -e OMP_NUM_THREADS=1 \
  -e NVCC_RECORD=/probe/results-integrated/nvcc.log --entrypoint /bin/bash \
  quay.io/gpustack/runner:cuda13.0-sglang0.5.15.post1 \
  -c 'exec nice -n 10 timeout --signal=TERM --kill-after=20s 15m python3 -u /probe/run_gguf_regression.py'
```

`run_gguf_regression.py` preserves the checked-in test logic, redirects GGUF
Python imports to the fresh isolated CUDA library, and maps fixed-revision
snapshot calls to LFS-verified local fixtures. It is a source-kernel harness,
not a replacement for installed-full-wheel CI. Allocator cap is 0.025.

## CUDA Graph two-projection test

Use the same GPU container flags as the GGUF test command, with
`PROBE_ROLE=integrated`, and replace the entry script with
`python3 -u /probe/gguf_graph_probe.py`. It uses production `fused_moe_gguf`
and the shared default activation, and checks bit-exact replay for FP16/BF16
and SiLU/GELU at M=128.

## Model smoke attempt

Use the GGUF GPU container flags with `--memory=10g --memory-swap=10g`,
`PROBE_ROLE=integrated`, `TRANSFORMERS_OFFLINE=1`, a new
`SGLANG_CACHE_DIR=/probe/model-cache-integrated`, and entrypoint
`bash /probe/model_smoke.sh`. No ports are published; client traffic is
inside the container. This launch **failed before model load** at the minimum
kernel-version check. The unmodified server and client logs are included;
no health or generation success is claimed for this smoke.

## Download and source precautions

Fixture retrieval runs in a separate CPU-only container with a writable
artifact mount and `--network host`; curl uses explicit
`--proxy http://127.0.0.1:17890`. Numeric test containers use `--network none`.
Pinned files are downloaded by `fetch_mirror_verified.py`, which verifies
sizes and LFS SHA-256 before use. The original full benchmark model is never
loaded or retimed.
