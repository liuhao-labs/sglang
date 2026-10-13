# Production GGUF MoE bindings and full-wheel build attempts, 2026-10-13

The official `sglang-kernel 0.4.9` production bindings passed the eight-IQ
MoE matrix on latest main and a clean verification merge of main+PR:
**32 cases per arm, all finite, all within the existing `atol=1, rtol=0.1`
quantization bounds, and all 32 paired output tensors bit-identical.**
The maximum reference absolute error was `0.003662109375`; peak Torch
allocator reserved memory was `127,926,272` bytes (122 MiB) per arm.

This validates the published-wheel **vector fallback** with the new shared
fast-math activation, not enabled IQ MMQ. The complete new PR wheel was not
produced: three actual builds hit container memory limits. No new complete
wheel ABI, enabled IQ-MMQ expert model, or official H100/B200 pass is claimed.

## Source provenance

- Original PR H: `0a39ec3a8754e3c4426950a5a93046c048a03f07`.
- PR base B: `6e7beace143964386ce2e4418401b7369f0dabd0`.
- Latest frozen main M: `27b68303d043ddbddf9861c55c26bc8617ea7c35`.
- Verification-only merge I: `449c32e493e80e67190fcf1f61976dc6045159e9`.
- Verification merge tree: `bb5c83cc334e80c51e6a0eb44045a5358c189162`.
- Earlier accepted main: `7ff3a112846a3fcc12a27bba09d644885a226e4d`.
- Earlier verification tree: `36909c6658f0ce1ed346db5b85a3568325b9357d`.

`latest-drift.json` records the 15 upstream commits since the earlier main.
There is no intersection with the PR's 15 files. GGUF, activation operators,
the JIT include/compile chain, primary dependency pins and AOT build metadata
are unchanged by blob. Three-way merge is clean; feature diff remains
15 files, +4127/-81. The original feature worktree was not changed or pushed.
Only detached verification trees and a separate evidence branch were used.

`full-wheel/source-equivalence.json` verifies all 291 tracked AOT files in the
staged full build against I; AOT tree is
`bbb2a30ede10936904ec0fd1d2d4922b1b810016`.

## Official-wheel production matrix

All runs use the prior isolated runtime overlay: Torch `2.14.1+cu130`,
CUDA toolkit `13.0.88`, official kernel `0.4.9`, Transformers `5.19.0`,
FlashInfer `0.7.1`, Triton `3.8.0`, CUTLASS DSL `4.8.0` and the matching
NCCL/cuDNN setup from the [preceding runtime report](../runtime-049/README.md).
The derived image SHA is
`e3d39aa55b029a98b1d91f8968329303cd4bb48fd16db898c7b35f6d6680f6ce`.

- Official wheel SHA256:
  `384a0d2158740f5ab42cfc64c872640bb3c80f3b0ec3b4666bc3efb553254bc4`.
- Loaded `sgl_kernel/sm100/common_ops.abi3.so` SHA256:
  `8b5ee02f8eb7c807e8b8051963e4a88f6d7174a72960bc5f2e3d9905c21456bf`.
- The real namespace is `torch.ops.sgl_kernel`; no operator or activation
  implementation was patched/replaced, and no version gate was disabled.

Each arm runs eight pure-IQ formats, FP16/BF16, SiLU/GELU, M=128, E=8,
top-k=2, input/intermediate/output width 256. Weights are cropped from the
same pinned, hash-verified real MoE GGUF fixtures. A two-projection reference
uses GGUF dequantization, einsum, FP32 SiLU/GELU and route reduction.
This is an actual expert-layer pipeline, not a complete served expert model.

| source pair | cases | paired outputs | operator route | result |
|---|---:|---|---|---|
| earlier frozen main / main+PR | 32+32 | all bit-identical | vector | PASS |
| latest M / I | 32+32 | all bit-identical | vector | PASS |

Every profiler trace records two `sgl_kernel::ggml_moe_a8_vec` calls and
one `sglang::_run_activation_inplace`; no `ggml_moe_a8` call occurs because
the published wheel has `capability=False`. Separate empty JIT caches were
used for each arm. The recorded real NVCC commands include
`--use_fast_math`, C++20, O3 and `compute_121a/sm_121a`. The earlier
precise-vs-fast component checks remain separately scoped to their prior
Torch/runtime and test namespace; these production tests do not relabel them.

All four Docker runs exited 0. Raw stdout/stderr, provenance and exit-code
files, compiler commands and CPU output tensors are under `runtime-049/`.
Serialized `.pt` file hashes differ because archive filenames differ;
`compare_published.py` compares loaded tensor values with `torch.equal`.

Commands (serial, source paths are recorded in the scripts):

```bash
bash runtime-049/run_published_moe.sh main
bash runtime-049/run_published_moe.sh integrated
bash runtime-049/run_published_moe.sh main-latest
bash runtime-049/run_published_moe.sh integrated-latest
# Run compare_published.py inside the same isolated runtime.
```

GPU launches enforce utilization <=10% and MemAvailable >=24 GiB. The
latest-main attempt was first deferred at 96% utilization, then both latest
arms ran during a verified 2% window. GPU containers use an 8 GiB hard limit,
no extra swap, one CPU and a 2.5% Torch allocator limit. Existing services
were not stopped/restarted; health checks remained HTTP 200.

## Complete source-wheel attempts: not passed

I staged the unmodified full AOT source, verified all seven FetchContent
archives against the upstream CMake SHA256 pins, and used the real project
PEP517/scikit-build wheel build with matched Torch `2.14.1+cu130`, GCC 13.3,
CUDA 13.0.88, C++20, one build worker and one NVCC thread. Logs contain exact
commands, paths, CMake arguments and exit codes.

Two setup errors were corrected before the resource attempts: the CUTLASS
submodule directory existed but was unextracted; a prior CMake cache also
retained old Torch libraries. The corrected clean build records exclusively
the new Torch include/library paths. Those failed/stopped attempts are kept
as diagnostics, not presented as product errors.

| build configuration | hard RAM/swap limit | failure |
|---|---:|---|
| default complete multi-architecture build | 10 GiB / 10 GiB | compiler exit 137, `es_fp8_blockwise.cu` |
| default complete multi-architecture build | 14 GiB / 14 GiB | compiler exit 137, `fp8_gemm_kernel.cu` |
| all source/targets, SM121a-only common_ops | 14 GiB / 14 GiB | compiler exit 137, same `fp8_gemm_kernel.cu` |

Docker emitted three `oom` events; raw JSON is in `full-wheel/oom-events.jsonl`.
The 14 GiB default build's cgroup sample reached exactly `15,032,385,536`
bytes with `oom_kill=1`. Both failing CUDA units have identical source blob
hashes in M and I. They are existing FP8 units, not the proposed GGUF kernels.
This is an observed resource failure, not a reproduced GGUF code regression.

The GB10 attempt used an external CMake configuration hook, preserved all
146 compilation units / all default package targets (FA3 is disabled by the
existing aarch64 default), and restricted only common_ops device code to
SM121a. No source, schema, registration or operator was removed. Exact
compile commands and the hook are included. This attempt also failed; it
is not a successful full-wheel or cross-architecture result.

```bash
python3 prepare_full_wheel.py
bash run_full_wheel.sh
PROBE_GB10_ONLY=1 bash run_full_wheel.sh
```

No `.whl` was emitted. Therefore the prepared 80 enabled-production cases,
four graphs and 153-test AOT run against a new full wheel were **not executed**.
Their scripts are retained, and `full-wheel/failure-summary.json` explicitly
records these false/unexecuted fields. The earlier 153-pass AOT/component
result does not fill this gap. The new build's memory limit was not raised
again on this shared host; initial available memory was about 27.8 GiB and
the existing vLLM service remained running. No temporary test/build container
was left running. Host memory/swap samples are included.

## Necessity and remaining validation

Latest main still has `MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES`
and no IQ-MMQ capability. The published production matrix still selects
the vector path. `full-wheel/necessity-summary.json` records the exact source
blobs. Thus the capability provided by this PR remains absent upstream.
This is not a new speed benchmark: the historical large-model +59.8%
prefill / -40% TTFT data keeps its original pinned runtime/checkpoint label.

Required outstanding evidence is a complete new PR wheel built with enough
memory, capability=true production expert validation, and official H100/B200
CI. The existing `run-ci` permission gate is still external; no repetitive
maintainer ping was sent. The new numerical evidence found no feature-code
change to make, so the PR head remains H.
