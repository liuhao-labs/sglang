# Official 0.4.9 wheel and current-source serving follow-up

This extends the component report in the parent directory. The initial
0.4.4 minimum-version startup failure was a fixable cached-image gap; it was
not treated as a final external blocker.

## Frozen source and actual runtime

Both arms use the sources from the parent report:

- U: `7ff3a112846a3fcc12a27bba09d644885a226e4d`.
- U+PR detached tree: `36909c6658f0ce1ed346db5b85a3568325b9357d`.
- Original PR H remains `0a39ec3a8754e3c4426950a5a93046c048a03f07` (clean).

The official aarch64 `sglang-kernel 0.4.9` wheel was downloaded and verified:

- wheel SHA256:
  `384a0d2158740f5ab42cfc64c872640bb3c80f3b0ec3b4666bc3efb553254bc4`.
- loaded `sm100/common_ops.abi3.so` SHA256:
  `8b5ee02f8eb7c807e8b8051963e4a88f6d7174a72960bc5f2e3d9905c21456bf`.
- Torch `2.14.1+cu130` wheel SHA256:
  `0b13acde294b401509830d85a8906cbf2fc97425a12a66cce1d9fa13c5106a2a`.

The isolated runtime overlays these primary pins:
Torch 2.14.1, torchvision 0.29.1, kernel 0.4.9, Transformers 5.19.0,
FlashInfer 0.7.1, CUTLASS DSL 4.8.0/CUDA13, Triton 3.8.0,
compressed-tensors 0.18.0, xgrammar 0.2.7, tokenizers 0.23.2,
safetensors 0.8.0, Hugging Face Hub 1.31.0, NCCL 2.30.7 and cuDNN 9.24.0.43.
Package provenance is recorded for both arms. Other supporting dependencies
remain from the pinned base image; this is an isolated verified overlay,
not a claim that every optional project extra was installed into a clean
canonical environment.

The cached image's optional FlashInfer cubin/JIT-cache 0.6.12 providers were
removed in a **new derived image**, selecting normal source/JIT fallback.
FlashInfer 0.7.1 refused to use the incompatible old providers. No version
check was disabled, and the original image, service, and host environment
were not edited. Derived image SHA256:
`e3d39aa55b029a98b1d91f8968329303cd4bb48fd16db898c7b35f6d6680f6ce`.

NCCL was placed in the isolated wheel's expected loader path after the first
Torch 2.14.1 import reported `undefined symbol: ncclCommResume`. The final
imports use the normal package layout (no LD_PRELOAD or fake metadata needed).
The diagnostic attempts and downloads are retained in the raw logs.

## Real paired model result

Both U and U+PR, with **the same unmodified official 0.4.9 wheel**, loaded
the exact prior `Qwen2.5-0.5B-Instruct.IQ3_S.gguf` file (SHA256
`b7685f565fc6e49a8f9bc0fe09b9bb22ca0d5e4b10635cd4b181597f113794b6`).
Both health and `/generate` returned **HTTP 200**, 8 prompt / 16 completion
tokens, zero cached tokens, and exactly matching output token IDs and text:

```text
 One reason to work outdoors is to gain a better understanding of the environment and its
```

Canonical output SHA256 (text, IDs, prompt/completion/cached counts):
`2604448b1bc1b2c9a10961f6b562bed70aab5041db4cb8a7db54e83f9cfde287`.
Full client JSON includes request IDs/timestamps/latencies, so it is not
claimed byte-identical. The comparison script explicitly checks token IDs
and text. The passing final server logs contain no traceback, CUDA error,
OOM or illegal-memory failure.

Both final runs use exactly the same argv: TP1, one request, Triton attention,
PyTorch sampling, CUDA graphs disabled, random seed 35019,
`--mem-fraction-static 0.05`, `--max-total-tokens 1024`. No ports were published;
requests run inside separate network-disabled containers. Memory limit is
10 GiB/no extra swap, MAX_JOBS=1, source/model/wheel readonly where applicable.
The first main attempt at 0.03 failed its real KV-budget guard (minimum 0.035)
under current host occupancy, so **both final arms** use 0.05. The actual
allocated cache can be smaller than the 1024-token cap; this was a short
correctness smoke, not a memory or latency benchmark.

The official wheel does **not** expose the PR capability, so U+PR records
`capability=False` and uses the published fallback. There is no test-namespace
binding replacement in these serving runs. This proves published 0.4.9 ABI
loads with matched Torch and that this small dense GGUF serving/fallback path
works on current source. It **does not** prove the new PR-built full wheel's
ABI, IQ MMQ enabled full-model serving, or routed-expert model correctness.

An extra eight-format published-wheel MoE matrix was prepared in
`published_moe_probe.py`, but no such run is claimed: repeated post-serving
GPU samples remained at 95-96% under the existing vLLM service, above the <=10%
launch gate. The prepared scripts remain for the next idle window. The already
completed 160x2 MoE chain cases in the parent report used the separate freshly
compiled source/test namespace, not the published 0.4.9 wheel.

## Attempts and evidence interpretation

- Direct full dependency install stalled downloading cuDNN via TLS/Range;
  all failed install and successful segmented-download logs are included.
  Every used downloaded wheel is recorded by size and SHA256.
- A core-package import initially inherited old NCCL and failed; matching
  NCCL/cuDNN and the isolated loader layout corrected it.
- FlashInfer 0.7.1 first rejected old optional prebuilt providers and then
  old CUTLASS; a new image without the old providers plus DSL 4.8 corrected it.
- The first integrated serving smoke succeeded before FlashInfer/Triton were
  aligned; it is kept as historical setup evidence, not the final comparison.
- One integrated attempt's client aborted on a transient startup
  `/freeze_gc` connection retry traceback even though the server subsequently
  became ready. The client was corrected to distinguish real scheduler/version
  failure; the final rerun has no such traceback and generated successfully.
- Existing `qwen35-9b-official` continued serving throughout; final `/health`
  returned 200. Only named temporary containers created by this task were
  stopped. No main/PR branch mutation or rebase was made.

The parent component proofs, this released-wheel smoke, and remaining full
PR-wheel/official H100/B200 CI each have distinct scopes. The historical
large-model +59.8%/-40% performance figures are not new-current-main data.

## Reproduction

Use `launch_smoke.sh main` followed by `launch_smoke.sh integrated`; they
enforce MemAvailable>=24 GiB and GPU utilization<=10% at launch. Exact
package layout, source paths, Dockerfile and logs are included. Run
`compare_results.py` only after both final `client.json` files exist.
There is no `FLASHINFER_DISABLE_VERSION_CHECK` or
`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` in the passing launch environment.
