# GGUF activation-policy revalidation (2026-10-13)

## Outcome and scope

No PR-related numerical regression was observed in the bounded GB10 tests after
main enabled `--use_fast_math` for Blackwell JIT activations in
[#43236](https://github.com/sgl-project/sglang/pull/43236). This is fresh
component/source-integration evidence, **not a rebuilt full `sglang-kernel`
wheel or a passing latest-runtime serving test**. The original PR branch was
left clean and unchanged; no new feature commit, force-push, or maintainer ping
was made.

## Pinned sources and actual call chain

| item | SHA |
| --- | --- |
| remote PR head H (unchanged) | `0a39ec3a8754e3c4426950a5a93046c048a03f07` |
| PR base | `6e7beace143964386ce2e4418401b7369f0dabd0` |
| tested upstream main U | `7ff3a112846a3fcc12a27bba09d644885a226e4d` |
| U's parent (old Blackwell precise policy) | `d8f852234850ac01a36adbb1c337a0acc0ff3912` |
| isolated U+H integration tree | `36909c6658f0ce1ed346db5b85a3568325b9357d` |
| local detached integration commit | `15d569117ce2d059ed2a3203ed53701063ace606` |

`git merge-tree --write-tree U H` produced that integration tree without
conflicts. Its diff from U is still exactly **15 files, +4127/-81** and
`git diff --check U <integration>` passed. The local commit is only a detached
verification object, not a new remote PR head.

The inspected CUDA path is:

```text
fused_moe_gguf(..., activation="silu"|"gelu")
  -> silu_and_mul / gelu_and_mul (shared activation.py)
  -> _run_activation_inplace
  -> activation_module(dtype)  # fast_math=True by default
  -> get_activation_cuda_cflags()
  -> load_jit(... extra_cuda_cflags=["--use_fast_math"])
```

U's parent returned `[]` on SM>=100; U returns `["--use_fast_math"]` on CUDA.
Precise control used the **unrounded** `run_activation` entry of
`activation_module(dtype, fast_math=False)`, not the separately rounded
activation API. The same current kernel source, inputs, and flags were used
for both controls, with the fast-math flag as the intended difference.

Earlier 153-node AOT tests use the AOT SiLU function. Earlier no-JIT production
smoke probed the first projection and composed an AOT-only two-projection
calculation. Neither covers this new shared JIT policy. The historical
Qwen2.5 smoke is a small dense-model test, not routed-MoE proof. These are why
the fresh activation and full GGUF-chain probes below were necessary.

## Runtime and binary provenance

- NVIDIA GB10, CC 12.1; aarch64; Python 3.12.3.
- Torch `2.11.0+cu130`; nvcc `13.0.88`; `apache-tvm-ffi 0.1.11`.
- Frozen runner image ID:
  `sha256:c5d229150e9f0815f0d66ba64918c9383afccfdc7a548d318a1fb6d38fd2f06c`.
- Runner repository digest:
  `sha256:28aabc1a96ec840d35a56b862bcacbcd198edddde3850a2862f70eb99cbc0e56`.
- Installed auxiliary `sglang-kernel 0.4.4` wheel SHA-256:
  `727e4bc53abeade20260186f99199200320b9fa51f8de7af90c01524cff73e5d`.
  The installed `sm100/common_ops.abi3.so` was verified against that downloaded
  wheel: `672a1cb425d2d2815b8138c34fd8a1e89603bac9a8986dc9db7883e04be535d3`.
- Fresh main GGUF test library SHA-256:
  `6f3c63eeaa7c1588bec229fbbb4f40e5164829636068d77e2df15dcec77f5e93`.
- Fresh U+PR GGUF test library SHA-256:
  `ee85ba3df98e21a7fda4598961da08a48aacf8fb3cdf67b18000ae03bde062d2`.

GGUF source was compiled separately for U and U+PR, in a `gguf_probe` test
namespace. The harness redirects only the GGUF Python operator bindings to
these actual newly compiled C++/CUDA functions; it does not overwrite any
installed Torch operator, mock numerical results, or substitute an old GGUF
binary. Shared default activations and the production `fused_moe_gguf` function
are executed directly. AOT SiLU/reduction used by the unchanged AOT tests are
auxiliary ops from the recorded installed wheel. Full wheel registration and
ABI validation are therefore still separate requirements.

Both roles began with separate empty JIT cache directories. All fast/precise
activation modules were compiled independently and their exact nvcc command
lines were recorded by a transparent wrapper that delegates to the real
compiler. The main and U+PR activation binaries have identical SHA-256 hashes.
Fast compile commands contain `--use_fast_math`; precise commands do not.
All device compilation targets `compute_121a/sm_121a` with C++20/O3. The GGUF
test libraries use neither activation fast-math policy nor cached GGUF code.

See [provenance.json](provenance.json), build logs, and `results-*/nvcc.log`.

## Fresh results

| check | result |
| --- | --- |
| registered routing CPU suite, U+PR | **10 passed**, 23 warnings, 12.63 s |
| registered activation correctness/out-param subset | **54 passed**, 56 deselected, 2 warnings, 12.11 s |
| standalone activation comparisons, U | **60 passed**, plus 4 exact CUDA Graph replays |
| standalone activation comparisons, U+PR | **60 passed**, plus 4 exact CUDA Graph replays |
| two-projection GGUF + shared JIT activation, U | **160 cases passed** |
| two-projection GGUF + shared JIT activation, U+PR | **160 cases passed** |
| unchanged AOT `tests/test_gguf.py`, fresh U+PR GGUF bindings | **153 passed**, 2 warnings, 85.00 s |
| U+PR full two-projection MoE CUDA Graph replay | **4 passed**, bit-identical to eager output |

Standalone cases cover FP16/BF16, SiLU/GELU, **every finite representable gate
input** (including subnormals, signed zeros, and exponent extremes) with up=1,
and random gated/multiply inputs at M=1/127/128/129 with scales 1/3/10.
Comparisons are fast vs precise, fast vs FP64 reference, and precise vs FP64.
All outputs are finite. The existing activation tolerances are retained:
`atol=0.01, rtol=0.01`; the registered FP32 cases retain `1e-4/1e-4`.

The combined tolerance ratio is
`abs(actual-reference)/(atol+rtol*abs(reference))`; **<=1 is passing**.

| standalone fast vs precise | max absolute difference | max tolerance ratio |
| --- | ---: | ---: |
| FP16 SiLU | 0.0625 | 0.075643 |
| FP16 GELU | 0 | 0 |
| BF16 SiLU | 2.3842e-7 | 2.3841e-5 |
| BF16 GELU | 1.1755e-38 | 1.1755e-36 |

The largest standalone fast-vs-FP64 tolerance ratio is **0.256411**.
Fast math is not claimed bit-identical to precise math. The BF16 exhaustive
differences include very small values affected by flush-to-zero.

GGUF chain cases use real pinned fixture **submatrices** E=8/H=256/I=256,
top-k=2; eight pure IQ types and both IQ3_S/Q5_0 directions, FP16/BF16,
SiLU/GELU, M=1/127/128/129. The same source wrapper is used for both policies;
the precise control replaces only its activation callable with the explicit
precise JIT module. The numerical reference dequantizes the same fixed
weights and computes both projections.

- Max chain fast-vs-precise difference (both U and U+PR): **3.05176e-5**;
  max tolerance ratio **0.002993** at `0.01/0.01`.
- Max U+PR fast-vs-dequantized-reference difference: **0.00317383**;
  max ratio **0.003151** under original GGUF MoE `atol=1, rtol=0.1`.
- Cross U-vs-U+PR: max absolute output difference **0.00241089**;
  all chain outputs also meet the tighter `0.01/0.01` bound. Of 180 saved
  activation/chain outputs, 110 are bit-identical; others differ as expected
  between vector and batched quantized paths.
- Profiler records two vector projections on U at every M; U+PR records two
  vector projections at M=1/127 and two batched projections at M=128/129,
  with the shared `sglang::_run_activation_inplace` between them.
- Fresh U `block_size(IQ3_S)` returns 0; fresh U+PR returns 4. U still lacks
  IQ MMQ implementation, so #43236 does not make #36122 redundant.

The 43 fixture files use revisions `d82b8773934ef260d8d8a896a7c197bc69a0fac1`
and `2b77eb27ae2ea4b3f68cf3042961509a9f5847b8`. Every downloaded file's size and
SHA-256 were verified against its pinned repository LFS metadata. Downloads
used `hf-mirror.com` because direct HF TLS/Hub metadata requests failed; tests
then ran with network disabled. The LFS manifest and retrieval logs are included.

## Model smoke and remaining limits

The exact prior Qwen2.5 IQ3_S file was redownloaded and hash verified:
`b7685f565fc6e49a8f9bc0fe09b9bb22ca0d5e4b10635cd4b181597f113794b6`.
A real isolated latest-source server startup was attempted. It stopped before
model load with:

```text
sglang-kernel is installed with version 0.4.4, which is less than
the minimum required version 0.4.9.
```

No version guard was bypassed. Current main pins Torch `2.14.1` and kernel
`0.4.9`; the available cached toolchain is Torch 2.11/kernel 0.4.4. The startup
guard is identical on U and U+PR and unrelated to this feature. A matching
full-runtime/full-wheel model serving smoke, H100/B200 official CI, and formal
review are still pending. This run does not verify the current published
0.4.9 wheel's fallback integration or the full new wheel's ABI.

Resource limits: serial GPU phases, container RAM 8 GiB/no extra swap for
numerics (10 GiB for the attempted model smoke), allocator fractions
0.015/0.025, 1 compile worker and low priority. Peak reserved: activation
**34 MiB**, chain **156 MiB**, full GGUF suite **2890 MiB**. The original vLLM
service remained running, `/health` returned 200 before/after, and its container
was never entered, stopped, or restarted. Original PR working tree remains
clean at H. System swap use rose from ~0.9 to ~2.3 GiB during the session;
there was no OOM or service failure. See `state-final.log` for the final snapshot.

The old **+59.8% throughput / -40% TTFT** values remain historical results on
the original pinned runtime and reconstructed model. They were **not rerun on
this latest main** and are not a new performance claim.

## Reproduction

The scripts are supplied as task artifacts; none was added to the feature diff.
Use the exact image digest, pins, and test binding details above. Mount source
read-only, use isolated caches, and preserve existing services. Commands used:

```bash
git fetch --no-tags origin 7ff3a112846a3fcc12a27bba09d644885a226e4d
git merge-tree --write-tree 7ff3a112846a3fcc12a27bba09d644885a226e4d \
  0a39ec3a8754e3c4426950a5a93046c048a03f07
# Create detached main and integration verification worktrees from the pins.
# Fix probe_dir in the supplied launch scripts to your artifact directory.
bash run_activation.sh main
bash run_activation.sh integrated
bash build_gguf.sh integrated
bash build_gguf.sh main
# fetch_mirror_verified.py: fixed revisions + LFS size/SHA256 verification.
bash run_chain.sh integrated
bash run_chain.sh main
# Run run_registered.py and run_gguf_regression.py using the recorded Docker
# commands in commands.md (same mounts, flags, and environment).
```

Original stdout/stderr is included in `results-main/` and
`results-integrated/`. The initial launch-failure log preserves a corrected
nvcc-wrapper symlink mistake; the first CPU-collection log preserves a missing
container USER setting, corrected for the passing rerun. These were harness
setup failures, not source regressions. The first main graph-only diagnostic
ratio printed NaN from 0/0 at a zero tolerance; output finiteness and exact
equality assertions passed. The integrated run corrects that reporting field.
`SHA256SUMS` covers all published files.
