# PR #36122 reviewer evidence packet

This packet records the current provenance and the evidence available before
the repository-owned CI gate is enabled.

## Current source and merge state

- PR head: `0a39ec3a8754e3c4426950a5a93046c048a03f07`
- upstream `main` base: `6e7beace143964386ce2e4418401b7369f0dabd0`
- three-way merge: clean
- feature diff against that base: 15 files, `+4127/-81`
- `git diff --check`: passed
- final-head Python syntax checks for the SRT selector and registered routing
  test: passed
- final-head symbol/selector audit: capability handshake, eight-format set,
  MoE M=128 gate, alignment guards, and fail-fast unsupported-type paths are
  present
- the 67 main commits since the previous sync have no path overlap with the
  15 feature files; the 4 earlier overlapping paths only contained AOT cleanup
  and formatting, not GGUF MMQ logic.

## Local correctness evidence

The following was run on one NVIDIA GB10 (`sm_121`, CUDA 13, aarch64) using
the PR-built AOT wheel, with source, wheel, and fixtures mounted read-only:

| suite | result |
| --- | --- |
| registered routing and old-wheel compatibility | 10 passed |
| focused GGUF CUDA matrix | 57 passed, 5 warnings |
| `python/sglang/kernels/aot/tests/test_gguf.py` | 153 passed, 5 warnings |
| installed-wheel production smoke | passed |

The focused matrix covers all eight added formats (`IQ1_S`, `IQ2_XXS`,
`IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`), dense MMQ and
dequantization boundaries, pure and mixed MoE, K-alignment rejection, Q8_1
fail-fast behavior, invalid-expert zero output, and the scalar capability
dispatch.

The production route log contains these kernel events:

[captured route log](https://github.com/liuhao-labs/sglang/blob/pr-assets-36122/36122/route-log-local.txt)

```text
dense M=16  -> sgl_kernel::ggml_mul_mat_a8
dense M=17  -> sgl_kernel::ggml_dequantize
MoE M=127   -> sgl_kernel::ggml_moe_a8_vec
MoE M=128   -> sgl_kernel::ggml_moe_a8
MoE M=2050  -> sgl_kernel::ggml_moe_a8
CUDA Graph capture/replay: exact output match
```

The old released-wheel pairing passed with `capability=False` and retained
the dequant/vector fallbacks. The PR wheel pairing returned `capability=True`.
The small IQ3_S serving check returned HTTP 200 for health and generation in
both pairings, zero cached prompt tokens, and byte-identical client JSON
(`bd3c11d390f7a373e534f5361807119b2c040bc325cfe1f6fd2c8165d4426933`).

## External routed-MoE evidence

The issue reporter's pinned H-source run is recorded in
[comment 5428479426](https://github.com/sgl-project/sglang/pull/36122#issuecomment-5428479426)
and [follow-up 5428566143](https://github.com/sgl-project/sglang/pull/36122#issuecomment-5428566143).
The two measured arms used identical H source and identical serving arguments;
only the kernel wheel changed:

| metric, 4k cold prefill | H + released wheel | H + H-built wheel |
| --- | ---: | ---: |
| prefill throughput | 60.93 tok/s | 97.37 tok/s |
| median TTFT | 26,513 ms | 15,912 ms |
| decode TPOT control | 61.60 ms | 62.09 ms |
| minimum free unified memory | 2.06 GiB | 3.06 GiB |

The resulting deltas are +59.8% prefill, -40.0% TTFT, +0.8% decode TPOT,
and +1 GiB free memory. Route logs show vector MoE for the released wheel and
batched MMQ at `M>=128` for the H-built wheel.

The checkpoint was reconstructed from surviving shards: 99.18 GiB, 1328
tensors, SHA-256 `17f57289e0d41f913ada93132b645e0920029391f1b3d973b3e5356840c49988`.
Two MXFP4 routed-expert tensors were converted to Q8_0; the remaining tensors
were copied byte-for-byte. The result is strong routed-MoE evidence, but it is
not a bit-identical copy of the deleted original checkpoint.

The same reconstructed checkpoint reproduced the issue-scale baseline:
57.69 tok/s versus the issue's 56.3 tok/s (+2.5%).

The checked-in benchmark visualizations are available as immutable PNG assets:

- [dense crossover heatmap](https://raw.githubusercontent.com/liuhao-labs/sglang/pr-assets-36122/36122/dense-crossover-heatmap.png)
- [dense M=16 latency comparison](https://raw.githubusercontent.com/liuhao-labs/sglang/pr-assets-36122/36122/dense-m16-benchmark.png)
- [batched MoE speedup](https://raw.githubusercontent.com/liuhao-labs/sglang/pr-assets-36122/36122/moe-speedup.png)

The plots are directional GB10 measurements with a live service; they are
supporting evidence for the conservative gates, not cross-architecture claims.

## Gate evidence and remaining verification

The latest base and extra workflows still stop before build/test. For example,
[latest base gate job 113663936023](https://github.com/sgl-project/sglang/actions/runs/37882097010/job/113663936023)
records:

[captured gate log](https://github.com/liuhao-labs/sglang/blob/pr-assets-36122/36122/gate-0a39ec3a.log)

```text
PR Draft: false
Require run-ci: true
Missing required label 'run-ci'
Process completed with exit code 1.
```

The downstream wheel, AOT, Arm64, H100, B200, and other platform jobs are
therefore skipped. This is a repository permission gate, not a kernel test
failure. The author's token cannot add the label (`403 Must have admin rights
to Repository`).

The remaining evidence required for merge is repository-owned: add `run-ci`,
run the official CI matrix, and obtain AOT/SRT CODEOWNER approval. The local
and external data support merging because issue #35019 is still open, main has
no equivalent IQ MMQ implementation, the old wheel remains safe through the
capability handshake, and the routed-MoE path shows a large measured gain.
The evidence does not justify claiming cross-architecture performance or a
full exact-checkpoint 16k result; those remain outside the current proof.
