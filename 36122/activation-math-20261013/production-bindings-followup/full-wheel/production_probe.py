"""Validate full-wheel production GGUF bindings; never replace torch.ops."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import gguf
import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(.025)
import sgl_kernel
from sglang.srt.layers.quantization import gguf as g
from sglang.kernels.jit.utils.arch import get_activation_cuda_cflags

root = Path('/probe/full-wheel')
binary = Path(sgl_kernel.common_ops.__file__).resolve()
assert binary.is_relative_to(root/'site'), binary
assert Path(sgl_kernel.__file__).resolve().is_relative_to(root/'site')
assert torch.__version__.split('+')[0] == '2.14.1'
assert g._supports_iq_mmq is True
assert torch.ops.sgl_kernel.ggml_supports_iq_mmq.default() is True
assert torch.ops.sgl_kernel.ggml_moe_get_block_size.default(int(gguf.GGMLQuantizationType.Q5_0)) > 0
print(json.dumps({'kind': 'provenance', 'common_ops': str(binary),
                  'common_ops_sha256': hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
                  'python_wrapper': sgl_kernel.__file__, 'gguf_source': g.__file__,
                  'kernel_version': metadata.version('sglang-kernel'),
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'capability': g._supports_iq_mmq,
                  'activation_flags': get_activation_cuda_cflags(),
                  'namespace': 'sgl_kernel (full PR wheel)',
                  'source': json.loads((root/'source-manifest.json').read_text()),
                  'runtime_source_equivalence': json.loads((root/'source-equivalence.json').read_text())['latest_integration_sha'],
                  'gpu': torch.cuda.get_device_name(0)}), flush=True)

names = ('IQ1_S', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'IQ4_NL', 'IQ4_XS')
pairs = [(name, name) for name in names] + [('IQ3_S', 'Q5_0'), ('Q5_0', 'IQ3_S')]
fixtures = Path('/probe/fixtures/moe/2b77eb27ae2ea4b3f68cf3042961509a9f5847b8')
published = {row['case']: row['out'] for row in torch.load(
    '/probe/runtime-049/published-moe-integrated.pt', weights_only=True)}
outputs = []
max_quant_error = 0.
max_fallback_difference = 0.
for first, second in pairs:
    q1 = getattr(gguf.GGMLQuantizationType, first)
    q2 = getattr(gguf.GGMLQuantizationType, second)
    tensor1 = gguf.GGUFReader(fixtures/f'Quant_{first}_512.gguf').tensors[0]
    tensor2 = gguf.GGUFReader(fixtures/f'Quant_{second}_512.gguf').tensors[1]
    block1, packed1 = gguf.GGML_QUANT_SIZES[q1]
    block2, packed2 = gguf.GGML_QUANT_SIZES[q2]
    a = np.array(tensor1.data[:8, :512, :256//block1*packed1], copy=True)
    b = np.array(tensor2.data[:8, :256, :256//block2*packed2], copy=True)
    w1 = torch.tensor(a, device='cuda')
    w2 = torch.tensor(b, device='cuda')
    for dtype in (torch.float16, torch.bfloat16):
        dw1 = torch.tensor(gguf.dequantize(a, q1), device='cuda', dtype=dtype)
        dw2 = torch.tensor(gguf.dequantize(b, q2), device='cuda', dtype=dtype)
        for op in ('silu', 'gelu'):
            for tokens in (127, 128):
                torch.manual_seed(35019+int(q1))
                x = torch.randn(tokens, 256, device='cuda', dtype=dtype)*.25
                ids = torch.randint(0, 8, (tokens, 2), device='cuda', dtype=torch.int32)
                weights = torch.full((tokens, 2), .5, device='cuda', dtype=dtype)
                eligible = g._should_use_moe_mmq(int(q1), int(q2), tokens, 8, 2, 256, 256)
                assert eligible == (tokens == 128)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    out = g.fused_moe_gguf(x, w1, w2, weights, ids, int(q1), int(q2), op)
                events = {e.key: e.count for e in prof.key_averages()
                          if 'sgl_kernel::ggml_moe' in e.key or '_run_activation' in e.key}
                selected = 'sgl_kernel::ggml_moe_a8' if eligible else 'sgl_kernel::ggml_moe_a8_vec'
                other = 'sgl_kernel::ggml_moe_a8_vec' if eligible else 'sgl_kernel::ggml_moe_a8'
                assert events.get(selected) == 2 and events.get(other, 0) == 0, events
                gate = torch.einsum('mh,mknh->mkn', x, dw1[ids.long()])
                left, right = gate.chunk(2, dim=-1)
                activation = (torch.nn.functional.silu(left.float()) if op == 'silu' else
                              torch.nn.functional.gelu(left.float(), approximate='none'))*right.float()
                ref = (torch.einsum('mki,mkoi->mko', activation.to(dtype), dw2[ids.long()])
                       *weights[:, :, None]).sum(1)
                assert torch.isfinite(out).all()
                torch.testing.assert_close(out, ref, atol=1., rtol=.1)
                error = float((out-ref).float().abs().max())
                max_quant_error = max(max_quant_error, error)
                case = f'{first}_{dtype}_{op}'
                fallback_difference = None
                if tokens == 128 and first == second:
                    prior = published[case].to(out.device)
                    torch.testing.assert_close(out, prior, atol=.01, rtol=.01)
                    fallback_difference = float((out-prior).float().abs().max())
                    max_fallback_difference = max(max_fallback_difference, fallback_difference)
                print(json.dumps({'kind': 'moe_case', 'first': first, 'second': second,
                                  'tokens': tokens, 'dtype': str(dtype), 'activation': op,
                                  'status': 'PASS', 'events': events, 'max_abs': error,
                                  'published_fallback_max_abs': fallback_difference}), flush=True)
                outputs.append({'case': f'{first}_{second}_{dtype}_{op}_{tokens}', 'out': out.cpu()})
            # A real two-projection pipeline capture after both routes are warm.
            if first == second == 'IQ3_S':
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        g.fused_moe_gguf(x, w1, w2, weights, ids, int(q1), int(q2), op)
                torch.cuda.current_stream().wait_stream(stream)
                expected = out.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = g.fused_moe_gguf(x, w1, w2, weights, ids, int(q1), int(q2), op)
                graph.replay()
                assert torch.equal(captured, expected)
                graph.replay()
                assert torch.equal(captured, expected)
                torch.testing.assert_close(captured, ref, atol=1., rtol=.1)
                print(json.dumps({'kind': 'graph', 'dtype': str(dtype), 'activation': op,
                                  'status': 'PASS', 'exact_replay': True}), flush=True)
                del graph, captured, expected
    del w1, w2, dw1, dw2
    torch.cuda.empty_cache()

torch.save(outputs, root/'production-moe-outputs.pt')
assert len(outputs) == 80
payload = {'status': 'PASS', 'moe_cases': len(outputs), 'graphs': 4,
           'max_quant_reference_abs': max_quant_error,
           'max_published_fallback_abs': max_fallback_difference,
           'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
           'scope': 'real full wheel, production two-projection expert layer; no complete expert-model serving'}
root.joinpath('production-summary.json').write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps(payload), flush=True)
