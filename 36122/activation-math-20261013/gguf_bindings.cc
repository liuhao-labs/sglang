#include <sgl_kernel_ops.h>

// Test-only namespace: no replacement of the installed sgl_kernel operators.
TORCH_LIBRARY(gguf_probe, m) {
  m.def("dequantize", &ggml_dequantize);
  m.def("mul_mat", &ggml_mul_mat_a8);
  m.def("mul_mat_vec", &ggml_mul_mat_vec_a8);
  m.def("moe", &ggml_moe_a8);
  m.def("moe_vec", &ggml_moe_a8_vec);
  m.def("block_size", &ggml_moe_get_block_size);
#ifdef PROBE_PR
  m.def("supports_iq_mmq", &ggml_supports_iq_mmq);
#endif
}
