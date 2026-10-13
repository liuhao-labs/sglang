FROM quay.io/gpustack/runner@sha256:28aabc1a96ec840d35a56b862bcacbcd198edddde3850a2862f70eb99cbc0e56
# Old optional prebuilt providers are incompatible with the new FlashInfer.
# Removing them selects FlashInfer's normal source/JIT fallback; mandatory
# source/runtime version checks stay enabled.
RUN python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache
