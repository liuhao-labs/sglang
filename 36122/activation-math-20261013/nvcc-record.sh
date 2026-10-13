#!/bin/bash
set -euo pipefail
printf 'nvcc' >> "${NVCC_RECORD}"
printf ' %q' "$@" >> "${NVCC_RECORD}"
printf '\n' >> "${NVCC_RECORD}"
exec /usr/local/cuda/bin/nvcc "$@"
