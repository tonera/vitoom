#!/usr/bin/env bash
set -euo pipefail

: "${FLASHINFER_VERSION:=0.5.3}"
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu130}"

# vLLM 0.14.0 wheel links against CUDA 12 runtime even when torch is cu130.
python -m pip install "nvidia-cuda-runtime-cu12" --extra-index-url "${TORCH_INDEX_URL}"

python -m pip install qwen-asr
python -m pip install "flashinfer-python==${FLASHINFER_VERSION}" "flashinfer-cubin==${FLASHINFER_VERSION}" -f https://flashinfer.ai

# qwen-asr / vllm may replace cu130 torch with a CPU build; restore the CUDA stack last.
python -m pip install \
  "torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1" \
  --index-url "${TORCH_INDEX_URL}"

python - <<'PY'
import glob
import os
import site
import sys

roots: list[str] = []
try:
    roots.extend(site.getsitepackages())
except Exception:
    pass

for root in roots:
    cudart = os.path.join(root, "nvidia", "cuda_runtime", "lib", "libcudart.so.12")
    if os.path.isfile(cudart):
        print(f"[post-install] verified {cudart}", file=sys.stderr)
        break
else:
    sys.stderr.write(
        "ERROR: nvidia-cuda-runtime-cu12 install did not provide libcudart.so.12\n"
    )
    raise SystemExit(1)

for root in roots:
    libtorch_cuda = os.path.join(root, "torch", "lib", "libtorch_cuda.so")
    if os.path.isfile(libtorch_cuda):
        print(f"[post-install] verified {libtorch_cuda}", file=sys.stderr)
        raise SystemExit(0)

sys.stderr.write(
    "ERROR: torch cu130 install did not provide torch/lib/libtorch_cuda.so\n"
)
raise SystemExit(1)
PY
