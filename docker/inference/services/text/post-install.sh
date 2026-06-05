#!/usr/bin/env bash
set -euo pipefail

: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu130}"
: "${VLLM_VERSION:=0.21.0}"

python -m pip install "vllm==${VLLM_VERSION}" \
  --extra-index-url "${TORCH_INDEX_URL}"
