#!/usr/bin/env bash
# Prefer the CUDA toolkit ptxas over Triton wheel's bundled binary.
# GB10 / NVIDIA Spark (sm_121a) needs CUDA 13.x ptxas; older bundled ptxas fails with:
#   ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'
set -euo pipefail

cuda_ptxas="${CUDA_HOME:-/usr/local/cuda}/bin/ptxas"
if [ ! -x "$cuda_ptxas" ]; then
  echo "[link-triton-ptxas] skip: ${cuda_ptxas} not found" >&2
  exit 0
fi

triton_ptxas="$(python - <<'PY'
import os

import triton

print(os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas"))
PY
)"

if [ ! -e "$triton_ptxas" ]; then
  echo "[link-triton-ptxas] skip: ${triton_ptxas} not found" >&2
  exit 0
fi

triton_dir="$(dirname "$triton_ptxas")"
bundled_backup="${triton_ptxas}.vitoom-bundled"

if [ -L "$triton_ptxas" ]; then
  current_target="$(readlink -f "$triton_ptxas" || true)"
  if [ "$current_target" = "$(readlink -f "$cuda_ptxas")" ]; then
    echo "[link-triton-ptxas] already linked: ${triton_ptxas} -> ${cuda_ptxas}" >&2
    exit 0
  fi
  rm -f "$triton_ptxas"
elif [ -f "$triton_ptxas" ]; then
  if [ ! -e "$bundled_backup" ]; then
    mv "$triton_ptxas" "$bundled_backup"
  else
    rm -f "$triton_ptxas"
  fi
fi

ln -sf "$cuda_ptxas" "$triton_ptxas"
echo "[link-triton-ptxas] linked ${triton_ptxas} -> ${cuda_ptxas}" >&2
