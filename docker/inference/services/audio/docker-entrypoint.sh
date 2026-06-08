#!/usr/bin/env bash
set -euo pipefail

export VITOOM_SERVICE_GROUP=audio
export VITOOM_SUPERVISOR_CONF=/app/docker/inference/services/audio/supervisord.conf

# vLLM 0.14 wheel expects libcudart.so.12 and libtorch_cuda.so from torch/lib.
_native_ld="$(
  PYTHONPATH="/app:/app/inference" python - <<'PY'
from common.cuda_libs_bootstrap import discover_native_lib_dirs

print(":".join(discover_native_lib_dirs()))
PY
)"
if [ -n "$_native_ld" ]; then
  export LD_LIBRARY_PATH="${_native_ld}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec /app/docker/inference/common/docker-entrypoint.sh "$@"
