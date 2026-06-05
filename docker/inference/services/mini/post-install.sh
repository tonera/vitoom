#!/usr/bin/env bash
set -euo pipefail

: "${VITOOM_WHEELS_DIR:?VITOOM_WHEELS_DIR is required}"
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu130}"

install_flash_attn_wheel() {
  local matches=()
  while IFS= read -r wheel; do
    matches+=("$wheel")
  done < <(find "$VITOOM_WHEELS_DIR" -maxdepth 1 -type f -name 'flash_attn-*torch2.11*.whl' | sort)

  if [ "${#matches[@]}" -eq 1 ]; then
    python -m pip install --no-deps "${matches[0]}"
    return
  fi

  if [ "${#matches[@]}" -eq 0 ]; then
    echo "Missing required prebuilt wheel for MINI_FLASH_ATTN_WHEEL. Expected exactly one flash_attn-*torch2.11*.whl in ${VITOOM_WHEELS_DIR}." >&2
    echo "Run: python scripts/setup_vitoom.py" >&2
  else
    echo "Multiple flash_attn wheels found; keep only one flash_attn-*torch2.11*.whl in ${VITOOM_WHEELS_DIR}." >&2
    printf '  %s\n' "${matches[@]}" >&2
  fi
  return 1
}

python -m pip install uv
python -m pip install "vllm==0.21.0" \
  --extra-index-url "${TORCH_INDEX_URL}"
uv pip install --system -r /app/docker/inference/services/mini/requirements.txt
install_flash_attn_wheel
