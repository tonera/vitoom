#!/usr/bin/env bash
set -euo pipefail

: "${VITOOM_WHEELS_DIR:?VITOOM_WHEELS_DIR is required}"

install_required_wheel() {
  local label="$1"
  local pattern="$2"

  local matches=()
  while IFS= read -r wheel; do
    matches+=("$wheel")
  done < <(find "$VITOOM_WHEELS_DIR" -maxdepth 1 -type f -name "$pattern" | sort)

  if [ "${#matches[@]}" -eq 1 ]; then
    python -m pip install --no-deps "${matches[0]}"
    return
  fi

  if [ "${#matches[@]}" -eq 0 ]; then
    echo "Missing required prebuilt wheel for ${label}. Expected exactly one ${pattern} in ${VITOOM_WHEELS_DIR}." >&2
    echo "Run: python scripts/setup_vitoom.py" >&2
  else
    echo "Multiple candidate wheels found for ${label}; keep only one ${pattern} in ${VITOOM_WHEELS_DIR}." >&2
    printf '  %s\n' "${matches[@]}" >&2
  fi
  return 1
}

python -m pip install tb-nightly -i https://pypi.org/simple
python -m pip install --no-deps -r /app/inference/requirements-sr.txt

install_required_wheel "FLASH_ATTN_WHEEL" "flash_attn-*torch2.11*.whl"
install_required_wheel "SPAS_SAGE_ATTN_WHEEL" "spas_sage_attn-*.whl"
install_required_wheel "NUNCHAKU_WHEEL" "nunchaku-*.whl"
install_required_wheel "TURBODIFFUSION_WHEEL" "turbodiffusion-*.whl"

python - <<'PY'
import importlib

modules = [
    "nunchaku",
    "nunchaku._C",
    "nunchaku.pipeline.pipeline_flux_pulid",
]

for module in modules:
    importlib.import_module(module)
    print(f"import ok: {module}")
PY
