python - <<'PY'
import json
from safetensors import safe_open

P = "/home/tonera/weights/nunchaku-flux.1-dev/svdq-fp4_r32-flux.1-dev.safetensors"  # 改成你的官方文件路径

with safe_open(P, framework="pt", device="cpu") as f:
    md = f.metadata() or {}
print("metadata keys:", sorted(md.keys()))

# 关键：quantization_config
qc = md.get("quantization_config", None)
print("\n=== quantization_config (raw) ===")
print(qc)

print("\n=== quantization_config (parsed) ===")
try:
    print(json.dumps(json.loads(qc), indent=2, ensure_ascii=False))
except Exception as e:
    print("parse failed:", e)

# 同时把 config 也贴一下（有时会用到）
cfg = md.get("config", None)
print("\n=== config exists? ===", cfg is not None)
PY

打印出键信息
python - <<'PY'
from safetensors import safe_open
P = "/home/tonera/weights/nunchaku-flux.1-dev/svdq-fp4_r32-flux.1-dev.safetensors"
with safe_open(P, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print("num_tensors:", len(keys))
print("head_keys:", keys[:30])
PY