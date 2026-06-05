import glob
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import torch


def pick_largest(paths: list[str]) -> str:
    return max(paths, key=lambda p: os.path.getsize(p) if os.path.exists(p) else -1)


def is_fp8_supported() -> bool:
    """
    Best-effort runtime check for FP8 availability.
    We avoid env vars and avoid assuming a specific torch version.
    """
    try:
        if not torch.cuda.is_available():
            return False
        # Newer torch may provide a direct helper.
        if hasattr(torch.cuda, "is_fp8_supported"):
            try:
                return bool(torch.cuda.is_fp8_supported())  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback: require FP8 dtype + Hopper-class (SM90+) GPU.
        if not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")):
            return False
        major, _minor = torch.cuda.get_device_capability()
        return int(major) >= 9
    except Exception:
        return False


def select_umt5_ckpt_in_roots(roots: Iterable[Path], *, required: bool = True) -> Optional[str]:
    """
    Select umT5 encoder checkpoint with preference:
    - if fp8 supported: fp8.pth > int8.pth > bf16.pth > other .pth > safetensors
    - else:            int8.pth > bf16.pth > other .pth > safetensors
    """
    fp8_ok = is_fp8_supported()
    patterns: list[str] = []
    if fp8_ok:
        patterns.append("models_t5_umt5-xxl-enc-fp8.pth")
    patterns.extend(
        [
            "models_t5_umt5-xxl-enc-int8.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "models_t5_umt5-xxl-enc-*.pth",
            "*umt5*enc*.pth",
            "*umt5*.pth",
            "*t5*enc*.pth",
            "*t5*.pth",
            # fallback: packaged safetensors (will be converted to .pth by caller)
            "umt5_fp8.safetensors",
            "*umt5*enc*.safetensors",
            "*umt5*.safetensors",
            "*t5*enc*.safetensors",
            "*t5*.safetensors",
        ]
    )
    return find_one_in_roots(roots, patterns, required=required)


def find_one(root: Path, patterns: list[str], *, required: bool = True) -> Optional[str]:
    tried: list[str] = []
    for pat in patterns:
        tried.append(pat)
        matches = sorted(glob.glob(str(root / pat)))
        if matches:
            return pick_largest(matches)
    if required:
        raise FileNotFoundError(f"Missing model file. patterns={tried}, root={root}")
    return None


def find_one_in_roots(roots: Iterable[Path], patterns: list[str], *, required: bool = True) -> Optional[str]:
    """
    Search patterns in multiple roots in order, returning the first match.
    This is used for shared components (umt5/vae/tokenizer) with fallback roots.
    """
    checked: list[str] = []
    for r in roots:
        try:
            rr = Path(r).expanduser().resolve()
        except Exception:
            rr = Path(r)
        checked.append(str(rr))
        if not rr.exists():
            continue
        hit = find_one(rr, patterns, required=False)
        if hit:
            return hit
    if required:
        raise FileNotFoundError(f"Missing model file. patterns={patterns}, roots={checked}")
    return None


def list_pth(root: Path) -> list[str]:
    return sorted(glob.glob(str(root / "*.pth")))


def _safe_basename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in s)[:180]


def _cache_path(prefix: str, src: Path) -> Path:
    try:
        stat = src.stat()
        sig = f"{src}:{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:
        sig = str(src)
    h = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / "vitoom_turbo_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{prefix}_{_safe_basename(src.stem)}_{h}.pth"


def ensure_pth_from_safetensors(path: str, *, prefix: str) -> str:
    """
    Convert `.safetensors` -> `.pth` and cache in /tmp (or OS temp dir).
    """
    p = str(path)
    if p.endswith(".pth") or p.endswith(".pt"):
        return p
    if not p.endswith(".safetensors"):
        raise ValueError(f"Unsupported checkpoint format: {p} (expected .pth/.pt or .safetensors)")

    src = Path(p).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    out_pth = _cache_path(prefix, src)
    if out_pth.exists():
        return str(out_pth)

    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Need `safetensors` to convert .safetensors -> .pth. "
            "Please `pip install safetensors` (or provide a .pth checkpoint)."
        ) from e

    state = load_file(str(src), device="cpu")
    # Default cast to bf16 for compatibility (fp8 weights may appear in packaged models).
    for k, v in list(state.items()):
        try:
            if hasattr(v, "dtype") and v.dtype != torch.bfloat16:
                state[k] = v.to(dtype=torch.bfloat16)
        except Exception:
            state[k] = v

    torch.save(state, str(out_pth))
    return str(out_pth)


def resolve_tokenizer_dir(model_root: Path) -> Optional[str]:
    cand = (model_root / "google" / "umt5-xxl").resolve()
    return str(cand) if cand.exists() else None


def resolve_tokenizer_dir_in_roots(roots: Iterable[Path]) -> Optional[str]:
    """
    Resolve tokenizer directory from roots in order.
    Expected structure: <root>/google/umt5-xxl/{spiece.model,...}
    """
    for r in roots:
        try:
            rr = Path(r).expanduser().resolve()
        except Exception:
            rr = Path(r)
        if not rr.exists():
            continue
        cand = (rr / "google" / "umt5-xxl").resolve()
        if cand.exists():
            return str(cand)
    return None


def _is_aux_ckpt(name: str) -> bool:
    s = name.lower()
    return ("vae" in s) or ("umt5" in s) or ("t5" in s and "enc" in s)


def select_t2v_dit(root: Path) -> str:
    cands = [p for p in list_pth(root) if not _is_aux_ckpt(os.path.basename(p))]
    if not cands:
        raise FileNotFoundError(f"Missing DiT checkpoint (.pth) in model dir: {root}")
    return pick_largest(cands)


def select_i2v_high_low(root: Path) -> tuple[str, str]:
    cands = [p for p in list_pth(root) if not _is_aux_ckpt(os.path.basename(p))]
    if not cands:
        raise FileNotFoundError(f"Missing DiT checkpoints (.pth) in model dir: {root}")
    high = [p for p in cands if "high" in os.path.basename(p).lower()]
    low = [p for p in cands if "low" in os.path.basename(p).lower()]
    if not high or not low:
        raise FileNotFoundError(
            f"Missing I2V high/low checkpoints in {root}. "
            f"Expected filenames containing 'high' and 'low'. Found: {[os.path.basename(p) for p in cands]}"
        )
    return pick_largest(high), pick_largest(low)


def infer_quant_linear(*, model_cfg: Optional[dict], ckpt_paths: list[str]) -> bool:
    if isinstance(model_cfg, dict) and "quant_linear" in model_cfg:
        return bool(model_cfg.get("quant_linear"))
    for p in ckpt_paths:
        if "quant" in os.path.basename(p).lower():
            return True
    return False

