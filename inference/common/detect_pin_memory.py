import os, torch, typing
from typing import Any


_PIN_MEMORY_AUTO_CACHE: dict[int | None, bool] = {}


def _env_flag(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip().lower()
    return v if v else None


def _auto_pin_memory_probe(device: torch.device) -> bool:
    """
    One-time per-process probe to decide whether to enable pin_memory for many-small H2D copies.
    This avoids hard-coded heuristics (e.g. tying behavior to CPU architecture) and instead
    uses the current runtime's observed cost ratio between pageable and pinned CPU->GPU copies.
    Environment overrides:
      - NUNCHAKU_PIN_MEMORY=1/0      : force enable/disable (highest priority)
      - NUNCHAKU_PIN_MEMORY_AUTO_RATIO: ratio threshold (default: 5.0)
      - NUNCHAKU_PIN_MEMORY_PROBE_N/ROWS/COLS: probe tensor config (defaults: 64 , 1024, 1024)
      - NUNCHAKU_PIN_MEMORY_DEBUG=1  : print probe details
    """
    override = _env_flag("NUNCHAKU_PIN_MEMORY")
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False

    try:
        ratio_threshold = float(os.environ.get("NUNCHAKU_PIN_MEMORY_AUTO_RATIO", "5.0"))
    except Exception:
        ratio_threshold = 5.0

    n = int(os.environ.get("NUNCHAKU_PIN_MEMORY_PROBE_N", "64"))
    rows = int(os.environ.get("NUNCHAKU_PIN_MEMORY_PROBE_ROWS", "1024"))
    cols = int(os.environ.get("NUNCHAKU_PIN_MEMORY_PROBE_COLS", "1024"))
    debug = _env_flag("NUNCHAKU_PIN_MEMORY_DEBUG") in ("1", "true", "yes", "on")

    if device.type != "cuda" or not torch.cuda.is_available():
        return False

    # Best-effort: reduce probe size if allocations fail.
    while n >= 16:
        try:
            with torch.no_grad():
                # Do NOT pre-touch CPU pages here; cold first-touch behavior is relevant to real loads.
                tensors = [torch.empty((rows, cols), dtype=torch.float16, device="cpu") for _ in range(n)]

                torch.cuda.synchronize(device)
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                for t in tensors:
                    _ = t.to(device, non_blocking=False)
                t1.record()
                torch.cuda.synchronize(device)
                pageable_ms = t0.elapsed_time(t1)

                pinned = [t.pin_memory() for t in tensors]
                torch.cuda.synchronize(device)
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                for t in pinned:
                    _ = t.to(device, non_blocking=True)
                t1.record()
                torch.cuda.synchronize(device)
                pinned_ms = t0.elapsed_time(t1)

            ratio = float("inf") if pinned_ms <= 0 else pageable_ms / pinned_ms
            decision = ratio >= ratio_threshold
            if debug:
                print(
                    f"[nunchaku] pin_memory auto probe: device={device} n={n} shape=({rows},{cols}) "
                    f"pageable={pageable_ms/1000:.3f}s pinned={pinned_ms/1000:.3f}s "
                    f"ratio={ratio:.2f} threshold={ratio_threshold:.2f} -> {decision}"
                )
            return decision
        except Exception as e:
            if debug:
                print(f"[nunchaku] pin_memory auto probe failed (n={n}): {type(e).__name__}: {e}")
            n //= 2

    return False


def resolve_pin_memory(pin_memory: bool | str, device: str | torch.device) -> bool:
    """
    Resolve pin_memory behavior for loaders.
    - If device is not CUDA: always False
    - If pin_memory is True/False: return as-is
    - If pin_memory is "auto": run a one-time per-process probe (cached per CUDA device index)
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device.type != "cuda":
        return False
    if pin_memory != "auto":
        return bool(pin_memory)

    key = device.index
    if key in _PIN_MEMORY_AUTO_CACHE:
        return _PIN_MEMORY_AUTO_CACHE[key]
    decision = _auto_pin_memory_probe(device)
    _PIN_MEMORY_AUTO_CACHE[key] = decision
    return decision


def pin_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    """
    Pin CPU tensors in a state_dict to accelerate many small H2D copies.
    Returns a new dict with pinned tensors where possible; non-tensors (or non-CPU tensors) are preserved.
    """
    out: dict[str, Any] = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.device.type == "cpu" and v.numel() > 0:
            try:
                out[k] = v if v.is_pinned() else v.pin_memory()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out