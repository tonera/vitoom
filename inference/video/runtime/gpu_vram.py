"""
GPU VRAM probing helpers (best-effort).

Used by video handlers to decide whether to force low-VRAM/offload mode
based on current free VRAM.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def _to_cuda_device_str(device: Optional[str]) -> Optional[str]:
    if device is None:
        return None
    d = str(device).strip().lower()
    if not d:
        return None
    if d == "cuda":
        return "cuda:0"
    return d


def probe_cuda_free_gib(device: Optional[str] = None) -> Optional[float]:
    """
    Return current free CUDA VRAM in GiB for the given device (best-effort).
    - If CUDA is unavailable or probing fails, returns None.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None

    try:
        dev = _to_cuda_device_str(device)
        if dev is None:
            try:
                idx = int(torch.cuda.current_device())
            except Exception:
                idx = 0
            dev = f"cuda:{idx}"

        free_b, _total_b = torch.cuda.mem_get_info(dev)
        return float(free_b) / float(1024**3)
    except Exception:
        return None


def decide_force_offload(
    *,
    requested: bool,
    threshold_gib: float = 40.0,
    device: Optional[str] = None,
    logger: Optional[Any] = None,
    log_prefix: str = "",
) -> Tuple[bool, Optional[float]]:
    """
    Decide whether to force offload/low_vram based on current free VRAM.
    Returns: (force_offload, free_gib)
    """
    free_gib = probe_cuda_free_gib(device=device)
    force = bool(requested)

    try:
        thr = float(threshold_gib)
    except Exception:
        thr = 40.0

    if free_gib is not None and free_gib < thr:
        if (not force) and logger is not None:
            try:
                logger.info(
                    f"{log_prefix}free_vram_gib={free_gib:.2f} < {thr:.2f} -> enabling force_offload/low_vram"
                )
            except Exception:
                pass
        force = True

    return force, free_gib

