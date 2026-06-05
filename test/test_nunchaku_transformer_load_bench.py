"""
Benchmark + diagnose NunchakuFluxTransformer2dModel.from_pretrained load time with:
  - pin_memory (auto/on/off)
  - disable_mmap (on/off)
  - warm-cache vs cold-cache behavior (best-effort)

It also measures:
  - H2D bandwidth (pageable vs pinned)
  - many small H2D copies (pageable vs pinned)

Run on target machine (e.g. GB10):
  python test/test_nunchaku_transformer_load_bench.py \
    --weight /home/tonera/weights/nunchaku-flux.1-dev/svdq-fp4_r32-flux.1-dev.safetensors \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from nunchaku import NunchakuFluxTransformer2dModel


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

def _try_drop_caches() -> bool:
    """
    Best-effort "cold cache" helper for Linux.
    - If running as root: write to /proc/sys/vm/drop_caches
    - Else: try sudo (may require password; if so, it will fail fast)
    """
    if os.name != "posix":
        return False
    drop_path = "/proc/sys/vm/drop_caches"
    if not os.path.exists(drop_path):
        return False

    try:
        os.sync()
    except Exception:
        pass

    try:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            with open(drop_path, "w", encoding="utf-8") as f:
                f.write("3\n")
            return True
    except Exception:
        return False

    # Try sudo (non-interactive). If password is required, this will fail.
    try:
        subprocess.run(
            ["bash", "-lc", "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _mount_info_for(path: str) -> str:
    """
    Best-effort mount info for Linux, useful to diagnose mmap/page-cache behavior.
    """
    try:
        p = Path(path).resolve()
        if os.name != "posix" or not Path("/proc/mounts").exists():
            return "mount=unknown"

        best = None
        best_len = -1
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                dev, mnt, fstype = parts[0], parts[1], parts[2]
                try:
                    mnt_path = Path(mnt)
                except Exception:
                    continue
                if str(p).startswith(str(mnt_path)) and len(str(mnt_path)) > best_len:
                    best = (dev, mnt, fstype)
                    best_len = len(str(mnt_path))
        if best is None:
            return "mount=unknown"
        dev, mnt, fstype = best
        return f"mount={mnt} fstype={fstype} dev={dev}"
    except Exception:
        return "mount=unknown"


def h2d_bandwidth_test(device: torch.device, *, gb: float = 1.0) -> None:
    if device.type != "cuda":
        print(f"[{now()}] H2D test skipped (device={device})")
        return

    # Allocate ~gb GiB float32 on CPU.
    nbytes = int(gb * (1024**3))
    n = nbytes // 4
    x = torch.empty(n, dtype=torch.float32, device="cpu")

    sync(device)
    t0 = time.perf_counter()
    _ = x.to(device, non_blocking=False)
    sync(device)
    dt = time.perf_counter() - t0
    print(f"[{now()}] H2D pageable: {nbytes/dt/1e9:.3f} GB/s (size={gb:.2f}GiB, dt={dt:.3f}s)")

    xpin = x.pin_memory()
    sync(device)
    t0 = time.perf_counter()
    _ = xpin.to(device, non_blocking=True)
    sync(device)
    dt = time.perf_counter() - t0
    print(f"[{now()}] H2D pinned:   {nbytes/dt/1e9:.3f} GB/s (size={gb:.2f}GiB, dt={dt:.3f}s)")


def small_copy_test(device: torch.device, *, n: int = 2500) -> None:
    if device.type != "cuda":
        print(f"[{now()}] small-copy test skipped (device={device})")
        return

    # ~2MB each (1024x1024 fp16)
    tensors = [torch.empty((1024, 1024), dtype=torch.float16, device="cpu") for _ in range(n)]
    sync(device)
    t0 = time.perf_counter()
    _ = [t.to(device) for t in tensors]
    sync(device)
    dt = time.perf_counter() - t0
    print(f"[{now()}] small copies pageable: n={n} dt={dt:.3f}s")

    tensors_pin = [t.pin_memory() for t in tensors]
    sync(device)
    t0 = time.perf_counter()
    _ = [t.to(device, non_blocking=True) for t in tensors_pin]
    sync(device)
    dt = time.perf_counter() - t0
    print(f"[{now()}] small copies pinned:   n={n} dt={dt:.3f}s")


def load_once(
    weight_path: str,
    device: torch.device,
    *,
    pin_memory: Optional[Any],
    disable_mmap: Optional[bool],
):
    sync(device)
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {"device": device}
    if pin_memory is not None:
        kwargs["pin_memory"] = pin_memory
    if disable_mmap is not None:
        kwargs["disable_mmap"] = disable_mmap
    m = NunchakuFluxTransformer2dModel.from_pretrained(weight_path, **kwargs)
    sync(device)
    t1 = time.perf_counter()
    return m, t1 - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help="Path to nunchaku flux safetensors, e.g. /home/tonera/weights/...safetensors")
    ap.add_argument("--device", default="cuda:0", help='Device, e.g. "cuda:0" or "cpu"')
    ap.add_argument("--repeat", type=int, default=2, help="Repeat each case N times (default: 2). Run#2 is typically warm-cache.")
    ap.add_argument("--drop-caches", action="store_true", help="Try to drop Linux page cache before each run (requires root or passwordless sudo).")
    ap.add_argument("--skip-h2d", action="store_true", help="Skip H2D bandwidth test.")
    ap.add_argument("--skip-small-copies", action="store_true", help="Skip small-copy test.")
    ap.add_argument("--h2d-gb", type=float, default=1.0, help="H2D test size in GiB (default: 1.0).")
    ap.add_argument("--small-copies-n", type=int, default=2500, help="Number of small tensors in small-copy test (default: 2500).")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"[{now()}] weight={args.weight}")
    print(f"[{now()}] device={device}")
    print(f"[{now()}] platform.machine={platform.machine()} python={platform.python_version()} torch={torch.__version__}")
    print(f"[{now()}] {_mount_info_for(args.weight)}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"[{now()}] gpu={props.name} capability={props.major}.{props.minor} mem={props.total_memory/1024**3:.2f}GB")

    if not args.skip_h2d:
        h2d_bandwidth_test(device, gb=args.h2d_gb)
    if not args.skip_small_copies:
        small_copy_test(device, n=args.small_copies_n)

    cases = [
        ("baseline(defaults)", dict(pin_memory=None, disable_mmap=None)),
        ("pin_memory=auto", dict(pin_memory="auto", disable_mmap=None)),
        ("disable_mmap=True (pin default)", dict(pin_memory=None, disable_mmap=True)),
        ("disable_mmap=True (pin_memory=False)", dict(pin_memory=False, disable_mmap=True)),
        ("pin_memory=auto + disable_mmap=True", dict(pin_memory="auto", disable_mmap=True)),
    ]

    for name, kw in cases:
        print(f"\n[{now()}] ===== CASE: {name} =====")
        for i in range(args.repeat):
            # Make runs as comparable as possible.
            reset_cuda(device)
            gc.collect()
            if args.drop_caches:
                ok = _try_drop_caches()
                print(f"[{now()}] drop_caches={'OK' if ok else 'SKIP'} (run={i+1})")

            m, dt = load_once(args.weight, device, **kw)
            print(f"[{now()}] run={i+1}/{args.repeat} load_time={dt:.3f}s kwargs={kw}")

            # Release
            del m
            gc.collect()
            reset_cuda(device)


if __name__ == "__main__":
    main()


