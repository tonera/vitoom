"""
Measure the runtime overhead of a "pin_memory auto probe".

The probe is intentionally simple and close to what we'd embed in an "auto"
decision: measure the ratio between many small H2D copies from:
  - pageable CPU tensors
  - pinned CPU tensors

This script prints a breakdown so you can decide if probing cost is acceptable.

Example (GB10):
  python test/test_pin_memory_probe_cost.py --device cuda:0 --n 2500
  python test/test_pin_memory_probe_cost.py --device cuda:0 --n 500 --repeat 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from datetime import datetime

import torch


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class ProbeResult:
    alloc_s: float
    touch_s: float
    pageable_copy_s: float
    pin_s: float
    pinned_copy_s: float

    @property
    def total_s(self) -> float:
        return self.alloc_s + self.pageable_copy_s + self.pin_s + self.pinned_copy_s

    @property
    def ratio(self) -> float:
        # pageable / pinned
        if self.pinned_copy_s <= 0:
            return float("inf")
        return self.pageable_copy_s / self.pinned_copy_s


def small_copy_probe(
    device: torch.device,
    *,
    n: int,
    rows: int,
    cols: int,
    dtype: str,
    reuse_alloc: bool,
    touch: bool,
) -> ProbeResult:
    if dtype == "fp16":
        dt = torch.float16
    elif dtype == "bf16":
        dt = torch.bfloat16
    elif dtype == "fp32":
        dt = torch.float32
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # 1) allocate CPU tensors
    t0 = time.perf_counter()
    tensors = [torch.empty((rows, cols), dtype=dt, device="cpu") for _ in range(n)]
    alloc_s = time.perf_counter() - t0

    # Optional: touch pages so timing reflects copy overhead, not first-touch/page faults.
    t0 = time.perf_counter()
    if touch:
        for t in tensors:
            t.zero_()
    touch_s = time.perf_counter() - t0

    # 2) pageable small copies
    sync(device)
    t0 = time.perf_counter()
    _ = [t.to(device, non_blocking=False) for t in tensors]
    sync(device)
    pageable_copy_s = time.perf_counter() - t0

    # 3) pin cost (page-lock)
    t0 = time.perf_counter()
    pinned = [t.pin_memory() for t in tensors]
    pin_s = time.perf_counter() - t0

    # 4) pinned small copies
    sync(device)
    t0 = time.perf_counter()
    _ = [t.to(device, non_blocking=True) for t in pinned]
    sync(device)
    pinned_copy_s = time.perf_counter() - t0

    # Optionally keep allocations alive for callers that want to reuse them.
    if reuse_alloc:
        _ = tensors, pinned

    return ProbeResult(
        alloc_s=alloc_s,
        touch_s=touch_s,
        pageable_copy_s=pageable_copy_s,
        pin_s=pin_s,
        pinned_copy_s=pinned_copy_s,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=500, help="number of small tensors (default: 500)")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--repeat", type=int, default=5, help="repeat probe N times (default: 5)")
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="warmup runs to discard from summary (default: 1). Useful because run#1 often pays first-touch/page-fault costs.",
    )
    ap.add_argument(
        "--touch",
        action="store_true",
        help="touch (zero) CPU tensors before timing copies, to reduce first-touch/page-fault noise (recommended).",
    )
    ap.add_argument(
        "--reuse-alloc",
        action="store_true",
        help="keep allocations alive (closer to embedding probe after a model load), mostly for experimentation",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"[{now()}] device={device} torch={torch.__version__}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        print(f"[{now()}] gpu={p.name} capability={p.major}.{p.minor} mem={p.total_memory/1024**3:.2f}GB")

    results: list[ProbeResult] = []
    for i in range(args.repeat):
        # best-effort cleanup between repeats (still not a perfect "cold" condition)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        r = small_copy_probe(
            device,
            n=args.n,
            rows=args.rows,
            cols=args.cols,
            dtype=args.dtype,
            reuse_alloc=args.reuse_alloc,
            touch=args.touch,
        )
        total_wall = time.perf_counter() - t0
        results.append(r)
        print(
            f"[{now()}] run={i+1}/{args.repeat} "
            f"alloc={r.alloc_s:.3f}s touch={r.touch_s:.3f}s pageable_copy={r.pageable_copy_s:.3f}s "
            f"pin={r.pin_s:.3f}s pinned_copy={r.pinned_copy_s:.3f}s "
            f"ratio={r.ratio:.2f} total={r.total_s:.3f}s wall={total_wall:.3f}s"
        )

    kept = results[args.warmup :] if args.warmup > 0 else results
    if not kept:
        kept = results
    totals = [r.total_s for r in kept]
    ratios = [r.ratio for r in kept]
    print(
        f"[{now()}] summary(kept={len(kept)}/{len(results)}, warmup_discarded={args.warmup}): "
        f"total_s median={statistics.median(totals):.3f}s "
        f"min={min(totals):.3f}s max={max(totals):.3f}s | "
        f"ratio median={statistics.median(ratios):.2f} min={min(ratios):.2f} max={max(ratios):.2f}"
    )


if __name__ == "__main__":
    main()


