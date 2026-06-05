"""
Benchmark FluxPipeline.to("cuda") on GB10 (and other GPUs) with different CPU-side preconditioning.

Motivation:
- On some platforms (e.g., ARM64 + unified/remote memory setups), `pipeline.to("cuda")` can be extremely slow,
  often due to many small H2D copies + CPU page faults/staging.
- Forcing weights to become "local" (clone) or at least page-touched before `.to("cuda")` can drastically reduce time.

Usage example:
  python test/test_flux_pipeline_to_cuda_bench.py \
    --model-dir /home/tonera/models/FLUX.1-dev \
    --weight-dir /home/tonera/weights \
    --precision auto \
    --mode baseline,clone \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Iterable

import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.utils import get_precision


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _mem() -> str:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) / (1024**3)
        reserv = torch.cuda.memory_reserved(idx) / (1024**3)
        return f"cuda_mem_alloc={alloc:.2f}GB reserved={reserv:.2f}GB"
    return "cuda_unavailable"


def _clone_module_params_buffers(module: torch.nn.Module) -> None:
    """Force parameters/buffers to materialize into fresh CPU storage (expensive in RAM, often fastest for `.to(cuda)`)."""
    with torch.no_grad():
        for p in module.parameters(recurse=True):
            # Replace storage to detach from e.g. mmapped/remote/unified pages.
            p.data = p.data.clone()
        for b in module.buffers(recurse=True):
            if isinstance(b, torch.Tensor) and hasattr(b, "data"):
                b.data = b.data.clone()


def _touch_module_params_buffers(module: torch.nn.Module) -> None:
    """Page-touch weights without allocating new storage (best-effort warmup).

    This may reduce page-fault storms but might not help if underlying storage is remote/unified.
    """
    with torch.no_grad():
        for p in module.parameters(recurse=True):
            if isinstance(p, torch.Tensor) and p.device.type == "cpu" and p.numel() > 0:
                _ = p.view(-1)[0].item()
        for b in module.buffers(recurse=True):
            if isinstance(b, torch.Tensor) and b.device.type == "cpu" and b.numel() > 0:
                _ = b.view(-1)[0].item()


def _apply_preconditioning(pipe: FluxPipeline, mode: str) -> None:
    modules: Iterable[str] = ("text_encoder", "text_encoder_2", "vae", "unet", "transformer")
    for name in modules:
        if hasattr(pipe, name):
            m = getattr(pipe, name)
            if not isinstance(m, torch.nn.Module):
                continue
            print(f"[{_now()}] precond {mode}: {name}")
            if mode == "clone":
                _clone_module_params_buffers(m)
            elif mode == "touch":
                _touch_module_params_buffers(m)
            else:
                raise ValueError(f"unknown mode={mode}")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Path to FLUX.1-dev (diffusers) model dir, e.g. /home/tonera/models/FLUX.1-dev")
    ap.add_argument("--weight-dir", required=True, help="Path to nunchaku weights dir, e.g. /home/tonera/weights")
    ap.add_argument("--precision", default="auto", help="nunchaku precision: auto/int4/fp4 (default: auto)")
    ap.add_argument("--device", default="cuda:0", help='Target device for pipeline.to(), e.g. "cuda:0"')
    ap.add_argument(
        "--mode",
        default="baseline,clone,touch",
        help='Comma-separated modes: baseline,clone,touch (default: "baseline,clone,touch")',
    )
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"[{_now()}] device={device} {_mem()}")

    precision = get_precision(args.precision, device, args.weight_dir)
    print(f"[{_now()}] precision={precision}")

    # Build nunchaku transformer on GPU (this part is not the benchmark target; we keep it constant).
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"{args.weight_dir}/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
        torch_dtype=None,
        device=device,
    )

    # Optional: attention impl switch (safe if available)
    try:
        transformer.set_attention_impl("nunchaku-fp16")
    except Exception as e:
        print(f"[{_now()}] WARN: transformer.set_attention_impl failed: {e!r}")

    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
        f"{args.weight_dir}/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
    )

    print(f"[{_now()}] building pipeline... {_mem()}")
    pipe = FluxPipeline.from_pretrained(
        args.model_dir,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
    )
    print(f"[{_now()}] pipeline built. {_mem()}")

    for mode in [m.strip() for m in args.mode.split(",") if m.strip()]:
        # Rebuild pipeline each time to keep CPU-side initial state consistent across modes.
        # This avoids 'first run' caching affecting later modes.
        print(f"\n[{_now()}] ===== mode={mode} =====")
        pipe2 = FluxPipeline.from_pretrained(
            args.model_dir,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )

        _sync_if_cuda(device)
        t0 = time.perf_counter()

        if mode in ("clone", "touch"):
            _apply_preconditioning(pipe2, mode)

        t1 = time.perf_counter()
        pipe2.to(device)
        _sync_if_cuda(device)
        t2 = time.perf_counter()

        print(f"[{_now()}] precond_time={t1 - t0:.3f}s to_cuda_time={t2 - t1:.3f}s total={t2 - t0:.3f}s {_mem()}")


if __name__ == "__main__":
    main()


