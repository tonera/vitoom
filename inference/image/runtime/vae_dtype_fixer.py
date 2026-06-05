"""
VAE dtype 修复器（运行时兜底）

问题背景：
- 部分环境/模型组合下，SDXL pipeline 的 VAE 内部会出现参数 dtype 不一致（例如 post_quant_conv.bias=float32，
  但 decode 输入 latents 为 float16），从而触发：
    RuntimeError: Input type (c10::Half) and bias type (float) should be the same

策略：
- 在每次调用 pipeline 前做一次轻量检查：
  - 以 UNet dtype 为准（通常是 fp16/bf16）
  - 若发现 VAE 的关键层 dtype 与 UNet 不一致，或 weight/bias dtype 不一致，则对齐到目标 dtype
"""

from __future__ import annotations

from typing import Any, Optional


def ensure_vae_dtype(pipe: Any, *, logger: Any = None) -> None:
    try:
        import torch
    except Exception:
        return

    vae = getattr(pipe, "vae", None)
    unet = getattr(pipe, "unet", None)
    if vae is None or unet is None:
        return

    # 目标 dtype：以 unet 为准
    target_dtype: Optional[torch.dtype] = getattr(unet, "dtype", None)
    if target_dtype is None:
        try:
            target_dtype = next(iter(unet.parameters())).dtype
        except Exception:
            target_dtype = None
    if target_dtype is None:
        return

    pqc = getattr(vae, "post_quant_conv", None)
    if pqc is None:
        # 兜底：仍尝试把整个 vae 对齐到 target_dtype
        try:
            if getattr(vae, "dtype", None) != target_dtype:
                vae.to(dtype=target_dtype)
        except Exception:
            pass
        return

    try:
        w = getattr(pqc, "weight", None)
        b = getattr(pqc, "bias", None)
        w_dtype = getattr(w, "dtype", None)
        b_dtype = getattr(b, "dtype", None)
    except Exception:
        w_dtype = None
        b_dtype = None

    # 仅在确实有问题时才转换，避免每次推理都 to()
    need_fix = False
    if w_dtype is not None and w_dtype != target_dtype:
        need_fix = True
    if b_dtype is not None and b_dtype != target_dtype:
        need_fix = True
    if w_dtype is not None and b_dtype is not None and w_dtype != b_dtype:
        need_fix = True

    if not need_fix:
        return

    try:
        if logger is not None:
            logger.warning(
                f"VAE dtype mismatch detected; fixing to {target_dtype}. "
                f"post_quant_conv.weight={w_dtype}, bias={b_dtype}, unet={target_dtype}"
            )
    except Exception:
        pass

    # 1) 先整体对齐
    try:
        vae.to(dtype=target_dtype)
    except Exception:
        pass

    # 2) 再确保 post_quant_conv 的 weight/bias 同 dtype（某些情况下整体 to() 可能不会覆盖到 bias）
    try:
        w = getattr(pqc, "weight", None)
        b = getattr(pqc, "bias", None)
        if w is not None and getattr(w, "dtype", None) != target_dtype:
            w.data = w.data.to(dtype=target_dtype)
        if b is not None and getattr(b, "dtype", None) != target_dtype:
            b.data = b.data.to(dtype=target_dtype)
    except Exception:
        pass


