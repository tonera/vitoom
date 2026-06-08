from __future__ import annotations

import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from common.model_metadata import safetensors_load_excluding_prefixes, safetensors_load_subset


@dataclass(frozen=True)
class ZImageSingleFileBuildContext:
    ckpt_path: str
    config_dir: str
    torch_dtype: torch.dtype
    local_files_only: bool
    logger: Any

def _remap_to_target_keys(
    tensors: Dict[str, torch.Tensor],
    target_keys: Iterable[str],
    *,
    prefix_to_strip: str,
    also_strip_model_prefix: bool,
) -> Dict[str, torch.Tensor]:
    target_keys = set(target_keys)
    remapped: Dict[str, torch.Tensor] = {}
    plen = len(prefix_to_strip)
    for k, v in tensors.items():
        if not k.startswith(prefix_to_strip):
            continue
        kk = k[plen:]
        if also_strip_model_prefix and kk.startswith("model."):
            kk = kk[len("model.") :]
        if kk in target_keys:
            remapped[kk] = v
    return remapped


def build_zimage_text_encoder_from_single_file(ctx: ZImageSingleFileBuildContext) -> torch.nn.Module:
    """
    ZImage 的 text_encoder 通常是 Qwen3Model（非 CLIP/T5），diffusers 的 single_file loader 不会自动灌权重。
    这里从 ckpt 抽取权重并映射到 Qwen3Model 的 state_dict key。
    """
    prefixes = [
        "text_encoders.qwen3_4b.transformer.",
        "text_encoders.qwen3_4b.",
        "text_encoder.",
    ]
    tensors = safetensors_load_subset(Path(ctx.ckpt_path), tuple(prefixes))
    if not tensors:
        raise RuntimeError(
            "在 checkpoint 里没找到 text encoder 权重前缀；请确认是否存在 `text_encoders.qwen3_4b.*` 或 `text_encoder.*`"
        )

    from transformers import AutoConfig, Qwen3Model  # type: ignore

    text_cfg = AutoConfig.from_pretrained(
        ctx.config_dir,
        subfolder="text_encoder",
        local_files_only=ctx.local_files_only,
    )

    # 避免超慢初始化：能用 accelerate 就用 init_empty_weights + assign load
    assign_supported = "assign" in inspect.signature(torch.nn.Module.load_state_dict).parameters
    init_ctx = nullcontext
    use_assign = False
    try:
        from accelerate import init_empty_weights  # type: ignore

        if assign_supported:
            init_ctx = init_empty_weights
            use_assign = True
    except Exception:
        init_ctx = nullcontext
        use_assign = False

    with init_ctx():
        text_encoder = Qwen3Model(text_cfg)

    target_keys = text_encoder.state_dict().keys()
    candidates: list[tuple[str, bool]] = [
        ("text_encoders.qwen3_4b.transformer.", True),
        ("text_encoders.qwen3_4b.", True),
        ("text_encoder.", True),
        ("text_encoders.qwen3_4b.transformer.", False),
        ("text_encoders.qwen3_4b.", False),
        ("text_encoder.", False),
    ]
    best: Dict[str, torch.Tensor] = {}
    best_desc: Optional[tuple[str, bool]] = None
    for p, strip_model in candidates:
        remapped = _remap_to_target_keys(tensors, target_keys, prefix_to_strip=p, also_strip_model_prefix=strip_model)
        if len(remapped) > len(best):
            best = remapped
            best_desc = (p, strip_model)

    if not best:
        raise RuntimeError("无法将 checkpoint 的 text encoder 权重映射到 Qwen3Model（可能是前缀/结构不一致）")

    load_kwargs = {"assign": True} if use_assign else {}
    incompat = text_encoder.load_state_dict(best, strict=False, **load_kwargs)
    missing = len(getattr(incompat, "missing_keys", []))
    unexpected = len(getattr(incompat, "unexpected_keys", []))
    try:
        ctx.logger.info(f"[zimage-single-file] text_encoder mapped={len(best)} missing={missing} unexpected={unexpected} via={best_desc}")
    except Exception:
        pass

    text_encoder = text_encoder.to(dtype=ctx.torch_dtype)
    text_encoder.eval()
    return text_encoder


def build_vae_from_single_file(ctx: ZImageSingleFileBuildContext) -> torch.nn.Module:
    """
    复用 diffusers 的 LDM->diffusers VAE 转换器；优先尝试直接 strip 前缀匹配。
    """
    from diffusers import AutoencoderKL  # type: ignore

    vae_cfg = AutoencoderKL.load_config(ctx.config_dir, subfolder="vae", local_files_only=ctx.local_files_only)
    vae = AutoencoderKL.from_config(vae_cfg)

    vae_tensors = safetensors_load_subset(Path(ctx.ckpt_path), ("vae.", "first_stage_model."))
    if not vae_tensors:
        raise RuntimeError("在 checkpoint 里没找到 VAE 权重前缀：`vae.*` 或 `first_stage_model.*`")

    target_keys = vae.state_dict().keys()
    direct_best: Dict[str, torch.Tensor] = {}
    for p in ["vae.", "first_stage_model."]:
        remapped = _remap_to_target_keys(vae_tensors, target_keys, prefix_to_strip=p, also_strip_model_prefix=False)
        if len(remapped) > len(direct_best):
            direct_best = remapped

    loaded = False
    if direct_best:
        incompat = vae.load_state_dict(direct_best, strict=False)
        missing = len(getattr(incompat, "missing_keys", []))
        unexpected = len(getattr(incompat, "unexpected_keys", []))
        try:
            ctx.logger.info(f"[zimage-single-file] vae direct_mapped={len(direct_best)} missing={missing} unexpected={unexpected}")
        except Exception:
            pass
        loaded = len(direct_best) > 1000

    if not loaded:
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint  # type: ignore

        converted = convert_ldm_vae_checkpoint(vae_tensors, vae_cfg)
        incompat = vae.load_state_dict(converted, strict=False)
        missing = len(getattr(incompat, "missing_keys", []))
        unexpected = len(getattr(incompat, "unexpected_keys", []))
        try:
            ctx.logger.info(f"[zimage-single-file] vae converted_mapped={len(converted)} missing={missing} unexpected={unexpected}")
        except Exception:
            pass

    vae = vae.to(dtype=ctx.torch_dtype)
    vae.eval()
    return vae


def build_zimage_transformer_from_single_file(ctx: ZImageSingleFileBuildContext) -> torch.nn.Module:
    from diffusers import ZImageTransformer2DModel  # type: ignore

    transformer_ckpt = safetensors_load_excluding_prefixes(
        Path(ctx.ckpt_path),
        (
            "text_encoders.",
            "text_encoder.",
            "vae.",
            "first_stage_model.",
        ),
    )

    transformer = ZImageTransformer2DModel.from_single_file(
        transformer_ckpt,
        config=ctx.config_dir,
        subfolder="transformer",
        torch_dtype=ctx.torch_dtype,
        local_files_only=ctx.local_files_only,
    )
    transformer.eval()
    return transformer


def build_zimage_overrides_for_single_file(
    *,
    ckpt_path: str,
    config_dir: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    logger: Any,
    need_text_encoder: bool,
    need_vae: bool,
    need_transformer: bool,
) -> dict[str, Any]:
    """
    按需构建 overrides（避免 model_config 已指定时重复构建）。
    """
    p = Path(str(ckpt_path))
    if not p.exists() or not p.is_file():
        raise ValueError(f"ckpt_path must be an existing file: {ckpt_path!r}")
    cfg = Path(str(config_dir))
    if not cfg.exists() or not cfg.is_dir():
        raise ValueError(f"config_dir must be an existing directory: {config_dir!r}")

    ctx = ZImageSingleFileBuildContext(
        ckpt_path=str(p),
        config_dir=str(cfg),
        torch_dtype=torch_dtype,
        local_files_only=bool(local_files_only),
        logger=logger,
    )

    out: dict[str, Any] = {}
    if need_text_encoder:
        out["text_encoder"] = build_zimage_text_encoder_from_single_file(ctx)
    if need_vae:
        out["vae"] = build_vae_from_single_file(ctx)
    if need_transformer:
        out["transformer"] = build_zimage_transformer_from_single_file(ctx)
    return out

