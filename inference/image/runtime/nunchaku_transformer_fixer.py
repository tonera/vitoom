"""
最小版 QwenImage + nunchaku 运行时补丁（可复用）

目标：只保留已验证必要的两点修复
1) diffusers 的 apply_rotary_emb_qwen：当 freqs 比 query 更长时裁剪 freqs
2) nunchaku 的 transformer.pos_embed.forward：注入 max_txt_seq_len（避免 ValueError）

以及一个常用的显存配置：
- transformer 常驻 GPU，其它组件 enable_sequential_cpu_offload（可选再开 vae tiling）
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import functools

_CTX: dict[str, Any] = {"video_fhw": None}


def set_qwen_video_fhw_from_size(
    width: int, height: int, *, latent_scale_factor: int = 16
) -> Tuple[int, int, int]:
    fhw = (1, int(height) // int(latent_scale_factor), int(width) // int(latent_scale_factor))
    _CTX["video_fhw"] = fhw
    return fhw


def patch_apply_rotary_emb_qwen() -> None:
    try:
        from diffusers.models.transformers import transformer_qwenimage as _dq  # type: ignore
    except Exception:
        return
    if not hasattr(_dq, "apply_rotary_emb_qwen"):
        return
    if getattr(_dq.apply_rotary_emb_qwen, "__vitoom_patched__", False):
        return

    _orig = _dq.apply_rotary_emb_qwen

    @functools.wraps(_orig)
    def _patched(x, freqs_cis, use_real: bool = False):
        try:
            xl = int(x.shape[1])
            fs = getattr(freqs_cis, "shape", None)
        except Exception:
            return _orig(x, freqs_cis, use_real=use_real)
        if fs is not None and len(fs) >= 2:
            # 覆盖我们已观测到的主流形状：2D/3D/4D，序列维分别在 0/1/2
            # 经验上：编辑/多图场景下 img_query 可能是多段/多张图 token 拼接，
            # 但 freqs 仍按单张图的 H*W 生成，导致 freqs 序列维 < query 序列维。
            # 这里做一个“尽量不破坏语义”的兜底：在序列维上 tile 到 >= xl，再 slice 到 xl。
            def _tile_to_len(t, seq_dim: int, target: int):
                try:
                    cur = int(t.shape[seq_dim])
                except Exception:
                    return t
                if cur <= 0 or cur == target:
                    return t
                if cur > target:
                    # 过长：直接裁剪
                    sl = [slice(None)] * int(t.ndim)
                    sl[seq_dim] = slice(0, target)
                    return t[tuple(sl)]
                # 过短：tile + slice
                reps = (target + cur - 1) // cur
                tile_reps = [1] * int(t.ndim)
                tile_reps[seq_dim] = int(reps)
                try:
                    t2 = t.repeat(*tile_reps)
                except Exception:
                    # repeat 失败就退回原逻辑（让上游报错，便于暴露真实形状问题）
                    return t
                sl = [slice(None)] * int(t2.ndim)
                sl[seq_dim] = slice(0, target)
                return t2[tuple(sl)]

            if len(fs) == 2:
                freqs_cis = _tile_to_len(freqs_cis, 0, xl)
            elif len(fs) == 3:
                freqs_cis = _tile_to_len(freqs_cis, 1, xl)
            elif len(fs) == 4:
                freqs_cis = _tile_to_len(freqs_cis, 2, xl)
        return _orig(x, freqs_cis, use_real=use_real)

    _patched.__vitoom_patched__ = True
    _dq.apply_rotary_emb_qwen = _patched

    # nunchaku 里可能复制了同名引用（from-import），同步 patch
    try:
        import nunchaku.models.attention_processors.qwenimage as _nq  # type: ignore

        if hasattr(_nq, "apply_rotary_emb_qwen"):
            _nq.apply_rotary_emb_qwen = _patched
    except Exception:
        pass


def _infer_max_txt_seq_len(
    pipe: Any,
    prompt: Any,
    negative_prompt: Any,
    *,
    default: int = 256,
    margin: int = 8,
    align: int = 8,
) -> int:
    tok = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
    if tok is None:
        return int(default)

    def _len(text: Any) -> int:
        if text is None:
            return 0
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        ids = tok(text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True).input_ids
        return int(ids.shape[-1])

    v = max(_len(prompt), _len(negative_prompt)) or int(default)
    v = v + int(margin)
    if align and align > 1:
        v = ((v + (align - 1)) // align) * align
    return int(v)


def patch_nunchaku_qwenembedrope_forward(transformer: Any, *, fallback_max_txt_seq_len: int = 256) -> None:
    pos = getattr(transformer, "pos_embed", None)
    if pos is None or not hasattr(pos, "forward"):
        return
    if getattr(pos.forward, "__vitoom_patched__", False):
        return

    orig_forward = pos.forward

    @functools.wraps(orig_forward)
    def _patched_forward(*args, **kwargs):
        txt_seq_lens = args[1] if len(args) >= 2 else kwargs.get("txt_seq_lens", None)
        kwargs.pop("txt_seq_lens", None)
        kwargs["max_txt_seq_len"] = int(fallback_max_txt_seq_len)
        video_fhw = _CTX.get("video_fhw", None) or (args[0] if len(args) >= 1 else None)
        return orig_forward(video_fhw, txt_seq_lens, **kwargs) if len(args) >= 2 else orig_forward(video_fhw, **kwargs)

    _patched_forward.__vitoom_patched__ = True
    pos.forward = _patched_forward


def enable_qwenimage_nunchaku_compat(
    pipe: Any,
    transformer: Any,
    *,
    prompt: Any = None,
    negative_prompt: Any = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    latent_scale_factor: int = 16,
    fallback_max_txt_seq_len: Optional[int] = None,
) -> None:
    patch_apply_rotary_emb_qwen()
    if width is not None and height is not None:
        set_qwen_video_fhw_from_size(width, height, latent_scale_factor=latent_scale_factor)
    if fallback_max_txt_seq_len is None:
        fallback_max_txt_seq_len = _infer_max_txt_seq_len(pipe, prompt, negative_prompt)
    patch_nunchaku_qwenembedrope_forward(transformer, fallback_max_txt_seq_len=int(fallback_max_txt_seq_len))


def configure_hybrid_offload(
    pipe: Any,
    transformer: Any,
    *,
    device: str = "cuda",
    enable_vae_tiling: bool = True,
) -> None:
    def _is_bnb_quantized_model(m: Any) -> bool:
        # transformers bitsandbytes integration commonly sets these flags
        try:
            if bool(getattr(m, "is_loaded_in_4bit", False)) or bool(getattr(m, "is_loaded_in_8bit", False)):
                return True
        except Exception:
            pass
        # fallback: inspect a few submodules' class names (avoid importing bitsandbytes)
        try:
            for _name, sm in getattr(m, "named_modules", lambda: [])():
                cn = sm.__class__.__name__
                if "Linear4bit" in cn or "Linear8bit" in cn or "Params4bit" in cn:
                    return True
        except Exception:
            pass
        return False

    # transformer stays on GPU
    try:
        transformer.to(device)
    except Exception:
        pass
    # reduce VAE decode peak
    if enable_vae_tiling:
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
    # sequential cpu offload for the rest
    try:
        ex = getattr(pipe, "_exclude_from_cpu_offload", None)
        if isinstance(ex, list) and "transformer" not in ex:
            ex.append("transformer")
    except Exception:
        pass
    # 4/8 位量化（bnb）通常不兼容 sequential cpu offload，优先使用 model cpu offload
    use_model_cpu_offload = False
    try:
        te = getattr(pipe, "text_encoder", None)
        if te is not None and _is_bnb_quantized_model(te):
            use_model_cpu_offload = True
    except Exception:
        pass

    if use_model_cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
    else:
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass

