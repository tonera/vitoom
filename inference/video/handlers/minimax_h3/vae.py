"""视频 VAE（int8 convrot）和音频 VAE（折叠的 weight_norm）装进 diffusers。"""

from __future__ import annotations

import json
import re

import torch
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from safetensors import safe_open

from video.handlers.minimax_h3.quant import (
    QuantLinear,
    build_qtensor,
    group_quant_tensors,
    parse_quant,
    set_module,
    split_rows,
    swap_halves,
)


VIDEO_RENAME = [
    (r"^encoder\.down\.(\d+)\.block\.(\d+)\.nin_shortcut\.", r"encoder.down_blocks.\1.resnets.\2.conv_shortcut."),
    (r"^encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2."),
    (r"^encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0."),
    (r"^decoder\.x_embedder\.", "decoder.proj_in."),
    (r"\.attn\.to_out\.", ".attn.to_out.0."),
    (r"\.ff\.w1\.", ".ff.net.0.proj."),
    (r"\.ff\.w2\.", ".ff.net.2."),
]


def _video_rename(key: str) -> str:
    for pattern, repl in VIDEO_RENAME:
        key = re.sub(pattern, repl, key)
    return key


def _read(path: str, device: str):
    with safe_open(path, "pt", device=device) as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        meta = handle.metadata() or {}
    return tensors, meta


def _interleaved_to_stacked(tensors: dict[str, torch.Tensor], heads: int) -> dict[str, torch.Tensor]:
    """[head0: q k v, head1: q k v, ...] -> [q; k; v]。"""
    out = {}
    for name, value in tensors.items():
        if name == "comfy_quant" or value.ndim == 0:
            out[name] = value
            continue
        rows = value.shape[0]
        out[name] = (
            value.reshape(heads, 3, rows // (3 * heads), *value.shape[1:])
            .transpose(0, 1)
            .reshape(value.shape)
            .contiguous()
        )
    return out


def load_video_vae(path: str, device: str = "cuda", log=print) -> AutoencoderKLMiniMaxH3:
    tensors, meta = _read(path, device)
    info = json.loads(meta["minimax_h3_video_vae"])
    with torch.device(device):
        vae = AutoencoderKLMiniMaxH3(latents_mean=info["latents_mean"], latents_std=info["latents_std"])
    groups, plain = group_quant_tensors(tensors)
    for prefix, layer_tensors in groups.items():
        cfg = parse_quant(layer_tensors["comfy_quant"])
        bias = layer_tensors.pop("bias", None)
        target = _video_rename(prefix).rstrip(".")
        if target.endswith(".attn.to_qkv"):
            base = target[: -len("to_qkv")]
            heads = vae.config.decoder_num_attention_heads
            layer_tensors = _interleaved_to_stacked(layer_tensors, heads)
            if bias is not None:
                bias = _interleaved_to_stacked({"b": bias}, heads)["b"]
            biases = bias.chunk(3) if bias is not None else (None, None, None)
            for name, piece, piece_bias in zip(("to_q", "to_k", "to_v"), split_rows(cfg, layer_tensors, 3), biases):
                set_module(vae, base + name, QuantLinear(build_qtensor(cfg, piece), piece_bias, out_dtype=None))
            continue
        if target.endswith(".ff.net.0.proj"):
            layer_tensors = swap_halves(layer_tensors)
            if bias is not None:
                bias = torch.cat(bias.chunk(2)[::-1])
        set_module(vae, target, QuantLinear(build_qtensor(cfg, layer_tensors), bias, out_dtype=None))

    mask_token = plain.pop("decoder.mask_token", None)
    for key in ("latents_mean", "latents_std"):
        plain.pop(key, None)
    state = {_video_rename(key): (value.float() if value.is_floating_point() else value) for key, value in plain.items()}
    missing, unexpected = vae.load_state_dict(state, strict=False, assign=True)
    held = {_video_rename(prefix).rstrip(".") for prefix in groups}
    held |= {path.replace("to_qkv", name) for path in held for name in ("to_q", "to_k", "to_v")}
    missing = [name for name in missing if name.rsplit(".", 1)[0] not in held]
    mask_abs = None if mask_token is None else float(mask_token.abs().max().item())
    log(
        f"[video_vae] quant={len(groups)} plain={len(state)} missing={len(missing)} {missing[:6]} "
        f"unexpected={len(unexpected)} {unexpected[:6]} mask_token_absmax={mask_abs}"
    )
    if missing or unexpected:
        raise RuntimeError(f"视频 VAE 键未对齐 missing={missing[:8]} unexpected={unexpected[:8]}")
    return vae.eval().requires_grad_(False)


def load_audio_vae(path: str, device: str = "cuda", log=print) -> AutoencoderKLMiniMaxH3Audio:
    tensors, meta = _read(path, device)
    info = json.loads(meta["minimax_h3_audio_vae"])
    kwargs = {key: info[key] for key in ("latents_mean", "latents_std") if key in info}
    vae = AutoencoderKLMiniMaxH3Audio(**kwargs).to(device)
    reference = vae.state_dict()
    state = {}
    for key, value in tensors.items():
        if key in ("latents_mean", "latents_std"):
            continue
        if key.endswith(".weight") and key[: -len("weight")] + "weight_g" in reference:
            base = key[: -len("weight")]
            g_shape = reference[base + "weight_g"].shape
            dims = [dim for dim in range(value.ndim) if g_shape[dim] == 1]
            state[base + "weight_g"] = torch.linalg.vector_norm(value, dim=dims, keepdim=True)
            state[base + "weight_v"] = value
        else:
            state[key] = value
    missing, unexpected = vae.load_state_dict(state, strict=False)
    log(
        f"[audio_vae] keys={len(state)} missing={len(missing)} {missing[:6]} "
        f"unexpected={len(unexpected)} {unexpected[:6]}"
    )
    if missing or unexpected:
        raise RuntimeError(f"音频 VAE 键未对齐 missing={missing[:8]} unexpected={unexpected[:8]}")
    return vae.eval().requires_grad_(False)
