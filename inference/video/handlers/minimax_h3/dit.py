"""把 pruned 量化 DiT 装进 diffusers 的 MiniMaxH3Transformer3DModel。"""

from __future__ import annotations

import functools
import torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel
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


RENAME = [
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("video_patch_proj.", "proj_in."),
    ("audio_patch_proj.", "audio_proj_in."),
    ("condition_proj.", "context_embedder."),
    ("final_layer.norm.", "norm_out.norm."),
    ("final_layer.adaln_proj.linear.", "norm_out.linear."),
    ("final_layer.video_out.", "proj_out."),
    ("final_layer.audio_out.", "audio_proj_out."),
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
    (".attn.out_proj.", ".attn.to_out.0."),
    (".mlp.fc1.", ".ff.net.0.proj."),
    (".mlp.fc2.", ".ff.net.2."),
]


def rename_dit_key(key: str) -> str:
    if key.startswith("blocks."):
        key = "transformer_blocks." + key[len("blocks.") :]
    for src, dst in RENAME:
        key = key.replace(src, dst)
    return key


class PrunedTimeBasis(nn.Module):
    """t∈[0,1] 上的秩 8 系数表。第 0 行对应 t=0（噪声端），格点之间线性插值。"""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table.float(), persistent=False)
        self.on_step = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.on_step is not None:
            self.on_step()
        n = self.table.shape[0] - 1
        t = t.float().clamp(0, 1)
        pos = t * n
        lo = pos.floor().long().clamp(0, n - 1)
        weight = (pos - lo).unsqueeze(-1)
        return self.table[lo] * (1 - weight) + self.table[lo + 1] * weight


class PrunedModulation(nn.Module):
    """adaLN 直接吃 8 维系数，不再过 SiLU。"""

    def __init__(self, hidden_size: int, linear: nn.Linear):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = linear

    def forward(self, coeff: torch.Tensor):
        temb = self.linear(coeff.to(self.linear.weight.dtype)).float().to(torch.bfloat16)
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class PrunedNormOut(nn.Module):
    def __init__(self, norm: nn.Module, linear: nn.Linear):
        super().__init__()
        self.norm = norm
        self.linear = linear

    def forward(self, hidden_states, coeff, timestep_indices):
        shift, scale = self.linear(coeff.to(self.linear.weight.dtype)).to(torch.bfloat16).chunk(2, dim=-1)
        # transformer 随后会 .to(float32) 送进输出头，直接写 float32 省掉一份整段的 bf16 结果。
        return _modulated_norm(self.norm, hidden_states, scale, shift, timestep_indices, out_dtype=torch.float32)


_ROW_CHUNK = 8192


def _modulated_norm(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """norm(x) * (1 + scale[idx]) + shift[idx]，按行分段，不把逐 token 的 scale / shift 物化成整段。

    逐元素的运算和 dtype 与 diffusers 原式相同，结果逐位一致。packed 序列的 batch 恒为 1。
    """
    rows = hidden_states.view(-1, hidden_states.shape[-1])
    out = torch.empty(rows.shape, dtype=out_dtype or hidden_states.dtype, device=rows.device)
    for start in range(0, rows.shape[0], _ROW_CHUNK):
        end = min(start + _ROW_CHUNK, rows.shape[0])
        idx = indices[start:end]
        out[start:end] = norm(rows[start:end]) * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)
    return out.view(hidden_states.shape)


def _gated_residual_(hidden_states: torch.Tensor, update: torch.Tensor, gate: torch.Tensor, indices: torch.Tensor) -> None:
    """hidden += gate[idx] * update，按行分段原地累加。"""
    rows = hidden_states.view(-1, hidden_states.shape[-1])
    updates = update.view(-1, update.shape[-1])
    for start in range(0, rows.shape[0], _ROW_CHUNK):
        end = min(start + _ROW_CHUNK, rows.shape[0])
        rows[start:end] += gate.index_select(0, indices[start:end]) * updates[start:end]


def _lean_block_forward(block, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    """MiniMaxH3TransformerBlock.forward 的等价写法：调制分段计算，残差原地写回输入。

    输入是 transformer 循环里上一个 block 的输出，之后不会再被读取，所以可以原地改。
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
    normed = _modulated_norm(block.norm1, hidden_states, scale_msa, shift_msa, adaln_indices)
    update = block.attn(normed, rotary_emb, attention_mask)
    del normed
    _gated_residual_(hidden_states, update, gate_msa, adaln_indices)
    del update
    normed = _modulated_norm(block.norm2, hidden_states, scale_mlp, shift_mlp, adaln_indices)
    update = block.ff(normed)
    del normed
    _gated_residual_(hidden_states, update, gate_mlp, adaln_indices)
    return hidden_states


class _Identity(nn.Module):
    def __init__(self):
        super().__init__()
        # diffusers 用 get_parameter_dtype(time_embedder) 决定 temb 精度。
        self.register_buffer("_dtype_anchor", torch.zeros((), dtype=torch.float32), persistent=False)

    def forward(self, x):
        return x


def _inside_quant_linear(root: nn.Module, key: str) -> bool:
    module: nn.Module = root
    for part in key.split(".")[:-1]:
        if isinstance(module, QuantLinear):
            return True
        module = module[int(part)] if part.isdigit() else getattr(module, part, None)
        if module is None:
            return False
    return isinstance(module, QuantLinear)


def _read_checkpoint(path: str, device: str) -> dict[str, torch.Tensor]:
    with safe_open(path, "pt", device=device) as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


_SEQ_CHUNK = 1024


def _norm_rope_inplace(hidden_states: torch.Tensor, norm: nn.Module, rotary_emb) -> None:
    """先 qk-norm 再 RoPE，与 diffusers 同式，但按序列分段原地写回。

    整段做 RMSNorm 会产生两三份与 q / k 同大的临时张量，15 秒序列下每份约 1.1 GiB。
    """
    seq = hidden_states.shape[1]
    if rotary_emb is not None:
        cos, sin = (item.to(dtype=hidden_states.dtype) for item in rotary_emb)
        rotary_dim = cos.shape[-1]
        half = rotary_dim // 2
    for start in range(0, seq, _SEQ_CHUNK):
        end = min(start + _SEQ_CHUNK, seq)
        piece = hidden_states[:, start:end]
        piece.copy_(norm(piece))
        if rotary_emb is None:
            continue
        rot = piece[..., :rotary_dim]
        x1 = rot[..., :half].clone()
        x2 = rot[..., half:].clone()
        freq = cos[start:end].view(1, end - start, 1, rotary_dim)
        wave = sin[start:end].view(1, end - start, 1, rotary_dim)
        rot[..., :half] = x1 * freq[..., :half] - x2 * wave[..., :half]
        rot[..., half:] = x2 * freq[..., half:] + x1 * wave[..., half:]


class _ChunkedRopeProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        _norm_rope_inplace(query, attn.norm_q, rotary_emb)
        _norm_rope_inplace(key, attn.norm_k, rotary_emb)
        dtype = query.dtype
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        # 15 秒序列的 q/k/v 各约 1.1 GiB，留到 to_out 投影时会多占 3 GiB 以上。
        del query, key, value
        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _install_chunked_rope(model: nn.Module) -> None:
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.set_processor(_ChunkedRopeProcessor())


def _install_chunked_mlp(model: nn.Module, chunk_tokens: int = 4096) -> None:
    """MLP 的中间激活按 token 分段，避免 15 秒序列一次分配数 GiB。"""
    from diffusers.models.attention import FeedForward

    for module in model.modules():
        if not isinstance(module, FeedForward):
            continue
        original = module.forward

        def forward(hidden_states, *args, _orig=original, _chunk=chunk_tokens, **kwargs):
            if hidden_states.ndim < 2 or hidden_states.shape[-2] <= _chunk:
                return _orig(hidden_states, *args, **kwargs)
            out = torch.empty_like(hidden_states)
            seq = hidden_states.shape[-2]
            for start in range(0, seq, _chunk):
                end = min(start + _chunk, seq)
                out[..., start:end, :] = _orig(hidden_states[..., start:end, :], *args, **kwargs)
            return out

        module.forward = forward


def _lean_transformer_forward(
    model,
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    timestep,
    timestep_indices,
    token_tags,
    position_ids,
    video_indices,
    audio_indices,
    text_indices,
    attention_kwargs=None,
    return_dict: bool = True,
):
    """MiniMaxH3Transformer3DModel.forward 的等价写法：各模态嵌入拷进打包序列后立即释放。

    原实现把 float32 的 video_embeds（约两份隐藏状态大小）留到 50 个 block 全部跑完。
    attention_kwargs 只服务 PEFT 的 LoRA 缩放，这里的 LoRA 是 QuantLinear 旁路，用不到。
    """
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM,
        MiniMaxH3TransformerOutput,
    )

    del attention_kwargs
    rotary_emb = model.rope(position_ids)
    text_embeds = model.token_refiner(
        model.context_embedder(encoder_hidden_states.to(get_parameter_dtype(model.context_embedder)))
    )
    packed = text_embeds.new_zeros((text_embeds.shape[0], position_ids.shape[0], text_embeds.shape[-1]))
    packed.index_copy_(1, text_indices, text_embeds)
    del text_embeds
    video_embeds = model.proj_in(hidden_states.to(get_parameter_dtype(model.proj_in)))
    packed.index_copy_(1, video_indices, video_embeds.to(packed.dtype))
    del video_embeds
    audio_embeds = model.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(model.audio_proj_in)))
    packed.index_copy_(1, audio_indices, audio_embeds.to(packed.dtype))
    del audio_embeds

    temb = model.time_embedder(model.time_proj(timestep).to(get_parameter_dtype(model.time_embedder)))
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
    for block in model.transformer_blocks:
        packed = block(packed, temb, adaln_indices, rotary_emb)

    out = model.norm_out(packed, temb, timestep_indices).to(get_parameter_dtype(model.proj_out))
    del packed
    video_output = model.proj_out(out).index_select(1, video_indices)
    audio_output = model.audio_proj_out(out).index_select(1, audio_indices)
    if not return_dict:
        return (video_output, audio_output)
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def _install_lean_forward(model: nn.Module) -> None:
    model.forward = functools.partial(_lean_transformer_forward, model)
    for block in model.transformer_blocks:
        block.forward = functools.partial(_lean_block_forward, block)


def load_pruned_dit(path: str, device: str = "cuda", log=print) -> MiniMaxH3Transformer3DModel:
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel()

    groups, plain = group_quant_tensors(_read_checkpoint(path, device))
    for prefix, tensors in groups.items():
        cfg = parse_quant(tensors["comfy_quant"])
        bias = tensors.pop("bias", None)
        target = rename_dit_key(prefix).rstrip(".")
        out_dtype = torch.float32 if target == "proj_out" else torch.bfloat16
        if target.endswith(".attn.qkv_proj"):
            base = target[: -len("qkv_proj")]
            for name, piece in zip(("to_q", "to_k", "to_v"), split_rows(cfg, tensors, 3)):
                set_module(model, base + name, QuantLinear(build_qtensor(cfg, piece), None))
            continue
        if target.endswith(".ff.net.0.proj"):
            tensors = swap_halves(tensors)
        set_module(model, target, QuantLinear(build_qtensor(cfg, tensors), bias, out_dtype=out_dtype))

    table = plain.pop("adaln_t_table")
    plain.pop("rope.inv_freq", None)
    model.time_proj = PrunedTimeBasis(table).to(device)
    model.time_embedder = _Identity().to(device)
    hidden = model.config.hidden_size
    for index, block in enumerate(model.transformer_blocks):
        weight = plain.pop(f"blocks.{index}.adaln_proj.linear.weight")
        bias = plain.pop(f"blocks.{index}.adaln_proj.linear.bias")
        linear = nn.Linear(weight.shape[1], weight.shape[0], device=device, dtype=weight.dtype)
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        block.adaln_proj = PrunedModulation(hidden, linear)
    weight = plain.pop("final_layer.adaln_proj.linear.weight")
    bias = plain.pop("final_layer.adaln_proj.linear.bias")
    linear = nn.Linear(weight.shape[1], weight.shape[0], device=device, dtype=weight.dtype)
    linear.weight.data.copy_(weight)
    linear.bias.data.copy_(bias)
    model.norm_out = PrunedNormOut(model.norm_out.norm, linear)

    state = {rename_dit_key(key): value for key, value in plain.items()}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [
        name
        for name in missing
        if not name.endswith("rope.inv_freq")
        and "adaln_proj" not in name
        and not name.startswith("norm_out.linear")
        and not _inside_quant_linear(model, name)
    ]
    log(
        f"[dit] quant layers={len(groups)} plain tensors={len(state)} "
        f"missing={missing[:8]} ({len(missing)}) unexpected={unexpected[:8]} ({len(unexpected)})"
    )
    if missing or unexpected:
        raise RuntimeError(f"DiT 键未对齐 missing={missing[:12]} unexpected={unexpected[:12]}")

    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    model.rope = MiniMaxH3RotaryPosEmbed(model.config.rope_freq_dim, model.config.rope_theta).to(device)
    _install_chunked_rope(model)
    _install_chunked_mlp(model)
    _install_lean_forward(model)
    leftovers = [name for name, param in list(model.named_parameters()) + list(model.named_buffers()) if param.device.type == "meta"]
    if leftovers:
        raise RuntimeError(f"DiT 仍有 meta 张量: {leftovers[:10]}")
    return model.eval().requires_grad_(False)
