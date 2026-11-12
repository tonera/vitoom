"""把 Comfy 命名的 Turbo LoRA 以旁路方式挂到 QuantLinear 上。"""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors import safe_open

from video.handlers.minimax_h3.dit import rename_dit_key
from video.handlers.minimax_h3.quant import QuantLinear


def detach_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.lora.clear()


def attach_comfy_lora(model: nn.Module, path: str, strength: float = 1.0, device: str = "cuda", log=print) -> int:
    detach_lora(model)
    with safe_open(path, "pt", device=device) as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    count = 0
    for key in [name for name in state if name.endswith(".lora_A.weight")]:
        base = key[: -len(".lora_A.weight")]
        down = state[key].to(torch.bfloat16)
        up = state[base + ".lora_B.weight"].to(torch.bfloat16)
        alpha = state.get(base + ".alpha")
        scale = strength * (float(alpha.item()) / down.shape[0] if alpha is not None else 1.0)
        target = rename_dit_key(base[len("diffusion_model.") :] + ".").rstrip(".")
        if target.endswith(".attn.qkv_proj"):
            stem = target[: -len("qkv_proj")]
            for name, up_part in zip(("to_q", "to_k", "to_v"), up.chunk(3, dim=0)):
                model.get_submodule(stem + name).lora.append((down, up_part.contiguous(), scale))
            count += 1
            continue
        if target.endswith(".ff.net.0.proj"):
            gate, value = up.chunk(2, dim=0)
            up = torch.cat([value, gate], 0)
        model.get_submodule(target).lora.append((down, up.contiguous(), scale))
        count += 1
    log(f"[lora] attached {count} adapters from {path.rsplit('/', 1)[-1]} (strength={strength})")
    return count
