"""comfy-kitchen 量化层包装。

量化按输出行分组，convrot 只旋转输入维，因此按行切分或重排是无损的。
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy_kitchen.tensor.base import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout
from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout


def parse_quant(raw: torch.Tensor) -> dict:
    return json.loads(bytes(raw.cpu().tolist()).decode())


def group_quant_tensors(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """把带 comfy_quant 的层按前缀（含末尾的点）归组，其余张量原样返回。"""
    prefixes = {key[: -len("comfy_quant")] for key in tensors if key.endswith(".comfy_quant")}
    groups: dict[str, dict[str, torch.Tensor]] = {}
    plain: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        prefix = key[: key.rfind(".") + 1]
        if prefix in prefixes:
            groups.setdefault(prefix, {})[key[len(prefix) :]] = value
        else:
            plain[key] = value
    return groups, plain


def build_qtensor(cfg: dict, tensors: dict[str, torch.Tensor], dtype=torch.bfloat16) -> QuantizedTensor:
    fmt = cfg["format"]
    weight = tensors["weight"]
    if fmt == "w6a8_int8":
        k = tensors["weight_s_rel"].shape[1] * cfg["group_size"]
        params = AsymW4A8Int8Layout.Params(
            scale=tensors["weight_s_rel"],
            s_channel=tensors["weight_s_channel"],
            orig_dtype=dtype,
            orig_shape=(weight.shape[0], k),
            group_size=cfg["group_size"],
            convrot_groupsize=cfg.get("convrot_groupsize", 256),
        )
        return QuantizedTensor(weight, "AsymW4A8Int8Layout", params)
    if fmt == "int8_tensorwise":
        params = TensorWiseINT8Layout.Params(
            scale=tensors["weight_scale"].float(),
            orig_dtype=dtype,
            orig_shape=tuple(weight.shape),
            convrot=cfg.get("convrot", False),
            convrot_groupsize=cfg.get("convrot_groupsize", 256),
        )
        return QuantizedTensor(weight, "TensorWiseINT8Layout", params)
    if fmt == "nvfp4":
        params = TensorCoreNVFP4Layout.Params(
            scale=tensors["weight_scale_2"].float(),
            orig_dtype=dtype,
            orig_shape=(weight.shape[0], weight.shape[1] * 2),
            block_scale=tensors["weight_scale"],
        )
        return QuantizedTensor(weight, "TensorCoreNVFP4Layout", params)
    raise ValueError(f"unsupported quant format {fmt}")


def split_rows(cfg: dict, tensors: dict[str, torch.Tensor], chunks: int) -> list[dict[str, torch.Tensor]]:
    """按输出行切开量化层。cfg 保留给调用方，切分本身不改格式。"""
    del cfg
    out = [dict() for _ in range(chunks)]
    for name, value in tensors.items():
        if name == "comfy_quant" or value.ndim == 0:
            for piece in out:
                piece[name] = value
            continue
        for piece, part in zip(out, value.chunk(chunks, dim=0)):
            piece[name] = part.contiguous()
    return out


def swap_halves(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """把 [gate; value] 换成 diffusers SwiGLU 要的 [value; gate]。"""
    left, right = split_rows({}, tensors, 2)
    swapped: dict[str, torch.Tensor] = {}
    for name in right:
        right_part = right[name]
        left_part = left[name]
        if right_part.ndim and name != "comfy_quant":
            swapped[name] = torch.cat([right_part, left_part], 0)
        else:
            swapped[name] = right_part
    return swapped


class QuantLinear(nn.Module):
    """量化线性层，LoRA 走旁路，不把低秩权重合并进量化矩阵。"""

    def __init__(
        self,
        qweight: QuantizedTensor,
        bias: torch.Tensor | None,
        out_dtype=torch.bfloat16,
        pre_quant_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.qweight = qweight
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.out_dtype = out_dtype
        self.pre_quant_scale = pre_quant_scale
        self.out_features, self.in_features = qweight.shape
        self.lora: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    @property
    def weight(self):
        # diffusers / transformers 有时只拿 weight.dtype 判断精度。
        if self.bias is not None:
            return self.bias
        return torch.empty(0, dtype=torch.bfloat16, device=self.qweight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out_dtype = self.out_dtype or x.dtype
        flat = x.reshape(-1, shape[-1]).to(torch.bfloat16)
        if self.pre_quant_scale is not None:
            flat = flat * self.pre_quant_scale
        y = F.linear(flat, self.qweight, None if self.bias is None else self.bias.to(torch.bfloat16))
        for down, up, scale in self.lora:
            # 原地加，避免 fc1 上再分配约 2 GiB 的临时结果。
            y.addmm_(flat @ down.T, up.T, alpha=scale)
        return y.reshape(*shape[:-1], -1).to(out_dtype)


def set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent, _, name = path.rpartition(".")
    owner = root.get_submodule(parent) if parent else root
    if name.isdigit():
        owner[int(name)] = module
    else:
        setattr(owner, name, module)
