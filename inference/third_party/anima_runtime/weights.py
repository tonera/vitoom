from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .safetensors_utils import load_split_safetensors_state_dict


def strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def only_keep_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_state_dict_any(
    path: str,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    strip_net_prefix: bool = False,
) -> dict[str, torch.Tensor]:
    sd = load_split_safetensors_state_dict(path, device=device, dtype=dtype)
    if strip_net_prefix:
        sd = strip_prefix(sd, "net.")
    return sd


@dataclass(frozen=True)
class StateDictLoadReport:
    missing_keys: list[str]
    unexpected_keys: list[str]


def load_model_state_dict_strictish(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    allow_missing_prefixes: tuple[str, ...] = (),
) -> StateDictLoadReport:
    """
    Similar spirit to sd-scripts: strict=False but fail on suspicious missing/unexpected.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)

    if allow_missing_prefixes:
        filtered = []
        for k in missing:
            if any(k.startswith(p) for p in allow_missing_prefixes):
                continue
            filtered.append(k)
        missing = filtered

    if unexpected:
        raise RuntimeError(f"权重包含未预期的 keys（示例前 10 个）: {unexpected[:10]}")
    if missing:
        raise RuntimeError(f"权重缺少 keys（示例前 10 个）: {missing[:10]}")

    return StateDictLoadReport(missing_keys=missing, unexpected_keys=unexpected)

