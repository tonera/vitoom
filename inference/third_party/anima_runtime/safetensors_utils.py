from __future__ import annotations

import os
import re
from typing import Optional, Union

import torch
from safetensors.torch import load_file


def get_split_weight_filenames(file_path: str) -> Optional[list[str]]:
    """
    Support HuggingFace-style split safetensors: *00001-of-00004.safetensors
    """
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if not match:
        return None

    prefix = basename[: match.start(2)]
    count = int(match.group(3))
    out: list[str] = []
    for i in range(count):
        filename = f"{prefix}{i + 1:05d}-of-{count:05d}.safetensors"
        full = os.path.join(os.path.dirname(file_path), filename)
        if not os.path.exists(full):
            raise FileNotFoundError(f"缺少分片权重文件: {full}")
        out.append(full)
    return out


def load_safetensors_state_dict(
    path: str,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    """
    Minimal safetensors loader for inference.
    """
    sd = load_file(path, device=str(device))
    if dtype is not None:
        for k in list(sd.keys()):
            sd[k] = sd[k].to(dtype=dtype)
    return sd


def load_split_safetensors_state_dict(
    path: str,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    paths = get_split_weight_filenames(path)
    if paths is None:
        return load_safetensors_state_dict(path, device=device, dtype=dtype)

    sd: dict[str, torch.Tensor] = {}
    for p in paths:
        sd.update(load_safetensors_state_dict(p, device=device, dtype=dtype))
    return sd

