"""UNet wrapper + PositionalEncoding (vendored from LiveTalking
``avatars/musetalk/models/unet.py``)。

去掉了上游 ``__main__`` 块（无意义的空 ``UNet()`` 调用）。其它实现一字未改。
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 384, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x


class UNet():
    def __init__(self, unet_config: str, model_path: str,
                 use_float16: bool = False, device=None):
        with open(unet_config, 'r') as f:
            unet_config_obj = json.load(f)
        self.model = UNet2DConditionModel(**unet_config_obj)
        self.pe = PositionalEncoding(d_model=384)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = (
            torch.load(model_path)
            if torch.cuda.is_available()
            else torch.load(model_path, map_location=self.device)
        )
        self.model.load_state_dict(weights)
        if use_float16:
            self.model = self.model.half()
        self.model.to(self.device)


__all__ = ["UNet", "PositionalEncoding"]
