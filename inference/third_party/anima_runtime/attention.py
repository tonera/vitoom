from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class AttentionParams:
    """
    Minimal attention params for Anima inference (torch SDPA only).

    Kept compatible with the call site in `anima_models.py`.
    """

    attn_mode: Optional[str] = "torch"
    split_attn: bool = False
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    @property
    def supports_fp32(self) -> bool:
        return True

    @property
    def requires_same_dtype(self) -> bool:
        return False

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        # For portability we only implement torch SDPA.
        if attn_mode not in (None, "torch"):
            raise ValueError(f"当前 anima_runtime 仅支持 attn_mode='torch'，收到: {attn_mode}")
        return AttentionParams(attn_mode="torch", split_attn=bool(split_attn))


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute attention output with PyTorch scaled_dot_product_attention.

    Expected input layout at call site:
    - q,k,v: [B, L, H, D]
    Output: [B, L, H*D]
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q = qkv_or_q
        assert k is not None and v is not None

    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # Convert to SDPA layout: [B, H, L, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # We don't use masks in current inference path (prompt padding already zeroed).
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop_rate)

    # Back: [B, L, H, D] -> [B, L, H*D]
    x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
    return x

