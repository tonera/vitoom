"""
family 归一化工具。
"""

from __future__ import annotations

from common.model_registry import MODEL_REGISTRY


def to_model_family(v: object) -> str:
    """
    将用户/系统输入的 family 值归一化为 canonical family。
    """
    return MODEL_REGISTRY.to_family(v)
