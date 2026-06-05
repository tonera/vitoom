"""
推理参数构建相关的小工具函数（供 specs 与 builder 复用）
"""

from __future__ import annotations

from typing import Any

from common.logger import get_logger
from common.image_utils import calc_fit_size_with_multiple

logger = get_logger(__name__)


def resolve_edit_size_1024_square_multiple16(images: list[Any]) -> tuple[int, int]:
    """
    仅对编辑输入图计算推理用的 width/height（不对图片做 resize）：
    - 保持原图比例
    - 宽高不超过 1024x1024
    - 宽高为 16 的倍数
    返回：(width, height)，取第一张图的尺寸计算结果。
    """
    if not images:
        return 0, 0

    sizes: list[tuple[int, int]] = []
    for img in images:
        try:
            sizes.append(getattr(img, "size", None) or (0, 0))
        except Exception:
            sizes.append((0, 0))

    ow, oh = sizes[0]
    w, h = calc_fit_size_with_multiple(int(ow), int(oh), 1024, 1024, multiple=16)
    if any(s != (ow, oh) for s in sizes[1:]):
        logger.warning(
            "JT_ED tpl_list contains images with different original sizes; "
            "using first image size to compute height/width: %sx%s, sizes=%s",
            ow,
            oh,
            sizes,
        )
    return int(w), int(h)

