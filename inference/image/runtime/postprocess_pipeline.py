"""
统一图片后处理流程（供所有 handler 复用）

规则（与现有项目一致）：
- upscale: 仅在 upscale=2/4 时执行；同时支持 face_enhance。
- remove_bg: 必须最后执行，且输出格式强制 png（alpha 通道）。
"""

from __future__ import annotations

from typing import Any

from common.postprocess import remove_bg_image, upscale_image
from schemas import InferenceRequestParams


def apply_postprocess(image: Any, params: InferenceRequestParams, *, force_remove_bg: bool = False) -> Any:
    """对单张图片应用统一后处理。"""
    upscale = int(getattr(params, "upscale", 0) or 0)
    face_enhance = bool(getattr(params, "face_enhance", False))
    arch = str(getattr(params, "arch", "clean") or "clean")

    # 1) upscale / face enhance
    if upscale in (2, 4) or face_enhance:
        image = upscale_image(image, upscale, face_enhance=face_enhance, arch=arch)

    # 2) remove_bg (must be last)
    if force_remove_bg or bool(getattr(params, "remove_bg", False)):
        image = remove_bg_image(image)
        params.file_type = "png"

    return image


