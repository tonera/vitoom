"""
后处理模块（图片）

规则（按当前项目约定）：
- upscale: 仅在 upscale=2/4 时执行；同时支持 face_enhance（GFPGAN）。
- remove_bg: 必须最后执行，且输出格式强制 png（alpha 通道）。

实现参考：
- 超分/人脸增强：aiservice/diffusers/tonera/UpscaleEnhance.py
- 去背景：aiservice/RMBG-1.4/RbgProcessor.py
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from .config_loader import load_inference_config
from .rbg_processor import RbgConfig, RbgProcessor, resolve_rmbg_backend_and_dir
from .upscale_enhance import UpscaleEnhance, UpscaleEnhanceConfig


def upscale_image(image: Any, scale: int, *, face_enhance: bool = False, arch: str = "clean") -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"upscale_image expects PIL.Image.Image, got {type(image)}")
    if scale not in (2, 4) and not face_enhance:
        return image

    inference_config = load_inference_config()
    cfg = UpscaleEnhanceConfig(
        weights_dir=inference_config.models_dir,
        mode="normal",
        arch=arch or "clean",
        upscale=scale,
        face_enhance=bool(face_enhance),
    )
    processor = UpscaleEnhance(cfg)
    return processor.run([image])[0]


def remove_bg_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"remove_bg_image expects PIL.Image.Image, got {type(image)}")

    inference_config = load_inference_config()
    backend, model_dir = resolve_rmbg_backend_and_dir(inference_config.models_dir)
    processor = RbgProcessor(RbgConfig(model_dir=model_dir, backend=backend))
    out = processor.generate([image])[0]
    return out.convert("RGBA")


