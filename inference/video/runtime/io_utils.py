"""
视频推理 I/O 工具：
- 下载远程 URL 到本地临时文件
- 读取图片为 PIL.Image
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

from common.logger import get_logger
from common.io_utils import download_url_to_tempfile

logger = get_logger(__name__)


async def load_image_from_url_or_path(
    url_or_path: str,
    *,
    size: Optional[Tuple[int, int]] = None,
    rgb: bool = True,
    resize_mode: str = "crop",
    max_long_edge: Optional[int] = 2048,
    timeout_seconds: float = 30.0,
) -> Image.Image:
    """
    加载图片（支持 http(s) URL 或本地路径）。
    size: 若提供，则将图片变换到 (width, height)：
      - resize_mode="crop"（默认）：先按目标宽高比做 center-crop（取最大内接矩形），再缩放到目标尺寸（避免拉伸变形）
      - resize_mode="stretch"：直接拉伸到目标尺寸（不推荐，可能变形）
      - resize_mode="pad"：等比缩放后居中补边到目标尺寸（不丢内容但可能引入边框区域）
    max_long_edge: 若参考图过大，先将最长边缩到该上限（保持比例），降低内存与耗时；None 表示不预缩小。
    """
    path = await download_url_to_tempfile(
        url_or_path,
        default_suffix=".png",
        timeout_seconds=timeout_seconds,
        max_bytes=50 * 1024 * 1024,  # 50MB
    )

    def _open() -> Image.Image:
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im)
            if rgb and img.mode != "RGB":
                img = img.convert("RGB")

            if max_long_edge is not None:
                try:
                    mle = int(max_long_edge)
                except Exception:
                    mle = 0
                if mle > 0:
                    w0, h0 = img.size
                    long_edge = max(w0, h0)
                    if long_edge > mle:
                        scale = float(mle) / float(long_edge)
                        nw = max(1, int(round(w0 * scale)))
                        nh = max(1, int(round(h0 * scale)))
                        img = img.resize((nw, nh), Image.Resampling.LANCZOS)

            if size is not None:
                tw, th = int(size[0]), int(size[1])
                if tw <= 0 or th <= 0:
                    raise ValueError(f"Invalid size (width,height)={size}")

                mode = str(resize_mode or "crop").strip().lower()
                if mode in ("crop", "crop_center", "center_crop", "fit"):
                    # ImageOps.fit: center-crop to aspect ratio, then resize to exact size
                    img = ImageOps.fit(img, (tw, th), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                elif mode in ("stretch", "resize"):
                    img = img.resize((tw, th), Image.Resampling.LANCZOS)
                elif mode in ("pad", "letterbox", "contain"):
                    # contain keeps aspect ratio and fits within (tw,th); then pad to exact (tw,th)
                    fitted = ImageOps.contain(img, (tw, th), method=Image.Resampling.LANCZOS)
                    if rgb:
                        canvas = Image.new("RGB", (tw, th), (0, 0, 0))
                    else:
                        canvas = Image.new(fitted.mode, (tw, th))
                    x = (tw - fitted.size[0]) // 2
                    y = (th - fitted.size[1]) // 2
                    canvas.paste(fitted, (x, y))
                    img = canvas
                else:
                    raise ValueError(f"Unsupported resize_mode={resize_mode!r} (expected crop/stretch/pad)")

            return img

    return await asyncio.to_thread(_open)


async def download_and_preprocess_image_to_tempfile(
    url_or_path: str,
    *,
    size: Tuple[int, int],
    rgb: bool = True,
    resize_mode: str = "crop",
    max_long_edge: Optional[int] = 2048,
    timeout_seconds: float = 30.0,
    out_suffix: str = ".png",
) -> Path:
    """
    下载/读取图片后做统一预处理，并写入临时文件（用于需要 image_path 输入的后端）。
    - 输出图片默认保存为 PNG（避免 JPEG 反复压缩带来的质量损失）。
    """
    src_path = await download_url_to_tempfile(
        url_or_path,
        default_suffix=".png",
        timeout_seconds=timeout_seconds,
        max_bytes=50 * 1024 * 1024,  # 50MB
    )
    img = await load_image_from_url_or_path(
        str(src_path),
        size=size,
        rgb=rgb,
        resize_mode=resize_mode,
        max_long_edge=max_long_edge,
        timeout_seconds=timeout_seconds,
    )

    fd, tmp_path = tempfile.mkstemp(prefix="vitoom_video_img_", suffix=str(out_suffix or ".png"))
    try:
        os.close(fd)
    except Exception:
        pass
    out_path = Path(tmp_path)

    def _save() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 强制写入（覆盖 mkstemp 创建的空文件）
        img.save(out_path)

    await asyncio.to_thread(_save)
    return out_path

