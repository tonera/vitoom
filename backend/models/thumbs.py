"""
模型缩略图处理（从 routes.py 抽离）。

职责：
- 将 http(s) thumb 下载并安全落盘到 outputs/models 下
- 将任意图片等比缩放到宽高不超过 768x768，并统一转码为 webp
- 返回可持久化的相对 outputs 根目录路径（例如：models/202601/uuid.webp）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from io import BytesIO

import anyio
from PIL import Image, ImageOps

from backend.core.config import get_config
from backend.core.logger import get_app_logger
from backend.utils import generate_uuid
from backend.utils.http_utils import HTTPClient

logger = get_app_logger(__name__)


def resolve_outputs_dir() -> Path:
    """
    解析 outputs 目录（storage.local.base_path），并规范为绝对路径。
    """
    outputs_dir = Path(get_config("storage.local.base_path", "resources/outputs"))
    if not outputs_dir.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        outputs_dir = (project_root / outputs_dir).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def _scale_to_webp_bytes(raw: bytes) -> bytes:
    """
    将任意图片 bytes 等比缩放至最大 768x768，并转码为 webp。

    注意：该函数是 CPU 密集型，同步执行；上层应放到线程池中运行，避免阻塞事件循环。
    """
    max_w = int(get_config("models.thumb_max_width", 768))
    max_h = int(get_config("models.thumb_max_height", 768))
    # webp 质量与编码参数
    quality = int(get_config("models.thumb_webp_quality", 85))
    method = int(get_config("models.thumb_webp_method", 6))

    bio = BytesIO(raw)
    with Image.open(bio) as im:
        # 处理 EXIF 旋转，保证方向正确
        im = ImageOps.exif_transpose(im)

        # 保留透明通道（webp 支持 alpha）
        if im.mode in ("RGBA", "LA"):
            pass
        elif im.mode == "P":
            # 调色板图像可能带透明，统一转 RGBA 更稳妥
            im = im.convert("RGBA")
        else:
            # 其他模式（如 CMYK）转 RGB
            im = im.convert("RGB")

        # 等比缩放：thumbnail 会保证不超过给定尺寸
        try:
            resample = Image.Resampling.LANCZOS  # Pillow>=9
        except Exception:
            resample = Image.LANCZOS
        im.thumbnail((max_w, max_h), resample=resample)

        out = BytesIO()
        # Pillow 若未编译 webp 支持，这里会抛异常；上层会记录日志并返回 400
        im.save(out, format="WEBP", quality=quality, method=method)
        return out.getvalue()


async def download_thumb_to_outputs_models(url: str) -> str:
    """
    当 thumb 是 http(s) URL 时，下载图片到 outputs/models/ 下，返回相对 outputs 根目录的路径：
    - 统一转码为 webp；返回示例：models/202601/uuid.webp
    """
    u = str(url or "").strip()
    if not (u.lower().startswith("http://") or u.lower().startswith("https://")):
        raise ValueError("thumb url must start with http(s)")

    max_size = int(get_config("upload.max_size", 50 * 1024 * 1024))
    # 缩略图下载大小限制：
    # - 旧逻辑默认 10MB，遇到“超大 PNG”会在转码缩放前就被拒绝
    # - 新逻辑默认跟随 upload.max_size（50MB），可用 models.thumb_download_max_size 单独调整
    thumb_download_max = int(get_config("models.thumb_download_max_size", max_size))
    limit = min(max_size, thumb_download_max)

    headers = {
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
    }

    ct = ""
    cl = ""
    status = None
    final_url = u
    data: bytes = b""
    try:
        async with HTTPClient(timeout=30.0) as client:
            resp = await client.get(u, headers=headers, follow_redirects=True)
            status = getattr(resp, "status_code", None)
            try:
                final_url = str(getattr(resp, "url", u))
            except Exception:
                final_url = u

            # 预判大小（如果服务端给了 content-length）
            cl = str(resp.headers.get("Content-Length", "") or "").strip()
            ct = str(resp.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()

            resp.raise_for_status()

            if cl:
                try:
                    if int(cl) > limit:
                        raise ValueError("thumb too large")
                except Exception:
                    # content-length 不可信就跳过
                    pass

            data = await resp.aread()
            if not data:
                raise ValueError("empty thumb")
            if len(data) > limit:
                raise ValueError("thumb too large")
    except Exception as e:
        # 打详细日志，便于定位（不要只靠猜）
        head_hex = ""
        try:
            head_hex = data[:32].hex()
        except Exception:
            head_hex = ""
        logger.error(
            "Thumb download failed: url=%s final_url=%s status=%s content_type=%s content_length=%s downloaded=%sB head_hex=%s err=%s",
            u,
            final_url,
            status,
            ct,
            cl,
            (len(data) if isinstance(data, (bytes, bytearray)) else -1),
            head_hex,
            str(e),
            exc_info=True,
        )
        raise

    # 统一处理：缩放 + 转 webp（放到线程池，避免阻塞事件循环）
    try:
        data = await anyio.to_thread.run_sync(_scale_to_webp_bytes, data)
    except Exception as e:
        logger.error(
            "Thumb convert to webp failed: url=%s final_url=%s content_type=%s downloaded=%sB err=%s",
            u,
            final_url,
            ct,
            len(data) if isinstance(data, (bytes, bytearray)) else -1,
            str(e),
            exc_info=True,
        )
        raise

    yyyymm = datetime.utcnow().strftime("%Y%m")
    rel_path = f"models/{yyyymm}/{generate_uuid()}.webp"

    outputs_dir = resolve_outputs_dir()
    abs_path = (outputs_dir / rel_path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    # 防目录穿越
    abs_path.relative_to(outputs_dir.resolve())
    abs_path.write_bytes(data)
    return rel_path

