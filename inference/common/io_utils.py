"""
通用 I/O 工具。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from common.logger import get_logger

logger = get_logger(__name__)


def _guess_suffix(url: str, default_suffix: str) -> str:
    try:
        path = urlparse(url).path
        suffix = Path(path).suffix
        if suffix:
            return suffix
    except Exception:
        pass
    return default_suffix


async def download_url_to_tempfile(
    url: str,
    *,
    default_suffix: str,
    timeout_seconds: float = 60.0,
    max_bytes: Optional[int] = None,
) -> Path:
    """
    下载 URL 到临时文件并返回路径。
    - max_bytes: 可选的大小上限（超过直接报错）
    """
    if not url or not isinstance(url, str):
        raise ValueError("url is required")
    if not (url.startswith("http://") or url.startswith("https://")):
        # 允许本地路径直通
        p = Path(url)
        if p.exists():
            return p
        raise ValueError(f"Unsupported url (must be http(s) or existing local path): {url}")

    suffix = _guess_suffix(url, default_suffix)
    fd, tmp_path = tempfile.mkstemp(prefix="vitoom_download_", suffix=suffix)
    Path(tmp_path).unlink(missing_ok=True)  # 让 aiohttp 重新创建

    try:
        Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    finally:
        try:
            import os

            os.close(fd)
        except Exception:
            pass

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            total = 0
            out_path = Path(tmp_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise ValueError(f"Downloaded file too large (> {max_bytes} bytes): {url}")
                    f.write(chunk)
            logger.info(f"Downloaded: url={url} -> {out_path} ({total} bytes)")
            return out_path
