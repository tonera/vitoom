"""
URL 拼接与规范化工具

目的：
- 统一生成“对外可访问的绝对 URL”，避免跨机器服务拿到 localhost 导致不可达
- 优先使用配置 server.public_base_url；未配置则回退到 request.base_url
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import Request

from backend.core.config import get_config


_ABS_URL_RE = re.compile(r"^(https?:)?//", re.IGNORECASE)
_DATA_URL_RE = re.compile(r"^data:", re.IGNORECASE)


def get_public_base_url(request: Request, *, config_key: str = "server.public_base_url") -> str:
    """
    返回对外可访问的 base url（不带末尾 /）。
    - 优先使用配置 server.public_base_url
    - 未配置则回退到当前请求的 base_url（例如 http://127.0.0.1:8888）
    """
    public_base_url = str(get_config(config_key, "") or "").strip().rstrip("/")
    if public_base_url:
        return public_base_url
    return str(request.base_url).rstrip("/")


def is_absolute_url(value: str) -> bool:
    s = (value or "").strip()
    if not s:
        return False
    return bool(_ABS_URL_RE.match(s) or _DATA_URL_RE.match(s))


def normalize_outputs_path(value: str) -> str:
    """
    将各种可能的 outputs 表达形式统一转换成以 "/" 开头的路径：
    - "/outputs/xxx.png" -> "/outputs/xxx.png"
    - "outputs/xxx.png" -> "/outputs/xxx.png"
    - "resources/outputs/xxx.png" -> "/outputs/xxx.png"
    - "xxx.png" -> "/outputs/xxx.png"
    """
    u = (value or "").strip()
    if not u:
        return ""

    if u.startswith("/"):
        return u
    if u.startswith("outputs/"):
        return f"/{u}"
    if u.startswith("resources/outputs/"):
        return f"/{u.replace('resources/', '', 1)}"
    return f"/outputs/{u.lstrip('/')}"


def to_absolute_outputs_url(request: Request, value: Optional[str]) -> Optional[str]:
    """
    将 outputs 相关路径/URL 转换为对外可访问的绝对 URL。
    若 value 已经是绝对 URL（http(s)://、//、data:），则原样返回。
    """
    if value is None:
        return None
    u = str(value).strip()
    if not u:
        return u
    if is_absolute_url(u):
        return u
    path = normalize_outputs_path(u)
    if not path:
        return ""
    return f"{get_public_base_url(request)}{path}"


