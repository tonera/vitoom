"""
生成产物 / 用户上传的 storage 规范化与访问 URL 解析。

约定：
- 推理侧可使用 local | server | s3 | oss；local 不提供访问 URL。
- Backend 写入侧将 local 视为 server；访问 URL 仅对 server | s3 | oss 生成。
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import Request

from backend.core.config import get_config
from backend.utils.url_utils import normalize_outputs_path, to_absolute_outputs_url

ArtifactStorage = Literal["local", "server", "s3", "oss"]
BackendWritableStorage = Literal["server", "s3", "oss"]

_VALID_INFERENCE = frozenset({"local", "server", "s3", "oss"})
_VALID_BACKEND_WRITE = frozenset({"server", "s3", "oss"})


def normalize_storage_for_write(storage: Optional[str]) -> BackendWritableStorage:
    """
    Backend 落盘/上传使用的 storage。
    入参 local（或空且默认）→ server。
    """
    s = str(storage or "").strip().lower()
    if s == "local" or not s:
        return "server"
    if s in _VALID_BACKEND_WRITE:
        return s  # type: ignore[return-value]
    return "server"


def normalize_storage_label(storage: Optional[str]) -> ArtifactStorage:
    """校验并规范化 storage 标签（用于 DB 记录，保留 local）。"""
    s = str(storage or "").strip().lower()
    if s in _VALID_INFERENCE:
        return s  # type: ignore[return-value]
    return "local"


def resolve_artifact_public_url(
    storage: Optional[str],
    key: Optional[str],
    request: Optional[Request] = None,
) -> Optional[str]:
    """
    解析产物/上传的对外访问 URL。
    local 或空 key → None。
    """
    st = str(storage or "").strip().lower()
    rel = str(key or "").strip().lstrip("/")
    if not rel or st == "local":
        return None

    if st == "server":
        if request is not None:
            return to_absolute_outputs_url(request, rel)
        path = normalize_outputs_path(rel)
        public = str(get_config("server.public_base_url", "") or "").strip().rstrip("/")
        if public:
            return f"{public}{path}"
        return path

    if st == "s3":
        base = get_config("storage.s3.public_base_url", None)
        if base:
            return f"{str(base).strip().rstrip('/')}/{rel}"
        return None

    if st == "oss":
        base = get_config("storage.oss.public_base_url", None)
        if base:
            return f"{str(base).strip().rstrip('/')}/{rel}"
        return None

    return None


def enrich_file_http_url(
    file_dict: dict,
    request: Optional[Request] = None,
) -> dict:
    """按 storage + storage_path 刷新 http_url（用于列表 API）。"""
    if not isinstance(file_dict, dict):
        return file_dict
    storage = file_dict.get("storage")
    key = file_dict.get("storage_path")
    url = resolve_artifact_public_url(storage, key, request)
    if url:
        file_dict["http_url"] = url
    elif str(storage or "").strip().lower() == "local":
        file_dict["http_url"] = None
    return file_dict
