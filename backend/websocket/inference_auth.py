"""推理器 WebSocket 连接鉴权（/ws/inference/{service_id}）。"""

from __future__ import annotations

import secrets
from typing import Mapping, Optional

from backend.core.dotenv_loader import load_project_dotenv
from backend.core.config import get_config
from backend.core.logger import get_app_logger

logger = get_app_logger(__name__)


def get_configured_inference_token() -> str:
    """读取期望的推理 WS token；为空表示不启用鉴权。"""
    load_project_dotenv()
    return str(get_config("inference.token", "") or "").strip()


def extract_client_inference_token(
    *,
    query_token: Optional[str],
    headers: Mapping[str, str],
) -> str:
    """从 query / Authorization / X-Inference-Token 提取客户端 token。"""
    if query_token and str(query_token).strip():
        return str(query_token).strip()

    auth = str(headers.get("authorization") or headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
        if bearer:
            return bearer

    header_token = str(
        headers.get("x-inference-token") or headers.get("X-Inference-Token") or ""
    ).strip()
    return header_token


def is_inference_ws_authorized(
    *,
    query_token: Optional[str],
    headers: Mapping[str, str],
) -> bool:
    """校验推理 WS 连接 token。

    - 未配置 ``inference.token``：放行（兼容历史部署）
    - 已配置：客户端必须提供匹配的 token
    """
    expected = get_configured_inference_token()
    if not expected:
        return True

    provided = extract_client_inference_token(query_token=query_token, headers=headers)
    if not provided:
        return False

    try:
        return secrets.compare_digest(provided, expected)
    except TypeError:
        return False


def reject_unauthorized_inference_ws(
    *,
    service_id: str,
    query_token: Optional[str],
    headers: Mapping[str, str],
) -> bool:
    """若鉴权失败则记录日志并返回 True（调用方应拒绝连接）。"""
    if is_inference_ws_authorized(query_token=query_token, headers=headers):
        return False

    logger.warning(
        "Inference WebSocket auth rejected for service_id=%s "
        "(token_configured=%s, query_token_present=%s, "
        "authorization_header_present=%s, x_inference_token_present=%s)",
        service_id,
        bool(get_configured_inference_token()),
        bool(query_token and str(query_token).strip()),
        bool(
            str(headers.get("authorization") or headers.get("Authorization") or "").strip()
        ),
        bool(
            str(headers.get("x-inference-token") or headers.get("X-Inference-Token") or "").strip()
        ),
    )
    return True
