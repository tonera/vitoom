"""
下载相关的 websocket 广播与来源字段校验（从 routes.py 抽离）。

职责：
- 规范化/校验 source.provider/source.repo_id
- 封装对 download worker 的广播消息结构
"""

from typing import Any, Dict, Tuple

from fastapi import HTTPException

from backend.utils import utc_now


def normalize_source(source: Dict[str, Any]) -> Tuple[str, str]:
    src = source if isinstance(source, dict) else {}
    provider = str(src.get("provider") or "").strip().lower()
    repo_id = str(src.get("repo_id") or "").strip()
    return provider, repo_id


def validate_source(provider: str, repo_id: str) -> None:
    if provider not in {"huggingface", "modelscope", "civitai"} or not repo_id:
        raise HTTPException(status_code=400, detail="source.provider/source.repo_id is required")


async def broadcast_download_message(
    *,
    model_key: str,
    source: Dict[str, Any],
    asset_type: str,
    message_type: str,
) -> int:
    """
    统一的 download worker 广播入口（download / download_cancel）。
    """
    from backend.websocket.manager import get_websocket_manager

    manager = get_websocket_manager()
    message = {
        "type": message_type,
        "model_key": model_key,
        "source": dict(source or {}),
        "asset_type": asset_type,
        "timestamp": utc_now().isoformat(),
    }
    return await manager.broadcast_to_download_services(message)

