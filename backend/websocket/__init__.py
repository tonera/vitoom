"""WebSocket 模块入口。

``router`` 聚合：

    - ``/ws/task/{task_id}``：任务进度订阅（routes.py）
    - ``/ws/model/{model_key}``：模型下载订阅（routes.py）
    - ``/ws/inference/{service_id}``：推理服务回连（routes.py）
    - ``/ws/chat/{session_id}``：统一会话实时通道（chat_routes.py）
"""

from fastapi import APIRouter

from .manager import WebSocketManager, get_websocket_manager

__all__ = [
    "WebSocketManager",
    "get_websocket_manager",
    "router",
]


def _build_router() -> APIRouter:
    from .chat_routes import router as chat_router
    from .routes import router as base_router

    aggregate = APIRouter()
    aggregate.include_router(base_router)
    aggregate.include_router(chat_router)
    return aggregate


def __getattr__(name):
    if name == "router":
        return _build_router()
    raise AttributeError(name)
