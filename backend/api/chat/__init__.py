"""统一会话 HTTP 接入层。

对应重构计划 §4：/v1/chat/sessions + /v1/chat/sessions/{id}/messages。
"""

from .routes import router

__all__ = ["router"]
