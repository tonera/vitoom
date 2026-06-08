"""统一会话 (chat) 服务模块。

对应重构计划 P2 / P3 的后端实现。包含：
    - LoadNameRouter：按 ``load_name`` 精确路由到推理服务；
    - session：ChatSession / Turn / SessionRuntime 状态机；
    - master_runtime：MasterAgentRuntime，同步执行一次 Run。

该模块承载统一 chat 会话的后端实现与运行时编排。
"""

from .master_runtime import MasterAgentRuntime
from .router import LoadNameRouter, get_load_name_router
from .session import (
    ChatSessionRuntime,
    InputMode,
    SessionRuntime,
    SessionState,
    Turn,
    TurnAssembler,
)

__all__ = [
    "LoadNameRouter",
    "get_load_name_router",
    "MasterAgentRuntime",
    "ChatSessionRuntime",
    "SessionRuntime",
    "SessionState",
    "InputMode",
    "Turn",
    "TurnAssembler",
]
