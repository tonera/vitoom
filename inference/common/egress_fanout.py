"""
Egress 抽象与 fanout 实现：
- 统一 send_result / send_task_status / is_connected 接口
- 允许将多种输出端（WS / Redis list / 未来更多）聚合成一个“逻辑 egress”

注意：为保持与现有代码兼容，我们让该模块的接口形状尽量贴近 WebSocketClient。
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable, List


@runtime_checkable
class EgressClient(Protocol):
    def is_connected(self) -> bool: ...

    async def send_result(self, result_message: dict) -> bool: ...

    async def send_stream_event(self, message: dict) -> bool: ...

    async def send_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> bool: ...


class FanoutEgress:
    """
    将多个 egress 组合在一起：
    - 发送时依次尝试，每个都发送
    - 返回值：只要有一个成功即 True
    """

    def __init__(self, clients: List[EgressClient]):
        self._clients = [c for c in clients if c is not None]

    def is_connected(self) -> bool:
        # 任意一个可用即可视为“可连接”
        for c in self._clients:
            try:
                if c.is_connected():
                    return True
            except Exception:
                continue
        return False

    async def send_result(self, result_message: dict) -> bool:
        ok = False
        for c in self._clients:
            try:
                ok = bool(await c.send_result(result_message)) or ok
            except Exception:
                # 单个通道异常不影响其他通道
                continue
        return ok

    async def send_stream_event(self, message: dict) -> bool:
        ok = False
        for c in self._clients:
            sender = getattr(c, "send_stream_event", None)
            if sender is None:
                continue
            try:
                ok = bool(await sender(message)) or ok
            except Exception:
                continue
        return ok

    async def send_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        ok = False
        for c in self._clients:
            try:
                ok = bool(await c.send_task_status(task_id=task_id, status=status, error=error, **kwargs)) or ok
            except Exception:
                continue
        return ok

