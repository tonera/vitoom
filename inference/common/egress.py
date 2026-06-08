"""
Egress（结果/状态出口）抽象与实现。

目标：
- 保持现有消息格式（result + task_status）不变
- 允许将消息发送到不同通道（WebSocket / Redis List 等）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from .logger import get_logger
from .message_cache import MessageCache

logger = get_logger(__name__)


class EgressClient(Protocol):
    def is_connected(self) -> bool: ...

    async def send_result(self, result_message: dict) -> bool: ...

    async def send_task_status(
        self, task_id: str, status: str, error: Optional[str] = None, **kwargs
    ) -> bool: ...


def build_task_status_message(
    *, task_id: str, status: str, error: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    构造 task_status 消息 dict（与 WebSocketClient.send_task_status 的格式保持一致）。
    """
    msg: Dict[str, Any] = {
        "type": "task_status",
        "task_id": task_id,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if error:
        msg["error"] = error
    msg.update(kwargs)
    return msg


@dataclass
class MultiEgressClient:
    """
    将一条消息 fan-out 到多个 egress（例如未来需要同时发 WS + Redis 时）。
    默认语义：
    - 只要任意一个成功返回 True，即视为成功（返回 True）
    """

    clients: List[EgressClient]

    def is_connected(self) -> bool:
        return any(getattr(c, "is_connected", lambda: True)() for c in self.clients)

    async def send_result(self, result_message: dict) -> bool:
        ok = False
        for c in self.clients:
            try:
                ok = (await c.send_result(result_message)) or ok
            except Exception as e:
                logger.warning(f"Egress send_result failed: {e}")
        return ok

    async def send_task_status(
        self, task_id: str, status: str, error: Optional[str] = None, **kwargs
    ) -> bool:
        ok = False
        for c in self.clients:
            try:
                ok = (await c.send_task_status(task_id, status, error=error, **kwargs)) or ok
            except Exception as e:
                logger.warning(f"Egress send_task_status failed: {e}")
        return ok


class RedisListEgressClient:
    """
    Redis List 作为结果出口：
    - result/task_status 都写入同一个 list key（reschannle）
    - 使用 LPUSH（保持与部分既有系统一致；也可改 RPUSH，但需对齐消费端）
    """

    def __init__(
        self,
        *,
        redis,  # redis.asyncio.Redis
        list_key: str,
        message_cache: Optional[MessageCache] = None,
        lpush: bool = True,
    ):
        self._redis = redis
        self._list_key = str(list_key)
        self._message_cache = message_cache
        self._lpush = bool(lpush)

    def is_connected(self) -> bool:
        # redis-py 的连接池是惰性的；这里返回 True，具体失败交由 send_* 捕获
        return True

    async def _push(self, payload: Dict[str, Any], *, task_id: Optional[str], status: str) -> bool:
        try:
            raw = json.dumps(payload, ensure_ascii=False)
            if self._lpush:
                await self._redis.lpush(self._list_key, raw)
            else:
                await self._redis.rpush(self._list_key, raw)
            return True
        except Exception as e:
            logger.warning(f"Redis egress push failed: key={self._list_key}, err={e}")
            if self._message_cache and task_id:
                try:
                    await self._message_cache.save_status_result(task_id, status, payload)
                except Exception:
                    pass
            return False

    async def send_result(self, result_message: dict) -> bool:
        task_id = result_message.get("task_id")
        status = str(result_message.get("status") or "unknown")
        payload = dict(result_message)
        # 结果消息类型：确保为 result（保持与现有约定一致）
        payload["type"] = "result"
        return await self._push(payload, task_id=task_id, status=status)

    async def send_task_status(
        self, task_id: str, status: str, error: Optional[str] = None, **kwargs
    ) -> bool:
        payload = build_task_status_message(task_id=task_id, status=status, error=error, **kwargs)
        return await self._push(payload, task_id=task_id, status=status)

