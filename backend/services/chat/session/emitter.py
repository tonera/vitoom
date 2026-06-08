"""WS event sender + state change broadcaster.

唯一负责：把任何后端事件包成 envelope 发给前端。runtime / service 不再各自
拼 `try / except / sequence / server_ts`。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING

from backend.core.logger import get_app_logger
from backend.services.chat.session._models import Turn

if TYPE_CHECKING:
    from backend.services.chat.session.runtime import SessionRuntime

logger = get_app_logger(__name__)
EventSender = Callable[..., Awaitable[None]]


class WsEmitter:
    def __init__(self, *, session_id: str, emit: EventSender) -> None:
        self.session_id = session_id
        self._emit = emit
        self._sequence = 0

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def send(
        self,
        event_type: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        turn: Optional[Turn] = None,
        binary: Optional[bytes] = None,
    ) -> None:
        msg = {
            "type": event_type,
            "session_id": self.session_id,
            "turn_id": turn.turn_id if turn else None,
            "run_id": turn.run_id if turn else None,
            "sequence": self.next_sequence(),
            "server_ts": datetime.utcnow().isoformat() + "Z",
            "payload": payload or {},
        }
        try:
            await self._emit(msg, binary=binary)
        except Exception as exc:
            logger.warning("emit failed session=%s type=%s err=%s", self.session_id, event_type, exc)

    async def set_state(
        self,
        runtime: "SessionRuntime",
        new_state: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        prev = runtime.state
        if prev == new_state:
            return
        runtime.state = new_state
        payload: Dict[str, Any] = {"state": new_state, "prev": prev}
        if extra:
            payload.update(extra)
        await self.send("status_changed", payload=payload, turn=runtime.current_turn)

    async def error(
        self,
        runtime: "SessionRuntime",
        code: str,
        message: str,
        *,
        recoverable: bool = True,
    ) -> None:
        await self.send(
            "error",
            payload={"code": code, "message": message, "recoverable": bool(recoverable)},
            turn=runtime.current_turn,
        )


__all__ = ["EventSender", "WsEmitter"]
