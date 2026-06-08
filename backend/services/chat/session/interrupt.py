"""interrupt：取消 TTS、ASR commit、派生 task、master run，并发布最终事件。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from backend.core.logger import get_app_logger
from backend.database import AgentRun, Task
from backend.services.chat.avatar.livetalking_client import get_livetalking_client
from backend.services.chat.session._models import INTERRUPT_ALLOWED, SessionState
from backend.services.conversation import append_message

if TYPE_CHECKING:
    from backend.services.chat.session.runtime import SessionRuntime

logger = get_app_logger(__name__)


class InterruptCoordinator:
    def __init__(self, runtime: "SessionRuntime") -> None:
        self._runtime = runtime

    async def handle_request(self) -> None:
        rt = self._runtime
        if rt.state not in INTERRUPT_ALLOWED:
            await rt._emitter.error(rt, "interrupt_not_allowed", f"state={rt.state}")
            return
        await self.execute()

    async def execute(self, *, reset_vad: bool = True) -> None:
        rt = self._runtime
        rt._interrupt_requested = True
        turn = rt.current_turn
        rt._tts.reset_playback_window()

        await self._cancel_tts()
        await self._cancel_avatar()
        await self._asr_commit_if_audio(turn)
        await self._cancel_derived_tasks(turn)
        await self._cancel_run_task()

        await rt._emitter.set_state(rt, SessionState.INTERRUPTED)
        if turn:
            await self._publish_partial_assistant(turn)

        rt._finalize_turn(reset_vad=reset_vad)
        await rt._emitter.set_state(rt, SessionState.READY)

    async def _cancel_tts(self) -> None:
        rt = self._runtime
        if rt._inference_session is None or not rt._inference_session.active_tts_request_id:
            return
        try:
            await rt._inference_session.tts_cancel()
        except Exception as exc:
            logger.debug("tts_cancel during interrupt failed session=%s err=%s", rt.session_id, exc)

    async def _cancel_avatar(self) -> None:
        """旁路通知数字人 sidecar 清队列。

        plan 不变量第 1 条配套：interrupt 是低频允许 await，但 client 内部已
        包 200ms 超时 + swallow，绝不影响主链路 interrupt 后续步骤（cancel
        ASR commit / derived tasks / run task / 状态切 ready）。
        """
        rt = self._runtime
        try:
            await get_livetalking_client().interrupt(rt.session_id)
        except Exception as exc:
            logger.debug(
                "livetalking interrupt swallowed session=%s err=%s",
                rt.session_id, exc,
            )

    async def _asr_commit_if_audio(self, turn) -> None:
        rt = self._runtime
        if not turn or not turn.is_audio or rt._inference_session is None:
            return
        try:
            await rt._inference_session.asr_commit(rt._next_sequence())
        except Exception as exc:
            logger.debug("asr_commit during interrupt failed session=%s err=%s", rt.session_id, exc)

    async def _cancel_derived_tasks(self, turn) -> None:
        rt = self._runtime
        if not turn or not turn.run_id:
            return
        try:
            rows = Task.list_by_agent_run_id(turn.run_id, limit=100)
        except Exception as exc:
            logger.debug("list tasks failed run=%s err=%s", turn.run_id, exc)
            return
        for row in rows:
            tid = str(row.get("id") or "").strip()
            if tid:
                turn.bind_task_id(tid)
        if not turn.derived_task_ids:
            return
        try:
            from backend.websocket.manager import get_websocket_manager
            ws = get_websocket_manager()
            for tid in list(turn.derived_task_ids):
                await ws.send_cancel_signal_to_inference_service(tid)
        except Exception as exc:
            logger.warning(
                "cancel derived tasks failed session=%s run=%s err=%s",
                rt.session_id, turn.run_id, exc,
            )

    async def _cancel_run_task(self) -> None:
        task = self._runtime._run_task
        if not task or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def _publish_partial_assistant(self, turn) -> None:
        rt = self._runtime
        partial = turn.assistant_text()
        files = turn.files_snapshot()
        await rt._emitter.send(
            "message_completed",
            payload={
                "role": "assistant",
                "content_type": "text",
                "content": partial,
                "files": files,
                "interrupt_reason": "user_interrupt",
            },
            turn=turn,
        )
        if str(partial or "").strip() or files:
            try:
                append_message(
                    conversation_id=rt.session_id,
                    role="assistant",
                    content=partial,
                    agent_run_id=turn.run_id,
                    turn_id=turn.turn_id,
                    metadata={"status": "interrupted", "files": files},
                    user_id=rt.user_id,
                )
            except Exception:
                pass
        if turn.run_id:
            try:
                AgentRun.update(turn.run_id, status="interrupted")
            except Exception:
                pass


__all__ = ["InterruptCoordinator"]
