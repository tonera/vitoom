"""Chat session runtime：连接生命周期 + WS 消息分发 + Run 收尾。

核心契约：
  * `open()` → emit `session_ready` → 进入 `READY`
  * 每条 WS 消息走 `on_client_message()`，分发到 audio/transcript/interrupt service
  * `complete_run()` / `fail_run()` 由 `master_runtime` 在 Run 完成时调用
  * audio/transcript/interrupt service 直接读写 runtime 的 turn / assembler 字段
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from backend.core.logger import get_app_logger
from backend.database import AgentRun
from backend.services.chat.artifacts import normalize_chat_file
from backend.services.chat.avatar.livetalking_client import get_livetalking_client
from backend.services.chat.inference_session import InferenceSessionManager
from backend.services.chat.session._models import (
    InputMode,
    SessionState,
    Turn,
    TurnAssembler,
    USER_INPUT_ALLOWED,
)
from backend.services.chat.session.audio_turn import AudioTurnService
from backend.services.chat.session.emitter import EventSender, WsEmitter
from backend.services.chat.session.inference_caps import InferenceCoordinator
from backend.services.chat.session.interrupt import InterruptCoordinator
from backend.services.chat.session.transcript import TranscriptProcessor
from backend.services.chat.session.tts_out import TtsCoordinator
from backend.services.conversation import append_message
from backend.utils import generate_uuid

logger = get_app_logger(__name__)

MasterRun = Callable[["SessionRuntime", Turn], Awaitable[None]]
"""单轮 Run 的异步执行函数（实现在 master_runtime.py）。

契约：用 emit_message_*/enter_* 推流，结束调 complete_run() 或 fail_run()；
回写 conversation_messages.role=assistant 由 complete_run 统一负责。
"""


class SessionRuntime:
    """状态机宿主。每条 ``/ws/chat/{session_id}`` 连接一个实例。

    生命周期：
        1. WS handler 构造 ``SessionRuntime(session_id, user_id, emit, master_run)``
        2. 调 ``await runtime.open()``
        3. 每条客户端消息调 ``await runtime.on_client_message(msg)``
        4. WS 断开前调 ``await runtime.close()``
    """

    def __init__(
        self,
        *,
        session_id: str,
        user_id: str,
        emit: EventSender,
        master_run: MasterRun,
        input_mode: str = InputMode.TEXT,
        output_mode: str = "text_stream",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inference_session: Optional[InferenceSessionManager] = None,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.agent_id = agent_id
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.metadata = dict(metadata or {})

        self.state: str = SessionState.OPENING
        self._master_run = master_run
        self._inference_session = inference_session
        self._emitter = WsEmitter(session_id=session_id, emit=emit)
        self._inference = InferenceCoordinator(self)
        self._tts = TtsCoordinator(self)
        self._transcript = TranscriptProcessor(self)
        self._audio = AudioTurnService(self)
        self._interrupt = InterruptCoordinator(self)

        self.current_turn: Optional[Turn] = None
        self.current_assembler: Optional[TurnAssembler] = None
        self.current_message_id: Optional[str] = None  # assistant message row id

        self._lock = asyncio.Lock()
        self._run_task: Optional[asyncio.Task] = None
        self._interrupt_requested = False
        self._audio_turn_committed = False

    # ------------------------------------------------------------------
    # 内部基础设施
    # ------------------------------------------------------------------

    def _next_sequence(self) -> int:
        return self._emitter.next_sequence()

    def _finalize_turn(self, *, reset_vad: bool = True) -> None:
        """清掉一切 turn 级别的状态，回到 idle。状态转换由调用方控制。"""
        self.current_turn = None
        self.current_assembler = None
        self._run_task = None
        self._audio_turn_committed = False
        if reset_vad:
            self._audio.reset_vad_detector()

    async def _emit(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        binary: Optional[bytes] = None,
    ) -> None:
        await self._emitter.send(event_type, payload=payload, turn=self.current_turn, binary=binary)

    @property
    def interrupt_requested(self) -> bool:
        return self._interrupt_requested

    # ------------------------------------------------------------------
    # 状态切换 API（master_runtime / task_event_adapter 调用）
    # ------------------------------------------------------------------

    async def enter_reasoning(self) -> None:
        await self._emitter.set_state(self, SessionState.REASONING)

    async def enter_tool_running(self) -> None:
        await self._emitter.set_state(self, SessionState.TOOL_RUNNING)

    async def enter_streaming_output(self) -> None:
        await self._emitter.set_state(self, SessionState.STREAMING_OUTPUT)

    async def enter_waiting_task(self) -> None:
        await self._emitter.set_state(self, SessionState.WAITING_TASK)

    # ------------------------------------------------------------------
    # 事件发射 API（master_runtime / voice_reply / task_event_adapter）
    # ------------------------------------------------------------------

    async def emit_message_started(self, *, role: str = "assistant", content_type: str = "text") -> None:
        await self._emit("message_started", {"role": role, "content_type": content_type})

    async def emit_message_delta(self, delta: str, *, role: str = "assistant") -> None:
        if not delta:
            return
        if self.current_turn:
            self.current_turn.append_assistant_delta(delta)
        await self._emit("message_delta", {"role": role, "delta": delta})

    async def emit_tool_call_started(self, *, payload: Dict[str, Any]) -> None:
        await self._emit("tool_call_started", payload)

    async def emit_tool_call_completed(self, *, payload: Dict[str, Any]) -> None:
        await self._emit("tool_call_completed", payload)

    async def emit_tool_call_failed(self, *, payload: Dict[str, Any]) -> None:
        await self._emit("tool_call_failed", payload)

    async def emit_artifact_created(self, *, payload: Dict[str, Any]) -> None:
        normalized = normalize_chat_file(payload) if isinstance(payload, dict) else None
        if normalized and self.current_turn:
            self.current_turn.add_file(normalized)
        await self._emit("artifact_created", normalized or payload)

    async def emit_task_status(self, *, payload: Dict[str, Any]) -> None:
        """把派生 task 的生命周期投影到统一 chat 事件流。"""
        merged: Dict[str, Any] = {"state": self.state}
        if isinstance(payload, dict):
            merged.update(payload)
        await self._emit("status_changed", merged)

    async def emit_transcript_delta(self, *, text: str, is_final: bool) -> None:
        await self._emit("transcript_delta", {"text": text, "is_final": bool(is_final)})

    async def emit_audio_delta(
        self,
        *,
        pcm_bytes: Optional[bytes],
        mime: str,
        is_final: bool,
        sample_rate: Optional[int] = None,
    ) -> None:
        raw = pcm_bytes or b""
        payload: Dict[str, Any] = {
            "mime": str(mime or "audio/wav"),
            "is_final": bool(is_final),
            "bytes_len": len(raw),
        }
        if sample_rate is not None:
            payload["sample_rate"] = sample_rate
        await self._emit("audio_delta", payload, binary=raw if raw else None)
        if raw:
            self._tts.advance_playback_estimate(raw, sample_rate=sample_rate, mime=mime)

    async def emit_error_event(self, *, code: str, message: str, recoverable: bool = True) -> None:
        await self._emitter.error(self, code, message, recoverable=recoverable)

    async def submit_tts(
        self,
        *,
        text: str,
        voice_cfg: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: float = 240.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return await self._tts.submit(
            text=text,
            voice_cfg=voice_cfg,
            request_id=request_id,
            timeout=timeout,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # 连接生命周期
    # ------------------------------------------------------------------

    async def open(
        self,
        *,
        input_mode: Optional[str] = None,
        output_mode: Optional[str] = None,
    ) -> None:
        """WS accept 后首次调用：进入 ready 并下发 session_ready。"""
        if input_mode:
            self.input_mode = input_mode
        if output_mode:
            self.output_mode = output_mode

        asr_av = await self._inference.ensure_asr_session_opened()
        tts_av = await self._inference.ensure_tts_session_opened()
        await self._inference.set_audio_capabilities(
            supports_audio_input=asr_av,
            supports_audio_output=tts_av,
        )

        await self._emit("session_ready", {
            "mode": "chat",
            "input_mode": self.input_mode,
            "output_mode": self.output_mode,
            "capabilities": self._inference.capabilities_payload(),
            "agent_id": self.agent_id,
            "conversation_id": self.session_id,
        })
        await self._emitter.set_state(self, SessionState.READY)

        # 音频会话：fire-and-forget 后台预热 VAD 模型，消除用户首次开口的 ~4s 延迟。
        # 不能 await——session_ready 必须立即发出（前端 5s readyTimer）。详见文档 §3.7。
        if self._inference.is_audio_input_mode():
            asyncio.create_task(
                self._audio.warmup_models(),
                name=f"vad-warmup-{self.session_id}",
            )

    async def close(self, *, reason: str = "client_requested") -> None:
        if self.state == SessionState.CLOSED:
            return
        if self.current_turn and self.state in {
            SessionState.REASONING,
            SessionState.TOOL_RUNNING,
            SessionState.STREAMING_OUTPUT,
            SessionState.WAITING_TASK,
            SessionState.TURN_BUFFERING,
        }:
            try:
                await self._interrupt.execute()
            except Exception:
                pass
        if self._inference_session is not None:
            try:
                await self._inference_session.close_all()
            except Exception:
                pass
        self._inference.reset_opened_flags()
        self._audio.reset_vad_detector()
        self._tts.reset_playback_window()
        # 数字人副链路收尾：释放 sidecar WS / queue / consumer task。
        # 失败 swallow——主链路 session_closed 必须照常下发。
        try:
            await get_livetalking_client().close(self.session_id)
        except Exception as exc:
            logger.debug(
                "livetalking close swallowed session=%s err=%s",
                self.session_id, exc,
            )
        await self._emit("session_closed", {"reason": reason})
        self.state = SessionState.CLOSED

    # ------------------------------------------------------------------
    # WS 消息分发
    # ------------------------------------------------------------------

    async def on_client_message(self, msg: Dict[str, Any]) -> None:
        mtype = str(msg.get("type") or "").strip()
        payload = msg.get("payload") or {}
        if not mtype:
            await self._emitter.error(self, "invalid_payload", "missing type")
            return

        if mtype == "interrupt":
            await self._interrupt.handle_request()
        elif mtype == "session_close":
            await self.close(reason="client_requested")
        elif mtype == "session_open":
            await self._handle_session_open(payload)
        elif mtype == "ack":
            return
        elif mtype == "user_message":
            await self._handle_user_message(payload)
        elif mtype == "audio_chunk":
            await self._audio.handle_chunk(msg)
        elif mtype == "session_commit":
            await self._audio.handle_session_commit()
        elif mtype == "avatar_toggle":
            self._handle_avatar_toggle(payload)
        else:
            await self._emitter.error(self, "unsupported_message_type", f"unsupported type: {mtype}")

    def _handle_avatar_toggle(self, payload: Dict[str, Any]) -> None:
        """前端开关数字人时调用：更新本 session 在 livetalking_client 内的 enabled 标记。

        ``avatar_toggle`` 是装饰性副链路控制消息，故意做成同步、零异常路径：
        * 失败 swallow，不影响主链路
        * 不发 status_changed / error 事件，避免污染聊天主链路状态机
        """
        enabled = bool(payload.get("enabled"))
        try:
            get_livetalking_client().set_enabled(self.session_id, enabled)
        except Exception as exc:
            logger.debug(
                "livetalking set_enabled swallowed session=%s enabled=%s err=%s",
                self.session_id, enabled, exc,
            )

    async def _handle_session_open(self, payload: Dict[str, Any]) -> None:
        imode = str(payload.get("input_mode") or self.input_mode)
        omode = str(payload.get("output_mode") or self.output_mode)
        if self.state == SessionState.OPENING:
            await self.open(input_mode=imode, output_mode=omode)
        else:
            self.input_mode, self.output_mode = imode, omode
        await self._inference.ensure_asr_session_opened()
        await self._inference.ensure_tts_session_opened()

    async def _handle_user_message(self, payload: Dict[str, Any]) -> None:
        if self.state not in USER_INPUT_ALLOWED:
            await self._emitter.error(self, "busy", f"state={self.state} refuses user_message")
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            await self._emitter.error(self, "invalid_payload", "user_message.payload.text is required")
            return

        # turn_buffering（音频流偶尔混入文字）→ 追加到 assembler；否则开新文本 turn。
        if self.state == SessionState.TURN_BUFFERING and self.current_assembler:
            self.current_assembler.append_text(text)
            return

        turn_id = generate_uuid()
        self.current_turn = Turn(turn_id=turn_id, input_mode=InputMode.TEXT, user_text=text)
        try:
            append_message(
                conversation_id=self.session_id,
                role="user",
                content=text,
                turn_id=turn_id,
                user_id=self.user_id,
            )
        except Exception as exc:
            logger.warning("persist user message failed session=%s err=%s", self.session_id, exc)
        await self._start_run_for_current_turn()

    # ------------------------------------------------------------------
    # 推理侧事件（session.transcript.* / session.audio.* / session.error 等）
    # ------------------------------------------------------------------

    async def on_inference_session_event(self, event: Dict[str, Any]) -> None:
        etype = str(event.get("type") or "").strip()
        if not etype.startswith("session."):
            return
        if etype == "session.transcript.delta":
            await self.emit_transcript_delta(text=str(event.get("text") or ""), is_final=False)
        elif etype == "session.transcript.final":
            await self._transcript.handle_final(event)
        elif etype == "session.audio.start":
            logger.debug(
                "session.audio.start session=%s request_id=%s sr=%s",
                self.session_id, event.get("request_id"), event.get("sample_rate"),
            )
        elif etype == "session.audio.chunk":
            await self._tts.handle_audio_out(event, is_final=False)
        elif etype == "session.audio.end":
            await self._tts.handle_audio_out(event, is_final=True)
        elif etype == "session.error":
            await self._handle_inference_error(event)
        elif etype in {"session.ready", "session.closed"}:
            logger.debug(
                "inference %s session=%s inference_session_id=%s",
                etype, self.session_id, event.get("session_id"),
            )
        else:
            logger.debug("unhandled inference session event type=%s session=%s", etype, self.session_id)

    async def _handle_inference_error(self, event: Dict[str, Any]) -> None:
        msg = str(event.get("error") or event.get("message") or "audio session error")
        await self._emitter.error(
            self,
            str(event.get("code") or "audio_session_error"),
            msg,
            recoverable=bool(event.get("recoverable", True)),
        )
        rid = str(event.get("request_id") or "").strip()
        if rid and self._inference_session is not None:
            self._inference_session.resolve_tts_waiter(rid, error=msg)

    async def on_inference_services_changed(self, event: Optional[Dict[str, Any]] = None) -> None:
        await self._inference.handle_services_changed(event)

    # ------------------------------------------------------------------
    # Run 启停（audio_turn / _handle_user_message → _start_run_for_current_turn）
    # ------------------------------------------------------------------

    async def _start_run_for_current_turn(self) -> None:
        turn = self.current_turn
        if not turn:
            return
        self._interrupt_requested = False

        run_id = generate_uuid()
        try:
            created = AgentRun.create(
                id=run_id,
                user_id=self.user_id,
                task_id=None,
                agent_id=self.agent_id or "preset-master-agent",
                status="reasoning",
                turn_id=turn.turn_id,
                conversation_id=self.session_id,
                source_type="chat_ws",
                source_ref=self.session_id,
                input_payload={
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "input_mode": turn.input_mode,
                    "user_text": turn.user_text or "",
                },
            )
            if not created:
                # AgentRun.create 内部吞异常返回 None；显式失败避免后续 FK 冲突。
                raise RuntimeError("AgentRun.create returned None")
            turn.run_id = run_id
        except Exception as exc:
            logger.error("create AgentRun failed: %s", exc, exc_info=True)
            await self._emitter.error(self, "internal_error", f"create run failed: {exc}", recoverable=False)
            await self._fail_run_unchecked("internal_error")
            return

        await self._emitter.set_state(self, SessionState.REASONING)

        async def _runner() -> None:
            try:
                await self._master_run(self, turn)
            except asyncio.CancelledError:
                logger.info("master run cancelled session=%s turn=%s", self.session_id, turn.turn_id)
            except Exception as exc:
                from backend.services.chat.error_summary import summarize_chat_run_error

                log_msg, user_msg, _, _ = summarize_chat_run_error(exc)
                logger.error("master run exception: %s", log_msg)
                await self._emitter.error(self, "internal_error", user_msg, recoverable=False)
                await self._fail_run_unchecked("internal_error")

        self._run_task = asyncio.create_task(_runner())

    async def complete_run(
        self,
        *,
        assistant_text: Optional[str] = None,
        usage_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """master_runtime 在成功完成一次 Run 后调用。"""
        turn = self.current_turn
        if not turn:
            return
        final_text = assistant_text if assistant_text is not None else turn.assistant_text()
        files = turn.files_snapshot()
        await self._emit("message_completed", {
            "role": "assistant",
            "content_type": "text",
            "content": final_text,
            "files": files,
            "interrupt_reason": None,
            "usage_metrics": usage_metrics or {},
        })
        if str(final_text or "").strip() or files:
            try:
                append_message(
                    conversation_id=self.session_id,
                    role="assistant",
                    content=final_text,
                    agent_run_id=turn.run_id,
                    turn_id=turn.turn_id,
                    user_id=self.user_id,
                    metadata={"files": files},
                )
            except Exception as exc:
                logger.warning("persist assistant message failed session=%s err=%s", self.session_id, exc)
        if turn.run_id:
            try:
                AgentRun.update(
                    turn.run_id,
                    status="completed",
                    result_summary={"text": (final_text or "")[:500]},
                )
            except Exception:
                pass
        await self._emitter.set_state(self, SessionState.COMPLETED)
        self._finalize_turn()
        await self._emitter.set_state(self, SessionState.READY)

    async def fail_run(self, *, code: str = "internal_error", message: str = "") -> None:
        await self._emitter.error(self, code, message, recoverable=False)
        await self._fail_run_unchecked(code)

    async def _fail_run_unchecked(self, code: str) -> None:
        turn = self.current_turn
        if turn and turn.run_id:
            try:
                AgentRun.update(turn.run_id, status="failed", error_message=code)
            except Exception:
                pass
        await self._emitter.set_state(self, SessionState.FAILED)
        self._finalize_turn()
        await self._emitter.set_state(self, SessionState.READY)


# 对外别名：文档里也称作 ``ChatSessionRuntime``
ChatSessionRuntime = SessionRuntime


__all__ = [
    "ChatSessionRuntime",
    "InputMode",
    "SessionRuntime",
    "SessionState",
    "Turn",
    "TurnAssembler",
]
