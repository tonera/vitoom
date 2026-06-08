"""音频 turn：VAD detector、PCM 转发、auto-commit、barge-in 探测。

两条路径：
  - READY → 一次新音频 turn（用户主动开口）
  - REASONING/STREAMING_OUTPUT 等 → barge-in 监听（截断当前 turn）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from backend.core.logger import get_app_logger
from backend.services.chat.session._models import (
    AUDIO_CHUNK_ALLOWED,
    BARGE_IN_LISTENING_STATES,
    InputMode,
    SessionState,
    Turn,
    TurnAssembler,
)
from backend.services.chat.streaming_vad import StreamingVadDetector, VadEvent
from backend.services.chat.user_messages import unavailable_message
from backend.services.conversation import append_message
from backend.utils import generate_uuid

if TYPE_CHECKING:
    from backend.services.chat.session.runtime import SessionRuntime

logger = get_app_logger(__name__)


class AudioTurnService:
    def __init__(self, runtime: "SessionRuntime") -> None:
        self._runtime = runtime
        self._turn_vad_detector: Optional[StreamingVadDetector] = None
        self._barge_in_vad_detector: Optional[StreamingVadDetector] = None

    # ------------------------------------------------------------------
    # VAD detector lifecycle
    # ------------------------------------------------------------------

    def _vad_config(self, profile: str) -> Dict[str, Any]:
        meta = self._runtime.metadata
        raw = meta.get("vad") if isinstance(meta.get("vad"), dict) else meta.get("audio_vad")
        base = dict(raw) if isinstance(raw, dict) else {}
        nested = base.get(profile) if isinstance(base.get(profile), dict) else {}
        cfg = {k: v for k, v in base.items() if k not in {"turn", "barge_in"}}
        cfg.update(nested)
        # 两个 profile 都默认开能量地板：Silero 对人声/电视/广播都判 speech，地板
        # 用 RMS 兜底——背景噪声 RMS 远低于近场说话人，turn=0.02 能挡掉绝大多数
        # 环境噪声进 ASR；barge_in 同 0.02 + streak=2 应对 double-talk 下 echo 残留。
        # 调参依据见 docs/实时语音和文本聊天全生命周期流程.md §3.3。
        cfg.setdefault("energy_speech_floor", 0.02)
        if profile == "barge_in":
            cfg.setdefault("min_speech_ms", 300)
            cfg.setdefault("pre_roll_ms", 1500)
            cfg.setdefault("energy_floor_min_streak", 2)
        return cfg

    def _build_detector(self, profile: str) -> StreamingVadDetector:
        cfg = self._vad_config(profile)
        kwargs: Dict[str, Any] = {
            "sample_rate": int(cfg.get("sample_rate") or 16000),
            "frame_ms": int(cfg.get("frame_ms") or 20),
            "chunk_ms": int(cfg.get("chunk_ms") or 200),
            "pre_roll_ms": int(cfg.get("pre_roll_ms") or 300),
            "min_speech_ms": int(cfg.get("min_speech_ms") or 200),
            "silence_ms": int(cfg.get("silence_ms") or cfg.get("max_end_silence_time") or 2500),
            "max_end_silence_time": (
                int(cfg["max_end_silence_time"]) if cfg.get("max_end_silence_time") is not None else None
            ),
            "model_dir": str(cfg.get("model_dir") or ""),
            "speech_threshold": float(cfg.get("speech_threshold") or 0.5),
            "energy_threshold": float(cfg.get("energy_threshold") or 0.012),
            "energy_speech_floor": float(cfg.get("energy_speech_floor") or 0.0),
            "energy_floor_min_streak": int(cfg.get("energy_floor_min_streak") or 1),
            "use_neural_vad": bool(cfg.get("use_neural_vad", True)),
        }
        # 仅当 metadata 显式覆盖时才传 model_name/source，否则用 streaming_vad 默认值
        # （silero-vad + GitHub raw onnx URL）。
        if cfg.get("model_name"):
            kwargs["model_name"] = str(cfg["model_name"])
        if cfg.get("model_source"):
            kwargs["model_source"] = str(cfg["model_source"])
        return StreamingVadDetector(**kwargs)

    def _ensure_turn_detector(self) -> StreamingVadDetector:
        if self._turn_vad_detector is None:
            self._turn_vad_detector = self._build_detector("turn")
        return self._turn_vad_detector

    def _ensure_barge_in_detector(self) -> StreamingVadDetector:
        if self._barge_in_vad_detector is None:
            self._barge_in_vad_detector = self._build_detector("barge_in")
        return self._barge_in_vad_detector

    def reset_vad_detector(self) -> None:
        for det in (self._turn_vad_detector, self._barge_in_vad_detector):
            if det is not None:
                det.reset()
        self._turn_vad_detector = None
        self._barge_in_vad_detector = None

    async def warmup_models(self) -> None:
        """fire-and-forget 后台预热 VAD，消除首次开口的 onnxruntime cold-start 延迟。

        turn / barge_in 共用同一进程级 Silero ONNX session（按 onnx_path 缓存），
        预热一个 detector 即两个都热。Silero 比 fsmn-vad 轻得多（~100ms vs ~4s），
        即使没预热也几乎察觉不到，但保留 warmup 路径便于将来切大模型。
        """
        try:
            await self._ensure_turn_detector().warmup()
        except Exception as exc:
            logger.warning("vad warmup failed session=%s err=%s", self._runtime.session_id, exc)

    def begin_audio_turn(self, *, barge_in: bool = False) -> Turn:
        rt = self._runtime
        turn_id = generate_uuid()
        mode = InputMode.AUDIO_STREAM if rt.input_mode == InputMode.AUDIO_STREAM else InputMode.AUDIO_ONCE
        turn = Turn(turn_id=turn_id, input_mode=mode, barge_in=barge_in)
        rt.current_turn = turn
        rt.current_assembler = TurnAssembler(turn_id=turn_id, input_mode=mode)
        rt._audio_turn_committed = False
        return turn

    # ------------------------------------------------------------------
    # 入口：客户端 audio_chunk
    # ------------------------------------------------------------------

    async def handle_chunk(self, msg: Dict[str, Any]) -> None:
        rt = self._runtime
        payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
        pcm_in = msg.get("binary_bytes")
        if not isinstance(pcm_in, (bytes, bytearray)) or len(pcm_in) == 0:
            await rt._emitter.error(
                rt, "invalid_payload",
                "audio_chunk requires binary PCM frame after JSON meta (binary_bytes)",
            )
            return
        pcm_bytes = bytes(pcm_in)
        try:
            declared = int(payload.get("bytes_len") or 0)
        except (TypeError, ValueError):
            declared = 0
        if declared > 0 and declared != len(pcm_bytes):
            logger.warning(
                "audio_chunk bytes_len mismatch declared=%s actual=%s session=%s",
                declared, len(pcm_bytes), rt.session_id,
            )

        if rt.state not in AUDIO_CHUNK_ALLOWED:
            await rt._emitter.error(rt, "busy", f"state={rt.state} refuses audio_chunk")
            return

        # 已经 commit、等 transcript 中：所有 late chunk 都丢弃
        if (
            rt.current_turn and rt.current_turn.is_audio
            and rt._audio_turn_committed and not rt.current_turn.run_id
        ):
            return

        if (
            rt.state in BARGE_IN_LISTENING_STATES
            and rt.current_turn and rt.current_turn.run_id
        ):
            await self._handle_barge_in_chunk(payload, pcm_bytes)
            return
        if rt.state == SessionState.READY:
            await self._handle_ready_chunk(payload, pcm_bytes)
            return
        if rt.state == SessionState.TURN_BUFFERING:
            await self._handle_buffering_chunk(payload, pcm_bytes)
            return
        await rt._emitter.error(rt, "busy", f"state={rt.state} refuses audio_chunk")

    async def _handle_ready_chunk(self, payload: Dict[str, Any], pcm_bytes: bytes) -> None:
        rt = self._runtime
        events = await self._ensure_turn_detector().push(pcm_bytes)
        starts = [e for e in events if e.type == "speech_start"]
        if not starts:
            return
        # 后端虽 ready，前端 useAudioPlayback 队列里可能还有未播完的 TTS PCM。
        # 在估算窗口内开口 → 大概率是想"打断刚才的回复"，标 barge_in=True，让 transcript final
        # 的控制词捷径仍然有效（说"停"不会再触发 LLM）。
        playback_barge_in = rt._tts.is_within_playback_window()
        self.begin_audio_turn(barge_in=playback_barge_in)
        rt._tts.reset_playback_window()  # 进入新 turn 即清估算窗口
        extra: Dict[str, Any] = {"vad_state": "speech_start"}
        if playback_barge_in:
            extra["barge_in"] = True
        await rt._emitter.set_state(rt, SessionState.TURN_BUFFERING, extra=extra)
        await self._flush_start_frames(starts[-1], payload)
        await self._process_vad_events(events, payload)

    async def _handle_buffering_chunk(self, payload: Dict[str, Any], pcm_bytes: bytes) -> None:
        rt = self._runtime
        if not rt.current_turn or not rt.current_turn.is_audio or rt.current_assembler is None:
            await rt._emitter.error(rt, "busy", f"state={rt.state} refuses audio_chunk")
            return
        if not await self._append_and_send(pcm_bytes, payload.get("seq")):
            return
        await self._process_vad_events(await self._ensure_turn_detector().push(pcm_bytes), payload)

    async def _handle_barge_in_chunk(self, payload: Dict[str, Any], pcm_bytes: bytes) -> None:
        rt = self._runtime
        if not rt._inference.is_audio_input_mode():
            return
        detector = self._ensure_barge_in_detector()
        events = await detector.push(pcm_bytes)
        starts = [e for e in events if e.type == "speech_start"]
        if not starts:
            return

        logger.info(
            "barge-in detected session=%s old_turn=%s state=%s",
            rt.session_id,
            rt.current_turn.turn_id if rt.current_turn else None,
            rt.state,
        )
        await rt._interrupt.execute(reset_vad=False)
        # 复用刚检测到 speech 的 detector 作为新 turn 的 detector，避免再次冷启动。
        self._turn_vad_detector = detector
        self._barge_in_vad_detector = None
        self.begin_audio_turn(barge_in=True)
        await rt._emitter.set_state(
            rt, SessionState.TURN_BUFFERING,
            extra={"barge_in": True, "vad_state": "speech_start"},
        )
        await self._flush_start_frames(starts[-1], payload, fallback=pcm_bytes)
        await self._process_vad_events(events, payload)

    # ------------------------------------------------------------------
    # 公共 helper
    # ------------------------------------------------------------------

    async def _flush_start_frames(
        self,
        event: VadEvent,
        payload: Dict[str, Any],
        *,
        fallback: Optional[bytes] = None,
    ) -> None:
        flushed = event.frames or ([fallback] if fallback else [])
        for idx, frame in enumerate(flushed):
            seq = payload.get("seq") if idx == len(flushed) - 1 else None
            if not await self._append_and_send(frame, seq):
                return

    async def _append_and_send(self, pcm_bytes: bytes, seq: Any) -> bool:
        rt = self._runtime
        if not await rt._inference.ensure_asr_session_opened() or rt._inference_session is None:
            await self._abort_audio_turn_unavailable()
            return False
        if rt.current_assembler is not None:
            rt.current_assembler.append_audio(pcm_bytes)
        if not await rt._inference_session.asr_chunk(pcm_bytes, seq):
            await self._abort_audio_turn_unavailable()
            return False
        return True

    async def _process_vad_events(self, events: List[VadEvent], payload: Dict[str, Any]) -> None:
        rt = self._runtime
        if not rt.current_turn or not rt.current_turn.is_audio:
            return
        if any(e.type == "speech_end" for e in events):
            await rt._emitter.send(
                "status_changed",
                payload={"state": rt.state, "vad_state": "speech_end", "auto_commit": True},
                turn=rt.current_turn,
            )
            await self._commit_audio(seq=payload.get("seq"), auto=True)

    # ------------------------------------------------------------------
    # commit
    # ------------------------------------------------------------------

    async def handle_session_commit(self) -> None:
        rt = self._runtime
        if rt.state != SessionState.TURN_BUFFERING or not rt.current_turn:
            await rt._emitter.error(rt, "busy", "session_commit only allowed in turn_buffering")
            return
        if rt.current_turn.is_audio:
            await self._commit_audio(seq=rt._next_sequence(), auto=False)
            return
        await self._commit_text()

    async def _commit_text(self) -> None:
        rt = self._runtime
        assembler = rt.current_assembler
        turn = rt.current_turn
        text = assembler.text_so_far() if assembler else ""
        turn.user_text = text
        if assembler:
            turn.input_mode = assembler.input_mode
        try:
            append_message(
                conversation_id=rt.session_id,
                role="user",
                content=text,
                turn_id=turn.turn_id,
                metadata={"audio_chunks": assembler.audio_count() if assembler else 0},
                user_id=rt.user_id,
            )
        except Exception as exc:
            logger.warning("persist user audio turn failed session=%s err=%s", rt.session_id, exc)
        await rt._start_run_for_current_turn()

    async def _commit_audio(self, *, seq: Any = None, auto: bool = False) -> None:
        rt = self._runtime
        if not rt.current_turn or not rt.current_turn.is_audio or rt._audio_turn_committed:
            return
        rt._audio_turn_committed = True
        logger.info(
            "audio session_commit session=%s turn=%s state=%s auto=%s",
            rt.session_id, rt.current_turn.turn_id, rt.state, auto,
        )
        if rt._inference_session is None or not rt._inference.asr_session_opened:
            await self._abort_audio_turn_unavailable()
            return
        sent = await rt._inference_session.asr_commit(seq if seq is not None else rt._next_sequence())
        logger.info(
            "audio session.asr.commit forwarded session=%s turn=%s sent=%s auto=%s",
            rt.session_id, rt.current_turn.turn_id, sent, auto,
        )
        if not sent:
            await self._abort_audio_turn_unavailable()
            return
        await rt._emitter.set_state(
            rt, SessionState.STREAMING_OUTPUT,
            extra={"auto_commit": True} if auto else None,
        )

    async def _abort_audio_turn_unavailable(self) -> None:
        """ASR 链路在任何环节失败：发 model_not_available 错误并回退到 ready。"""
        rt = self._runtime
        rt._inference.asr_session_opened = False
        await rt._emitter.error(rt, "model_not_available", unavailable_message())
        rt._finalize_turn()
        if rt.state != SessionState.READY:
            await rt._emitter.set_state(rt, SessionState.READY)


__all__ = ["AudioTurnService"]
