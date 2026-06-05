"""音频 session 运行时（PR2 起统一 `session.*` 协议）。

本 runtime 同时承载 ASR 与 TTS 两种 role：

- ``role=asr``：沿用 qwen-asr 流式 session（``streaming_session_factory``），
  处理 ``session.asr.chunk`` / ``session.asr.commit``，回流
  ``session.transcript.delta`` / ``session.transcript.final``。
- ``role=tts``：通过 ``tts_engine_factory`` 构造 ``TtsEngine``，处理
  ``session.tts.request`` / ``session.tts.cancel``，回流
  ``session.audio.start`` / ``session.audio.chunk`` /
  ``session.audio.end``。

共同协议：``session.open`` / ``session.close`` / ``session.ready`` /
``session.closed`` / ``session.error``。

生命周期上一个 inference session 的 ``session_id`` 由后端按
``<chat_sid>:<role>`` 下发，runtime 本身不关心后端编码方式：它用 ``session_id``
作为状态桶主键，同时把 ``role`` 字段作为路由依据（缺省从 session_id 的
``:<suffix>`` 推断）。
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from audio.engines.tts_engine import AudioChunk, TtsEngine, VoiceConfig
from audio.runtime.audio_wav_utils import audio_tensor_to_pcm16_bytes
from common.logger import get_logger

logger = get_logger(__name__)


ASR_ROLE = "asr"
TTS_ROLE = "tts"


@dataclass
class AudioSessionState:
    session_id: str
    role: str = ASR_ROLE
    chunk_count: int = 0
    last_seq: Optional[int] = None
    last_text: str = ""
    # ASR 专属
    streaming_session: Any = None
    streaming_error: str = ""
    streaming_error_reported: bool = False
    # 通用 model 描述
    load_name: str = ""
    family: str = ""
    model_cfg: Optional[Dict[str, Any]] = None
    # TTS 专属
    tts_engine: Optional[TtsEngine] = None
    tts_init_error: str = ""
    active_tts_request_id: Optional[str] = None
    active_tts_task: Optional[asyncio.Task] = None
    cancelled_request_ids: set = field(default_factory=set)
    # === 会话级 voice_design seed（固定音色锚点） ==========================
    # 用户用自然语言设计音色时，不能每个 block 都重新 voice_design；否则同一长回复
    # 会像随机 speaker 轮流朗读。首次遇到该设计音色时合成一段 seed wav，后续
    # block 复用该 wav 做 clone/reference，直到音色签名变化或 session.close。
    tts_voice_seed_wav_path: Optional[str] = None
    tts_voice_seed_text: str = ""
    tts_voice_seed_signature: str = ""
    # === Chained prompt（VoxCPM2 Ultimate Cloning 专用，方案 D） =========
    # 上一段成功合成的整段 wav 临时文件路径（与 prompt_text 严格对应）。
    # 下一段会把这个 wav 注入 voice_cfg.prompt_wav_path，让模型在解码新文本时
    # 续上上一段的韵律 / 呼吸 / 节奏，缓解句间断档。
    tts_prompt_wav_path: Optional[str] = None
    # 上一段合成的文本：必须与 prompt_wav 严格对应（VoxCPM 强约束），
    # 否则 prosody encoder 会失配。
    tts_prompt_text: str = ""
    # 用于检测音色切换——speaker / voice_design 改了之后，旧 prompt 不能再用，
    # 否则会把上一个音色的韵律带到新音色上。
    tts_prompt_voice_signature: str = ""


StreamingSessionFactory = Callable[[AudioSessionState], Awaitable[Any]]
TtsEngineFactory = Callable[[AudioSessionState], Awaitable[TtsEngine]]

# 出站：JSON meta + 可选 binary（与 backend / inference WS 约定一致）
SessionOutboundSender = Callable[..., Awaitable[bool]]


def _utc_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_role(message: Dict[str, Any], session_id: str, fallback: str = ASR_ROLE) -> str:
    role = str(message.get("role") or "").strip().lower()
    if role:
        return role
    if ":" in session_id:
        suffix = session_id.rsplit(":", 1)[-1].strip().lower()
        if suffix:
            return suffix
    return fallback


class AudioSessionRuntime:
    """音频 session runtime，统一承载 ASR 与 TTS 两条实时链路。"""

    def __init__(
        self,
        sender: SessionOutboundSender,
        *,
        backend: str = "transformers",
        fixed_family: Optional[str] = None,
        streaming_session_factory: Optional[StreamingSessionFactory] = None,
        tts_engine_factory: Optional[TtsEngineFactory] = None,
        stream_lock: Optional[asyncio.Lock] = None,
    ):
        self._sender = sender
        self._sessions: Dict[str, AudioSessionState] = {}
        self._backend = str(backend or "transformers").strip().lower() or "transformers"
        self._fixed_family = str(fixed_family or "").strip()
        self._streaming_session_factory = streaming_session_factory
        self._tts_engine_factory = tts_engine_factory
        # ASR/TTS 共用同一把锁：vLLM/TTS 单实例 generate() 非线程安全，且 ASR/TTS
        # 可能分时复用同一张卡，跨 session / task 并发会踩内部状态。
        self._stream_lock: asyncio.Lock = stream_lock or asyncio.Lock()

    # ------------------------------------------------------------------
    # 服务注册
    # ------------------------------------------------------------------

    async def register_service(
        self,
        *,
        service_type: str = "audio",
        supported_models: list[str],
        capabilities: list[str],
        fixed_model: str | None = None,
        fixed_family: str | None = None,
    ) -> bool:
        return await self._sender(
            {
                "type": "service_register",
                "service_type": service_type,
                "supports_task": True,
                "supported_models": [str(item).strip() for item in supported_models if str(item).strip()],
                "capabilities": [
                    str(item).strip().lower() for item in capabilities if str(item).strip()
                ],
                "fixed_model": str(fixed_model or "").strip(),
                "fixed_family": str(fixed_family or "").strip(),
                "timestamp": _utc_iso(),
            }
        )

    # ------------------------------------------------------------------
    # 公共回流辅助
    # ------------------------------------------------------------------

    async def _send_session_error(
        self,
        *,
        session_id: str,
        seq: Any,
        error: str,
        code: str = "realtime_not_supported",
        role: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        payload: Dict[str, Any] = {
            "type": "session.error",
            "session_id": session_id,
            "seq": seq,
            "code": code,
            "error": error,
            "timestamp": _utc_iso(),
        }
        if role:
            payload["role"] = role
        if request_id:
            payload["request_id"] = request_id
        return await self._sender(payload)

    async def _emit_transcript(
        self,
        *,
        state: AudioSessionState,
        seq: Any,
        payload: Dict[str, Any],
        final: bool,
    ) -> None:
        text = str(payload.get("text", "") or "").strip()
        if not text and not final:
            return
        state.last_text = text
        await self._sender(
            {
                "type": "session.transcript.final" if final else "session.transcript.delta",
                "session_id": state.session_id,
                "role": state.role,
                "seq": seq,
                "text": text,
                "delta": str(payload.get("delta", "") or ""),
                # qwen-asr 流式尾部 unfixed token 可能被改写，replaced 表示被改写
                # 的旧尾部，便于累积 delta 的消费者做回退覆盖。
                "replaced": str(payload.get("replaced", "") or ""),
                "language": str(payload.get("language", "") or ""),
                "chunk_count": state.chunk_count,
                "timestamp": _utc_iso(),
            }
        )

    # ------------------------------------------------------------------
    # PCM 解码
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_pcm16_bytes(raw: bytes) -> np.ndarray:
        if not raw:
            return np.zeros((0,), dtype=np.float32)
        pcm16 = np.frombuffer(raw, dtype=np.int16)
        if pcm16.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return (pcm16.astype(np.float32) / 32768.0).copy()

    # ------------------------------------------------------------------
    # ASR engine 懒加载
    # ------------------------------------------------------------------

    async def _ensure_streaming_session(self, state: AudioSessionState) -> Any:
        """懒加载 ASR streaming session。必须在 self._stream_lock 里调用。"""
        if state.streaming_session is not None:
            return state.streaming_session
        if self._streaming_session_factory is None:
            return None
        if state.streaming_error:
            return None
        try:
            state.streaming_session = await self._streaming_session_factory(state)
        except Exception as exc:
            state.streaming_error = str(exc) or exc.__class__.__name__
            state.streaming_session = None
        return state.streaming_session

    async def _ensure_tts_engine(self, state: AudioSessionState) -> Optional[TtsEngine]:
        if state.tts_engine is not None:
            return state.tts_engine
        if self._tts_engine_factory is None:
            return None
        if state.tts_init_error:
            return None
        try:
            state.tts_engine = await self._tts_engine_factory(state)
        except Exception as exc:
            state.tts_init_error = str(exc) or exc.__class__.__name__
            state.tts_engine = None
            logger.exception(
                "[audio-session] tts engine init failed session_id=%s family=%s",
                state.session_id,
                state.family,
            )
        return state.tts_engine

    # ------------------------------------------------------------------
    # 协议派发
    # ------------------------------------------------------------------

    async def handle_message(self, message: Dict[str, Any]) -> bool:
        message_type = str(message.get("type") or "").strip()
        session_id = str(message.get("session_id") or "").strip()
        if not session_id:
            return False
        if not message_type.startswith("session."):
            return False

        if message_type == "session.open":
            return await self._handle_session_open(message, session_id)
        if message_type == "session.close":
            return await self._handle_session_close(message, session_id)
        if message_type == "session.asr.chunk":
            return await self._handle_asr_chunk(message, session_id)
        if message_type == "session.asr.commit":
            return await self._handle_asr_commit(message, session_id)
        if message_type == "session.tts.request":
            return await self._handle_tts_request(message, session_id)
        if message_type == "session.tts.cancel":
            return await self._handle_tts_cancel(message, session_id)

        logger.debug(
            "[audio-session] ignoring unsupported session.* type=%s session_id=%s",
            message_type,
            session_id,
        )
        return False

    # ------------------------------------------------------------------
    # session.open / session.close
    # ------------------------------------------------------------------

    async def _handle_session_open(self, message: Dict[str, Any], session_id: str) -> bool:
        role = _parse_role(message, session_id)
        state = self._sessions.get(session_id) or AudioSessionState(session_id=session_id, role=role)
        state.role = role
        state.last_seq = message.get("seq")
        state.streaming_session = None
        state.streaming_error = ""
        state.streaming_error_reported = False
        state.tts_engine = None
        state.tts_init_error = ""
        state.active_tts_request_id = None
        state.active_tts_task = None
        state.cancelled_request_ids = set()
        state.load_name = str(message.get("load_name") or state.load_name or "").strip()
        # 推理服务配置里的 fixed_family 是当前实例的权威 runtime family；
        # 仅当它为空时，才使用 session.open / 请求侧传入的 family。
        state.family = str(
            self._fixed_family
            or message.get("family")
            or state.family
            or ""
        ).strip()
        incoming_model_cfg = message.get("model_cfg")
        if isinstance(incoming_model_cfg, dict):
            state.model_cfg = dict(incoming_model_cfg)
        self._sessions[session_id] = state

        logger.info(
            "[audio-session] session.open session_id=%s role=%s load_name=%s family=%s",
            session_id,
            role,
            state.load_name or "<empty>",
            state.family or "<empty>",
        )

        # TTS 角色在 open 阶段预热 engine；失败时也不急着报错，等 request 到来时再精细报
        if role == TTS_ROLE and self._tts_engine_factory is not None:
            await self._ensure_tts_engine(state)

        await self._sender(
            {
                "type": "session.ready",
                "session_id": session_id,
                "role": role,
                "seq": message.get("seq"),
                "timestamp": _utc_iso(),
            }
        )
        return True

    async def _handle_session_close(self, message: Dict[str, Any], session_id: str) -> bool:
        state = self._sessions.pop(session_id, None)
        role = state.role if state else _parse_role(message, session_id)
        # ASR：若尚有未提交音频，finish 一把把残留 transcript 落出
        if state and role == ASR_ROLE and state.chunk_count > 0:
            payload: Optional[Dict[str, Any]] = None
            async with self._stream_lock:
                streaming_session = state.streaming_session
                if streaming_session is None and self._streaming_session_factory is not None:
                    streaming_session = await self._ensure_streaming_session(state)
                if streaming_session is not None:
                    payload = await asyncio.to_thread(streaming_session.finish)
                    state.streaming_session = None
            if payload is not None:
                await self._emit_transcript(
                    state=state,
                    seq=message.get("seq"),
                    payload=payload,
                    final=True,
                )
        # TTS：取消正在跑的合成
        if state and role == TTS_ROLE and state.active_tts_task is not None:
            if state.active_tts_request_id:
                state.cancelled_request_ids.add(state.active_tts_request_id)
            task = state.active_tts_task
            state.active_tts_task = None
            try:
                if not task.done():
                    task.cancel()
                    await asyncio.shield(_await_silently(task))
            except Exception:
                pass
        # 方案 D：会话关闭时清掉 chained prompt 的临时 wav 文件，避免堆积
        if state and role == TTS_ROLE:
            self._purge_voice_seed_cache(state)
            self._purge_continuation_cache(state)

        await self._sender(
            {
                "type": "session.closed",
                "session_id": session_id,
                "role": role,
                "seq": message.get("seq"),
                "timestamp": _utc_iso(),
            }
        )
        return True

    # ------------------------------------------------------------------
    # ASR 子协议
    # ------------------------------------------------------------------

    async def _handle_asr_chunk(self, message: Dict[str, Any], session_id: str) -> bool:
        state = self._sessions.get(session_id)
        if state is None:
            state = AudioSessionState(session_id=session_id, role=ASR_ROLE)
            self._sessions[session_id] = state
        state.role = ASR_ROLE
        state.chunk_count += 1
        state.last_seq = message.get("seq")

        if self._streaming_session_factory is None:
            await self._send_session_error(
                session_id=session_id,
                seq=message.get("seq"),
                error=f"audio realtime ASR is not enabled on backend={self._backend}",
                role=ASR_ROLE,
            )
            return True

        pcm_bytes = message.get("binary_bytes")
        if not isinstance(pcm_bytes, (bytes, bytearray)) or len(pcm_bytes) == 0:
            await self._send_session_error(
                session_id=session_id,
                seq=message.get("seq"),
                error="session.asr.chunk requires binary_bytes (PCM16 LE payload)",
                code="invalid_audio_chunk",
                role=ASR_ROLE,
            )
            return True

        audio_chunk = self._decode_pcm16_bytes(bytes(pcm_bytes))
        async with self._stream_lock:
            streaming_session = await self._ensure_streaming_session(state)
            if streaming_session is not None:
                payload = await asyncio.to_thread(streaming_session.push_chunk, audio_chunk)

        if streaming_session is None:
            if not state.streaming_error_reported:
                state.streaming_error_reported = True
                await self._send_session_error(
                    session_id=session_id,
                    seq=message.get("seq"),
                    error=state.streaming_error or "streaming session factory is unavailable",
                    code=(
                        "streaming_session_init_failed"
                        if state.streaming_error
                        else "realtime_not_supported"
                    ),
                    role=ASR_ROLE,
                )
            return True

        await self._emit_transcript(
            state=state,
            seq=message.get("seq"),
            payload=payload,
            final=False,
        )
        return True

    async def _handle_asr_commit(self, message: Dict[str, Any], session_id: str) -> bool:
        state = self._sessions.get(session_id)
        logger.info(
            "[audio-session] session.asr.commit session_id=%s chunk_count=%s streaming_session=%s",
            session_id,
            getattr(state, "chunk_count", 0),
            bool(getattr(state, "streaming_session", None)),
        )
        if state is None or state.chunk_count <= 0:
            return True

        state.role = ASR_ROLE
        payload: Optional[Dict[str, Any]] = None
        async with self._stream_lock:
            streaming_session = state.streaming_session
            if streaming_session is None and self._streaming_session_factory is not None:
                streaming_session = await self._ensure_streaming_session(state)
            if streaming_session is not None:
                logger.info("[audio-session] finish_streaming start session_id=%s", session_id)
                payload = await asyncio.to_thread(streaming_session.finish)
                logger.info(
                    "[audio-session] finish_streaming done session_id=%s final_text_len=%s",
                    session_id,
                    len(str(payload.get("text") or "")),
                )
                state.streaming_session = None
        if payload is not None:
            await self._emit_transcript(
                state=state,
                seq=message.get("seq"),
                payload=payload,
                final=True,
            )
        state.chunk_count = 0
        state.last_text = ""
        state.streaming_error = ""
        state.streaming_error_reported = False
        return True

    # ------------------------------------------------------------------
    # TTS 子协议
    # ------------------------------------------------------------------

    async def _handle_tts_request(self, message: Dict[str, Any], session_id: str) -> bool:
        request_id = str(message.get("request_id") or "").strip()
        if not request_id:
            await self._send_session_error(
                session_id=session_id,
                seq=message.get("seq"),
                error="session.tts.request requires request_id",
                code="invalid_payload",
                role=TTS_ROLE,
            )
            return True

        state = self._sessions.get(session_id)
        if state is None:
            # 允许惰性补一个 state（容忍客户端乱序），但必须能走到 _ensure_tts_engine
            state = AudioSessionState(session_id=session_id, role=TTS_ROLE)
            self._sessions[session_id] = state
        state.role = TTS_ROLE

        # 同 session 内存在活跃 request 时，自动 barge-in
        if state.active_tts_task is not None and state.active_tts_request_id:
            prev_rid = state.active_tts_request_id
            logger.info(
                "[audio-session] tts barge-in session_id=%s prev_request_id=%s new_request_id=%s",
                session_id,
                prev_rid,
                request_id,
            )
            state.cancelled_request_ids.add(prev_rid)
            prev_task = state.active_tts_task
            state.active_tts_task = None
            state.active_tts_request_id = None
            try:
                if not prev_task.done():
                    prev_task.cancel()
                await _await_silently(prev_task)
            except Exception:
                pass

        engine = await self._ensure_tts_engine(state)
        if engine is None:
            await self._send_session_error(
                session_id=session_id,
                seq=message.get("seq"),
                error=state.tts_init_error or "tts engine is not available",
                code="tts_engine_init_failed",
                role=TTS_ROLE,
                request_id=request_id,
            )
            # 仍发一条 end 作为终态（cancelled 语义不合适，这里用 error_code）
            await self._emit_audio_end(
                state=state,
                request_id=request_id,
                sample_rate=0,
                seq=message.get("seq"),
                cancelled=False,
                error_code="tts_engine_init_failed",
            )
            return True

        voice_cfg_raw = message.get("voice_cfg") if isinstance(message.get("voice_cfg"), dict) else {}
        try:
            voice_cfg = _voice_config_from_dict(voice_cfg_raw)
        except Exception as exc:
            await self._send_session_error(
                session_id=session_id,
                seq=message.get("seq"),
                error=f"invalid voice_cfg: {exc}",
                code="invalid_voice_cfg",
                role=TTS_ROLE,
                request_id=request_id,
            )
            await self._emit_audio_end(
                state=state,
                request_id=request_id,
                sample_rate=0,
                seq=message.get("seq"),
                cancelled=False,
                error_code="invalid_voice_cfg",
            )
            return True

        text = str(message.get("text") or "")
        state.active_tts_request_id = request_id
        task = asyncio.create_task(
            self._run_tts_stream(
                state=state,
                engine=engine,
                text=text,
                voice_cfg=voice_cfg,
                request_id=request_id,
                seq=message.get("seq"),
            )
        )
        state.active_tts_task = task
        return True

    async def _handle_tts_cancel(self, message: Dict[str, Any], session_id: str) -> bool:
        state = self._sessions.get(session_id)
        if state is None:
            return True
        state.role = TTS_ROLE
        request_id = str(message.get("request_id") or "").strip() or state.active_tts_request_id or ""
        if not request_id:
            return True
        state.cancelled_request_ids.add(request_id)
        logger.info(
            "[audio-session] session.tts.cancel session_id=%s request_id=%s",
            session_id,
            request_id,
        )
        # 取消正在运行的任务；真正的 end 事件由 _run_tts_stream 的 finally 发出
        task = state.active_tts_task
        if task and request_id == state.active_tts_request_id and not task.done():
            task.cancel()
        return True

    async def _run_tts_stream(
        self,
        *,
        state: AudioSessionState,
        engine: TtsEngine,
        text: str,
        voice_cfg: VoiceConfig,
        request_id: str,
        seq: Any,
    ) -> None:
        sequence = 0
        started = False
        final_sample_rate = 0
        cancelled = False
        error_code: Optional[str] = None
        error_msg: Optional[str] = None

        # 固定音色锚点优先：voice_design 首次请求会在 stream lock 内生成会话级
        # seed wav，后续 block 使用该 seed 做 clone/reference，避免每段重新抽样音色。

        # 累积本次合成产出的所有 PCM chunk，成功完成后写为下一段的 prompt_wav。
        # 仅在未取消、无错误时才会落盘，避免把残缺音频喂给下一段模型。
        accumulated_pcm: List[np.ndarray] = []

        async def _cancel_check() -> bool:
            return request_id in state.cancelled_request_ids

        try:
            async with self._stream_lock:
                voice_cfg = await self._maybe_apply_voice_design_seed(
                    state=state,
                    engine=engine,
                    voice_cfg=voice_cfg,
                    cancel_check=_cancel_check,
                )
                # 方案 D（Chained Prompting / Ultimate Cloning）：把上一段成功合成的
                # 整段 wav + 文本注入到本次 voice_cfg.prompt_wav_path / prompt_text，
                # 让 VoxCPM2 在解码新文本时续上韵律。详见 _maybe_inject_continuation_prompt。
                voice_cfg = self._maybe_inject_continuation_prompt(state, voice_cfg)
                async for chunk in engine.synthesize_stream(  # type: ignore[attr-defined]
                    text=text,
                    voice_cfg=voice_cfg,
                    cancel_check=_cancel_check,
                    stream_mode=True,
                ):
                    if request_id in state.cancelled_request_ids:
                        cancelled = True
                        break
                    if not isinstance(chunk, AudioChunk):
                        continue
                    sample_rate = int(chunk.sample_rate or 0)
                    if sample_rate > 0:
                        final_sample_rate = sample_rate
                    if not started and chunk.pcm is not None and getattr(chunk.pcm, "size", 0) > 0:
                        started = True
                        await self._sender(
                            {
                                "type": "session.audio.start",
                                "session_id": state.session_id,
                                "role": TTS_ROLE,
                                "seq": seq,
                                "request_id": request_id,
                                "sample_rate": sample_rate,
                                "timestamp": _utc_iso(),
                            }
                        )
                    if chunk.is_final:
                        # 终态 chunk 由 finally 统一回送为 session.audio.end
                        break
                    if chunk.pcm is None or getattr(chunk.pcm, "size", 0) == 0:
                        continue
                    # chained prompt：累积 numpy 副本（避免引用外部 buffer 被覆盖）。
                    try:
                        accumulated_pcm.append(np.asarray(chunk.pcm, dtype=np.float32).reshape(-1).copy())
                    except Exception:
                        pass
                    sequence += 1
                    chunk_sr = int(sample_rate or 24000)
                    # session 通道下发**裸 PCM16 LE**：无 WAV header、无格式封装，
                    # 采样率通过 `audio/pcm;rate=N` mime + `sample_rate` 字段双重
                    # 冗余带出。前端 `useAudioPlayback` 走 PCM 分支直接解码为
                    # Float32 / AudioBuffer，省去 RIFF 解析与跨 chunk header 抖动。
                    pcm_bytes = audio_tensor_to_pcm16_bytes(chunk.pcm)
                    if not pcm_bytes:
                        continue
                    await self._sender(
                        {
                            "type": "session.audio.chunk",
                            "session_id": state.session_id,
                            "role": TTS_ROLE,
                            "seq": seq,
                            "sequence": sequence,
                            "request_id": request_id,
                            "bytes_len": len(pcm_bytes),
                            "mime": f"audio/pcm;rate={chunk_sr}",
                            "sample_rate": chunk_sr,
                            "is_final": False,
                            "timestamp": _utc_iso(),
                        },
                        binary=pcm_bytes,
                    )
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            error_code = exc.__class__.__name__
            error_msg = str(exc) or error_code
            logger.exception(
                "[audio-session] tts synthesize failed session_id=%s request_id=%s",
                state.session_id,
                request_id,
            )
        finally:
            if request_id in state.cancelled_request_ids:
                cancelled = True
            if error_code and not cancelled:
                await self._send_session_error(
                    session_id=state.session_id,
                    seq=seq,
                    error=error_msg or error_code,
                    code="tts_synthesize_failed",
                    role=TTS_ROLE,
                    request_id=request_id,
                )
            await self._emit_audio_end(
                state=state,
                request_id=request_id,
                sample_rate=final_sample_rate or 24000,
                seq=seq,
                cancelled=cancelled,
                error_code=error_code,
            )
            # 方案 D：仅在"完整且健康"地合成完一段后，才把它登记为下一段的 prompt。
            # 取消（barge-in）/ 错误中断时维持旧 cache 不变，避免把残缺音频喂给下一段。
            if (
                not cancelled
                and not error_code
                and accumulated_pcm
                and final_sample_rate > 0
                and text.strip()
            ):
                try:
                    self._update_continuation_cache(
                        state=state,
                        pcm_chunks=accumulated_pcm,
                        sample_rate=final_sample_rate,
                        text=text,
                        voice_cfg=voice_cfg,
                    )
                except Exception:
                    logger.exception(
                        "[chained-prompt] update cache failed session_id=%s request_id=%s",
                        state.session_id,
                        request_id,
                    )
            if state.active_tts_request_id == request_id:
                state.active_tts_request_id = None
                state.active_tts_task = None
            state.cancelled_request_ids.discard(request_id)

    async def _emit_audio_end(
        self,
        *,
        state: AudioSessionState,
        request_id: str,
        sample_rate: int,
        seq: Any,
        cancelled: bool,
        error_code: Optional[str],
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "session.audio.end",
            "session_id": state.session_id,
            "role": TTS_ROLE,
            "seq": seq,
            "request_id": request_id,
            "sample_rate": int(sample_rate or 0),
            "is_final": True,
            "timestamp": _utc_iso(),
        }
        if cancelled:
            payload["cancelled"] = True
        if error_code:
            payload["error_code"] = error_code
        await self._sender(payload)

    # ------------------------------------------------------------------
    # Stable voice seed（voice_design → fixed clone/reference）
    # ------------------------------------------------------------------

    _VOICE_DESIGN_SEED_TEXT = "你好，我是你的语音助手，很高兴和你继续对话。"
    _QWEN_DEFAULT_CLONE_BASE_MODEL = "Qwen3-TTS-12Hz-0.6B-Base"

    async def _maybe_apply_voice_design_seed(
        self,
        *,
        state: AudioSessionState,
        engine: TtsEngine,
        voice_cfg: VoiceConfig,
        cancel_check: Optional[Callable[[], Awaitable[bool]]],
    ) -> VoiceConfig:
        """把实时 voice_design 转成会话级固定 seed 的 clone/reference 请求。

        task/drama 通道各自有 seed 编排；这里仅处理 session.tts.request。
        """
        if voice_cfg is None:
            return voice_cfg
        mode = str(voice_cfg.tts_mode or "").strip().lower()
        if mode != "voice_design":
            return voice_cfg
        # 显式 reference/clone 请求由调用方负责，不覆盖。
        if voice_cfg.prompt_wav_path or voice_cfg.ref_audio:
            return voice_cfg

        signature = self._compute_voice_signature(voice_cfg)
        cached_path = state.tts_voice_seed_wav_path
        if (
            cached_path
            and state.tts_voice_seed_signature == signature
            and state.tts_voice_seed_text
            and os.path.exists(cached_path)
        ):
            return self._voice_cfg_from_seed(
                state=state,
                voice_cfg=voice_cfg,
                seed_path=cached_path,
                seed_text=state.tts_voice_seed_text,
            )

        if cached_path:
            self._purge_voice_seed_cache(state)

        seed_text = str(voice_cfg.design_seed_text or "").strip() or self._VOICE_DESIGN_SEED_TEXT
        seed_cfg = replace(
            voice_cfg,
            tts_mode="voice_design",
            speaker_name=None,
            voice_preset=None,
            prompt_wav_path=None,
            prompt_text=None,
            ref_audio=None,
            ref_text=None,
            continuation_prompt=False,
            continuation_reference_wav_path=None,
        )
        seed_audio, seed_sr = await self._synthesize_seed_audio(
            engine=engine,
            text=seed_text,
            voice_cfg=seed_cfg,
            cancel_check=cancel_check,
        )
        if seed_audio.size == 0 or seed_sr <= 0:
            return voice_cfg

        seed_path = self._write_voice_seed_wav(state.session_id, seed_audio, seed_sr)
        if not seed_path:
            return voice_cfg

        state.tts_voice_seed_wav_path = seed_path
        state.tts_voice_seed_text = seed_text
        state.tts_voice_seed_signature = signature
        logger.info(
            "[voice-seed] cached session_id=%s backend=%s wav=%s seed_chars=%d",
            state.session_id,
            self._voice_seed_backend(state),
            seed_path,
            len(seed_text),
        )
        return self._voice_cfg_from_seed(
            state=state,
            voice_cfg=voice_cfg,
            seed_path=seed_path,
            seed_text=seed_text,
        )

    async def _synthesize_seed_audio(
        self,
        *,
        engine: TtsEngine,
        text: str,
        voice_cfg: VoiceConfig,
        cancel_check: Optional[Callable[[], Awaitable[bool]]],
    ) -> tuple[np.ndarray, int]:
        collected: List[np.ndarray] = []
        final_audio: Optional[np.ndarray] = None
        final_sr = 0
        async for chunk in engine.synthesize_stream(  # type: ignore[attr-defined]
            text=text,
            voice_cfg=voice_cfg,
            cancel_check=cancel_check,
            # VoxCPM session engine 固定为 realtime_tts，不允许 stream_mode=False。
            # 这里虽然只为落 seed wav 收集完整音频，也必须走 streaming 生成。
            stream_mode=True,
        ):
            if not isinstance(chunk, AudioChunk):
                continue
            sample_rate = int(chunk.sample_rate or 0)
            if sample_rate > 0:
                final_sr = sample_rate
            if chunk.is_final:
                if chunk.pcm is not None and getattr(chunk.pcm, "size", 0) > 0:
                    final_audio = np.asarray(chunk.pcm, dtype=np.float32).reshape(-1).copy()
                break
            if chunk.pcm is not None and getattr(chunk.pcm, "size", 0) > 0:
                collected.append(np.asarray(chunk.pcm, dtype=np.float32).reshape(-1).copy())
        if final_audio is None:
            final_audio = np.concatenate(collected, axis=0) if collected else np.zeros(0, dtype=np.float32)
        return final_audio.astype(np.float32, copy=False), int(final_sr or 24000)

    def _voice_cfg_from_seed(
        self,
        *,
        state: AudioSessionState,
        voice_cfg: VoiceConfig,
        seed_path: str,
        seed_text: str,
    ) -> VoiceConfig:
        backend = self._voice_seed_backend(state)
        if backend == "voxcpm":
            return replace(
                voice_cfg,
                tts_mode="custom_voice",
                speaker_name=None,
                voice_preset=None,
                prompt_wav_path=seed_path,
                prompt_text=seed_text,
                continuation_prompt=False,
                continuation_reference_wav_path=None,
                ref_audio=None,
                ref_text=None,
                design_seed_text=None,
                design_instruct=None,
            )

        return replace(
            voice_cfg,
            tts_mode="voice_clone",
            speaker_name=None,
            voice_preset=None,
            ref_audio=seed_path,
            ref_text=seed_text,
            prompt_wav_path=None,
            prompt_text=None,
            load_name=None,
            clone_base_load_name=voice_cfg.clone_base_load_name or self._QWEN_DEFAULT_CLONE_BASE_MODEL,
            design_seed_text=None,
            design_instruct=None,
        )

    @staticmethod
    def _voice_seed_backend(
        state: AudioSessionState,
    ) -> str:
        family = str(state.family or "").strip().lower()
        if "vox" in family:
            return "voxcpm"
        if "qwen" in family:
            return "qwen"
        # 无 fixed_family 且 session.open 未传 family 时，只能用 load_name
        # 做最后兜底；正常配置不应走到这里。
        marker = " ".join(
            [
                str(state.load_name or ""),
            ]
        ).lower()
        if "vox" in marker:
            return "voxcpm"
        return "qwen"

    @staticmethod
    def _write_voice_seed_wav(
        session_id: str, audio: np.ndarray, sample_rate: int
    ) -> Optional[str]:
        try:
            import soundfile as sf  # type: ignore
        except Exception:
            logger.warning("[voice-seed] soundfile unavailable, disable stable voice seed")
            return None

        safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in session_id)[:48]
        try:
            fd, path = tempfile.mkstemp(prefix=f"tts-voice-seed-{safe}-", suffix=".wav")
            os.close(fd)
        except OSError:
            logger.exception("[voice-seed] mkstemp failed session_id=%s", session_id)
            return None
        try:
            sf.write(path, np.clip(audio, -1.0, 1.0), int(sample_rate), format="WAV")
        except Exception:
            logger.exception("[voice-seed] soundfile.write failed path=%s", path)
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
            return None
        return path

    @staticmethod
    def _purge_voice_seed_cache(state: AudioSessionState) -> None:
        old = state.tts_voice_seed_wav_path
        if old and os.path.exists(old):
            try:
                os.unlink(old)
            except OSError:
                pass
        state.tts_voice_seed_wav_path = None
        state.tts_voice_seed_text = ""
        state.tts_voice_seed_signature = ""

    # ------------------------------------------------------------------
    # Chained prompt cache（方案 D：VoxCPM2 Ultimate Cloning 续韵律）
    # ------------------------------------------------------------------

    # Session 实时聊天默认不再启用 chained prompt：上一段生成音频递归喂回模型
    # 容易把语速/破音瑕疵滚雪球放大。保留环境变量用于听测回滚。
    _CONTINUATION_ENABLED = (
        os.environ.get("VITOOM_TTS_CHAINED_PROMPT", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    # 单段 prompt 的硬上限（秒）。仅在 _CONTINUATION_ENABLED 打开时生效。
    _CONTINUATION_MAX_SECONDS = 18.0

    @staticmethod
    def _compute_voice_signature(voice_cfg: VoiceConfig) -> str:
        """音色身份签名。任意一个字段变化都视为换人 / 换声线，需要丢弃旧 prompt。"""
        return "|".join(
            [
                str(getattr(voice_cfg, "tts_mode", "") or ""),
                str(getattr(voice_cfg, "speaker_name", "") or ""),
                str(getattr(voice_cfg, "voice_preset", "") or ""),
                str(getattr(voice_cfg, "design_instruct", "") or ""),
                str(getattr(voice_cfg, "instruct", "") or ""),
                str(getattr(voice_cfg, "ref_audio", "") or ""),
                str(getattr(voice_cfg, "language", "") or ""),
            ]
        )

    def _maybe_inject_continuation_prompt(
        self, state: AudioSessionState, voice_cfg: VoiceConfig
    ) -> VoiceConfig:
        """如果会话里已经有上一段的 prompt 缓存，把它注入到本次 voice_cfg。

        关键约束：
        1. 调用方已经在 voice_cfg 里显式给了 prompt_wav_path（task 通道里
           drama 用户自带 reference）→ 完全尊重显式覆盖，不动；
        2. voice_design 模式靠 instruct 文本驱动音色，硬续上一段 prompt
           会让设计指令失效，跳过；
        3. 缓存文件丢失（被外部清理）→ 顺便清掉 state；
        4. 音色签名变化（set_chat_voice 切换）→ 旧 prompt 不能再用，
           清掉 cache 后走 fallback。
        """
        if voice_cfg is None:
            return voice_cfg
        if not self._CONTINUATION_ENABLED:
            return voice_cfg
        if voice_cfg.prompt_wav_path:
            return voice_cfg
        mode = (voice_cfg.tts_mode or "").strip().lower()
        if mode == "voice_design":
            return voice_cfg

        cached_path = state.tts_prompt_wav_path
        cached_text = state.tts_prompt_text
        if not cached_path or not cached_text:
            return voice_cfg
        if not os.path.exists(cached_path):
            self._purge_continuation_cache(state)
            return voice_cfg

        new_signature = self._compute_voice_signature(voice_cfg)
        if new_signature != state.tts_prompt_voice_signature:
            logger.info(
                "[chained-prompt] voice changed, dropping cache session_id=%s old_sig=%s new_sig=%s",
                state.session_id,
                state.tts_prompt_voice_signature,
                new_signature,
            )
            self._purge_continuation_cache(state)
            return voice_cfg

        logger.debug(
            "[chained-prompt] inject session_id=%s prompt_wav=%s prompt_text_len=%d",
            state.session_id,
            cached_path,
            len(cached_text),
        )
        # continuation_prompt=True：通知 engine 把 prompt_wav 仅当作韵律 prompt 用，
        # reference_wav 仍走 speaker_name preset。否则上一段生成 wav 会被反向当成
        # 下一段的音色锚点，连续多段后音色会逐步漂移（详见 voxcpm_tts_engine
        # ._build_generation_kwargs 旧实现）。
        return replace(
            voice_cfg,
            prompt_wav_path=cached_path,
            prompt_text=cached_text,
            continuation_prompt=True,
        )

    def _update_continuation_cache(
        self,
        *,
        state: AudioSessionState,
        pcm_chunks: List[np.ndarray],
        sample_rate: int,
        text: str,
        voice_cfg: VoiceConfig,
    ) -> None:
        """把刚合成完的整段写入临时 wav，作为下一段的 chained prompt。

        - prompt_wav 必须与 prompt_text 严格对应：故 wav 写整段、text 也用整段；
        - 上层 ``voice_reply.py`` 已经按 max_chars≈220 切 block，单段实际
          时长大多在 5～15 秒，超过 _CONTINUATION_MAX_SECONDS 时跳过更新，
          以免把超长 prompt 喂给下一段拖慢推理。
        - 写新文件 / 删旧文件全部静默吞错：cache 只是 best-effort 增益，
          失败也不能影响主合成路径。
        """
        if not self._CONTINUATION_ENABLED:
            return
        if not pcm_chunks or sample_rate <= 0:
            return
        full_audio = np.concatenate(pcm_chunks).astype(np.float32, copy=False)
        if full_audio.size == 0:
            return
        duration = full_audio.size / float(sample_rate)
        if duration > self._CONTINUATION_MAX_SECONDS:
            logger.debug(
                "[chained-prompt] skip cache update session_id=%s duration=%.2fs > %.2fs",
                state.session_id,
                duration,
                self._CONTINUATION_MAX_SECONDS,
            )
            return

        new_path = self._write_continuation_wav(state.session_id, full_audio, sample_rate)
        if not new_path:
            return

        old_path = state.tts_prompt_wav_path
        if old_path and old_path != new_path and os.path.exists(old_path):
            try:
                os.unlink(old_path)
            except OSError:
                pass

        state.tts_prompt_wav_path = new_path
        state.tts_prompt_text = text.strip()
        state.tts_prompt_voice_signature = self._compute_voice_signature(voice_cfg)
        logger.debug(
            "[chained-prompt] cache updated session_id=%s wav=%s duration=%.2fs",
            state.session_id,
            new_path,
            duration,
        )

    @staticmethod
    def _write_continuation_wav(
        session_id: str, audio: np.ndarray, sample_rate: int
    ) -> Optional[str]:
        try:
            import soundfile as sf  # type: ignore
        except Exception:
            logger.warning("[chained-prompt] soundfile unavailable, disable chained prompt")
            return None

        safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in session_id)[:48]
        try:
            fd, path = tempfile.mkstemp(prefix=f"voxcpm-chain-{safe}-", suffix=".wav")
            os.close(fd)
        except OSError:
            logger.exception("[chained-prompt] mkstemp failed session_id=%s", session_id)
            return None
        try:
            sf.write(path, np.clip(audio, -1.0, 1.0), int(sample_rate), format="WAV")
        except Exception:
            logger.exception("[chained-prompt] soundfile.write failed path=%s", path)
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
            return None
        return path

    @staticmethod
    def _purge_continuation_cache(state: AudioSessionState) -> None:
        """清掉 cache 文件 + state 字段。session.close / 音色切换 / 文件丢失时调用。"""
        old = state.tts_prompt_wav_path
        if old and os.path.exists(old):
            try:
                os.unlink(old)
            except OSError:
                pass
        state.tts_prompt_wav_path = None
        state.tts_prompt_text = ""
        state.tts_prompt_voice_signature = ""


async def _await_silently(task: asyncio.Task) -> None:
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def _voice_config_from_dict(raw: Dict[str, Any]) -> VoiceConfig:
    """把协议里的 voice_cfg 字典映射到 VoiceConfig dataclass。

    仅保留引擎关心的字段；未识别字段忽略，避免引擎层出现脏键。
    """
    if not isinstance(raw, dict):
        raw = {}

    def _get_str(key: str) -> Optional[str]:
        value = raw.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_int(key: str) -> Optional[int]:
        value = raw.get(key)
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_float(key: str) -> Optional[float]:
        value = raw.get(key)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    generation_cfg = raw.get("generation_cfg")
    if not isinstance(generation_cfg, dict):
        generation_cfg = None

    return VoiceConfig(
        tts_mode=str(raw.get("tts_mode") or "custom_voice").strip().lower() or "custom_voice",
        speaker_name=_get_str("speaker_name"),
        voice_preset=_get_str("voice_preset"),
        instruct=_get_str("instruct"),
        ref_audio=_get_str("ref_audio"),
        ref_text=_get_str("ref_text"),
        x_vector_only=bool(raw.get("x_vector_only", False)),
        language=_get_str("language"),
        sample_rate=_get_int("sample_rate"),
        file_type=str(raw.get("file_type") or "wav"),
        load_name=_get_str("load_name"),
        design_seed_text=_get_str("design_seed_text"),
        design_instruct=_get_str("design_instruct"),
        clone_base_load_name=_get_str("clone_base_load_name"),
        prompt_wav_path=_get_str("prompt_wav_path"),
        prompt_text=_get_str("prompt_text"),
        continuation_prompt=bool(raw.get("continuation_prompt", False)),
        continuation_reference_wav_path=_get_str("continuation_reference_wav_path"),
        guidance_scale=_get_float("guidance_scale"),
        num_inference_steps=_get_int("num_inference_steps"),
        generation_cfg=dict(generation_cfg) if generation_cfg else None,
    )


__all__ = [
    "AudioSessionRuntime",
    "AudioSessionState",
    "ASR_ROLE",
    "TTS_ROLE",
]
