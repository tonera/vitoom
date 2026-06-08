"""Streaming VAD wrapper used by chat (turn 起止 + barge-in 探测)。

主判定：Silero VAD v5 ONNX 模型——业界对嘈杂环境（电视/广播/远处对话/背景音乐）
鲁棒性最好的开源 VAD。模型权重 ~2MB，单帧推理 (16kHz=32ms 帧) ~1ms CPU。

兜底：能量地板（``energy_speech_floor``）+ 连续 streak。Silero 不可用时退回
纯能量阈值。能量地板还可以叠加在 Silero 之上做"低能量噪声二级过滤"（turn /
barge_in profile 都建议设值）。

设计要点：
  * push(pcm_bytes) → List[VadEvent]，事件类型仅 ``speech_start`` / ``speech_end``。
  * 内部把任意 chunk_ms 的 PCM 累积成 Silero 严格要求的 16kHz=512 samples / 帧
    送推理；剩下不满 512 的 PCM 留在 buffer 等下次。
  * Silero 是带 hidden state 的 LSTM——每个 detector 持自己的 state；同 onnx
    模型权重 / InferenceSession 在进程级 LRU 缓存。
"""

from __future__ import annotations

import array
import asyncio
import math
import threading
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, List, Optional, Sequence

from backend.core.config import get_config
from backend.core.logger import get_app_logger

logger = get_app_logger(__name__)


# ---------------------------------------------------------------------------
# Silero VAD ONNX session 进程级缓存
# ---------------------------------------------------------------------------

# Silero v5 / 16kHz：每次推理输入 `[1, context_size + frame_size]`，frame=512、context=64。
# 8kHz：frame=256、context=32。context 是上一帧的最后 N 个 samples（首次推理用 0），
# 保留来给 LSTM 提供边界连续性。直接喂 `[1, 512]` 不拼 context 模型会给基线输出（~0.0006），
# 与真实人声不可区分——这是 v5 相对 v4 的接口变更，必须严格遵守。详见 silero_vad pip
# 包 `OnnxWrapper.__call__`。
_SILERO_FRAME_SAMPLES_16K = 512
_SILERO_FRAME_SAMPLES_8K = 256
_SILERO_CONTEXT_SAMPLES_16K = 64
_SILERO_CONTEXT_SAMPLES_8K = 32
_SILERO_STATE_SHAPE = (2, 1, 128)

# 业界标准下载源：snakers4/silero-vad GitHub raw（v5 ONNX，~2MB）。
_SILERO_DEFAULT_ONNX_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_SILERO_DEFAULT_MODEL_NAME = "silero-vad"
_SILERO_ONNX_FILE_NAME = "silero_vad.onnx"

_SILERO_SESSION_CACHE: dict[str, Any] = {}
_SILERO_SESSION_LOAD_LOCK = threading.Lock()


def _load_or_get_silero_session_sync(onnx_path: str) -> Optional[Any]:
    """同步获取 onnxruntime InferenceSession，命中进程级缓存就直接返回。

    设计为在 ``asyncio.to_thread`` 里调用，避免 ONNX 模型 load 阻塞事件循环。
    """
    with _SILERO_SESSION_LOAD_LOCK:
        cached = _SILERO_SESSION_CACHE.get(onnx_path)
        if cached is not None:
            return cached
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            logger.info("onnxruntime import failed, using energy vad fallback: %s", exc)
            return None
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(
                onnx_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            logger.warning("silero onnx load failed path=%s err=%s", onnx_path, exc)
            return None
        _SILERO_SESSION_CACHE[onnx_path] = session
        logger.info("silero vad onnx loaded and cached path=%s", onnx_path)
        return session


# ---------------------------------------------------------------------------
# 公共数据结构
# ---------------------------------------------------------------------------


@dataclass
class VadEvent:
    type: str
    frames: List[bytes]
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# StreamingVadDetector
# ---------------------------------------------------------------------------


class StreamingVadDetector:
    """流式 VAD：Silero v5 + 能量地板兜底。

    chat 调用面：
      * ``push(pcm_bytes)`` 喂连续 PCM-S16LE，返回积累出的 VadEvent；
      * ``reset()`` 在 turn 切换时清状态；
      * ``warmup()`` 在 session ready 后台预热，避开首帧 ~100ms 的 onnxruntime
        warm 延迟（Silero 比 fsmn-vad 快得多，这步远没有 fsmn 的 ~4s 那么吃紧）；
      * ``diag_snapshot()`` 给 chat 层 barge-in 日志用。

    构造参数与 chat ``_vad_config`` 保持一致——只暴露真正需要由配置决定的项。
    fsmn 专有的 ``respect_funasr_utterance_edges`` / 4 态机已废弃。
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        chunk_ms: int = 200,
        pre_roll_ms: int = 300,
        min_speech_ms: int = 200,
        silence_ms: int = 2500,
        max_end_silence_time: Optional[int] = None,
        model_name: str = _SILERO_DEFAULT_MODEL_NAME,
        model_source: str = _SILERO_DEFAULT_ONNX_URL,
        model_dir: Optional[str] = None,
        model_factory: Optional[Callable[[str], Any]] = None,
        speech_threshold: float = 0.5,
        energy_threshold: float = 0.012,
        energy_speech_floor: float = 0.0,
        energy_floor_min_streak: int = 1,
        use_neural_vad: bool = True,
    ) -> None:
        self.sample_rate = int(sample_rate or 16000)
        self.frame_ms = int(frame_ms or 20)
        self.chunk_ms = int(chunk_ms or 200)
        self.pre_roll_ms = int(pre_roll_ms or 300)
        self.min_speech_ms = int(min_speech_ms or 200)
        self.silence_ms = int(max_end_silence_time or silence_ms or 2500)
        self.model_name = str(model_name or _SILERO_DEFAULT_MODEL_NAME)
        self.model_source = str(model_source or _SILERO_DEFAULT_ONNX_URL)
        self.model_dir = str(model_dir or "").strip()
        self._model_factory = model_factory
        self._speech_threshold = float(speech_threshold)
        self._energy_threshold = float(energy_threshold)
        # 能量地板二级过滤：Silero 即便误判 speech，也要求 RMS ≥ floor 才放行。
        # turn profile 推荐 0.02 挡掉低能量背景噪声（电视/远处对话）；barge_in
        # profile 一并设 0.02 + streak=2 应对 double-talk 下 echo 残留。
        self._energy_speech_floor = float(energy_speech_floor)
        self._energy_floor_min_streak = max(1, int(energy_floor_min_streak))
        self._energy_floor_streak = 0
        self._use_neural_vad = bool(use_neural_vad)

        self._silero: Optional[_SileroEngine] = None
        self._silero_load_failed = False
        self._lock: Optional[asyncio.Lock] = None

        self._pre_roll: Deque[bytes] = deque(maxlen=max(0, self.pre_roll_ms // self.frame_ms))
        self._chunk_frames: List[bytes] = []
        self._chunk_samples = 0
        self._speaking = False
        self._pending_speech_ms = 0
        self._silence_run_ms = 0
        self._speech_duration_ms = 0
        # 仅用于诊断日志：最近一次 chunk 的判决与 Silero probability / RMS。
        self._last_state: str = "init"
        self._last_prob: float = 0.0
        self._last_rms: float = 0.0

    # ------------------------------------------------------------------
    # 对外属性
    # ------------------------------------------------------------------

    @property
    def speaking(self) -> bool:
        return self._speaking

    def diag_snapshot(self) -> dict:
        """chat 层 barge-in 日志用：能告诉运维"detector 看到 speech 了吗、
        累计了多少、是被 silence 重置了吗"，无需触碰私有字段。
        """
        return {
            "speaking": self._speaking,
            "pending_speech_ms": self._pending_speech_ms,
            "silence_run_ms": self._silence_run_ms,
            "speech_duration_ms": self._speech_duration_ms,
            "buffered_chunk_ms": int(round(self._chunk_samples * 1000 / max(1, self.sample_rate))),
            "min_speech_ms": self.min_speech_ms,
            "silence_ms": self.silence_ms,
            "speech_threshold": round(self._speech_threshold, 3),
            "energy_speech_floor": round(self._energy_speech_floor, 4),
            "energy_floor_streak": self._energy_floor_streak,
            "energy_floor_min_streak": self._energy_floor_min_streak,
            "last_state": self._last_state,
            "last_prob": round(self._last_prob, 3),
            "last_rms": round(self._last_rms, 4),
            "model_loaded": self._silero is not None,
            "model_load_failed": self._silero_load_failed,
            "use_neural_vad": self._use_neural_vad,
        }

    # ------------------------------------------------------------------
    # 流式入口
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._pre_roll.clear()
        self._chunk_frames = []
        self._chunk_samples = 0
        self._speaking = False
        self._pending_speech_ms = 0
        self._silence_run_ms = 0
        self._speech_duration_ms = 0
        self._energy_floor_streak = 0
        self._last_state = "init"
        self._last_prob = 0.0
        self._last_rms = 0.0
        if self._silero is not None:
            self._silero.reset()

    async def push(self, pcm_bytes: bytes) -> List[VadEvent]:
        if not pcm_bytes:
            return []
        frame = bytes(pcm_bytes)
        self._pre_roll.append(frame)
        self._chunk_frames.append(frame)
        self._chunk_samples += max(0, len(frame) // 2)

        target_samples = max(1, int(self.sample_rate * self.chunk_ms / 1000))
        if self._chunk_samples < target_samples:
            return []

        frames = self._chunk_frames
        self._chunk_frames = []
        self._chunk_samples = 0
        return await self._process_frames(frames, is_final=False)

    async def finish(self) -> List[VadEvent]:
        frames = self._chunk_frames
        self._chunk_frames = []
        self._chunk_samples = 0
        events = await self._process_frames(frames, is_final=True) if frames else []
        if self._speaking:
            events.append(VadEvent(type="speech_end", frames=[], duration_ms=self._speech_duration_ms))
        self.reset()
        return events

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    async def _process_frames(self, frames: List[bytes], *, is_final: bool) -> List[VadEvent]:
        pcm = b"".join(frames)
        if not pcm:
            return []
        rms = _rms_pcm16(pcm)
        state = await self._detect_speech_state(pcm, rms=rms, is_final=is_final)
        chunk_duration_ms = int(round((len(pcm) / 2) * 1000 / self.sample_rate))
        self._last_state = state
        self._last_rms = rms
        return self._advance_state(state=state, duration_ms=chunk_duration_ms)

    async def _detect_speech_state(self, pcm: bytes, *, rms: float, is_final: bool) -> str:
        prob: Optional[float] = None
        if self._use_neural_vad and not self._silero_load_failed:
            try:
                if self._lock is None:
                    self._lock = asyncio.Lock()
                async with self._lock:
                    engine = await self._ensure_engine_async()
                    if engine is not None:
                        audio = _pcm16_to_float32_array(pcm)
                        prob = await asyncio.to_thread(engine.detect, audio)
            except Exception as exc:
                self._silero_load_failed = True
                logger.warning("silero vad unavailable, falling back to energy vad: %s", exc)
                prob = None
        self._last_prob = prob if prob is not None else 0.0

        if prob is None:
            # Silero 不可用 → 纯能量阈值兜底（与 fsmn 时代行为一致）。
            return "speech" if rms >= self._energy_threshold else "silence"

        # Silero 判 speech：还要过能量地板（如果配置了）。
        if prob >= self._speech_threshold:
            if self._energy_speech_floor > 0:
                if rms < self._energy_speech_floor:
                    # Silero 高分但实际能量低：典型电视/远处对话。streak 不增。
                    self._energy_floor_streak = 0
                    return "silence"
                self._energy_floor_streak += 1
                if self._energy_floor_streak < self._energy_floor_min_streak:
                    return "silence"
            else:
                self._energy_floor_streak = 0
            return "speech"

        # Silero 判 silence：streak 清零并听 Silero（不再因 RMS 高强判 speech）。
        # 旧的"FunASR silence + 能量过 floor → 强判 speech"通路是为 fsmn 在 double-talk
        # 下倾向 silence 设计的兜底；Silero 对 double-talk 鲁棒得多，不需要这条捷径。
        self._energy_floor_streak = 0
        return "silence"

    async def warmup(self) -> bool:
        """预加载 Silero ONNX session + 跑一帧 silence，把 onnxruntime 内部
        kernel cache / mem allocator 都热起来。

        Silero 单帧推理 ~1ms，warm 流程 < 200ms（远比 fsmn 的 ~4s 轻）。
        warmup 失败返回 False，上层落到 lazy load 路径不报错（首次 push 会
        承担一次性加载延迟，~100ms）。
        """
        if not self._use_neural_vad:
            return False
        try:
            engine = await self._ensure_engine_async()
            if engine is None:
                return False
            silence_samples = max(_SILERO_FRAME_SAMPLES_16K, int(self.sample_rate * self.chunk_ms / 1000))
            silence_pcm = b"\x00\x00" * silence_samples
            await asyncio.to_thread(engine.detect, _pcm16_to_float32_array(silence_pcm))
            engine.reset()  # 清掉 warm 用过的 hidden state，避免污染下次真实推理
            logger.info("silero vad warmed up profile=%s", self.model_name)
            return True
        except Exception as exc:
            logger.warning("silero vad warmup failed: %s", exc)
            return False

    async def _ensure_engine_async(self) -> Optional["_SileroEngine"]:
        if self._silero is not None:
            return self._silero
        # 测试注入路径：不走全局缓存（每个 fake engine 独立实例）。
        if self._model_factory is not None:
            self._silero = self._model_factory(self.model_name)
            return self._silero
        onnx_path = await asyncio.to_thread(self._ensure_local_onnx_sync)
        if not onnx_path:
            self._silero_load_failed = True
            return None
        try:
            session = await asyncio.to_thread(_load_or_get_silero_session_sync, onnx_path)
        except Exception as exc:
            self._silero_load_failed = True
            logger.warning("silero load via thread failed, falling back to energy vad: %s", exc)
            return None
        if session is None:
            self._silero_load_failed = True
            return None
        self._silero = _SileroEngine(session=session, sample_rate=self.sample_rate)
        return self._silero

    # ------------------------------------------------------------------
    # 模型文件下载（与 fsmn-vad 时代的目录结构 / storage_path 完全对齐）
    # ------------------------------------------------------------------

    def _ensure_local_onnx_sync(self) -> Optional[str]:
        """返回本地 onnx 文件绝对路径；不存在就按 ``model_source`` 下载。

        路径策略：``{models.storage_path}/{safe(model_name)}/silero_vad.onnx``
        与原 fsmn-vad 完全一致，便于运维统一管理。
        """
        local_dir = self._local_model_dir()
        if local_dir is None:
            logger.warning("models.storage_path not configured, silero vad cannot persist locally")
            return None
        onnx_path = local_dir / _SILERO_ONNX_FILE_NAME
        if onnx_path.exists() and onnx_path.stat().st_size > 0:
            logger.info("reuse local silero vad onnx path=%s", onnx_path)
            return str(onnx_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(self.model_source, str(onnx_path))
            logger.info(
                "downloaded silero vad onnx source=%s local=%s size=%d",
                self.model_source,
                onnx_path,
                onnx_path.stat().st_size,
            )
            return str(onnx_path)
        except Exception as exc:
            logger.warning(
                "download silero vad onnx to %s failed source=%s err=%s",
                onnx_path,
                self.model_source,
                exc,
            )
            try:
                if onnx_path.exists():
                    onnx_path.unlink()
            except Exception:
                pass
            return None

    def _local_model_dir(self) -> Optional[Path]:
        explicit = self.model_dir
        if explicit:
            return _resolve_repo_path(explicit)
        storage_path = str(get_config("models.storage_path", "resources/models") or "").strip()
        if not storage_path:
            return None
        return _resolve_repo_path(storage_path) / _safe_model_dir_name(self.model_name)

    # ------------------------------------------------------------------
    # 状态机：从 chunk 级 speech/silence → speech_start / speech_end 事件
    # ------------------------------------------------------------------

    def _advance_state(self, *, state: str, duration_ms: int) -> List[VadEvent]:
        events: List[VadEvent] = []
        speech_now = state == "speech"
        if speech_now:
            self._pending_speech_ms += duration_ms
            self._silence_run_ms = 0
        else:
            self._pending_speech_ms = 0
            if self._speaking:
                self._silence_run_ms += duration_ms

        if not self._speaking and self._pending_speech_ms >= self.min_speech_ms:
            self._speaking = True
            self._speech_duration_ms = self._pending_speech_ms
            events.append(VadEvent(type="speech_start", frames=list(self._pre_roll)))
            return events

        if self._speaking:
            if speech_now:
                self._speech_duration_ms += duration_ms
            if self._silence_run_ms >= self.silence_ms:
                events.append(VadEvent(type="speech_end", frames=[], duration_ms=self._speech_duration_ms))
                self.reset()
        return events


# ---------------------------------------------------------------------------
# Silero ONNX 推理引擎（detector 私有；onnx session 进程级共享）
# ---------------------------------------------------------------------------


class _SileroEngine:
    """对 ``InferenceSession`` 的薄包装。每个 detector 持自己的 hidden state +
    跨 chunk PCM buffer；onnx 模型权重共享。
    """

    def __init__(self, *, session: Any, sample_rate: int) -> None:
        self._session = session
        self._sample_rate = int(sample_rate)
        if self._sample_rate == 16000:
            self._frame_samples = _SILERO_FRAME_SAMPLES_16K
            self._context_size = _SILERO_CONTEXT_SAMPLES_16K
        else:
            self._frame_samples = _SILERO_FRAME_SAMPLES_8K
            self._context_size = _SILERO_CONTEXT_SAMPLES_8K
        # 输入 / 输出 名字按 onnx model 真实定义动态获取，避免硬编码在不同导出版本下漂移。
        self._input_audio_name = session.get_inputs()[0].name
        self._input_state_name = session.get_inputs()[1].name
        self._input_sr_name = session.get_inputs()[2].name
        self._output_prob_name = session.get_outputs()[0].name
        self._output_state_name = session.get_outputs()[1].name
        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - 应在 _ensure_engine 之前已拦下
            raise RuntimeError(f"numpy required for silero vad: {exc}") from exc
        self._np = np
        self._sr_input = np.array(self._sample_rate, dtype=np.int64)
        self._state = np.zeros(_SILERO_STATE_SHAPE, dtype=np.float32)
        # 上一帧最后 N 个 samples，下一帧推理时拼在前面。首次推理用 0 填充。
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._buffer = np.zeros((0,), dtype=np.float32)

    def reset(self) -> None:
        np = self._np
        self._state = np.zeros(_SILERO_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._buffer = np.zeros((0,), dtype=np.float32)

    def detect(self, audio: Any) -> float:
        """对一段 float32 audio 跑 Silero，返回该段内最大 speech probability。

        chunk 不一定刚好是 frame_samples 的整数倍——多出来 < frame_samples 的尾部
        留在 ``self._buffer`` 等下次 push 拼接，保证流式 LSTM state + context 正确。

        Silero v5 / 16kHz 协议：每次推理输入是 `[1, 64 + 512] = [1, 576]`，前 64
        个 samples 是上一帧的尾部 context（首次推理为 0），后 512 是当前帧。模型
        输出 prob 后，把当前 576 samples 的最后 64 个保存为下次的 context。
        直接喂 `[1, 512]` 不拼 context 模型只会输出基线值（~0.0006），与真实人声
        不可区分。
        """
        np = self._np
        if audio is None:
            return 0.0
        if hasattr(audio, "size"):
            if audio.size == 0:
                return 0.0
            audio_arr = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        else:
            audio_arr = np.asarray(audio, dtype=np.float32)
            if audio_arr.size == 0:
                return 0.0
        self._buffer = (
            np.concatenate([self._buffer, audio_arr]) if self._buffer.size else audio_arr.copy()
        )
        max_prob = 0.0
        while self._buffer.size >= self._frame_samples:
            frame = self._buffer[: self._frame_samples].reshape(1, -1)
            self._buffer = self._buffer[self._frame_samples :]
            x_with_ctx = np.concatenate([self._context, frame], axis=1)
            output, new_state = self._session.run(
                [self._output_prob_name, self._output_state_name],
                {
                    self._input_audio_name: x_with_ctx,
                    self._input_state_name: self._state,
                    self._input_sr_name: self._sr_input,
                },
            )
            self._state = new_state
            self._context = x_with_ctx[..., -self._context_size :]
            prob = float(output[0][0])
            if prob > max_prob:
                max_prob = prob
        return max_prob


# ---------------------------------------------------------------------------
# PCM helpers
# ---------------------------------------------------------------------------


def _pcm16_to_float32_array(raw: bytes) -> Any:
    samples = _pcm16_samples(raw)
    try:
        import numpy as np  # type: ignore
    except Exception:
        return [sample / 32768.0 for sample in samples]
    if not samples:
        return np.zeros((0,), dtype=np.float32)
    return (np.asarray(samples, dtype=np.float32) / 32768.0).copy()


def _pcm16_samples(raw: bytes) -> Sequence[int]:
    if not raw:
        return []
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    return samples


def _rms_pcm16(raw: bytes) -> float:
    samples = _pcm16_samples(raw)
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        normalized = sample / 32768.0
        total += normalized * normalized
    return math.sqrt(total / len(samples))


# ---------------------------------------------------------------------------
# 路径 helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return _repo_root() / path


def _safe_model_dir_name(model_name: str) -> str:
    normalized = str(model_name or _SILERO_DEFAULT_MODEL_NAME).strip().replace("\\", "/")
    return normalized.strip("/").replace("/", "--") or _SILERO_DEFAULT_MODEL_NAME


__all__ = ["StreamingVadDetector", "VadEvent"]
