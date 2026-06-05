"""TTS 提交、音频输出事件转发与 barge-in 播放窗口估算。"""

from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Optional, TYPE_CHECKING

import numpy as np

from backend.core.logger import get_app_logger
from backend.services.chat.avatar.livetalking_client import get_livetalking_client

if TYPE_CHECKING:
    from backend.services.chat.session.runtime import SessionRuntime

logger = get_app_logger(__name__)


class _BlockLoudnessSmoother:
    """跨 block 的响度平滑器（方案 C：rolling RMS smoothing）。

    问题：``voice_reply.VoiceReplyStream`` 把 LLM 流式输出按句聚合成 block，
    每个 block 都会调一次 ``submit_tts`` → 后端 VoxCPM 做一次**完全独立**的
    ``model.generate``。每段独立合成的 prosody / loudness 不一致，听感上句与
    句之间出现响度跳变和"换气式"断档。

    思路：维护跨 block 的 ``anchor_rms``（上一 block 尾段 RMS 的 EMA 历史值），
    新 block 的段头几百毫秒按需乘一个 gain，把段头的响度对齐到 anchor，再
    沿"渐回 1.0"曲线在 ~400ms 内退回模型自然输出，避免一刀切压平段内韵律。

    设计取舍：
      * 段头压平（不抬尾）：模型尾音自然下落是合理的，强行抬尾会失去韵律。
      * 仅 chunk 级 gain，不 buffer 整段：保持 chunk-level streaming 的低延迟。
      * baseline 阶段（首段，``anchor_rms=None``）字节透传，不引入量化误差。
        这条契约对单元测试也很关键（测试断言 emit 出去的 PCM == 输入 PCM）。
      * gain clip 到 ``[0.6, 1.6]``：极端差异不强压，避免失真和误判。
    """

    def __init__(self) -> None:
        self.enabled: bool = (
            os.environ.get("VITOOM_TTS_LOUDNESS_SMOOTH", "1").strip()
            in {"1", "true", "True"}
        )
        # 跨 block 的尾段 RMS（normalized 到 [0,1]），首段建立后保留并 EMA 更新。
        # 这是平滑的"目标响度基准"——下一 block 的段头会被压/拉到这个值附近。
        self._anchor_rms: Optional[float] = None
        # 当前 block 的状态（每个新 request_id 重置）。
        self._active_request_id: Optional[str] = None
        self._block_samples_seen: int = 0
        self._block_first_gain: float = 1.0
        # 尾段 ~200ms PCM 缓冲，用于在 is_final 时算 anchor。
        self._tail_buffer: Deque[np.ndarray] = deque()
        self._tail_buffer_samples: int = 0
        # 调参常量（保守取值；未来可挂到 settings）
        self._fade_seconds: float = 0.4
        self._tail_window_seconds: float = 0.2
        self._anchor_ema_alpha: float = 0.5
        self._gain_clip: tuple = (0.6, 1.6)
        # 静音段不参与 anchor 更新（避免段尾全 0 把 anchor 拉到 0）。
        self._min_meaningful_rms: float = 1e-3
        # |gain-1| 小于这个阈值时跳过乘法，直接字节透传。
        self._min_apply_gain_diff: float = 0.01

    def process_chunk(
        self,
        pcm_body: bytes,
        *,
        request_id: str,
        sample_rate: Optional[int],
    ) -> bytes:
        """对单个 PCM chunk 应用响度平滑。返回处理后的 int16 LE PCM bytes。

        路径上若任一前置条件不满足（disabled / 空 PCM / 无 sample_rate /
        gain ≈ 1），都**字节透传**原 ``pcm_body``，保证零侵入。
        """

        if not self.enabled or not pcm_body or not sample_rate or sample_rate <= 0:
            return pcm_body
        try:
            samples = np.frombuffer(pcm_body, dtype=np.int16)
        except Exception:
            return pcm_body
        if samples.size == 0:
            return pcm_body

        arr = samples.astype(np.float32) / 32768.0

        if request_id != self._active_request_id:
            # 进入新 block：reset 计数 + 算 first_gain。
            self._active_request_id = request_id
            self._block_samples_seen = 0
            self._tail_buffer.clear()
            self._tail_buffer_samples = 0
            if self._anchor_rms is None:
                # 首段建立 baseline，整段不动 PCM；尾段 RMS 仍会被采集。
                self._block_first_gain = 1.0
            else:
                block_first_rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
                if block_first_rms < self._min_meaningful_rms:
                    self._block_first_gain = 1.0
                else:
                    ratio = self._anchor_rms / max(block_first_rms, 1e-6)
                    self._block_first_gain = float(np.clip(ratio, *self._gain_clip))

        # 段头按"线性 ease 回 1.0"应用 gain：chunk 内取首尾进度的均值，
        # 避免单一 gain 在 chunk 边界出现可感知的台阶。
        fade_samples = max(1, int(sample_rate * self._fade_seconds))
        seen_before = self._block_samples_seen
        seen_after = seen_before + arr.size
        progress_start = min(1.0, seen_before / fade_samples)
        progress_end = min(1.0, seen_after / fade_samples)
        progress_avg = (progress_start + progress_end) / 2.0
        gain = self._block_first_gain * (1.0 - progress_avg) + 1.0 * progress_avg

        applied = abs(gain - 1.0) > self._min_apply_gain_diff
        if applied:
            arr = np.clip(arr * gain, -1.0, 1.0)
        self._block_samples_seen = seen_after

        # 累积尾段缓冲：保留最近约 ``_tail_window_seconds`` 的样本即可。
        # 缓冲存的是**应用 gain 后**的 PCM，这样下次 anchor 更新基于真实播放出去的能量。
        self._tail_buffer.append(arr)
        self._tail_buffer_samples += arr.size
        target_window = int(sample_rate * self._tail_window_seconds)
        while (
            len(self._tail_buffer) > 1
            and (self._tail_buffer_samples - self._tail_buffer[0].size)
            >= target_window
        ):
            dropped = self._tail_buffer.popleft()
            self._tail_buffer_samples -= dropped.size

        if not applied:
            # gain 没动 → 字节透传，不引入 float→int16 量化误差。
            return pcm_body
        out = np.clip(arr * 32767.0, -32768.0, 32767.0).astype(np.int16).tobytes()
        return out

    def finalize_block(self, *, request_id: str, sample_rate: Optional[int]) -> None:
        """block 的 ``is_final`` 到达：用尾段 RMS 更新 anchor 并清理状态。"""

        if not self.enabled:
            return
        if request_id and request_id != self._active_request_id:
            # 异常 / stale：本次 final 跟当前 block 不对应，直接丢弃。
            return
        if not self._tail_buffer or not sample_rate or sample_rate <= 0:
            self._reset_block_state()
            return

        target_window = max(1, int(sample_rate * self._tail_window_seconds))
        try:
            concatenated = np.concatenate(list(self._tail_buffer), axis=0)
        except ValueError:
            self._reset_block_state()
            return
        if concatenated.size == 0:
            self._reset_block_state()
            return
        tail = (
            concatenated[-target_window:]
            if concatenated.size > target_window
            else concatenated
        )
        new_anchor = float(np.sqrt(np.mean(tail * tail) + 1e-12))
        if new_anchor >= self._min_meaningful_rms:
            if self._anchor_rms is None:
                self._anchor_rms = new_anchor
            else:
                alpha = self._anchor_ema_alpha
                self._anchor_rms = (1.0 - alpha) * self._anchor_rms + alpha * new_anchor
        self._reset_block_state()

    def reset(self) -> None:
        """连接关闭等场景下整体重置（含 anchor）。"""

        self._anchor_rms = None
        self._reset_block_state()

    def _reset_block_state(self) -> None:
        self._active_request_id = None
        self._block_samples_seen = 0
        self._tail_buffer.clear()
        self._tail_buffer_samples = 0
        self._block_first_gain = 1.0


class TtsCoordinator:
    def __init__(self, runtime: "SessionRuntime") -> None:
        self._runtime = runtime
        self.pending_audio_playback_until_ts: Optional[datetime] = None
        self.barge_in_playback_grace_extra_ms = 1500
        # 跨 block 响度平滑器，详见 ``_BlockLoudnessSmoother``。
        self._loudness = _BlockLoudnessSmoother()

    async def submit(
        self,
        *,
        text: str,
        voice_cfg: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: float = 240.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        rt = self._runtime
        if rt._inference_session is None:
            raise RuntimeError("no inference session manager available for TTS")
        if not rt._inference.tts_session_opened:
            await rt._inference.ensure_tts_session_opened()
            if not rt._inference.tts_session_opened:
                raise RuntimeError("failed to open tts inference session")
        rid = await rt._inference_session.tts_request(
            text=text,
            voice_cfg=dict(voice_cfg or rt._inference.voice_output_config() or {}),
            request_id=request_id,
            metadata=metadata,
        )
        await rt._inference_session.await_tts_finish(rid, timeout=timeout)
        return rid

    async def handle_audio_out(self, event: Dict[str, Any], *, is_final: bool) -> None:
        rt = self._runtime
        request_id = str(event.get("request_id") or "").strip()
        is_active = bool(
            request_id
            and rt._inference_session is not None
            and rt._inference_session.active_tts_request_id == request_id
        )
        if is_active:
            pcm = event.get("binary_bytes") if not is_final else None
            pcm_body = bytes(pcm) if isinstance(pcm, (bytes, bytearray)) else b""
            mime = str(event.get("mime") or event.get("mime_type") or "audio/wav")
            try:
                sr: Optional[int] = int(event["sample_rate"]) if event.get("sample_rate") is not None else None
            except (TypeError, ValueError):
                sr = None
            # 跨 block 的响度平滑：把段头响度对齐到上一 block 尾段 RMS，再 ease 回模型自然输出。
            # 首 block / disabled / 空 PCM 时透传，保持原有字节级契约。
            if pcm_body:
                pcm_body = self._loudness.process_chunk(
                    pcm_body, request_id=request_id, sample_rate=sr
                )
            if is_final:
                self._loudness.finalize_block(request_id=request_id, sample_rate=sr)
            # 数字人副链路镜像（plan 不变量第 1 条：non-blocking，不带 await）。
            # `push_pcm` 内部完成 16k mono resample + bounded queue 入队 + lazy 起 consumer
            # task；sidecar 慢 / 卡死 / WS 断都不会反向阻塞下面的 emit_audio_delta。
            # `is_final=True` 路径调 flush 通知 sidecar 段结束。stale rid 在本方法
            # 入口 `is_active` 校验中已被过滤，这里不会送 stale chunk 到 sidecar。
            try:
                client = get_livetalking_client()
                if pcm_body:
                    client.push_pcm(
                        rt.session_id,
                        request_id,
                        pcm_body,
                        sample_rate=sr,
                        channels=1,
                    )
                if is_final:
                    client.flush(rt.session_id, request_id)
            except Exception as exc:
                # 任何异常都吞掉：装饰性数字人副链路绝不能影响 audio_delta SLA
                # 但用 warning 级别让默认 INFO 配置下也能看到（之前 debug 等价于丢弃，
                # 导致"三方无日志"的诊断盲区）。
                logger.warning(
                    "livetalking push_pcm/flush swallowed session=%s rid=%s err=%s",
                    rt.session_id, request_id, exc,
                )
            await rt.emit_audio_delta(pcm_bytes=pcm_body, mime=mime, is_final=is_final, sample_rate=sr)
        else:
            logger.debug(
                "drop stale tts audio chunk session=%s request_id=%s is_final=%s",
                rt.session_id,
                request_id or "<none>",
                is_final,
            )
        if not is_final or not request_id or rt._inference_session is None:
            return
        err_code = str(event.get("error_code") or "").strip()
        err_msg = f"{err_code}: {event.get('error') or ''}".strip(":  ") if err_code else None
        rt._inference_session.resolve_tts_waiter(request_id, error=err_msg)

    def advance_playback_estimate(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: Optional[int],
        mime: str,
    ) -> None:
        sr = _extract_sample_rate(sample_rate, mime)
        samples = max(0, len(pcm_bytes) // 2)
        if samples <= 0:
            return
        duration_ms = int(samples * 1000 / sr)
        if duration_ms <= 0:
            return
        now = datetime.utcnow()
        baseline = self.pending_audio_playback_until_ts or now
        if baseline < now:
            baseline = now
        self.pending_audio_playback_until_ts = baseline + timedelta(milliseconds=duration_ms)

    def is_within_playback_window(self) -> bool:
        until = self.pending_audio_playback_until_ts
        if until is None:
            return False
        grace = timedelta(milliseconds=max(0, self.barge_in_playback_grace_extra_ms))
        return datetime.utcnow() < until + grace

    def reset_playback_window(self) -> None:
        self.pending_audio_playback_until_ts = None
        # 连接关闭场景下顺手清掉响度平滑器的累积状态，避免 SessionRuntime 复用时跨会话泄漏。
        self._loudness.reset()


def _extract_sample_rate(sample_rate: Optional[int], mime: str) -> int:
    try:
        sr = int(sample_rate) if sample_rate is not None else 0
    except (TypeError, ValueError):
        sr = 0
    if sr > 0:
        return sr
    mime_lower = str(mime or "").lower()
    if "rate=" in mime_lower:
        try:
            return int(mime_lower.split("rate=", 1)[1].split(";", 1)[0])
        except (ValueError, IndexError):
            pass
    return 24000


__all__ = ["TtsCoordinator"]
