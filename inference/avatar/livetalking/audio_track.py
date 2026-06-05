"""``MuseTalkAudioTrack`` —— 把 ``MuseTalkRuntime`` 的音频帧接到 aiortc。

方案 A 的核心：sidecar 在同一 ``RTCPeerConnection`` 上推 video + audio 两条
RTP 流，浏览器用 RTP 时间戳天然对齐 lip-sync（这是 WebRTC 的本职工作），
端到端音视频时差降到 ≤ 50ms。

设计要点：

1. ``recv()`` 按 20ms wall-clock pacing 输出 16k mono ``av.AudioFrame``，
   PTS = 累计样本数（time_base = 1/16000）。aiortc 内部会再转 Opus@48k。
   选 16k mono 而不是手动 resample 到 48k stereo 的理由：上游 LiveTalking
   ``server/webrtc.py`` 的 ``PlayerStreamTrack`` 实测就是这么做的，
   ``aiortc.codecs.opus`` 接受 16k 输入，少一层 resample CPU 也少出 bug。

2. 每帧 ``FRAME_SAMPLES = 320``（= 20ms @ 16k），与 ``MuseTalkRuntime``
   ``_push_audio_chunks_for_frame`` 的 chunk 长度严格一致：1 video frame
   (40ms) → 2 audio chunks。pacing 由本 track 的 ``_next_timestamp`` 维护，
   不依赖 video track 的时钟。

3. ``__init__`` 时调 ``runtime.set_audio_out_active(True)`` 通知 runtime
   开始 push 音频；``stop()`` 时复位为 ``False`` 防止队列积压旧声。

4. runtime 队列暂时为空（首字延迟 / 静音段）→ silence 兜底，保持 PTS
   单调。永远不抛 exception 让 aiortc PC fail。

本文件只在 aiortc 已安装时被 import（``avatar_session.get_audio_track``
导入失败会返回 ``None``），与 ``video_track.py`` 一致。
"""

from __future__ import annotations

import asyncio
import fractions
import time
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from aiortc import MediaStreamTrack  # type: ignore[import-not-found]
from av import AudioFrame  # type: ignore[import-not-found]
from common.logger import get_logger  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from .musetalk import MuseTalkRuntime

logger = get_logger(__name__)


class MuseTalkAudioTrack(MediaStreamTrack):
    """aiortc 音频轨：从 ``MuseTalkRuntime`` 拉 16k mono float32 PCM。"""

    kind = "audio"

    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 320  # 20ms @ 16k；与 runtime feature_buffer.chunk_samples 对齐
    PTIME = 0.020  # 秒
    TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

    def __init__(self, runtime: "MuseTalkRuntime", *, recv_timeout: float = 0.1) -> None:
        super().__init__()
        self._runtime = runtime
        self._recv_timeout = recv_timeout
        # PTS / pacing：起始时由首次 recv 抓 wall-clock，后续每帧自增 FRAME_SAMPLES
        self._start_wall_time: Optional[float] = None
        self._pts: int = 0
        self._frame_count: int = 0
        # 通知 runtime 开始 push 音频；stop() 时复位
        try:
            runtime.set_audio_out_active(True)
        except Exception as exc:  # noqa: BLE001
            # runtime 没实现 set_audio_out_active 的极端情况：log 但继续，
            # 最坏退化成"音频全静音"——比让整个 PC 起不来好。
            logger.warning(
                "MuseTalkRuntime.set_audio_out_active(True) failed: %s "
                "(audio track will produce silence)",
                exc,
            )

    async def _next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        """方案 A AV 同步：与 ``MuseTalkVideoTrack`` **共享** runtime 提供的
        wall-clock T0，否则两个 track 各自第一次 ``recv()`` 时各自抓
        ``time.time()`` 会导致 ``T0_a ≠ T0_v``，浏览器侧 RTCP SR 把 audio /
        video 的 NTP 基准错开 → lip-sync 完全偏掉。

        * 首帧 ``await runtime.wait_av_t0()`` 拿到 runtime 第一次 push 帧的
          wall-clock，作为 PTS=0 的锚（与 video track 用同一个 T0）
        * 后续每帧 PTS += FRAME_SAMPLES，并 sleep 到 ``T0 + N*PTIME``
        * runtime T0 等待超时（5s 还没产帧）时回退到 ``time.time()``，至少
          不卡死——但此时 lip-sync 必然崩，靠 sidecar 端日志能定位

        浏览器 jitter buffer 看到的 RTP 间隔保持稳定 20ms。
        """
        if self._start_wall_time is None:
            t0 = await asyncio.to_thread(self._runtime.wait_av_t0, 5.0)
            if t0 is None:
                # runtime 5s 没产帧：极端异常，但不能卡 aiortc PC。退回独立 T0；
                # 此时 lip-sync 必然偏，让用户看到日志去查为什么没产帧。
                t0 = time.time()
                logger.warning(
                    "MuseTalkAudioTrack: runtime.wait_av_t0 timed out after 5s; "
                    "falling back to local time.time() — AV sync may be broken",
                )
            self._start_wall_time = t0
            self._pts = 0
            self._frame_count = 0
            # 第一帧立刻发出去：不 sleep，避免再叠一层延迟到 T0 之外
            return self._pts, self.TIME_BASE
        self._pts += self.FRAME_SAMPLES
        self._frame_count += 1
        target = self._start_wall_time + self._frame_count * self.PTIME
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        return self._pts, self.TIME_BASE

    async def recv(self) -> "AudioFrame":
        pts, time_base = await self._next_timestamp()

        # next_audio_frame 是 queue.get(timeout) → 阻塞 IO 线程，必须用 to_thread
        pcm = await asyncio.to_thread(
            self._runtime.next_audio_frame, timeout=self._recv_timeout,
        )
        if pcm is None:
            # 队列空：可能是首字延迟、静音段、或 runtime 暂停。出 silence 帧
            # 维持 PTS 推进，浏览器侧不会断流；下一帧自然会拿到真实音频。
            pcm = np.zeros(self.FRAME_SAMPLES, dtype=np.float32)
        elif pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32, copy=False)

        # 长度兜底：runtime 已统一切到 FRAME_SAMPLES，但极端情况下做防御性
        # padding/截断，不允许给 aiortc 丢非标准长度的 AudioFrame。
        if pcm.size != self.FRAME_SAMPLES:
            if pcm.size < self.FRAME_SAMPLES:
                pcm = np.pad(pcm, (0, self.FRAME_SAMPLES - pcm.size))
            else:
                pcm = pcm[: self.FRAME_SAMPLES]

        # float32 [-1, 1] → int16 LE，对齐 av.AudioFrame format='s16'
        pcm_int16 = np.clip(pcm * 32767.0, -32768.0, 32767.0).astype(np.int16)

        frame = AudioFrame(format="s16", layout="mono", samples=pcm_int16.size)
        frame.planes[0].update(pcm_int16.tobytes())
        frame.sample_rate = self.SAMPLE_RATE
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self) -> None:
        # 父类 stop 把 readyState 标记为 'ended'，aiortc 后续不再调 recv
        try:
            super().stop()
        finally:
            try:
                self._runtime.set_audio_out_active(False)
            except Exception as exc:  # noqa: BLE001
                logger.debug("set_audio_out_active(False) on stop failed: %s", exc)


__all__ = ["MuseTalkAudioTrack"]
