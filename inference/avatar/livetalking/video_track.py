"""``MuseTalkVideoTrack`` —— 把 ``MuseTalkRuntime`` 输出接到 aiortc。

aiortc ``VideoStreamTrack.recv()`` 是异步方法，要返回带正确时间戳的
``av.VideoFrame``。本实现：

1. ``recv()`` 内 ``await asyncio.to_thread(runtime.next_video_frame, timeout=...)``
   不阻塞事件循环
2. **重写** ``next_timestamp()``：PTIME 跟 ``runtime.fps`` 严格一致（aiortc
   自带的 ``VIDEO_PTIME=1/30`` 在 25fps 配置下会让 video_track 跑得比 runtime
   push 快 20%，触发频繁 last_frame 兜底 → AV 越来越脱离）。同时第一次
   ``recv()`` 时 ``await runtime.wait_av_t0()`` 拿到与 ``MuseTalkAudioTrack``
   **共享** 的 wall-clock T0，否则两个 track 各自抓 ``time.time()`` 的几十~
   几百毫秒抖动会让浏览器侧 RTCP SR 把 audio / video NTP 基准错开 → lip-sync
   失败（用户主观感受"对不上"）。
3. 没有新帧时（runtime 还没产出 / push 暂停）保持上一帧避免视频卡死。
   首帧前用 avatar 第 0 帧或纯黑兜底。
4. ``stop()`` 时不真正 stop runtime（runtime 生命周期归 ``AvatarSession``），
   只标记本 track 不再 recv。

这个文件只在 aiortc 已安装时被 import（见 ``avatar_session.get_video_track``），
否则 sidecar 走 ``/offer`` 503 的退化分支。
"""

from __future__ import annotations

import asyncio
import fractions
import time
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from aiortc import VideoStreamTrack  # type: ignore[import-not-found]
from av import VideoFrame  # type: ignore[import-not-found]
from common.logger import get_logger  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from .musetalk import MuseTalkRuntime

logger = get_logger(__name__)

# RFC 4566 / RFC 3551：H.264 RTP clock rate 固定 90 kHz，所有 video format
# 在 RTP 层都用这个时基；aiortc 也是这么配的常量。我们 PTS 在这个时基下推。
VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class MuseTalkVideoTrack(VideoStreamTrack):
    """aiortc 视频轨，从 ``MuseTalkRuntime`` 拉 BGR uint8 帧。"""

    kind = "video"

    def __init__(self, runtime: "MuseTalkRuntime", *, recv_timeout: float = 0.2):
        super().__init__()
        self._runtime = runtime
        self._recv_timeout = recv_timeout
        # PTIME 严格按 runtime fps 推算，避免 25fps runtime + 30fps default
        # PTIME 错配（每秒多拉 5 帧 → 5×last_frame 兜底 → PTS 推进比真实视频
        # 快 20% → AV 越来越脱节）
        self._ptime: float = 1.0 / float(runtime.fps if runtime.fps > 0 else 25)
        self._pts_step: int = int(self._ptime * VIDEO_CLOCK_RATE)
        self._start_wall_time: Optional[float] = None
        self._pts: int = 0
        self._frame_count: int = 0
        # 起始兜底帧：avatar 视频第 0 帧（如果 runtime 还没出新帧就先发它，
        # avoid 浏览器端 video 元素一直黑屏）
        self._last_frame_bgr: Optional[np.ndarray] = None
        if runtime.frame_list_cycle:
            self._last_frame_bgr = runtime.frame_list_cycle[0].copy()

    # ---------- 重写 aiortc 自带的 wall-clock pacing ----------
    async def next_timestamp(  # type: ignore[override]
        self,
    ) -> Tuple[int, fractions.Fraction]:
        """与 ``MuseTalkAudioTrack._next_timestamp`` 完全对称：

        * 首帧 ``await runtime.wait_av_t0()`` 拿共享 T0（PTS=0 的锚）
        * 后续每帧 PTS += ``_pts_step``，并 sleep 到 ``T0 + N*PTIME``
        * 超时回退到本地 ``time.time()``（lip-sync 必崩，但不卡死）

        与 audio_track 保持同一 ``time.time()`` 时间基 + 同一 T0，浏览器
        侧 RTCP SR 算出 NTP 时戳一致 → AV 同步。
        """
        if self._start_wall_time is None:
            t0 = await asyncio.to_thread(self._runtime.wait_av_t0, 5.0)
            if t0 is None:
                t0 = time.time()
                logger.warning(
                    "MuseTalkVideoTrack: runtime.wait_av_t0 timed out after 5s; "
                    "falling back to local time.time() — AV sync may be broken",
                )
            self._start_wall_time = t0
            self._pts = 0
            self._frame_count = 0
            return self._pts, VIDEO_TIME_BASE
        self._pts += self._pts_step
        self._frame_count += 1
        target = self._start_wall_time + self._frame_count * self._ptime
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        return self._pts, VIDEO_TIME_BASE

    async def recv(self) -> "VideoFrame":
        pts, time_base = await self.next_timestamp()

        # next_video_frame 内部用 queue.Queue.get(timeout=...)，会阻塞 IO 线程；
        # 用 to_thread 避免堵 event loop
        bgr = await asyncio.to_thread(self._runtime.next_video_frame, timeout=self._recv_timeout)
        if bgr is None:
            # runtime 暂时没新帧（可能在静音段或首帧前）→ 用上一帧兜底
            if self._last_frame_bgr is None:
                # 极端 fallback：纯黑帧（理论上不会走到，因为 runtime 构造时
                # 就 copy 了 avatar 第 0 帧）
                bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                bgr = self._last_frame_bgr
        else:
            self._last_frame_bgr = bgr

        # av.VideoFrame.from_ndarray 期望 BGR24 ndarray (H, W, 3)
        frame = VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


__all__ = ["MuseTalkVideoTrack"]
