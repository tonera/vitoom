"""``AvatarSession`` —— sidecar 协议层与 ``MuseTalkRuntime`` 的桥接。

职责非常窄：

* 接 ``WS /avatar_stream`` 推来的 ``int16 LE`` PCM bytes，转 ``float32 / 32768``
  喂给 ``MuseTalkRuntime.push_pcm()``
* 把 ``flush_request`` / ``interrupt`` 转成 ``runtime.flush()`` 或 ``flush_av()``
* 提供 ``get_video_track()`` / ``get_audio_track()`` 给 sidecar ``server.py``
  的 ``/offer`` 用，WebRTC 视频/音频帧从 ``runtime.next_*_frame()`` 取

设计决策：

* **不再有 stub fallback**：方案 C 落地后，sidecar 必须 vendor 完整的 MuseTalk
  推理内核（模型/avatar 资源缺失时直接抛 ``MuseTalkRuntimeError``）。开发期
  跑协议联调可以用 ``model="dummy"`` 走 ``DummyRuntime``，但默认禁用。
* **同步初始化**：``MuseTalkRuntime`` 构造里就加载模型 + avatar，几秒内不会
  返回；sidecar 第一次 ``open`` 消息触发懒加载，避免 sidecar 启动时就吃完
  GPU 资源。
* **线程安全**：``MuseTalkRuntime`` 内部的 push / flush / next_*_frame 都
  是线程安全的（基于 ``queue.Queue`` + ``threading.Event``），上层不需要锁。

音视频对齐模式（方案 A）：

由环境变量 ``VITOOM_LIVETALKING_AV_SYNC_MODE`` 控制：

* ``decorative``（默认）：兼容历史 D 方案。``get_audio_track()`` 返回
  ``None`` → ``/offer`` 不 addTrack(audio)，前端拿不到 remote audio，按
  自己本地 ``useAudioPlayback`` 出声；``interrupt`` 走 ``runtime.flush()``
  保留视频队列残帧。
* ``aligned``：方案 A 启用。``get_audio_track()`` 返回 ``MuseTalkAudioTrack``，
  ``/offer`` 把 audio track 也 addTrack 到同一 PeerConnection，浏览器自动
  lip-sync；``interrupt`` 走 ``runtime.flush_av()`` 同步清掉双队列。

同一 sidecar 进程的所有 session 共用同一模式（部署级 flag），运行时不切换。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from common.logger import get_logger  # type: ignore[import-not-found]

logger = get_logger(__name__)

# 与 protocol.py 一致：16k mono pcm_s16le，每 sample 2 字节
INT16_NORM = 32768.0

# 方案 A：对齐模式 flag。环境变量取值：``decorative``（默认）/ ``aligned``。
# 部署级配置：sidecar 启动时一次性读取，运行中不变。
_AV_SYNC_ENV_VAR = "VITOOM_LIVETALKING_AV_SYNC_MODE"
_AV_SYNC_MODE_ALIGNED = "aligned"


def av_sync_aligned_enabled() -> bool:
    """读取环境变量判断当前 sidecar 是否启用方案 A 对齐模式。

    支持 ``aligned`` / ``true`` / ``1``（大小写不敏感）；其余一律视为
    ``decorative``（=D 方案）。失败安全：env 缺失 / 非法值 = 默认关闭。
    """
    raw = os.environ.get(_AV_SYNC_ENV_VAR, "").strip().lower()
    return raw in {_AV_SYNC_MODE_ALIGNED, "true", "1"}


class AvatarSession:
    """绑定到一个 chat session 的 avatar runtime + WebRTC track 管理器。

    单 sidecar 进程允许多个 ``AvatarSession`` 共存（不同 session_id），但每个
    session 会独占一份 ``MuseTalkRuntime``（即各自一份 avatar 缓存帧 + 模型
    权重 + 推理线程组）。模型权重共享需要重构 ``MuseTalkRuntime`` 把 model 拆
    出来，TODO 后续优化。
    """

    def __init__(self, *, model: str, avatar_id: str, fps: int = 25,
                 batch_size: int = 8) -> None:
        self.model = model
        self.avatar_id = avatar_id
        self.fps = fps
        self.batch_size = batch_size

        # 延迟 import：runtime 拉 torch / diffusers，dev 环境不装也能 import 本模块
        from .musetalk.runtime import MuseTalkRuntime, MuseTalkRuntimeError

        if model != "musetalk":
            raise MuseTalkRuntimeError(
                f"AvatarSession only supports model='musetalk' for now (got '{model}')"
            )

        self._runtime = MuseTalkRuntime(
            avatar_id=avatar_id, fps=fps, batch_size=batch_size,
        )
        self._runtime.start()
        self._request_samples: dict[str, int] = {}
        self._video_track = None  # 懒创建：第一次 get_video_track() 时构造
        self._audio_track = None  # 懒创建 + 仅在 aligned 模式下；与 video 对称

    # ---------- PCM 入口 ----------
    def put_pcm16(self, request_id: str, pcm_bytes: bytes) -> None:
        """把 16k mono pcm_s16le 字节流喂给 runtime。

        转换路径：``bytes`` → ``np.int16`` → ``float32 / 32768``。
        本方法是 sidecar WS handler 的同步调用，不能阻塞；``runtime.push_pcm``
        内部走线程安全 Queue.put，不会阻塞。
        """
        if not pcm_bytes:
            return
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "failed to parse pcm16 (request_id=%s, bytes=%d): %s",
                request_id, len(pcm_bytes), exc,
            )
            return
        if samples.size == 0:
            return
        floats = samples.astype(np.float32) / INT16_NORM
        n = self._runtime.push_pcm(floats)
        prev = self._request_samples.get(request_id, 0)
        self._request_samples[request_id] = prev + n

    # ---------- 段结束 / 中断 ----------
    def flush_request(self, request_id: str) -> None:
        """段结束 hook：当前 runtime 不区分 request 边界，仅记录日志。"""
        total = self._request_samples.pop(request_id, 0)
        if total:
            logger.debug(
                "avatar flush request_id=%s total_chunks=%d", request_id, total,
            )

    def interrupt(self) -> None:
        """中断：清空 runtime 内部待处理音频和未消费 feature 批次。

        aligned（方案 A）模式下额外 drain 音视频输出队列，否则前端会出现
        "声音断了 200~500ms 后嘴还在动"的撕裂；decorative（D 方案）保留
        视频残帧让前端把已渲染的最后几帧自然播完，体感更柔和。
        """
        try:
            if av_sync_aligned_enabled() and self._runtime.audio_out_active:
                self._runtime.flush_av()
            else:
                self._runtime.flush()
        except Exception as exc:  # noqa: BLE001
            logger.error("MuseTalkRuntime.flush(_av) failed: %s", exc, exc_info=True)
            raise
        self._request_samples.clear()

    def shutdown(self) -> None:
        """sidecar 关闭 / session 关闭时调用：停推理线程。"""
        try:
            self._runtime.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("MuseTalkRuntime.stop failed: %s", exc, exc_info=True)

    # ---------- WebRTC 视频轨 ----------
    def get_video_track(self):
        """惰性构造一个 aiortc ``VideoStreamTrack``，从 runtime 拉帧。

        aiortc 未安装时返回 ``None``，由 ``server.py`` 决定如何回退。
        """
        if self._video_track is not None:
            return self._video_track
        try:
            from .video_track import MuseTalkVideoTrack  # 循环 import 防御
        except ImportError as exc:
            logger.warning("video_track import failed (aiortc missing?): %s", exc)
            return None
        self._video_track = MuseTalkVideoTrack(self._runtime)
        return self._video_track

    def get_audio_track(self):
        """方案 A 对齐模式下惰性构造一个 aiortc ``AudioStreamTrack``，从
        ``runtime.next_audio_frame()`` 拉 16k mono float32 PCM。

        模式判定：

        * decorative（默认）：返回 ``None``，``server.py`` 不 addTrack(audio)，
          前端拿不到 remote audio，按本地 ``useAudioPlayback`` 出声（D 方案）
        * aligned：返回 ``MuseTalkAudioTrack``，``server.py`` 把 audio 加到
          同一 PeerConnection，浏览器自动 lip-sync

        aiortc / av 未安装时返回 ``None``，由 ``server.py`` 决定如何回退。
        """
        if not av_sync_aligned_enabled():
            return None
        if self._audio_track is not None:
            return self._audio_track
        try:
            from .audio_track import MuseTalkAudioTrack  # 循环 import 防御
        except ImportError as exc:
            logger.warning(
                "audio_track import failed (aiortc/av missing? aligned mode disabled): %s",
                exc,
            )
            return None
        self._audio_track = MuseTalkAudioTrack(self._runtime)
        logger.info(
            "AvatarSession aligned-mode audio track created avatar_id=%s",
            self.avatar_id,
        )
        return self._audio_track


__all__ = ["AvatarSession", "INT16_NORM", "av_sync_aligned_enabled"]
