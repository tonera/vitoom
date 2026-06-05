"""流式 Whisper 特征 buffer（替代上游 ``WhisperASR`` + ``BaseASR``）。

上游 ``WhisperASR`` 跟 ``BaseAvatar`` 紧耦合（依赖 ``parent.custom_audiotype``
等字段）、且强行把 ASR 概念暴露给上层（实际只是 Whisper encoder 提特征，
不是真 ASR）。本实现是去 BaseAvatar 化的轻量版：

* 输入：外部 push 16k mono float32 PCM（任意长度），内部按 ``chunk_samples``
  (=320 / 20ms) 切片入队
* 输出：``step()`` 触发一次特征提取，产出 ``batch_size`` 个 384-channel
  whisper feature chunk（喂给 UNet）+ 同步的 ``2*batch_size`` 个 audio frame
  （供 inference 线程判断说话/静音段）
* 静音兜底：``step()`` 取 chunk 时如果输入队列空，自动用 zero PCM 补足并
  打 ``type=1`` 标记，让推理线程能跳过 silence 段直接走 full image 通道
* 滑动窗口：每次 ``step()`` 后保留尾部 ``stride_left + stride_right`` 个
  chunk 作为下次 ``audio2feat`` 的 left context，跟上游策略一致

对齐上游默认参数（25fps / batch=8 / l=r=10）：每次 ``step()`` 喂入 16 个
chunk（= 320ms 音频）→ 出 8 帧视频对应的 feature。
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


# 跟上游 BaseAvatar 一致：speak=0, silence=1, >1 为 custom（本内核不用）
FRAME_TYPE_SPEAK = 0
FRAME_TYPE_SILENCE = 1


@dataclass
class AudioFrame:
    data: NDArray[np.float32]   # shape=(chunk_samples,), float32
    type: int                   # FRAME_TYPE_SPEAK / FRAME_TYPE_SILENCE


class FeatureBuffer:
    """16k mono float32 PCM → Whisper feature 流式提取。"""

    def __init__(self, audio_processor, *, fps: int = 25, batch_size: int = 8,
                 stride_left: int = 10, stride_right: int = 10):
        self.audio_processor = audio_processor
        self.fps = fps
        self.batch_size = batch_size
        self.stride_left = stride_left
        self.stride_right = stride_right
        # 320 samples per chunk (16k / (25 * 2))
        self.chunk_samples = 16000 // (fps * 2)

        self._input_queue: "queue.Queue[NDArray[np.float32]]" = queue.Queue()
        self._output_queue: "queue.Queue[AudioFrame]" = queue.Queue()
        # 限 2：avoid runaway memory if downstream slows down
        self._feat_queue: "queue.Queue[List[NDArray[np.float32]]]" = queue.Queue(maxsize=2)
        self._frames: List[NDArray[np.float32]] = []

    # ---------- producer ----------
    def push_pcm(self, pcm_float32: NDArray[np.float32]) -> int:
        """喂 16k mono float32 PCM，按 ``chunk_samples`` 切片入队。

        末尾不足一帧的样本会被丢弃（上游 WhisperASR 同样行为）。返回入队
        chunk 数量，方便上层做 log 或 backpressure。
        """
        if pcm_float32.size == 0:
            return 0
        if pcm_float32.dtype != np.float32:
            pcm_float32 = pcm_float32.astype(np.float32, copy=False)
        n = pcm_float32.size // self.chunk_samples
        for i in range(n):
            chunk = pcm_float32[i * self.chunk_samples:(i + 1) * self.chunk_samples]
            self._input_queue.put(chunk)
        return n

    def flush(self) -> None:
        """中断时调用：清空输入 + 已算出但未消费的 feature 批次。

        保留 ``self._frames`` 窗口，下次说话时还能用上历史 left context；
        如果想"硬清"可以手动重置 _frames，但通常不需要。
        """
        _drain(self._input_queue)
        _drain(self._feat_queue)

    # ---------- inference-side consumer ----------
    def warm_up(self) -> None:
        """初始化：喂 ``stride_left + stride_right`` 个 silence chunk 让滑动
        窗口饱和。然后从 ``output_queue`` 提前消耗 ``stride_left`` 个，对齐
        第一个真实 chunk 在 output 端的相对位置（与上游 ``BaseASR.warm_up``
        逻辑一致）。
        """
        for _ in range(self.stride_left + self.stride_right):
            silence = np.zeros(self.chunk_samples, dtype=np.float32)
            self._frames.append(silence)
            self._output_queue.put(AudioFrame(data=silence, type=FRAME_TYPE_SILENCE))
        for _ in range(self.stride_left):
            self._output_queue.get()

    def step(self) -> bool:
        """触发一次 batch 特征提取。

        每次取 ``batch_size * 2`` 个 chunk（输入空就用 silence 补），
        累积到滑动窗口后调 ``audio2feat``，然后切成 ``batch_size`` 个 video-
        frame-aligned feature chunk 推入 ``feat_queue``。

        返回 ``True`` 表示本次产生了新的 feature batch；``False`` 表示窗口
        还没饱和（一般只在最初几次 step 出现）。
        """
        for _ in range(self.batch_size * 2):
            chunk, ftype = self._take_one()
            self._frames.append(chunk)
            self._output_queue.put(AudioFrame(data=chunk, type=ftype))

        if len(self._frames) <= self.stride_left + self.stride_right:
            return False

        inputs = np.concatenate(self._frames)
        whisper_feature = self.audio_processor.audio2feat(inputs)
        whisper_chunks = self._feature2chunks(whisper_feature)
        self._feat_queue.put(whisper_chunks)
        self._frames = self._frames[-(self.stride_left + self.stride_right):]
        return True

    def get_feat_batch(self, *, block: bool = True, timeout: float = 1.0) -> List[NDArray[np.float32]]:
        return self._feat_queue.get(block=block, timeout=timeout)

    def get_audio_frame(self) -> AudioFrame:
        return self._output_queue.get()

    # ---------- internals ----------
    def _take_one(self) -> Tuple[NDArray[np.float32], int]:
        """取一个 chunk；输入队列空就返回 silence。"""
        try:
            return self._input_queue.get(block=True, timeout=0.01), FRAME_TYPE_SPEAK
        except queue.Empty:
            return np.zeros(self.chunk_samples, dtype=np.float32), FRAME_TYPE_SILENCE

    def _feature2chunks(self, feature_array,
                        audio_feat_win=(0, 5), feature_idx_multiplier=2):
        start = self.stride_left / 2
        feature_chunks: List[NDArray[np.float32]] = []
        for i in range(self.batch_size):
            selected_feature, _ = self._get_sliced_feature(
                feature_array, vid_idx=i + start,
                audio_feat_win=audio_feat_win,
                feature_idx_multiplier=feature_idx_multiplier,
            )
            feature_chunks.append(selected_feature.reshape(-1, 384))
        return feature_chunks

    @staticmethod
    def _get_sliced_feature(feature_array, vid_idx, audio_feat_win, feature_idx_multiplier):
        length = feature_array.shape[0]
        center_idx = int(vid_idx * feature_idx_multiplier)
        left = int(center_idx - audio_feat_win[0] * feature_idx_multiplier)
        right = int(center_idx + audio_feat_win[1] * feature_idx_multiplier)
        selected = []
        selected_idx = []
        for idx in range(left, right):
            idx = max(0, min(length - 1, idx))
            selected.append(feature_array[idx])
            selected_idx.append(idx)
        return np.asarray(selected), selected_idx


def drain_queue(q: "queue.Queue") -> None:
    """非阻塞清空 queue（带容量上限的 Queue 用 q.queue.clear() 不安全，
    这里走 get_nowait 直到 empty）."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


# 内部历史名，保留导出以避免外部破坏（runtime 用过 _drain）。新代码用 drain_queue。
_drain = drain_queue


__all__ = [
    "AudioFrame",
    "FRAME_TYPE_SILENCE",
    "FRAME_TYPE_SPEAK",
    "FeatureBuffer",
    "drain_queue",
]
