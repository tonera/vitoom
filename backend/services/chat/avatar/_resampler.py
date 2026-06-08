"""16k mono pcm_s16le 流式 resample 器（跨 chunk 维护状态）。

为什么必须流式：单 chunk 独立 resample 会在 chunk 边界处出高频 artifact，
LiveTalking 模型对 mel 频谱敏感，会出垃圾口型。``Resampler`` 按
``(session_id, request_id)`` 维度持有：

* 残留样本 buffer（无法对齐 ratio 的尾部样本，留给下一 chunk）
* scipy 路径下，``signal.resample_poly`` 不天然带 stateful filter，所以
  fallback 用线性插值时维护"上一个样本"做 chunk 间无缝拼接

实现策略：
* 优先 ``scipy.signal.resample_poly``（多相滤波，高质量）
* fallback 用 numpy 线性插值（质量足够 wav2lip / musetalk 提取唇动特征）

任何 ``flush`` / ``interrupt`` / 切换 ``request_id`` 都要 ``reset()``。
"""

from __future__ import annotations

import logging
from math import gcd
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
INT16_MAX = 32767.0
INT16_MIN = -32768.0

try:
    from scipy.signal import resample_poly as _resample_poly  # type: ignore[import-not-found]

    _HAS_SCIPY = True
except ImportError:
    _resample_poly = None  # type: ignore[assignment]
    _HAS_SCIPY = False
    logger.info(
        "scipy not available; livetalking_client resampler will use numpy linear "
        "interpolation. Audio quality is still acceptable for lip-sync model "
        "feature extraction."
    )


def _to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """多声道 → mono；按声道交错布局解释 PCM。"""
    if channels <= 1:
        return samples
    if samples.size % channels != 0:
        # 末尾不齐的 sample 直接丢，不强行 reshape 引发异常
        usable = (samples.size // channels) * channels
        samples = samples[:usable]
    if samples.size == 0:
        return samples
    reshaped = samples.reshape(-1, channels)
    return reshaped.mean(axis=1)


class Resampler:
    """流式 resample：``(source_sr, source_channels) → 16k mono pcm_s16le``。

    每个 request_id 应该用独立实例，或在切换 request_id 时调 ``reset()``。
    """

    def __init__(self, *, target_sr: int = TARGET_SAMPLE_RATE) -> None:
        self.target_sr = target_sr
        self._tail_floats: np.ndarray = np.zeros(0, dtype=np.float32)  # 跨 chunk 残留
        # 线性插值 fallback 用：上一 chunk 的最后一个样本，新 chunk 起点做无缝拼接
        self._last_sample: float = 0.0
        self._linear_pos: float = 0.0  # 当前累计的"已消耗源样本"小数位置

    def reset(self) -> None:
        self._tail_floats = np.zeros(0, dtype=np.float32)
        self._last_sample = 0.0
        self._linear_pos = 0.0

    def process(
        self,
        pcm_bytes: bytes,
        *,
        source_sr: int,
        source_channels: int,
    ) -> bytes:
        """把任意采样率多声道 ``pcm_s16le`` 转成 ``16k mono pcm_s16le``。

        缺失采样率（``source_sr<=0``）→ 不做转换直接透传，由调用方决定如何 log。
        """
        if not pcm_bytes:
            return b""
        if source_sr <= 0:
            return pcm_bytes

        try:
            int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        except Exception:
            return pcm_bytes
        if int16.size == 0:
            return b""

        mono = _to_mono(int16, max(1, int(source_channels or 1)))
        if mono.size == 0:
            return b""

        floats = mono.astype(np.float32) / 32768.0

        # 已经是目标采样率：mono 转换后字节长度 = 原 mono 字节，直接打回 int16
        if source_sr == self.target_sr:
            return _to_int16_bytes(floats)

        if _HAS_SCIPY:
            return _to_int16_bytes(self._scipy_resample(floats, source_sr))
        return _to_int16_bytes(self._linear_resample(floats, source_sr))

    def _scipy_resample(self, floats: np.ndarray, source_sr: int) -> np.ndarray:
        # 把上次残留拼到本次开头，避免边界 click
        if self._tail_floats.size:
            floats = np.concatenate([self._tail_floats, floats])
            self._tail_floats = np.zeros(0, dtype=np.float32)

        g = gcd(source_sr, self.target_sr)
        up = self.target_sr // g
        down = source_sr // g
        # resample_poly 内部 anti-alias filter 比线性插值好得多
        resampled = _resample_poly(floats, up=up, down=down)
        return np.asarray(resampled, dtype=np.float32)

    def _linear_resample(self, floats: np.ndarray, source_sr: int) -> np.ndarray:
        """numpy 线性插值降级路径（无 scipy 时用）。

        实现：以源样本下标 i 为 x，目标样本下标 j 对应 ``j * source_sr / target_sr``
        的源位置 ``pos``，``np.interp`` 在 ``[i, i+1]`` 内做线性插值。``_last_sample`` /
        ``_linear_pos`` 跨 chunk 持有起点，避免段间相位跳变。
        """
        if floats.size == 0:
            return floats
        ratio = source_sr / self.target_sr  # 一个目标样本"消耗"多少源样本
        # 源序列拼上 _last_sample 作为索引 -1 的虚拟样本
        x_src = np.arange(-1, floats.size, dtype=np.float64)
        y_src = np.concatenate([[self._last_sample], floats]).astype(np.float64)
        # 目标位置序列
        target_count = max(0, int((floats.size - self._linear_pos) / ratio))
        if target_count == 0:
            self._linear_pos = max(0.0, self._linear_pos - floats.size)
            self._last_sample = float(floats[-1])
            return np.zeros(0, dtype=np.float32)
        positions = self._linear_pos + np.arange(target_count) * ratio
        # np.interp 用线性插值
        out = np.interp(positions, x_src, y_src)
        # 更新跨 chunk 状态
        consumed = positions[-1] + ratio
        self._linear_pos = consumed - floats.size
        self._last_sample = float(floats[-1])
        return out.astype(np.float32)


def _to_int16_bytes(floats: np.ndarray) -> bytes:
    if floats.size == 0:
        return b""
    clipped = np.clip(floats * 32767.0, INT16_MIN, INT16_MAX).astype(np.int16)
    return clipped.tobytes()


__all__ = ["Resampler", "TARGET_SAMPLE_RATE"]
