"""Ref2VA 参考素材：从 tpl_list / ref_video / ref_audio 下载并解码成 diffusers 的参考对象。

用 imageio-ffmpeg 解码，不依赖 PyAV 和 torchaudio：音轨由 ffmpeg 直接输出 32 kHz 双声道，diffusers 不必再重采样。
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3.references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)

from common.io_utils import download_url_to_tempfile
from video.handlers.minimax_h3.request import H3_AUDIO_RATE, H3ReferenceSources
from video.runtime.io_utils import load_image_from_url_or_path

# 生成时长最长 15.08 秒，diffusers 会把参考截到生成时长；多解码的部分只是为了不截短。
REFERENCE_DECODE_SECONDS = 16.0
_MAX_MEDIA_BYTES = 500 * 1024 * 1024


async def load_references(sources: H3ReferenceSources) -> list:
    """按打包顺序加载：图片（按 tpl_list）、视频、音频。"""
    references: list = []
    for url in sources.images:
        image = await load_image_from_url_or_path(url, size=None, max_long_edge=None)
        references.append(MiniMaxH3ImageReference(image=image))
    if sources.video:
        references.append(await _with_local_file(sources.video, ".mp4", decode_video_reference))
    if sources.audio:
        references.append(await _with_local_file(sources.audio, ".wav", decode_audio_reference))
    return references


async def _with_local_file(url_or_path: str, suffix: str, decode):
    path = await download_url_to_tempfile(url_or_path, default_suffix=suffix, timeout_seconds=120, max_bytes=_MAX_MEDIA_BYTES)
    try:
        return await asyncio.to_thread(decode, path)
    finally:
        if str(path) != url_or_path:
            Path(path).unlink(missing_ok=True)


def decode_video_reference(path: str | Path, max_seconds: float = REFERENCE_DECODE_SECONDS) -> MiniMaxH3VideoReference:
    """画面按容器帧率原样解码（ffmpeg 会按旋转元数据摆正），带音轨时一并取出。"""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path), output_params=["-t", f"{max_seconds:g}"])
    meta = next(reader)
    width, height = meta["size"]
    frames = [np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3) for frame in reader]
    if not frames:
        raise ValueError(f"参考视频里没有可解码的画面: {path}")
    fps = float(meta.get("fps") or 0.0)
    if fps <= 0:
        raise ValueError(f"无法读出参考视频的帧率: {path}")
    audio = _decode_audio(path, max_seconds) if _has_audio_stream(path) else None
    return MiniMaxH3VideoReference(
        frames=np.stack(frames),
        fps=fps,
        audio=audio,
        sample_rate=H3_AUDIO_RATE if audio is not None else None,
    )


def decode_audio_reference(path: str | Path, max_seconds: float = REFERENCE_DECODE_SECONDS) -> MiniMaxH3AudioReference:
    if not _has_audio_stream(path):
        raise ValueError(f"参考音频里没有音轨: {path}")
    return MiniMaxH3AudioReference(audio=_decode_audio(path, max_seconds), sample_rate=H3_AUDIO_RATE)


def _decode_audio(path: str | Path, max_seconds: float) -> torch.Tensor:
    """(2, samples) float32。单声道由 ffmpeg 复制成双声道，与 diffusers 的 expand 一致。"""
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v", "error",
        "-i", str(path),
        "-t", f"{max_seconds:g}",
        "-vn",
        "-ac", "2",
        "-ar", str(H3_AUDIO_RATE),
        "-f", "f32le",
        "-",
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"解码音轨失败: {result.stderr.decode(errors='replace').strip()}")
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size < 2:
        raise ValueError(f"音轨为空: {path}")
    return torch.from_numpy(samples.reshape(-1, 2).T.copy())


def _has_audio_stream(path: str | Path) -> bool:
    import imageio_ffmpeg

    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return re.search(r"Stream #\S+.*: Audio:", probe.stderr or "") is not None
