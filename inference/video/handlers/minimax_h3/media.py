"""画面用 imageio 写盘，音轨用 imageio 自带的 ffmpeg 合流。合流失败必须让任务失败。"""

from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch

from common.Constant import VIDEO_SAVE_QUALITY


def save_mp4(frames: np.ndarray, audio: torch.Tensor, sample_rate: int, path: str | Path, fps: int = 24) -> None:
    """frames: uint8 [T,H,W,3]；audio: float [2, S]。"""
    import imageio
    import imageio_ffmpeg

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    video_path = target.with_suffix(".video.mp4")
    wav_path = target.with_suffix(".wav")
    stereo = _cover_video(_stereo(audio), int(frames.shape[0]), fps, sample_rate)
    try:
        _write_wav(stereo, sample_rate, wav_path)
        writer = imageio.get_writer(str(video_path), fps=fps, quality=VIDEO_SAVE_QUALITY)
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(wav_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"H3 音视频合流失败: {detail or 'ffmpeg 返回 ' + str(result.returncode)}")
    finally:
        video_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)


def verify_mp4(
    path: str | Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: int = 24,
    sample_rate: int = 32000,
    audio_slack_seconds: float = 0.025,
) -> None:
    """校验分辨率、帧数、帧率，以及非静音的双声道 32 kHz 音轨。"""
    import imageio
    import imageio_ffmpeg

    target = Path(path)
    errors: list[str] = []
    reader = imageio.get_reader(str(target))
    try:
        meta = reader.get_meta_data()
        size = meta.get("size") or (0, 0)
        got_w, got_h = int(size[0]), int(size[1])
        got_fps = float(meta.get("fps") or 0.0)
        got_frames = int(reader.count_frames())
    finally:
        reader.close()
    if (got_w, got_h) != (int(width), int(height)):
        errors.append(f"分辨率 {got_w}x{got_h}，期望 {width}x{height}")
    if got_frames != int(num_frames):
        errors.append(f"帧数 {got_frames}，期望 {num_frames}")
    if abs(got_fps - float(fps)) > 0.1:
        errors.append(f"帧率 {got_fps}，期望 {fps}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(target)], capture_output=True, text=True)
    info = probe.stderr or ""
    audio_match = re.search(r"Audio:.*?(\d+)\s+Hz,\s+([A-Za-z0-9]+)", info)
    if audio_match is None:
        errors.append("没有音轨")
    else:
        rate = int(audio_match.group(1))
        layout = audio_match.group(2).lower()
        if rate != int(sample_rate):
            errors.append(f"音轨采样率 {rate}，期望 {sample_rate}")
        if layout not in {"stereo", "2"}:
            errors.append(f"音轨不是双声道: {layout}")

    extracted = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(target), "-vn", "-ac", "2", "-ar", str(sample_rate), "-f", "f32le", "-"],
        capture_output=True,
    )
    if extracted.returncode != 0 or not extracted.stdout:
        errors.append("无法解码音轨")
    else:
        samples = np.frombuffer(extracted.stdout, dtype=np.float32)
        if samples.size < 2 or samples.size % 2 != 0:
            errors.append(f"音轨样本数异常: {samples.size}")
        else:
            stereo = samples.reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.square(stereo))))
            if rms < 1e-4:
                errors.append(f"音轨近乎静音 rms={rms:.3e}")
            audio_seconds = stereo.shape[0] / float(sample_rate)
            video_seconds = int(num_frames) / float(fps)
            if abs(audio_seconds - video_seconds) > float(audio_slack_seconds):
                errors.append(
                    f"音视频时长相差 {abs(audio_seconds - video_seconds) * 1000:.1f} ms，"
                    f"超过一个音频块（{audio_slack_seconds * 1000:.0f} ms）"
                )
    if errors:
        raise RuntimeError("H3 产物校验失败: " + "；".join(errors))


def _cover_video(stereo: torch.Tensor, num_frames: int, fps: int, sample_rate: int) -> torch.Tensor:
    """把较短的音轨补到画面时长。否则 AAC 延迟会让 -shortest 裁掉末尾画面。"""
    needed = int(np.ceil(num_frames / float(fps) * sample_rate))
    short = needed - int(stereo.shape[-1])
    if short <= 0:
        return stereo
    return torch.nn.functional.pad(stereo, (0, short))


def _stereo(audio: torch.Tensor) -> torch.Tensor:
    waveform = audio.detach()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim != 2:
        raise RuntimeError(f"音频维度应为 2，收到 {tuple(audio.shape)}")
    if waveform.shape[0] != 2 and waveform.shape[-1] == 2:
        waveform = waveform.transpose(0, 1)
    if waveform.shape[0] != 2:
        raise RuntimeError(f"音频应为双声道，收到 {tuple(waveform.shape)}")
    return waveform.float().cpu().contiguous()


def _write_wav(stereo: torch.Tensor, sample_rate: int, path: Path) -> None:
    pcm = (stereo.clamp(-1, 1).transpose(0, 1).numpy() * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
