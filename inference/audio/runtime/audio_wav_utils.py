"""与具体 TTS 引擎无关的音频 / 转写文本小工具（原 vibevoice_bridge 中通用部分）。"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import soundfile as sf


def _to_mono_float32(audio: Any) -> np.ndarray:
    """把 numpy/torch/list 统一成 1D float32 mono 波形。

    对多维输入**只抽第一声道 / 第一候选**，绝不 ``reshape(-1)`` 串联。与
    ``audio.engines.tts_engine.normalize_audio_array`` 保持一致语义（防止
    多候选 / stereo 被错误串联成奇怪的"多人叠声"）。
    """
    if hasattr(audio, "detach"):
        arr = audio.detach().cpu().float().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)
    arr = np.squeeze(arr).astype(np.float32, copy=False)

    if arr.ndim == 0:
        return np.zeros((0,), dtype=np.float32)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        picked = arr[0] if h <= w else arr[:, 0]
        return np.ascontiguousarray(picked.astype(np.float32, copy=False))

    while arr.ndim > 1:
        arr = arr[0]
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def audio_tensor_to_base64(audio: Any, sample_rate: int) -> str:
    """将 1D 浮点波形（numpy 或 torch.Tensor）编码为 WAV 的 base64 字符串。"""
    arr = _to_mono_float32(audio)
    buf = io.BytesIO()
    sf.write(buf, arr, int(sample_rate), format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def audio_tensor_to_pcm16_bytes(audio: Any) -> bytes:
    """将 1D 浮点波形编码为裸 ``Int16 LE PCM`` 字节（**不含** WAV header）。

    与已移除的 ``audio_tensor_to_pcm16_base64`` 算法一致，仅输出原始 bytes，
    供 WebSocket binary frame 透传。
    """
    arr = _to_mono_float32(audio)
    if arr.size == 0:
        return b""
    clipped = np.clip(arr, -1.0, 1.0)
    i16 = (clipped * 32767.0).astype("<i2", copy=False)
    return i16.tobytes()


def build_transcript_text(raw_text: str, segments: list[dict[str, Any]]) -> str:
    """拼装 ASR 落盘的 .txt 内容：正文 + 可选时间戳分段。"""
    lines: list[str] = []
    body = (raw_text or "").strip()
    if body:
        lines.append(body)
    if segments:
        if lines:
            lines.append("")
        lines.append("# Timestamped segments")
        for seg in segments:
            start = str(seg.get("start_time", "") or "").strip()
            end = str(seg.get("end_time", "") or "").strip()
            speaker = str(seg.get("speaker_id", "") or "").strip()
            content = str(seg.get("content", "") or "").strip()
            if not content:
                continue
            span = f"{start} - {end}".strip()
            if speaker:
                lines.append(f"[{span}] speaker={speaker}: {content}")
            else:
                lines.append(f"[{span}] {content}")
    return "\n".join(lines).strip()
