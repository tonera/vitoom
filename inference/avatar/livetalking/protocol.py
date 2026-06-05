"""``WS /avatar_stream`` 协议消息 schema + 入口校验。

入口契约（绝对硬约束）：所有进 sidecar 的 PCM 必须是 ``16k mono pcm_s16le``，
转换工作由 vitoom 后端 ``livetalking_client`` 完成；sidecar 这一层只做严格
校验，违规一律 ``400`` 关闭连接，不做兜底转换，避免协议博弈。

消息类型（JSON meta + 可选 binary frame，跟 vitoom chat WS 协议一致）：

* ``open`` —— 必须是首条消息，绑定到 LiveTalking session
* ``audio_chunk`` —— JSON meta 后紧跟一帧 binary PCM（int16 LE）
* ``audio_flush`` —— 标记本 ``request_id`` 段结束
* ``interrupt`` —— 调 ``avatar_session.flush_talk()`` 清队列
* ``close`` —— 主动关闭

binary 字节数检查：``pcm_s16le`` 每样本 2 字节，必须为偶数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_FORMAT = "pcm_s16le"
EXPECTED_CHANNELS = 1
SAMPLE_BYTES = 2  # int16 LE


class ProtocolError(Exception):
    """sidecar 客户端协议违约。处理方应回 ``400`` 并关闭连接。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class OpenMessage:
    session_id: str
    request_id: str
    sample_rate: int
    format: str
    channels: int


@dataclass(frozen=True)
class AudioChunkMessage:
    request_id: str
    seq: Optional[int]


@dataclass(frozen=True)
class AudioFlushMessage:
    request_id: str


@dataclass(frozen=True)
class InterruptMessage:
    session_id: str


def parse_open(payload: Dict[str, Any]) -> OpenMessage:
    """严格校验 open 消息；任意字段不符 → ``ProtocolError``。"""
    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        raise ProtocolError("missing_session_id", "open.session_id is required")
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise ProtocolError("missing_request_id", "open.request_id is required")
    try:
        sample_rate = int(payload.get("sample_rate") or 0)
    except (TypeError, ValueError):
        sample_rate = 0
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ProtocolError(
            "unsupported_sample_rate",
            f"sample_rate must be {EXPECTED_SAMPLE_RATE}, got {sample_rate}",
        )
    fmt = str(payload.get("format") or "").strip().lower()
    if fmt != EXPECTED_FORMAT:
        raise ProtocolError(
            "unsupported_format",
            f"format must be '{EXPECTED_FORMAT}', got '{fmt}'",
        )
    try:
        channels = int(payload.get("channels") or EXPECTED_CHANNELS)
    except (TypeError, ValueError):
        channels = -1
    if channels != EXPECTED_CHANNELS:
        raise ProtocolError(
            "unsupported_channels",
            f"channels must be {EXPECTED_CHANNELS}, got {channels}",
        )
    return OpenMessage(
        session_id=session_id,
        request_id=request_id,
        sample_rate=sample_rate,
        format=fmt,
        channels=channels,
    )


def parse_audio_chunk(payload: Dict[str, Any]) -> AudioChunkMessage:
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise ProtocolError("missing_request_id", "audio_chunk.request_id is required")
    seq_raw = payload.get("seq")
    try:
        seq = int(seq_raw) if seq_raw is not None else None
    except (TypeError, ValueError):
        seq = None
    return AudioChunkMessage(request_id=request_id, seq=seq)


def parse_audio_flush(payload: Dict[str, Any]) -> AudioFlushMessage:
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise ProtocolError("missing_request_id", "audio_flush.request_id is required")
    return AudioFlushMessage(request_id=request_id)


def parse_interrupt(payload: Dict[str, Any]) -> InterruptMessage:
    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        raise ProtocolError("missing_session_id", "interrupt.session_id is required")
    return InterruptMessage(session_id=session_id)


def validate_pcm_bytes(data: bytes) -> None:
    """字节数必须为 ``pcm_s16le`` 每样本 2 字节的偶数。"""
    if not isinstance(data, (bytes, bytearray)):
        raise ProtocolError("invalid_binary", "audio_chunk binary must be bytes")
    if len(data) == 0:
        raise ProtocolError("empty_binary", "audio_chunk binary is empty")
    if len(data) % SAMPLE_BYTES != 0:
        raise ProtocolError(
            "misaligned_binary",
            f"audio_chunk binary length {len(data)} is not a multiple of {SAMPLE_BYTES}",
        )


__all__ = [
    "AudioChunkMessage",
    "AudioFlushMessage",
    "EXPECTED_CHANNELS",
    "EXPECTED_FORMAT",
    "EXPECTED_SAMPLE_RATE",
    "InterruptMessage",
    "OpenMessage",
    "ProtocolError",
    "SAMPLE_BYTES",
    "parse_audio_chunk",
    "parse_audio_flush",
    "parse_interrupt",
    "parse_open",
    "validate_pcm_bytes",
]
