"""ChatSession 数据模型 + 状态枚举 + 输入许可集合。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.services.chat.artifacts import chat_file_identity, normalize_chat_file


class SessionState:
    OPENING = "opening"
    READY = "ready"
    TURN_BUFFERING = "turn_buffering"
    REASONING = "reasoning"
    TOOL_RUNNING = "tool_running"
    STREAMING_OUTPUT = "streaming_output"
    WAITING_TASK = "waiting_task"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CLOSED = "closed"


class InputMode:
    TEXT = "text"
    TEXT_STREAM = "text_stream"
    AUDIO_ONCE = "audio_once"
    AUDIO_STREAM = "audio_stream"
    MIXED = "mixed"


USER_INPUT_ALLOWED = {SessionState.READY, SessionState.TURN_BUFFERING}

# audio_chunk 在助手输出期间也可进入后端 VAD 监听，用于 barge-in。
AUDIO_CHUNK_ALLOWED = {
    SessionState.READY,
    SessionState.TURN_BUFFERING,
    SessionState.REASONING,
    SessionState.TOOL_RUNNING,
    SessionState.STREAMING_OUTPUT,
    SessionState.WAITING_TASK,
}

BARGE_IN_LISTENING_STATES = {
    SessionState.REASONING,
    SessionState.TOOL_RUNNING,
    SessionState.STREAMING_OUTPUT,
    SessionState.WAITING_TASK,
}

INTERRUPT_ALLOWED = {
    SessionState.TURN_BUFFERING,
    SessionState.REASONING,
    SessionState.TOOL_RUNNING,
    SessionState.STREAMING_OUTPUT,
    SessionState.WAITING_TASK,
}


@dataclass
class TurnAssembler:
    """收集单轮 user 输入碎片（文本拼接或音频 PCM 分片）。"""

    turn_id: str
    input_mode: str = InputMode.TEXT
    text_fragments: List[str] = field(default_factory=list)
    audio_chunks: List[bytes] = field(default_factory=list)

    def append_text(self, chunk: str) -> None:
        if chunk:
            self.text_fragments.append(str(chunk))

    def append_audio(self, pcm_bytes: bytes) -> None:
        if pcm_bytes:
            self.audio_chunks.append(bytes(pcm_bytes))

    def text_so_far(self) -> str:
        return "".join(self.text_fragments).strip()

    def audio_count(self) -> int:
        return len(self.audio_chunks)


@dataclass
class Turn:
    """一次完整的用户轮次（user 输入 → Run → assistant 输出）。"""

    turn_id: str
    input_mode: str = InputMode.TEXT
    user_text: Optional[str] = None
    run_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    assistant_fragments: List[str] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    derived_task_ids: List[str] = field(default_factory=list)
    interrupted: bool = False
    barge_in: bool = False

    @property
    def is_audio(self) -> bool:
        return self.input_mode.startswith("audio")

    def append_assistant_delta(self, delta: str) -> None:
        if delta:
            self.assistant_fragments.append(str(delta))

    def assistant_text(self) -> str:
        return "".join(self.assistant_fragments)

    def add_file(self, file_info: Dict[str, Any]) -> None:
        normalized = normalize_chat_file(file_info)
        if not normalized:
            return
        new_key = chat_file_identity(normalized)
        for idx, existing in enumerate(self.files):
            if chat_file_identity(existing) == new_key:
                self.files[idx] = normalized
                return
        self.files.append(normalized)

    def files_snapshot(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.files]

    def bind_task_id(self, task_id: str) -> None:
        normalized = str(task_id or "").strip()
        if normalized and normalized not in self.derived_task_ids:
            self.derived_task_ids.append(normalized)


__all__ = [
    "AUDIO_CHUNK_ALLOWED",
    "BARGE_IN_LISTENING_STATES",
    "INTERRUPT_ALLOWED",
    "InputMode",
    "SessionState",
    "Turn",
    "TurnAssembler",
    "USER_INPUT_ALLOWED",
]
