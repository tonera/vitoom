from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any, Dict, Optional

from backend.services.chat.session import InputMode, SessionRuntime, Turn


_STRONG_SENT_END = re.compile(r"[。！？!?\n]")
_WEAK_SENT_END = re.compile(r"[，,；;：:]")

_MD_CODE_FENCE_LINE = re.compile(r"^\s*(?:`{3,}|~{3,})[\w+-]*\s*$")
_MD_HR = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_MD_HEADING_PREFIX = re.compile(r"^\s*#{1,6}\s+")
_MD_BLOCKQUOTE_PREFIX = re.compile(r"^\s*>\s?")
_MD_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_AUTOLINK = re.compile(r"<\s*(?:https?://|mailto:)[^>\s]+\s*>")
_MD_INLINE_CODE = re.compile(r"`+([^`]+)`+")
_MD_EMPHASIS = re.compile(
    r"(\*\*\*|___|~~|\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1",
    re.DOTALL,
)
_MD_TABLE_PIPE = re.compile(r"\s*\|\s*")
_MD_RESIDUAL_DECOR = re.compile(r"[`*_~#]+")
_BARE_URL = re.compile(
    r"(?i)(?:https?://[^\s\]`'\"<>）\]]+|www\.[^\s\]`'\"<>）\]]+|mailto:[^\s\]`'\"<>）\]]+)"
)
_URL_TRAILING_PUNCT = frozenset(".,;:!?、，。；：）)]}」』'\"")
_EMOJI_KEYCAP = re.compile(r"[0-9#*]\ufe0f?\u20e3")
_EMOJI_CHARS = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"  # regional indicator flags
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\ufe0e\ufe0f"
    "\U0001f3fb-\U0001f3ff"
    "\u200d"
    "]+"
)


def _strip_bare_urls(text: str) -> str:
    if not text:
        return ""

    def _trim_trailing_punct(u: str) -> str:
        while u and u[-1] in _URL_TRAILING_PUNCT:
            u = u[:-1]
        return u

    out: list[str] = []
    pos = 0
    for m in _BARE_URL.finditer(text):
        out.append(text[pos : m.start()])
        trimmed = _trim_trailing_punct(m.group(0))
        out.append(text[m.start() + len(trimmed) : m.end()])
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def _strip_tts_emoji(text: str) -> str:
    if not text:
        return ""
    return _EMOJI_CHARS.sub("", _EMOJI_KEYCAP.sub("", text))


def strip_markdown_for_tts(text: str, *, state: Dict[str, Any]) -> str:
    if not text:
        return ""
    out_lines: list[str] = []
    in_code = bool(state.get("in_code_block"))
    for raw in text.split("\n"):
        if _MD_CODE_FENCE_LINE.match(raw):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = raw
        if _MD_HR.match(line):
            continue
        line = _MD_HEADING_PREFIX.sub("", line)
        line = _MD_BLOCKQUOTE_PREFIX.sub("", line)
        line = _MD_LIST_PREFIX.sub("", line)
        out_lines.append(line)
    state["in_code_block"] = in_code

    s = "\n".join(out_lines)
    s = _MD_IMAGE.sub(lambda m: (m.group(1) or "").strip(), s)
    s = _MD_LINK.sub(r"\1", s)
    s = _MD_AUTOLINK.sub("", s)
    prev: Optional[str] = None
    while prev != s:
        prev = s
        s = _MD_EMPHASIS.sub(r"\2", s)
    s = _MD_INLINE_CODE.sub("", s)
    s = _MD_TABLE_PIPE.sub(" ", s)
    s = _strip_bare_urls(s)
    s = _MD_RESIDUAL_DECOR.sub("", s)
    s = _strip_tts_emoji(s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _extract_sentences_for_tts(
    buffer: str,
    *,
    weak_min: int = 15,
    flush_all: bool = False,
) -> tuple[list[str], str]:
    sentences: list[str] = []
    while True:
        m_strong = _STRONG_SENT_END.search(buffer)
        if m_strong:
            end = m_strong.end()
        else:
            m_weak = _WEAK_SENT_END.search(buffer)
            if m_weak and m_weak.end() >= weak_min:
                end = m_weak.end()
            else:
                break
        part = buffer[:end].strip()
        if part:
            sentences.append(part)
        buffer = buffer[end:]
    if flush_all:
        tail = buffer.strip()
        if tail:
            sentences.append(tail)
            buffer = ""
    return sentences, buffer


def resolve_voice_reply_cfg(runtime: SessionRuntime) -> Dict[str, Any]:
    metadata = dict(getattr(runtime, "metadata", {}) or {})
    voice_cfg = metadata.get("audio_output")
    if not isinstance(voice_cfg, dict):
        voice_cfg = metadata.get("tts")
    return dict(voice_cfg) if isinstance(voice_cfg, dict) else {}


def _session_allows_audio(runtime: SessionRuntime) -> bool:
    session_output = str(getattr(runtime, "output_mode", "") or "").strip().lower()
    return session_output in {"audio_once", "audio_stream", "multimodal", "multimodal_result"}


def should_stream_voice_reply(runtime: SessionRuntime, turn: Turn) -> bool:
    turn_input = str(getattr(turn, "input_mode", "") or "").strip().lower()
    return turn_input in {InputMode.AUDIO_ONCE, InputMode.AUDIO_STREAM} and _session_allows_audio(runtime)


def should_emit_voice_reply(runtime: SessionRuntime, turn: Turn, assistant_text: str) -> bool:
    if not str(assistant_text or "").strip():
        return False
    turn_input = str(getattr(turn, "input_mode", "") or "").strip().lower()
    return turn_input in {InputMode.AUDIO_ONCE, InputMode.AUDIO_STREAM} and _session_allows_audio(runtime)


async def synthesize_voice_reply(
    *,
    runtime: SessionRuntime,
    turn: Turn,
    assistant_text: str,
    logger: Any = None,
) -> None:
    voice_cfg = resolve_voice_reply_cfg(runtime)
    if not voice_cfg:
        if logger is not None:
            logger.info(
                "[master-run] skip voice reply run_id=%s (no audio_output config) output_mode=%s",
                turn.run_id,
                getattr(runtime, "output_mode", ""),
            )
        return

    spoken_text = strip_markdown_for_tts(assistant_text, state={})
    if not spoken_text:
        if logger is not None:
            logger.info(
                "[master-run] skip voice reply run_id=%s (assistant_text empty after markdown strip)",
                turn.run_id,
            )
        return

    await runtime.submit_tts(
        text=spoken_text,
        voice_cfg=voice_cfg,
        request_id=uuid.uuid4().hex,
        metadata={
            "turn_id": turn.turn_id,
            "run_id": turn.run_id or "",
        },
    )


class VoiceReplyStream:
    def __init__(
        self,
        *,
        runtime: SessionRuntime,
        turn: Turn,
        logger: Any,
        run_timeout_seconds: float,
    ) -> None:
        self.runtime = runtime
        self.turn = turn
        self.logger = logger
        self.run_timeout_seconds = run_timeout_seconds
        self.enabled = should_stream_voice_reply(runtime, turn)
        if self.enabled and not resolve_voice_reply_cfg(runtime):
            logger.info(
                "[master-run] voice stream disabled run_id=%s (no audio_output config) output_mode=%s",
                turn.run_id,
                getattr(runtime, "output_mode", ""),
            )
            self.enabled = False
        self._queue: Optional[asyncio.Queue[Optional[Dict[str, str]]]] = (
            asyncio.Queue() if self.enabled else None
        )
        self._task: Optional[asyncio.Task[None]] = (
            asyncio.create_task(self._consumer()) if self._queue is not None else None
        )
        self._block_cfg: Dict[str, Any] = {
            "max_sentences": 5,
            "max_chars": 220,
            "max_wait_sec": 1.2,
        }
        self._state: Dict[str, Any] = {
            "buffer": "",
            "first_flushed": False,
            "block_sentences": [],
            "block_started_at": None,
            "in_code_block": False,
        }

    async def _consumer(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                return
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            kind = str(item.get("kind") or "semantic_block_stream").strip() or "semantic_block_stream"
            try:
                await self.runtime.submit_tts(
                    text=text,
                    voice_cfg=resolve_voice_reply_cfg(self.runtime),
                    request_id=uuid.uuid4().hex,
                    metadata={
                        "turn_id": self.turn.turn_id,
                        "run_id": self.turn.run_id or "",
                        "kind": kind,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.warning(
                    "[master-run] streaming tts block failed run_id=%s kind=%s chars=%d err=%s",
                    self.turn.run_id,
                    kind,
                    len(text),
                    exc,
                )

    async def teardown(self) -> None:
        if self._task is None or self._task.done():
            return
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except Exception:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def push_chunk(self, chunk: str) -> None:
        if self._queue is None or not chunk:
            return
        self._state["buffer"] += chunk
        weak_min = 15 if not self._state["first_flushed"] else 40
        sentences, remainder = _extract_sentences_for_tts(
            self._state["buffer"],
            weak_min=weak_min,
        )
        self._state["buffer"] = remainder
        for sentence in sentences:
            await self._accept_sentence(sentence)
        if "\n" in chunk:
            await self._flush_block(force=True)

    async def drain(self) -> None:
        if self._queue is None or self._task is None:
            return
        try:
            tail = str(self._state["buffer"] or "").strip()
            if tail:
                await self._accept_sentence(tail)
                self._state["buffer"] = ""
            await self._flush_block(force=True)
            await self._queue.put(None)
            try:
                await asyncio.wait_for(self._task, timeout=self.run_timeout_seconds)
            except asyncio.TimeoutError:
                self.logger.warning(
                    "[master-run] streaming tts drain timed out run_id=%s",
                    self.turn.run_id,
                )
                self._task.cancel()
                try:
                    await self._task
                except Exception:
                    pass
                await self.runtime.emit_error_event(
                    code="tts_failed",
                    message="TTS streaming timed out",
                    recoverable=True,
                )
        except asyncio.CancelledError:
            if self._task is not None and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except Exception:
                    pass
            raise
        except Exception as exc:
            self.logger.warning(
                "[master-run] streaming tts drain failed run_id=%s err=%s",
                self.turn.run_id,
                exc,
                exc_info=True,
            )
            await self.runtime.emit_error_event(
                code="tts_failed",
                message=f"TTS failed: {exc}",
                recoverable=True,
            )

    async def _enqueue_text(self, text: str, *, kind: str) -> None:
        if self._queue is None:
            return
        spoken = str(text or "").strip()
        if spoken:
            await self._queue.put({"text": spoken, "kind": kind})

    async def _flush_block(self, *, force: bool = False) -> None:
        block = [
            str(item or "").strip()
            for item in list(self._state.get("block_sentences") or [])
            if str(item or "").strip()
        ]
        if not block:
            self._state["block_sentences"] = []
            self._state["block_started_at"] = None
            return

        now = asyncio.get_running_loop().time()
        started_at = self._state.get("block_started_at")
        elapsed = (now - float(started_at)) if isinstance(started_at, (int, float)) else 0.0
        total_chars = sum(len(item) for item in block)
        should_flush = (
            force
            or len(block) >= int(self._block_cfg["max_sentences"])
            or total_chars >= int(self._block_cfg["max_chars"])
            or elapsed >= float(self._block_cfg["max_wait_sec"])
        )
        if not should_flush:
            return

        await self._enqueue_text(" ".join(block), kind="semantic_block_stream")
        self._state["block_sentences"] = []
        self._state["block_started_at"] = None

    async def _accept_sentence(self, sentence: str) -> None:
        spoken = strip_markdown_for_tts(sentence, state=self._state)
        if not spoken:
            return
        if not self._state["first_flushed"]:
            self._state["first_flushed"] = True
            await self._enqueue_text(spoken, kind="first_sentence_stream")
            return

        block = self._state.setdefault("block_sentences", [])
        if not block:
            self._state["block_started_at"] = asyncio.get_running_loop().time()
        block.append(spoken)
        await self._flush_block(force=False)
