from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    return value


def _json_unescape(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value


def _clean_tool_text(value: Any) -> str:
    return str(value or "").strip()


def _tool_call_name(tool_call: Any) -> str:
    if tool_call is None:
        return ""
    if isinstance(tool_call, dict):
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        return _clean_tool_text(
            tool_call.get("name")
            or tool_call.get("tool_name")
            or function.get("name")
        )
    function = getattr(tool_call, "function", None)
    return _clean_tool_text(
        getattr(tool_call, "name", None)
        or getattr(tool_call, "tool_name", None)
        or getattr(function, "name", None)
    )


def _tool_call_arguments_delta(chunk: str, tool_call: Any) -> str:
    if chunk:
        return str(chunk)
    if tool_call is None:
        return ""
    if isinstance(tool_call, dict):
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        raw = (
            tool_call.get("arguments")
            or tool_call.get("args")
            or tool_call.get("delta")
            or function.get("arguments")
        )
    else:
        function = getattr(tool_call, "function", None)
        raw = (
            getattr(tool_call, "arguments", None)
            or getattr(tool_call, "args", None)
            or getattr(tool_call, "delta", None)
            or getattr(function, "arguments", None)
        )
    if isinstance(raw, str):
        return raw
    if raw is not None:
        try:
            return json.dumps(raw, ensure_ascii=False, default=str)
        except Exception:
            return str(raw)
    return ""


class AudioDramaToolCallStreamRenderer:
    """Render user-visible script text from streaming `audio_drama_tts` args."""

    _STRING_FIELD_RE_TEMPLATE = r'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"'
    _OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

    def __init__(self) -> None:
        self.active = False
        self.buffer = ""
        self.header_emitted = False
        self.roles_header_emitted = False
        self.dialogues_header_emitted = False
        self.completed_note_emitted = False
        self.character_names: Dict[str, str] = {}
        self.emitted_characters: set[str] = set()
        self.emitted_dialogues: set[str] = set()

    def feed_delta(self, *, chunk: str, call_type: Any = None, tool_call: Any = None) -> List[str]:
        normalized_call_type = str(getattr(call_type, "value", call_type) or "").strip().lower()
        if normalized_call_type != "tool_call" and tool_call is None:
            return []

        tool_name = _tool_call_name(tool_call)
        delta = _tool_call_arguments_delta(chunk, tool_call)
        if (
            tool_name == "audio_drama_tts"
            or "audio_drama_tts" in delta
            or '"characters"' in delta
            or '\\"characters\\"' in delta
            or '"dialogues"' in delta
            or '\\"dialogues\\"' in delta
        ):
            self.active = True
        if not self.active or not delta:
            return []

        self.buffer += delta
        if len(self.buffer) > 200_000:
            self.buffer = self.buffer[-120_000:]
        return self._render_available()

    def feed_complete_args(self, *, tool_name: str, args: Any) -> List[str]:
        if str(tool_name or "").strip() != "audio_drama_tts":
            return []
        self.active = True
        parsed = _json_loads_maybe(args)
        if isinstance(parsed, dict):
            return self._render_from_mapping(parsed)
        self.buffer += _tool_call_arguments_delta("", args)
        return self._render_available()

    def render_completed_note(self, *, tool_name: str, status: str) -> List[str]:
        if str(tool_name or "").strip() != "audio_drama_tts":
            return []
        if not self.active or self.completed_note_emitted:
            return []
        self.completed_note_emitted = True
        if status == "completed":
            return ["\n音频已生成，文件在附件区。\n"]
        return []

    def _render_available(self) -> List[str]:
        parsed = self._try_parse_buffer_object()
        if isinstance(parsed, dict):
            return self._render_from_mapping(parsed)
        return self._render_from_partial_json()

    def _try_parse_buffer_object(self) -> Any:
        text = self.buffer.strip()
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        return _json_loads_maybe(text[start : end + 1])

    def _render_from_mapping(self, payload: Dict[str, Any]) -> List[str]:
        chunks = self._header()
        characters = payload.get("characters")
        if isinstance(characters, list):
            for item in characters:
                if isinstance(item, dict):
                    chunks.extend(self._maybe_render_character(item))
        dialogues = payload.get("dialogues")
        if isinstance(dialogues, list):
            for item in dialogues:
                if isinstance(item, dict):
                    chunks.extend(self._maybe_render_dialogue(item))
        return chunks

    def _render_from_partial_json(self) -> List[str]:
        chunks = self._header()
        scan_buffer = self.buffer.replace('\\"', '"')
        for raw_obj in self._OBJECT_RE.findall(scan_buffer):
            if "speaker_id" in raw_obj and "text" in raw_obj:
                item = {
                    "speaker_id": self._field(raw_obj, "speaker_id"),
                    "text": self._field(raw_obj, "text"),
                    "emotion": self._field(raw_obj, "emotion"),
                }
                chunks.extend(self._maybe_render_dialogue(item))
            elif '"id"' in raw_obj and '"name"' in raw_obj:
                item = {
                    "id": self._field(raw_obj, "id"),
                    "name": self._field(raw_obj, "name"),
                    "instruct": self._field(raw_obj, "instruct"),
                    "speaker_name": self._field(raw_obj, "speaker_name"),
                }
                chunks.extend(self._maybe_render_character(item))
        return chunks

    def _field(self, raw_obj: str, field: str) -> str:
        pattern = re.compile(self._STRING_FIELD_RE_TEMPLATE.format(field=re.escape(field)), re.DOTALL)
        match = pattern.search(raw_obj)
        return _json_unescape(match.group(1)).strip() if match else ""

    def _header(self) -> List[str]:
        if self.header_emitted:
            return []
        self.header_emitted = True
        return ["下面是我生成的对白文本，音频会在文本完成后继续合成：\n\n"]

    def _maybe_render_character(self, item: Dict[str, Any]) -> List[str]:
        cid = _clean_tool_text(item.get("id"))
        name = _clean_tool_text(item.get("name"))
        if not cid or not name:
            return []
        self.character_names[cid] = name
        if cid in self.emitted_characters:
            return []
        self.emitted_characters.add(cid)
        voice = "；".join(
            part for part in [
                _clean_tool_text(item.get("speaker_name")),
                _clean_tool_text(item.get("instruct")),
            ]
            if part
        )
        chunks = []
        if not self.roles_header_emitted:
            self.roles_header_emitted = True
            chunks.append("## 角色设定\n")
        chunks.append(f"- {name}" + (f"：{voice}" if voice else "") + "\n")
        return chunks

    def _maybe_render_dialogue(self, item: Dict[str, Any]) -> List[str]:
        speaker_id = _clean_tool_text(item.get("speaker_id"))
        text = _clean_tool_text(item.get("text"))
        if not speaker_id or not text:
            return []
        key = f"{speaker_id}\n{text}"
        if key in self.emitted_dialogues:
            return []
        self.emitted_dialogues.add(key)
        name = self.character_names.get(speaker_id, speaker_id)
        emotion = _clean_tool_text(item.get("emotion"))
        chunks = []
        if not self.dialogues_header_emitted:
            if self.roles_header_emitted:
                chunks.append("\n")
            self.dialogues_header_emitted = True
            chunks.append("## 对白\n")
        chunks.append(f"{name}" + (f"（{emotion}）" if emotion else "") + f"：{text}\n")
        return chunks


class ToolCallStreamRendererRegistry:
    """Small registry for rendering selected tool-call argument streams as chat text."""

    def __init__(self) -> None:
        self._renderers = [AudioDramaToolCallStreamRenderer()]

    @property
    def active(self) -> bool:
        return any(bool(getattr(renderer, "active", False)) for renderer in self._renderers)

    def feed_delta(self, *, chunk: str, call_type: Any = None, tool_call: Any = None) -> List[str]:
        out: List[str] = []
        for renderer in self._renderers:
            out.extend(renderer.feed_delta(chunk=chunk, call_type=call_type, tool_call=tool_call))
        return out

    def feed_complete_args(self, *, tool_name: str, args: Any) -> List[str]:
        out: List[str] = []
        for renderer in self._renderers:
            out.extend(renderer.feed_complete_args(tool_name=tool_name, args=args))
        return out

    def render_completed_note(self, *, tool_name: str, status: str) -> List[str]:
        out: List[str] = []
        for renderer in self._renderers:
            out.extend(renderer.render_completed_note(tool_name=tool_name, status=status))
        return out
