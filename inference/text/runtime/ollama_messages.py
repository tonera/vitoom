from __future__ import annotations

import json
from typing import Any, Dict, List


def wrap_qwen_tool_response(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("<tool_response>") and "</tool_response>" in text:
        return text
    return f"<tool_response>\n{text}\n</tool_response>"


def fold_tool_roles_for_ollama(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """旧路径：把 role=tool 折成 user XML。Ollama ``tools=`` 不要走这条。"""
    folded: List[Dict[str, Any]] = []
    pending: List[str] = []

    def flush() -> None:
        if not pending:
            return
        folded.append({"role": "user", "content": "\n".join(pending)})
        pending.clear()

    for message in messages:
        if str(message.get("role") or "") == "tool":
            pending.append(wrap_qwen_tool_response(str(message.get("content") or "")))
            continue
        flush()
        folded.append(message)
    flush()
    return folded


def parse_tool_arguments(args_raw: Any) -> Any:
    if isinstance(args_raw, str):
        try:
            return json.loads(args_raw) if args_raw.strip() else {}
        except Exception:
            return {"_raw": args_raw}
    if isinstance(args_raw, (dict, list)):
        return args_raw
    return {}


def normalize_assistant_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    out: List[Dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        item: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "arguments": parse_tool_arguments(function.get("arguments")),
            },
        }
        call_id = str(call.get("id") or "").strip()
        if call_id:
            item["id"] = call_id
        out.append(item)
    return out


def serialize_assistant_tool_calls_to_text(tool_calls: Any) -> str:
    blocks: List[str] = []
    for call in normalize_assistant_tool_calls(tool_calls):
        payload = json.dumps(
            {"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
            ensure_ascii=False,
        )
        blocks.append(f"<tool_call>\n{payload}\n</tool_call>")
    return ("\n".join(blocks) + "\n") if blocks else ""


def keep_openai_tool_pairing(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ollama ``tools=`` 要 assistant.tool_calls ↔ role=tool。arguments 必须是对象。"""
    converted: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip() or "user"
        content = message.get("content")
        text = content if isinstance(content, str) else "" if content is None else str(content)
        entry: Dict[str, Any] = {"role": role, "content": text}
        if message.get("images"):
            entry["images"] = message["images"]
        if role == "assistant":
            tool_calls = normalize_assistant_tool_calls(message.get("tool_calls"))
            if tool_calls:
                entry["tool_calls"] = tool_calls
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            name = str(message.get("name") or message.get("tool_name") or "").strip()
            if name:
                entry["tool_name"] = name
        converted.append(entry)
    return converted


def inline_tool_calls_and_fold(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """当前会 500 的路径：tool_calls 写成 XML，role=tool 折成 user。"""
    converted: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip() or "user"
        content = message.get("content")
        text = content if isinstance(content, str) else "" if content is None else str(content)
        if role == "assistant":
            tool_text = serialize_assistant_tool_calls_to_text(message.get("tool_calls"))
            if tool_text:
                text = (text + ("\n" if text else "") + tool_text).rstrip()
        converted.append({"role": role, "content": text})
    return fold_tool_roles_for_ollama(converted)
