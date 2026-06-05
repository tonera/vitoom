from __future__ import annotations

from typing import Any, Dict, List, Tuple


def chunk_text(text: str, *, chunk_size: int = 24) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []

    if " " in raw:
        words = [piece for piece in raw.split(" ") if piece]
        chunks: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    return [raw[i:i + chunk_size] for i in range(0, len(raw), chunk_size)]


def _normalize_text_content(value: Any) -> str:
    return str(value or "")


def _normalize_content_part(part: Any) -> Dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        raise ValueError(f"Unsupported message content part type: {type(part)!r}")

    raw_type = str(part.get("type") or "").strip().lower()
    if raw_type in {"", "text", "input_text"}:
        return {
            "type": "text",
            "text": _normalize_text_content(part.get("text") or part.get("input_text")),
        }

    if raw_type in {"image", "image_url"}:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = str(image_url.get("url") or "").strip()
            detail = image_url.get("detail")
        elif isinstance(image_url, str):
            url = image_url.strip()
            detail = None
        else:
            url = str(part.get("image") or part.get("url") or "").strip()
            detail = part.get("detail")
        if not url:
            raise ValueError("image_url content item requires a non-empty url")
        payload: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
        if detail not in (None, ""):
            payload["image_url"]["detail"] = detail
        return payload

    if raw_type in {"video", "video_url"}:
        video_url = part.get("video_url")
        if isinstance(video_url, dict):
            url = str(video_url.get("url") or "").strip()
        elif isinstance(video_url, str):
            url = video_url.strip()
        else:
            url = str(part.get("video") or part.get("url") or "").strip()
        if not url:
            raise ValueError("video_url content item requires a non-empty url")
        return {"type": "video_url", "video_url": {"url": url}}

    raise ValueError(f"Unsupported content part type='{raw_type or 'unknown'}'")


def normalize_chat_content(content: Any) -> str | List[Dict[str, Any]]:
    if isinstance(content, list):
        return [_normalize_content_part(part) for part in content]
    if isinstance(content, dict):
        return [_normalize_content_part(content)]
    return _normalize_text_content(content)


def _normalize_assistant_tool_calls(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for call in value:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or "").strip()
        call_type = str(call.get("type") or "function").strip() or "function"
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or call.get("name") or "").strip()
        arguments = function.get("arguments", call.get("arguments"))
        if isinstance(arguments, (dict, list)):
            import json as _json

            arguments = _json.dumps(arguments, ensure_ascii=False)
        elif arguments is None:
            arguments = ""
        else:
            arguments = str(arguments)
        if not name:
            continue
        normalized.append(
            {
                "id": call_id,
                "type": call_type,
                "function": {"name": name, "arguments": arguments},
            }
        )
    return normalized


def normalize_chat_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    normalized: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = str(item.get("role") or "").strip()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role='{role or 'unknown'}'")

        message: Dict[str, Any] = {
            "role": role,
            "content": normalize_chat_content(item.get("content")),
        }

        if role == "assistant":
            tool_calls = _normalize_assistant_tool_calls(item.get("tool_calls"))
            if tool_calls:
                message["tool_calls"] = tool_calls

        if role == "tool":
            tool_call_id = str(item.get("tool_call_id") or "").strip()
            if tool_call_id:
                message["tool_call_id"] = tool_call_id
            tool_name = str(item.get("name") or "").strip()
            if tool_name:
                message["name"] = tool_name

        normalized.append(message)
    return normalized


def count_multimodal_parts(messages: List[Dict[str, Any]]) -> Tuple[int, int]:
    image_count = 0
    video_count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip().lower()
            if part_type in {"image", "image_url"}:
                image_count += 1
            elif part_type in {"video", "video_url"}:
                video_count += 1
    return image_count, video_count


def build_messages_from_prompt(prompt: str, *, system_prompt: str | None = None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    system_text = str(system_prompt or "").strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})

    user_text = str(prompt or "").strip()
    if user_text:
        messages.append({"role": "user", "content": user_text})
    return messages


def deep_merge_dict(base: Dict[str, Any] | None, override: Dict[str, Any] | None) -> Dict[str, Any]:
    left = dict(base or {})
    right = dict(override or {})
    merged: Dict[str, Any] = dict(left)
    for key, value in right.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(current, value)
        else:
            merged[key] = value
    return merged
