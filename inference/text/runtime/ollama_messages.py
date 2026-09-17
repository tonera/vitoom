from __future__ import annotations

from typing import Any, Dict, List


def wrap_qwen_tool_response(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("<tool_response>") and "</tool_response>" in text:
        return text
    return f"<tool_response>\n{text}\n</tool_response>"


def fold_tool_roles_for_ollama(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 ``role=tool`` 折成 Qwen 的 user ``<tool_response>``。

    本桥把 assistant.tool_calls 内联成 ``<tool_call>`` 文本。Ollama 原生
    ``tools=`` 仍要 assistant.tool_calls ↔ role=tool 配对；对不上会 500 EOF。
    """
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
