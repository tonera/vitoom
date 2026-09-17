from __future__ import annotations

from typing import Any, Dict, List


def wrap_qwen_tool_response(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("<tool_response>") and "</tool_response>" in text:
        return text
    return f"<tool_response>\n{text}\n</tool_response>"


def fold_tool_roles_for_ollama(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 ``role=tool`` 折成 Qwen 的 user ``<tool_response>``。

    Ollama 桥把 assistant.tool_calls 内联成 ``<tool_call>`` 文本，不再带结构化
    ``tool_calls`` 字段。若再把 ``role=tool`` 原样交给 Ollama（尤其还带着
    ``tools=``），daemon 按原生协议校验配对，对不上时 llama runner 500 EOF。
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
