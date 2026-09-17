from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

QWEN_TOOLS_MARK = "<tools>"

_QWEN_TOOLS_PREAMBLE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""

_QWEN_TOOLS_EPILOGUE = """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <arguments-json-object>}
</tool_call>
"""


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


def _tool_schema_lines(tools: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        payload = dict(tool)
        if payload.get("type") != "function" and "function" not in payload:
            payload = {"type": "function", "function": payload}
        elif "type" not in payload:
            payload = {"type": "function", **payload}
        try:
            lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        except TypeError:
            continue
    return "\n".join(lines)


def inject_qwen_tool_schema(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """把 OpenAI tools 写进 system 文本，不再走 Ollama 原生 ``tools=``。

    Qwen 家族用 ``<tool_call>`` / ``<tool_response>`` 文本协议。把 schema 交给
    ``client.chat(tools=...)`` 时，daemon 会按原生 function-calling 渲染；历史里
    一旦已有内联 XML，runner 经常 500 EOF。
    """
    if not tools:
        return messages
    schema = _tool_schema_lines(tools)
    if not schema:
        return messages
    block = f"{_QWEN_TOOLS_PREAMBLE}{schema}{_QWEN_TOOLS_EPILOGUE}".strip()
    injected: List[Dict[str, Any]] = [dict(message) for message in messages]
    for message in injected:
        if str(message.get("role") or "") != "system":
            continue
        content = str(message.get("content") or "")
        if "# Tools" in content and QWEN_TOOLS_MARK in content:
            return injected
        message["content"] = f"{content.rstrip()}\n\n{block}" if content.strip() else block
        return injected
    return [{"role": "system", "content": block}, *injected]
