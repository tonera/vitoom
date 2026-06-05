"""增量解析 Qwen / Hermes 系列模型的 ``<tool_call>...</tool_call>`` 输出。

``<tool_call>`` 块内部有两种常见 payload 格式，我们都要认：

1. **JSON 格式**（Qwen2.5、Qwen2-VL、Hermes-2-Pro 等使用）::

       <tool_call>
       {"name": "analyze_media", "arguments": {"url": "https://.../a.jpg"}}
       </tool_call>

2. **XML 格式**（Qwen3 系列的某些微调，如 Qwen3.5-A3B/Int4 实际输出；也出现在
   Llama-3.1 的 Hermes-style tool prompting 里）::

       <tool_call><function=analyze_media>
       <parameter=url>
       https://.../a.jpg
       </parameter>
       <parameter=question>
       这张图是什么？
       </parameter>
       </function></tool_call>

一次生成中可能出现多个 ``<tool_call>`` 片段，也可能与普通文本混排。由于推理走的是
token 级流式解码，我们需要一个有状态解析器，边消费 delta、边分离出：

* ``content``  —— 需要透传给客户端的普通文本增量；
* ``tool_calls`` —— 当一个 ``<tool_call>`` 块被完整闭合时解析出的结构化调用。

输出的 tool call 结构对齐 OpenAI chat completions: ``{"id", "type", "function":
{"name", "arguments"}}``；``arguments`` 统一为 JSON 字符串，便于下游直接放进
``message.tool_calls``。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"

# XML 风格 payload 用正则匹配：
#   <function=NAME>...</function> 里包含若干 <parameter=KEY>VALUE</parameter>。
# 函数名与参数名允许连字符与下划线；值用非贪婪 + DOTALL 以覆盖带换行的长文本。
_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)

# 模块加载指纹：每当 inference 进程加载本模块时在日志里打印一次。
# 如果你怀疑 inference 服务没加载到新代码，就看进程日志里是否有这行——
# 没有代表根本没走到这里，旧进程还在跑。
_PARSER_FINGERPRINT = "qwen_tool_parser v2 (json+xml payloads) loaded"
logging.getLogger(__name__).info(_PARSER_FINGERPRINT)


@dataclass
class QwenToolCall:
    """已闭合的单个 tool call。"""

    id: str
    name: str
    arguments: str  # JSON 字符串，始终非 None

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class QwenToolCallParser:
    """Qwen ``<tool_call>`` 增量解析器。

    只负责**文本层**的状态机，不做 JSON schema 校验。解析失败（JSON 不合法）的
    片段会被降级为普通文本原样透传，避免把明显的模型生成内容吞掉。
    """

    _buffer: str = ""
    _inside_tool_call: bool = False
    _current: List[str] = field(default_factory=list)
    _calls: List[QwenToolCall] = field(default_factory=list)
    # 尾缓冲要大到既能识别 `<tool_call>` 也能识别 `</tool_call>`——后者更长一个字符，
    # 若缓冲不够大，close tag 正好卡在 chunk 边界时，首个 `<` 会被当作 content 吐出去，
    # 从而丢失整个 tool_call。
    _tail_safe_len: int = max(len(_OPEN_TAG), len(_CLOSE_TAG)) - 1

    def feed(self, text: str) -> Tuple[str, List[QwenToolCall]]:
        """吃掉一段新增 text，返回 (可安全下发的 content 增量, 本次闭合的 tool calls)。

        实现思路：维护一个 rolling buffer——既要能识别完整 ``<tool_call>`` /
        ``</tool_call>`` 标签，又不能因为 token 恰好切在标签中间就提前下发残片。
        因此在"安全下发"时保留最后 ``len(tag)-1`` 个字符作尾缓冲。
        """
        if not text:
            return "", []
        self._buffer += text
        emitted_content_parts: List[str] = []
        emitted_tool_calls: List[QwenToolCall] = []

        while True:
            if self._inside_tool_call:
                close_pos = self._buffer.find(_CLOSE_TAG)
                if close_pos == -1:
                    # 标签未闭合，先吃进来但不 emit。
                    if len(self._buffer) > self._tail_safe_len:
                        safe = self._buffer[: -self._tail_safe_len]
                        self._current.append(safe)
                        self._buffer = self._buffer[-self._tail_safe_len :]
                    break
                self._current.append(self._buffer[:close_pos])
                self._buffer = self._buffer[close_pos + len(_CLOSE_TAG) :]
                self._inside_tool_call = False
                call_text = "".join(self._current).strip()
                self._current.clear()
                parsed = _parse_qwen_tool_call_payload(call_text)
                if parsed is not None:
                    self._calls.append(parsed)
                    emitted_tool_calls.append(parsed)
                else:
                    # JSON 无法解析：降级为普通文本原样透传（含包裹标签），方便排查。
                    emitted_content_parts.append(_OPEN_TAG + call_text + _CLOSE_TAG)
                continue

            open_pos = self._buffer.find(_OPEN_TAG)
            if open_pos == -1:
                # 普通文本：保留末尾少量字符以防标签被切断。
                if len(self._buffer) > self._tail_safe_len:
                    emitted_content_parts.append(self._buffer[: -self._tail_safe_len])
                    self._buffer = self._buffer[-self._tail_safe_len :]
                break

            if open_pos > 0:
                emitted_content_parts.append(self._buffer[:open_pos])
            self._buffer = self._buffer[open_pos + len(_OPEN_TAG) :]
            self._inside_tool_call = True

        return "".join(emitted_content_parts), emitted_tool_calls

    def flush(self) -> Tuple[str, List[QwenToolCall]]:
        """生成结束时调用，强制 drain 剩余 buffer。"""
        emitted_content_parts: List[str] = []
        emitted_tool_calls: List[QwenToolCall] = []

        if self._inside_tool_call:
            # 生成异常终止于 tool_call 内部：把已收集到的 payload 尝试解析，失败则原样吐回。
            partial = "".join(self._current) + self._buffer
            self._current.clear()
            self._buffer = ""
            self._inside_tool_call = False
            parsed = _parse_qwen_tool_call_payload(partial.strip())
            if parsed is not None:
                self._calls.append(parsed)
                emitted_tool_calls.append(parsed)
            else:
                emitted_content_parts.append(_OPEN_TAG + partial)
        elif self._buffer:
            emitted_content_parts.append(self._buffer)
            self._buffer = ""

        return "".join(emitted_content_parts), emitted_tool_calls

    @property
    def collected_tool_calls(self) -> List[QwenToolCall]:
        return list(self._calls)


def _parse_qwen_tool_call_payload(text: str) -> Optional[QwenToolCall]:
    """解析一段 ``<tool_call>`` 内部的 payload，兼容 JSON 与 XML 两种格式。"""
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    call = _parse_json_style_payload(stripped)
    if call is not None:
        return call

    call = _parse_xml_style_payload(stripped)
    if call is not None:
        return call

    return None


def _parse_json_style_payload(text: str) -> Optional[QwenToolCall]:
    """解析 ``{"name": ..., "arguments": {...}}`` 形式的 Qwen2.5 / Hermes-JSON payload。"""
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name") or "").strip()
    if not name:
        return None
    arguments_value: Any = payload.get("arguments")
    if arguments_value is None:
        arguments_str = "{}"
    elif isinstance(arguments_value, str):
        arguments_str = arguments_value
    else:
        try:
            arguments_str = json.dumps(arguments_value, ensure_ascii=False)
        except Exception:
            return None
    call_id = str(payload.get("id") or "").strip() or f"call_{uuid.uuid4().hex[:24]}"
    return QwenToolCall(id=call_id, name=name, arguments=arguments_str)


def _parse_xml_style_payload(text: str) -> Optional[QwenToolCall]:
    """解析 ``<function=NAME><parameter=K>V</parameter>...</function>`` 形式的 XML payload。

    这是 Qwen3 / Llama-3.1 / 某些 Hermes 微调模型在启用 function-calling 时会生成的格式。
    值一律当作字符串保留；若值本身是合法 JSON（数字、布尔、嵌套对象等），则解析为原生
    JSON 类型，以便下游按正确类型填入函数参数。
    """
    fn_match = _FUNCTION_BLOCK_RE.search(text)
    if fn_match is None:
        return None
    name = fn_match.group(1).strip()
    if not name:
        return None
    body = fn_match.group(2) or ""

    arguments: Dict[str, Any] = {}
    for param_match in _PARAMETER_RE.finditer(body):
        key = param_match.group(1).strip()
        if not key:
            continue
        raw_value = param_match.group(2) or ""
        value_str = raw_value.strip()
        if value_str == "":
            arguments[key] = ""
            continue
        arguments[key] = _coerce_xml_value(value_str)

    try:
        arguments_str = json.dumps(arguments, ensure_ascii=False)
    except Exception:
        return None
    return QwenToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        name=name,
        arguments=arguments_str,
    )


_JSON_LITERAL_RE = re.compile(r"^(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|\[.*\]|\{.*\})$", re.DOTALL)


def _coerce_xml_value(value_str: str) -> Any:
    """将 XML 风格参数值转换为合适的 Python / JSON 类型。

    为避免把普通字符串 ``"true"`` / ``"1"`` 误解为 bool / 数字，我们只在值整体匹配
    JSON 字面量（数字、布尔、null、数组、对象）时才执行 ``json.loads``；URL、中文问
    句、普通英文短语等都会原样保留为字符串。
    """
    if not _JSON_LITERAL_RE.match(value_str):
        return value_str
    try:
        return json.loads(value_str)
    except Exception:
        return value_str
