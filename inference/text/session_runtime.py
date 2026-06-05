from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from text.runtime.common import normalize_chat_messages
from text.runtime.qwen_tool_parser import QwenToolCallParser


_STREAM_STATS_KEYS = (
    "finish_reason",
    "prompt_tokens",
    "output_tokens",
    "ttft_seconds",
    "total_seconds",
    "decode_seconds",
    "tok_s_total",
    "tok_s_decode",
)


@dataclass
class TextSessionState:
    """推理侧每个 session 的轻量状态。

    Phase 3 重构后，session_runtime 不再在推理侧维护多轮历史——每次
    ``session_text_input`` 都由后端把拼好的 prompt / messages 完整送进来；
    这里保留 ``generation_revision`` / ``active_request_id`` 仅为支持
    ``session_interrupt`` 的打断语义（后端触发 interrupt 时 +1，正在跑的
    generation 检测到 revision 变化就自行退出）。
    """

    session_id: str
    turn_count: int = 0
    last_seq: Optional[int] = None
    load_name: str = ""
    family: str = ""
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: str = ""
    enable_thinking: Optional[bool] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mm_processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    generation_revision: int = 0
    active_request_id: str = ""


class TextSessionRuntime:
    """
    文本 session 最小运行时。

    第一版先保证统一文本会话链路跑通；
    后续再把这里替换为真实 OpenAI/vLLM/Ollama 适配层。
    """

    def __init__(
        self,
        sender: Callable[[Dict], Awaitable[bool]],
        stream_text: Callable[[Dict[str, Any]], AsyncIterator[Dict[str, Any]]],
        abort_request: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ):
        self._sender = sender
        self._stream_text = stream_text
        self._abort_request = abort_request
        self._sessions: Dict[str, TextSessionState] = {}

    def _apply_session_config(self, state: TextSessionState, message: Dict[str, Any]) -> None:
        load_name = str(message.get("load_name") or state.load_name or "").strip()
        family = str(message.get("family") or state.family or "").strip()
        system_prompt = str(message.get("system_prompt") or state.system_prompt or "").strip()
        runtime_config = message.get("runtime_config") if isinstance(message.get("runtime_config"), dict) else None

        state.load_name = load_name
        state.family = family
        if runtime_config:
            state.runtime_config = dict(runtime_config)
        state.system_prompt = system_prompt

        if "temperature" in message and message.get("temperature") is not None:
            try:
                state.temperature = float(message.get("temperature"))
            except Exception:
                pass
        if "max_tokens" in message and message.get("max_tokens") is not None:
            try:
                state.max_tokens = int(message.get("max_tokens"))
            except Exception:
                pass
        if "enable_thinking" in message and message.get("enable_thinking") is not None:
            state.enable_thinking = bool(message.get("enable_thinking"))
        if "top_p" in message and message.get("top_p") is not None:
            try:
                state.top_p = float(message.get("top_p"))
            except Exception:
                pass
        if "top_k" in message and message.get("top_k") is not None:
            try:
                state.top_k = int(message.get("top_k"))
            except Exception:
                pass
        if "presence_penalty" in message and message.get("presence_penalty") is not None:
            try:
                state.presence_penalty = float(message.get("presence_penalty"))
            except Exception:
                pass
        if "frequency_penalty" in message and message.get("frequency_penalty") is not None:
            try:
                state.frequency_penalty = float(message.get("frequency_penalty"))
            except Exception:
                pass
        if "mm_processor_kwargs" in message and isinstance(message.get("mm_processor_kwargs"), dict):
            state.mm_processor_kwargs = dict(message.get("mm_processor_kwargs") or {})
        if "tools" in message:
            raw_tools = message.get("tools")
            if isinstance(raw_tools, list):
                state.tools = [dict(item) for item in raw_tools if isinstance(item, dict)]
            elif raw_tools in (None, ""):
                state.tools = []
        if "tool_choice" in message:
            state.tool_choice = message.get("tool_choice")

    def _build_messages(
        self,
        state: TextSessionState,
        user_text: str,
        *,
        request_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """把本轮输入归一化成 ``[{role, content}]``。

        推理侧不再维护多轮历史：``request_messages`` 非空时直接采用后端
        拼好的完整 messages；否则把 ``user_text`` 作为单条 user 消息，
        并在需要时自动补上 ``system_prompt``。
        """
        if request_messages is not None:
            normalized = normalize_chat_messages(request_messages)
            has_system_message = any(str(item.get("role") or "") == "system" for item in normalized)
            if state.system_prompt and not has_system_message:
                return [{"role": "system", "content": state.system_prompt}, *normalized]
            return normalized

        messages: List[Dict[str, str]] = []
        if state.system_prompt:
            messages.append({"role": "system", "content": state.system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    async def _abort_active_generation(self, state: TextSessionState) -> None:
        if not self._abort_request or not state.active_request_id:
            state.active_request_id = ""
            return
        request = {
            "session_id": state.session_id,
            "request_id": state.active_request_id,
            "load_name": state.load_name,
            "family": state.family,
            "runtime_config": state.runtime_config,
        }
        state.active_request_id = ""
        try:
            await self._abort_request(request)
        except Exception:
            pass

    async def register_service(self, *, service_type: str = "text") -> bool:
        """向后端注册本推理进程。

        重构后：推理服务不再在注册时上报"可承载模型列表"。后端按
        ``service_type`` 选出 running 服务，真正加载哪个 LLM 权重
        由会话侧 ``session_text_input.payload.load_name`` 决定。
        """
        return await self._sender(
            {
                "type": "service_register",
                "service_type": service_type,
                "supports_task": True,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def handle_message(self, message: Dict) -> bool:
        message_type = str(message.get("type") or "").strip()
        session_id = str(message.get("session_id") or "").strip()
        if not session_id:
            return False

        if message_type == "session_open":
            state = self._sessions.get(session_id) or TextSessionState(session_id=session_id)
            self._apply_session_config(state, message)
            state.last_seq = message.get("seq")
            self._sessions[session_id] = state
            await self._sender(
                {
                    "type": "session_ready",
                    "session_id": session_id,
                    "seq": message.get("seq"),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return True

        if message_type == "session_text_input":
            state = self._sessions.get(session_id) or TextSessionState(session_id=session_id)
            self._apply_session_config(state, message)
            request_messages = message.get("messages") if isinstance(message.get("messages"), list) else None
            raw_text = str(
                message.get("text")
                or message.get("input_text")
                or message.get("prompt")
                or ""
            ).strip()
            explicit_messages = request_messages is not None
            if not state.load_name or not state.family:
                await self._sender(
                    {
                        "type": "session_error",
                        "session_id": session_id,
                        "seq": message.get("seq"),
                        "error": "text session requires load_name and family before text input",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                return True

            if state.active_request_id:
                await self._abort_active_generation(state)

            state.turn_count += 1
            state.last_seq = message.get("seq")
            state.generation_revision += 1
            self._sessions[session_id] = state

            current_revision = state.generation_revision
            messages = self._build_messages(state, raw_text, request_messages=request_messages)
            request_id = f"session:{session_id}:{current_revision}"
            state.active_request_id = request_id
            reply_parts: List[str] = []
            chunk_index = 0
            tool_parser = QwenToolCallParser()
            accumulated_tool_calls: List[Dict[str, Any]] = []
            try:
                try:
                    async for item in self._stream_text(
                        {
                            "session_id": session_id,
                            "request_id": request_id,
                            "load_name": state.load_name,
                            "family": state.family,
                            "runtime_config": state.runtime_config,
                            "messages": messages,
                            "temperature": state.temperature,
                            "max_tokens": state.max_tokens,
                            "enable_thinking": state.enable_thinking,
                            "top_p": state.top_p,
                            "top_k": state.top_k,
                            "presence_penalty": state.presence_penalty,
                            "frequency_penalty": state.frequency_penalty,
                            "mm_processor_kwargs": state.mm_processor_kwargs,
                            "tools": list(state.tools) if state.tools else None,
                            "tool_choice": state.tool_choice,
                        }
                    ):
                        latest_state = self._sessions.get(session_id)
                        if not latest_state or latest_state.generation_revision != current_revision:
                            break
                        raw_delta = str(item.get("delta") or "")
                        finished = bool(item.get("finished"))

                        if raw_delta:
                            content_delta, new_tool_calls = tool_parser.feed(raw_delta)
                        else:
                            content_delta, new_tool_calls = "", []

                        if finished:
                            flushed_content, flushed_tool_calls = tool_parser.flush()
                            content_delta = content_delta + flushed_content
                            new_tool_calls = new_tool_calls + flushed_tool_calls

                        for call in new_tool_calls:
                            accumulated_tool_calls.append(call.to_openai())

                        if content_delta:
                            reply_parts.append(content_delta)

                        should_emit = bool(content_delta) or bool(new_tool_calls) or finished
                        if should_emit:
                            chunk_index += 1
                            event: Dict[str, Any] = {
                                "type": "llm_text_delta",
                                "session_id": session_id,
                                "seq": message.get("seq"),
                                "delta": content_delta,
                                "turn_index": state.turn_count,
                                "chunk_index": chunk_index,
                                "is_final": finished,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            if new_tool_calls:
                                event["tool_calls_delta"] = [call.to_openai() for call in new_tool_calls]
                            # 先带上底层 stats（prompt_tokens / output_tokens / finish_reason 等）。
                            for key in _STREAM_STATS_KEYS:
                                if item.get(key) is not None:
                                    event[key] = item.get(key)
                            # 只要解析到任何 tool_call，finish_reason 就统一覆盖为 tool_calls；
                            # 模型同一轮里先说点话再 call 工具也算命中。
                            if finished and accumulated_tool_calls:
                                event["tool_calls"] = list(accumulated_tool_calls)
                                event["finish_reason"] = "tool_calls"
                            await self._sender(event)
                        await asyncio.sleep(0)
                except Exception as e:
                    latest_state = self._sessions.get(session_id)
                    if latest_state and latest_state.generation_revision == current_revision:
                        await self._sender(
                            {
                                "type": "session_error",
                                "session_id": session_id,
                                "seq": message.get("seq"),
                                "error": str(e),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    return True
            finally:
                latest_state = self._sessions.get(session_id)
                if latest_state and latest_state.active_request_id == request_id:
                    latest_state.active_request_id = ""

            latest_state = self._sessions.get(session_id)
            if not latest_state or latest_state.generation_revision != current_revision:
                return True

            # Phase 3 起推理侧不再保存 history；拼历史这件事统一由后端负责。
            _ = explicit_messages  # kept for compatibility with upstream message shape
            return True

        if message_type == "session_interrupt":
            state = self._sessions.get(session_id)
            if state:
                state.last_seq = message.get("seq")
                state.generation_revision += 1
                await self._abort_active_generation(state)
            return True

        if message_type == "session_close":
            state = self._sessions.pop(session_id, None)
            if state:
                await self._abort_active_generation(state)
            await self._sender(
                {
                    "type": "session_closed",
                    "session_id": session_id,
                    "seq": message.get("seq"),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return True

        return False
