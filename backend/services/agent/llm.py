from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from backend.database import Model
from backend.models.service import get_model_service

from .settings import (
    get_agent_effective_user_header_name,
    get_agent_internal_auth_token,
    get_agent_llm_base_url,
    get_agent_llm_model_name,
    get_agent_llm_timeout_seconds,
)

_CREWAI_OPENAI_STREAM_PATCHED = False


def _normalize_model_name(model_row: Dict[str, Any]) -> str:
    load_name = str(model_row.get("load_name") or "").strip()
    if load_name:
        return load_name
    raise RuntimeError("Resolved text model is missing load_name")


def resolve_agent_llm_model_name(preferred_model_name: Optional[str] = None) -> str:
    preferred = str(preferred_model_name or get_agent_llm_model_name() or "").strip()
    if preferred:
        by_load_name = Model.get_by_load_name(preferred)
        if by_load_name:
            return _normalize_model_name(by_load_name)
        by_name = Model.get_by_name(preferred)
        if by_name:
            return _normalize_model_name(by_name)
        return preferred

    model_service = get_model_service()
    active_models, _total = model_service.list_models(modality="text", service_status="active", limit=1, offset=0)
    if active_models:
        return _normalize_model_name(active_models[0])

    models, _total = model_service.list_models(modality="text", limit=1, offset=0)
    if models:
        return _normalize_model_name(models[0])

    raise RuntimeError(
        "No text model is available for Agent runtime. "
        "Please activate a text model or set agents.default_model / agents.llm.model."
    )


def _patch_crewai_openai_streaming_tool_calls() -> None:
    global _CREWAI_OPENAI_STREAM_PATCHED
    if _CREWAI_OPENAI_STREAM_PATCHED:
        return

    try:
        from crewai.events.types.llm_events import LLMCallType
        from crewai.llms.providers.openai.completion import OpenAICompletion
    except Exception:
        return

    if getattr(OpenAICompletion, "_vitoom_streaming_tool_calls_patched", False):
        _CREWAI_OPENAI_STREAM_PATCHED = True
        return

    def _format_streamed_tool_calls(tool_calls_by_index: Dict[int, Dict[str, Any]]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for _idx, call_data in sorted(tool_calls_by_index.items(), key=lambda item: item[0]):
            function_name = str(call_data.get("name") or "").strip()
            arguments = str(call_data.get("arguments") or "")
            if not function_name or not arguments:
                continue
            formatted.append(
                {
                    "id": call_data.get("id"),
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments,
                    },
                    "index": int(call_data.get("index") or 0),
                }
            )
        return formatted

    def _handle_streaming_completion_patched(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> Any:
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}

        if response_model:
            return OpenAICompletion._vitoom_original_handle_streaming_completion(
                self,
                params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        stream = self.client.chat.completions.create(**params)
        usage_data = {"total_tokens": 0}

        for completion_chunk in stream:
            response_id_stream = completion_chunk.id if hasattr(completion_chunk, "id") else None

            if hasattr(completion_chunk, "usage") and completion_chunk.usage:
                usage_data = self._extract_openai_token_usage(completion_chunk)
                continue

            if not completion_chunk.choices:
                continue

            choice = completion_chunk.choices[0]
            chunk_delta = choice.delta

            if chunk_delta.content:
                full_response += chunk_delta.content
                self._emit_stream_chunk_event(
                    chunk=chunk_delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id_stream,
                )

            if chunk_delta.tool_calls:
                for tool_call in chunk_delta.tool_calls:
                    tool_index = tool_call.index if tool_call.index is not None else 0
                    if tool_index not in tool_calls:
                        tool_calls[tool_index] = {
                            "id": tool_call.id,
                            "name": "",
                            "arguments": "",
                            "index": tool_index,
                        }
                    elif tool_call.id and not tool_calls[tool_index]["id"]:
                        tool_calls[tool_index]["id"] = tool_call.id

                    if tool_call.function and tool_call.function.name:
                        tool_calls[tool_index]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_calls[tool_index]["arguments"] += tool_call.function.arguments

                    self._emit_stream_chunk_event(
                        chunk=tool_call.function.arguments if tool_call.function and tool_call.function.arguments else "",
                        from_task=from_task,
                        from_agent=from_agent,
                        tool_call={
                            "id": tool_calls[tool_index]["id"],
                            "function": {
                                "name": tool_calls[tool_index]["name"],
                                "arguments": tool_calls[tool_index]["arguments"],
                            },
                            "type": "function",
                            "index": tool_calls[tool_index]["index"],
                        },
                        call_type=LLMCallType.TOOL_CALL,
                        response_id=response_id_stream,
                    )

        self._track_token_usage_internal(usage_data)

        if tool_calls and not available_functions:
            formatted_tool_calls = _format_streamed_tool_calls(tool_calls)
            if formatted_tool_calls:
                self._emit_call_completed_event(
                    response=formatted_tool_calls,
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )
                return formatted_tool_calls

        if tool_calls and available_functions:
            for call_data in tool_calls.values():
                function_name = call_data["name"]
                arguments = call_data["arguments"]

                if not function_name or not arguments:
                    continue

                if function_name not in available_functions:
                    logging.warning(
                        "Function '%s' not found in available functions",
                        function_name,
                    )
                    continue

                try:
                    function_args = json.loads(arguments)
                except json.JSONDecodeError as e:
                    logging.error("Failed to parse streamed tool arguments: %s", e)
                    continue

                result = self._handle_tool_execution(
                    function_name=function_name,
                    function_args=function_args,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                if result is not None:
                    return result

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
        )

        return self._invoke_after_llm_call_hooks(params["messages"], full_response, from_agent)

    async def _ahandle_streaming_completion_patched(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> Any:
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}

        if response_model:
            return await OpenAICompletion._vitoom_original_ahandle_streaming_completion(
                self,
                params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        stream = await self.async_client.chat.completions.create(**params)
        usage_data = {"total_tokens": 0}

        async for chunk in stream:
            response_id_stream = chunk.id if hasattr(chunk, "id") else None

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = self._extract_openai_token_usage(chunk)
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            chunk_delta = choice.delta

            if chunk_delta.content:
                full_response += chunk_delta.content
                self._emit_stream_chunk_event(
                    chunk=chunk_delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id_stream,
                )

            if chunk_delta.tool_calls:
                for tool_call in chunk_delta.tool_calls:
                    tool_index = tool_call.index if tool_call.index is not None else 0
                    if tool_index not in tool_calls:
                        tool_calls[tool_index] = {
                            "id": tool_call.id,
                            "name": "",
                            "arguments": "",
                            "index": tool_index,
                        }
                    elif tool_call.id and not tool_calls[tool_index]["id"]:
                        tool_calls[tool_index]["id"] = tool_call.id

                    if tool_call.function and tool_call.function.name:
                        tool_calls[tool_index]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_calls[tool_index]["arguments"] += tool_call.function.arguments

                    self._emit_stream_chunk_event(
                        chunk=tool_call.function.arguments if tool_call.function and tool_call.function.arguments else "",
                        from_task=from_task,
                        from_agent=from_agent,
                        tool_call={
                            "id": tool_calls[tool_index]["id"],
                            "function": {
                                "name": tool_calls[tool_index]["name"],
                                "arguments": tool_calls[tool_index]["arguments"],
                            },
                            "type": "function",
                            "index": tool_calls[tool_index]["index"],
                        },
                        call_type=LLMCallType.TOOL_CALL,
                        response_id=response_id_stream,
                    )

        self._track_token_usage_internal(usage_data)

        if tool_calls and not available_functions:
            formatted_tool_calls = _format_streamed_tool_calls(tool_calls)
            if formatted_tool_calls:
                self._emit_call_completed_event(
                    response=formatted_tool_calls,
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )
                return formatted_tool_calls

        if tool_calls and available_functions:
            for call_data in tool_calls.values():
                function_name = call_data["name"]
                arguments = call_data["arguments"]

                if not function_name or not arguments:
                    continue

                if function_name not in available_functions:
                    logging.warning(
                        "Function '%s' not found in available functions",
                        function_name,
                    )
                    continue

                try:
                    function_args = json.loads(arguments)
                except json.JSONDecodeError as e:
                    logging.error("Failed to parse streamed tool arguments: %s", e)
                    continue

                result = self._handle_tool_execution(
                    function_name=function_name,
                    function_args=function_args,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                if result is not None:
                    return result

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
        )

        return full_response

    OpenAICompletion._vitoom_original_handle_streaming_completion = OpenAICompletion._handle_streaming_completion
    OpenAICompletion._vitoom_original_ahandle_streaming_completion = OpenAICompletion._ahandle_streaming_completion
    OpenAICompletion._handle_streaming_completion = _handle_streaming_completion_patched
    OpenAICompletion._ahandle_streaming_completion = _ahandle_streaming_completion_patched
    OpenAICompletion._vitoom_streaming_tool_calls_patched = True
    _CREWAI_OPENAI_STREAM_PATCHED = True


def build_crewai_llm(
    *,
    preferred_model_name: Optional[str] = None,
    effective_user_id: Optional[str] = None,
    stream: bool = True,
) -> Any:
    """构建给 CrewAI 使用的 LLM 实例。

    Vitoom 的 ``/v1/chat/completions`` 端点已原生支持 OpenAI ``tools`` / ``tool_choice``
    / ``tool_calls`` 协议（见 ``backend/api/openai/routes.py``）。CrewAI 1.9.x 的
    ``_invoke_loop`` 会走 ``_invoke_loop_native_tools`` 分支——这是我们想要的路径。
    因此这里**不再**对 ``supports_function_calling`` 做任何手脚，保持 CrewAI 默认行为。

    ``stream=True`` 时让 CrewAI 走流式分支，外层通过订阅
    ``LLMStreamChunkEvent`` 把 token 级 delta 喂给统一会话 WS。
    """
    try:
        from crewai import LLM
    except Exception as e:
        raise RuntimeError("crewai is required to build agent llm") from e

    _patch_crewai_openai_streaming_tool_calls()

    default_headers = None
    normalized_user_id = str(effective_user_id or "").strip()
    if normalized_user_id:
        default_headers = {
            get_agent_effective_user_header_name(): normalized_user_id,
        }

    return LLM(
        model=resolve_agent_llm_model_name(preferred_model_name),
        base_url=get_agent_llm_base_url(),
        api_key=get_agent_internal_auth_token(),
        timeout=get_agent_llm_timeout_seconds(),
        default_headers=default_headers,
        stream=stream,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            # 请求流式 usage 收尾块，便于上层在真流式模式下仍能拿到 token 统计。
            "stream_options": {"include_usage": True},
        },
    )
