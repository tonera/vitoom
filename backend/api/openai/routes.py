from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from fastapi.security import HTTPAuthorizationCredentials

from backend.auth import get_current_user_id_or_api_key, security
from backend.database import Model, User
from backend.models.service import get_model_service
from backend.services.agent.settings import (
    get_agent_effective_user_header_name,
    get_agent_internal_auth_token,
    get_agent_internal_user_id,
)
from backend.services.chat.router import (
    DispatchSelectionError,
    DispatchSpec,
    get_dispatch_router,
)
from backend.utils import utc_now
from backend.websocket.manager import get_websocket_manager

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

_SESSION_TIMEOUT_SECONDS = 120.0


async def _get_openai_request_user_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    internal_token = get_agent_internal_auth_token()
    token = str(credentials.credentials or "").strip() if credentials else ""
    if internal_token and token == internal_token:
        return get_agent_internal_user_id()
    return await get_current_user_id_or_api_key(request, credentials)


def _resolve_effective_user_id(
    request: Request,
    *,
    authenticated_user_id: str,
) -> str:
    internal_user_id = get_agent_internal_user_id()
    if authenticated_user_id != internal_user_id:
        return authenticated_user_id

    header_name = get_agent_effective_user_header_name()
    effective_user_id = str(request.headers.get(header_name, "") or "").strip()
    if not effective_user_id:
        raise HTTPException(
            status_code=400,
            detail=f"{header_name} is required for internal agent LLM calls",
        )
    return effective_user_id


def _model_created_unix(created_at: Any) -> int:
    if created_at is None:
        return 0
    if isinstance(created_at, (int, float)):
        return int(created_at)
    s = str(created_at).strip()
    if not s:
        return 0
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp())
    except Exception:
        return 0


def _openai_model_object_from_row(
    model_row: Dict[str, Any],
    *,
    model_key: Optional[str] = None,
    root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Shape aligned with OpenAI /v1/models entries (some clients require permission/root)."""
    load_name = str(model_row.get("load_name") or "").strip()
    if not load_name:
        return None
    object_id = str(model_key or load_name).strip()
    if not object_id:
        return None
    created = _model_created_unix(model_row.get("created_at"))
    return {
        "id": object_id,
        "object": "model",
        "created": created,
        "owned_by": "vitoom",
        "permission": [],
        "root": str(root or load_name).strip() or load_name,
    }


def _list_text_models() -> List[Dict[str, Any]]:
    service = get_model_service()
    models, _total = service.list_models(
        modality="text",
        limit=500,
        offset=0,
    )
    return [m for m in models if isinstance(m, dict)]


def _resolve_openai_model_row(requested_model: str) -> tuple[str, Dict[str, Any]]:
    requested_name = str(requested_model or "").strip()
    if not requested_name:
        raise HTTPException(status_code=400, detail="model is required")

    direct_match = Model.get_by_load_name(requested_name)
    if direct_match:
        resolved_name = str(direct_match.get("load_name") or requested_name).strip()
        return resolved_name, direct_match

    raise HTTPException(status_code=400, detail=f"Invalid model: {requested_name}")


@router.get("/models")
async def list_openai_models(user_id: str = Depends(_get_openai_request_user_id)):
    """
    OpenAI-compatible model list for clients that call GET /v1/models (e.g. Cursor, IDE plugins).

    Returns `id` equal to `load_name` from `model_catalog`, matching POST /v1/chat/completions `model`.
    """
    del user_id
    models = _list_text_models()
    data: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for m in models:
        item = _openai_model_object_from_row(m)
        if item is None or item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        data.append(item)
    return {"object": "list", "data": data}


@router.get("/models/{model_key:path}")
async def get_openai_model(model_key: str, user_id: str = Depends(_get_openai_request_user_id)):
    del user_id
    resolved_name, model_row = _resolve_openai_model_row(model_key)
    item = _openai_model_object_from_row(model_row, model_key=model_key, root=resolved_name)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_key}")
    return item


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model name")
    messages: List[Dict[str, Any]] = Field(..., description="OpenAI-style message list")
    temperature: Optional[float] = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)
    top_p: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=None)
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    stream_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenAI stream_options, e.g. {'include_usage': true}",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="OpenAI function-calling tool schemas, e.g. [{'type':'function','function':{...}}]",
    )
    tool_choice: Optional[Any] = Field(
        default=None,
        description="auto / none / required / {'type':'function','function':{'name':...}}",
    )


def _deep_merge_dict(base: Dict[str, Any] | None, override: Dict[str, Any] | None) -> Dict[str, Any]:
    left = dict(base or {})
    right = dict(override or {})
    merged: Dict[str, Any] = dict(left)
    for key, value in right.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(current, value)
        else:
            merged[key] = value
    return merged


def _normalize_content_part(part: Any) -> Dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        raise HTTPException(status_code=400, detail="Invalid message content part")

    raw_type = str(part.get("type") or "").strip().lower()
    if raw_type in {"", "text", "input_text"}:
        return {"type": "text", "text": str(part.get("text") or part.get("input_text") or "")}
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
            raise HTTPException(status_code=400, detail="image_url item requires a non-empty url")
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
            raise HTTPException(status_code=400, detail="video_url item requires a non-empty url")
        return {"type": "video_url", "video_url": {"url": url}}
    raise HTTPException(status_code=400, detail=f"Unsupported content part type: {raw_type or 'unknown'}")


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
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif arguments is None:
            arguments = ""
        else:
            arguments = str(arguments)
        if not name:
            # OpenAI 协议允许过渡态缺 name，但我们只接受完整可执行的 tool_call。
            continue
        normalized.append(
            {
                "id": call_id,
                "type": call_type,
                "function": {"name": name, "arguments": arguments},
            }
        )
    return normalized


def _normalize_chat_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    normalized: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="each message must be an object")
        role = str(item.get("role") or "").strip()
        if role not in {"system", "user", "assistant", "tool"}:
            raise HTTPException(status_code=400, detail=f"Unsupported message role: {role or 'unknown'}")
        content = item.get("content")
        if isinstance(content, list):
            normalized_content: Any = [_normalize_content_part(part) for part in content]
        elif isinstance(content, dict):
            normalized_content = [_normalize_content_part(content)]
        else:
            normalized_content = str(content or "")

        message: Dict[str, Any] = {"role": role, "content": normalized_content}

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


def _normalize_request_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if tools in (None, ""):
        return None
    if not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="tools must be a list when provided")
    normalized: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise HTTPException(status_code=400, detail="each tool must be an object")
        tool_type = str(tool.get("type") or "function").strip() or "function"
        if tool_type != "function":
            # 目前仅支持 function 类型；其他类型（如 code_interpreter）后续按需扩展。
            raise HTTPException(status_code=400, detail=f"Unsupported tool type: {tool_type}")
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if not function:
            raise HTTPException(status_code=400, detail="tool.function object is required")
        name = str(function.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="tool.function.name is required")
        entry: Dict[str, Any] = {
            "type": "function",
            "function": {"name": name},
        }
        description = function.get("description")
        if description not in (None, ""):
            entry["function"]["description"] = str(description)
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            entry["function"]["parameters"] = parameters
        normalized.append(entry)
    return normalized or None


def _resolve_text_model(load_name: str) -> tuple[str, str, Dict[str, Any]]:
    # family 只从 `model_catalog` 表字段读取，不做任何基于加载名的字符串猜测。
    # 这样 DB 里怎么配就是怎么路由，新增型号只改数据不改代码；也避免了
    # "Qwen3-Audio" 名字里含 'qwen' 被误判成 Qwen-text 之类的静默串台。
    resolved_load_name, model_dict = _resolve_openai_model_row(load_name)
    family = str(model_dict.get("family") or "").strip()
    if not family:
        raise HTTPException(
            status_code=400,
            detail=(
                f"model_catalog.family is empty for load_name={resolved_load_name}; "
                "please populate the `family` column (e.g. 'qwen.text', 'gemma')."
            ),
        )
    runtime_config = model_dict.get("runtime_config") if isinstance(model_dict.get("runtime_config"), dict) else {}
    return resolved_load_name, family, dict(runtime_config)


def _build_model_cfg(
    *,
    base_model_cfg: Dict[str, Any],
    messages: List[Dict[str, Any]],
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    # 注意：vLLM 的 `limit_mm_per_prompt` 属于 engine 构造期常量，改动即意味着重新
    # 初始化 AsyncLLMEngine（推理侧的 bundle 缓存键也包含 engine_kwargs）。
    # 因此这里不能再"按当前 messages 里的图片/视频数量"动态注入——否则纯文本轮和
    # 多模态轮会命中不同的 cache_key，导致 text 服务在第一次看到图片时尝试拉起
    # 第二个 vLLM engine，直接 OOM。如需开启多模态能力，请在 runtime_config
    # 里把 runtime.engine_kwargs.limit_mm_per_prompt 预置好（例如 {"image": 8}）。
    del messages
    runtime_override: Dict[str, Any] = {}
    if isinstance(extra_body.get("runtime"), dict):
        runtime_override = {"runtime": dict(extra_body["runtime"])}

    return _deep_merge_dict(base_model_cfg, runtime_override)


def _build_openai_usage(event: Dict[str, Any]) -> Optional[Dict[str, int]]:
    prompt_tokens = event.get("prompt_tokens")
    completion_tokens = event.get("output_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return None
    prompt_value = int(prompt_tokens or 0)
    completion_value = int(completion_tokens or 0)
    return {
        "prompt_tokens": prompt_value,
        "completion_tokens": completion_value,
        "total_tokens": prompt_value + completion_value,
    }


def _resolve_stream_include_usage(
    *,
    stream_options: Optional[Dict[str, Any]],
    extra_body: Dict[str, Any],
) -> bool:
    merged_options: Dict[str, Any] = {}
    if isinstance(extra_body.get("stream_options"), dict):
        merged_options.update(dict(extra_body.get("stream_options") or {}))
    if isinstance(stream_options, dict):
        merged_options.update(dict(stream_options or {}))
    return bool(merged_options.get("include_usage"))


def _chunk_payload(
    *,
    completion_id: str,
    created_at: int,
    response_model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str],
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": response_model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _usage_chunk_payload(
    *,
    completion_id: str,
    created_at: int,
    response_model: str,
    usage: Dict[str, int],
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": response_model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _format_openai_tool_calls(raw_tool_calls: Any) -> List[Dict[str, Any]]:
    formatted_tool_calls: List[Dict[str, Any]] = []
    if not isinstance(raw_tool_calls, list):
        return formatted_tool_calls
    for call in raw_tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        formatted_tool_calls.append(
            {
                "id": str(call.get("id") or ""),
                "type": str(call.get("type") or "function"),
                "function": {
                    "name": name,
                    "arguments": str(function.get("arguments") or ""),
                },
            }
        )
    return formatted_tool_calls


def _tool_calls_message_chunk_payload(
    *,
    completion_id: str,
    created_at: int,
    response_model: str,
    tool_calls: List[Dict[str, Any]],
    answer_text: Optional[str] = None,
    finish_reason: str = "tool_calls",
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": response_model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                # 给标准 SSE chunk 额外带一个完整 assistant message，
                # 兼容 CrewAI/litellm 在 stream=True + tool_calls-only 场景下
                # 仅从最后一帧读取 message.tool_calls 的实现。
                "message": {
                    "role": "assistant",
                    "content": answer_text if answer_text else None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _wait_for_session_event(queue: asyncio.Queue, *, timeout_seconds: float) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout_seconds)
    except asyncio.TimeoutError as e:
        raise HTTPException(status_code=504, detail="Timed out while waiting for text service response") from e


async def _dispatch_to_session(
    *,
    session: Dict[str, Any],
    client_message_type: str,
    service_message: Dict[str, Any],
) -> None:
    """把一条会话消息送到绑定的推理服务。

    统一会话重构后，旧会话选服逻辑已下线；这里改为按
    ``session.metadata.load_name`` 找一台正在运行的推理服务，直接通过
    WebSocketManager 投递。
    """
    _ = client_message_type
    ws_manager = get_websocket_manager()
    load_name = ""
    meta = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    if isinstance(meta, dict):
        load_name = str(meta.get("load_name") or "").strip()
    if not load_name:
        raise HTTPException(status_code=400, detail="session.metadata.load_name is required")
    connected_service_ids = await ws_manager.get_connected_inference_service_ids()
    try:
        service = get_dispatch_router().pick_service(
            DispatchSpec(
                service_type="text",
                reason="service_type matching 'text'",
                load_name=load_name,
            ),
            connected_service_ids=connected_service_ids,
        )
    except DispatchSelectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    service_id = str(service.get("id") or "")
    if not service_id:
        raise HTTPException(status_code=503, detail="No connected text inference service available")
    if not await ws_manager.send_message_to_inference_service(service_id, service_message):
        raise HTTPException(status_code=503, detail="No connected text inference service available")


async def _close_session_safely(session: Dict[str, Any], *, session_id: str, seq: int) -> None:
    try:
        await _dispatch_to_session(
            session=session,
            client_message_type="session_close",
            service_message={
                "type": "session_close",
                "session_id": session_id,
                "seq": seq,
                "timestamp": utc_now().isoformat(),
            },
        )
    except Exception:
        # 路由层已经无状态；这里只是兜底失败：忽略即可
        pass


async def _open_session_and_wait_ready(
    *,
    session: Dict[str, Any],
    session_id: str,
    queue: asyncio.Queue,
    open_message: Dict[str, Any],
) -> None:
    await _dispatch_to_session(
        session=session,
        client_message_type="session_open",
        service_message=open_message,
    )
    while True:
        event = await _wait_for_session_event(queue, timeout_seconds=_SESSION_TIMEOUT_SECONDS)
        message_type = str(event.get("type") or "")
        if message_type == "session_ready":
            return
        if message_type == "session_error":
            raise HTTPException(status_code=502, detail=str(event.get("error") or "text service error"))


@router.post("/chat/completions")
async def create_chat_completion(
    body: ChatCompletionRequest,
    http_request: Request,
    user_id: str = Depends(_get_openai_request_user_id),
):
    effective_user_id = _resolve_effective_user_id(request=http_request, authenticated_user_id=user_id)
    if not User.get_by_id(effective_user_id):
        raise HTTPException(status_code=400, detail=f"Invalid effective user id: {effective_user_id}")
    normalized_messages = _normalize_chat_messages(body.messages)
    load_name, family, base_runtime_config = _resolve_text_model(body.model)
    extra_body = dict(body.extra_body or {})
    runtime_config = _build_model_cfg(
        base_model_cfg=base_runtime_config,
        messages=normalized_messages,
        extra_body=extra_body,
    )

    temperature = body.temperature
    max_tokens = body.max_tokens
    top_p = body.top_p if body.top_p is not None else extra_body.get("top_p")
    top_k = body.top_k if body.top_k is not None else extra_body.get("top_k")
    presence_penalty = (
        body.presence_penalty if body.presence_penalty is not None else extra_body.get("presence_penalty")
    )
    frequency_penalty = (
        body.frequency_penalty if body.frequency_penalty is not None else extra_body.get("frequency_penalty")
    )
    mm_processor_kwargs = extra_body.get("mm_processor_kwargs") if isinstance(extra_body.get("mm_processor_kwargs"), dict) else None
    normalized_tools = _normalize_request_tools(body.tools)
    tool_choice = body.tool_choice
    include_stream_usage = _resolve_stream_include_usage(
        stream_options=body.stream_options,
        extra_body=extra_body,
    )

    # 统一会话重构后：OpenAI 兼容入口不再维护旧实时会话表记录；
    # 每次请求用一个临时 session_id 串起 register_session_subscriber →
    # session_open / session_text_input / session_close 的收发链路。
    session_id = uuid4().hex
    session = {
        "id": session_id,
        "user_id": effective_user_id,
        "metadata": {
            "load_name": load_name,
            "family": family,
            "effective_user_id": effective_user_id,
            "authenticated_user_id": user_id,
        },
    }
    ws_manager = get_websocket_manager()
    queue = await ws_manager.register_session_subscriber(session_id)

    created_at = int(time.time())
    completion_id = f"chatcmpl_{uuid4().hex}"
    open_message = {
        "type": "session_open",
        "session_id": session_id,
        "seq": 1,
        "load_name": load_name,
        "family": family,
        "runtime_config": runtime_config,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "mm_processor_kwargs": mm_processor_kwargs or {},
        "tools": normalized_tools,
        "tool_choice": tool_choice,
        "timestamp": utc_now().isoformat(),
    }
    input_message = {
        "type": "session_text_input",
        "session_id": session_id,
        "seq": 2,
        "load_name": load_name,
        "family": family,
        "runtime_config": runtime_config,
        "messages": normalized_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "mm_processor_kwargs": mm_processor_kwargs or {},
        "tools": normalized_tools,
        "tool_choice": tool_choice,
        "timestamp": utc_now().isoformat(),
    }

    stream_response_created = False
    try:
        await _open_session_and_wait_ready(
            session=session,
            session_id=session_id,
            queue=queue,
            open_message=open_message,
        )

        if body.stream:
            async def event_stream() -> AsyncIterator[str]:
                answer_parts: List[str] = []
                next_tool_call_index = 0
                try:
                    yield _chunk_payload(
                        completion_id=completion_id,
                        created_at=created_at,
                        response_model=body.model,
                        delta={"role": "assistant"},
                        finish_reason=None,
                    )
                    await _dispatch_to_session(
                        session=session,
                        client_message_type="input_text",
                        service_message=input_message,
                    )
                    while True:
                        event = await _wait_for_session_event(queue, timeout_seconds=_SESSION_TIMEOUT_SECONDS)
                        message_type = str(event.get("type") or "")
                        if message_type == "session_error":
                            raise RuntimeError(str(event.get("error") or "text service error"))
                        if message_type != "llm_text_delta":
                            continue

                        delta_text = str(event.get("delta") or "")
                        if delta_text:
                            answer_parts.append(delta_text)
                            yield _chunk_payload(
                                completion_id=completion_id,
                                created_at=created_at,
                                response_model=body.model,
                                delta={"content": delta_text},
                                finish_reason=None,
                            )

                        tool_calls_delta = event.get("tool_calls_delta")
                        if isinstance(tool_calls_delta, list) and tool_calls_delta:
                            # 我们的 session_runtime 按 <tool_call> 边界一次性吐完整结构，
                            # 流式里就直接把整段 arguments 作为单个增量发出——OpenAI SSE 规范
                            # 允许 delta.tool_calls 一次性给出完整字段，客户端会按 index 累加。
                            formatted_delta_calls = []
                            for call in tool_calls_delta:
                                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                                formatted_delta_calls.append(
                                    {
                                        "index": next_tool_call_index,
                                        "id": str(call.get("id") or ""),
                                        "type": str(call.get("type") or "function"),
                                        "function": {
                                            "name": str(function.get("name") or ""),
                                            "arguments": str(function.get("arguments") or ""),
                                        },
                                    }
                                )
                                next_tool_call_index += 1
                            yield _chunk_payload(
                                completion_id=completion_id,
                                created_at=created_at,
                                response_model=body.model,
                                delta={"tool_calls": formatted_delta_calls},
                                finish_reason=None,
                            )

                        if bool(event.get("is_final")):
                            usage = _build_openai_usage(event)
                            final_tool_calls = _format_openai_tool_calls(event.get("tool_calls"))
                            finish_reason = str(event.get("finish_reason") or "stop")
                            if final_tool_calls:
                                if include_stream_usage and usage is not None:
                                    yield _usage_chunk_payload(
                                        completion_id=completion_id,
                                        created_at=created_at,
                                        response_model=body.model,
                                        usage=usage,
                                    )
                                yield _tool_calls_message_chunk_payload(
                                    completion_id=completion_id,
                                    created_at=created_at,
                                    response_model=body.model,
                                    tool_calls=final_tool_calls,
                                    answer_text="".join(answer_parts),
                                    finish_reason=finish_reason,
                                )
                            else:
                                yield _chunk_payload(
                                    completion_id=completion_id,
                                    created_at=created_at,
                                    response_model=body.model,
                                    delta={},
                                    finish_reason=finish_reason,
                                )
                            if (not final_tool_calls) and include_stream_usage and usage is not None:
                                yield _usage_chunk_payload(
                                    completion_id=completion_id,
                                    created_at=created_at,
                                    response_model=body.model,
                                    usage=usage,
                                )
                            yield "data: [DONE]\n\n"
                            return
                finally:
                    await _close_session_safely(session, session_id=session_id, seq=3)
                    await ws_manager.unregister_session_subscriber(session_id, queue)

            stream_response_created = True
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        await _dispatch_to_session(
            session=session,
            client_message_type="input_text",
            service_message=input_message,
        )
        answer_parts: List[str] = []
        final_event: Dict[str, Any] = {}
        while True:
            event = await _wait_for_session_event(queue, timeout_seconds=_SESSION_TIMEOUT_SECONDS)
            message_type = str(event.get("type") or "")
            if message_type == "session_error":
                raise HTTPException(status_code=502, detail=str(event.get("error") or "text service error"))
            if message_type != "llm_text_delta":
                continue
            delta = str(event.get("delta") or "")
            if delta:
                answer_parts.append(delta)
            if bool(event.get("is_final")):
                final_event = event
                break

        usage = _build_openai_usage(final_event)

        formatted_tool_calls = _format_openai_tool_calls(final_event.get("tool_calls"))

        assistant_message: Dict[str, Any] = {"role": "assistant"}
        answer_text = "".join(answer_parts)
        if formatted_tool_calls:
            # OpenAI 规定 tool_calls 场景下 content 允许为 null；保留任何模型在 tool_call
            # 之外同时输出的解释文字，没有就 null，避免客户端把空串当作实际回答。
            assistant_message["content"] = answer_text if answer_text else None
            assistant_message["tool_calls"] = formatted_tool_calls
            finish_reason = "tool_calls"
        else:
            assistant_message["content"] = answer_text
            finish_reason = str(final_event.get("finish_reason") or "stop")

        response_payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_at,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            response_payload["usage"] = usage
        return response_payload
    finally:
        if (not body.stream) or (body.stream and not stream_response_created):
            await _close_session_safely(session, session_id=session_id, seq=3)
            await ws_manager.unregister_session_subscriber(session_id, queue)
