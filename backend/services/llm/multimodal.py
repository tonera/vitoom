"""多模态 Chat Completion 共享 helper。

为 agent 工具（如 analyze_media）提供一次性、非流式的多模态 LLM 调用能力。
实现上直接复用本进程内已有的 ``/v1/chat/completions`` OpenAI 兼容接口，保持
所有 session / 观测 / 限流逻辑在同一处维护。

调用路径：Agent tool → run_multimodal_completion → HTTP(localhost)
   → /v1/chat/completions → inference WS session → 底层多模态模型
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from backend.services.agent.llm import resolve_agent_llm_model_name
from backend.services.agent.settings import (
    get_agent_effective_user_header_name,
    get_agent_internal_auth_token,
    get_agent_llm_base_url,
    get_agent_llm_timeout_seconds,
)

logger = logging.getLogger(__name__)


class MultimodalCompletionError(RuntimeError):
    """运行多模态 completion 失败时抛出。"""


def run_multimodal_completion(
    *,
    user_id: str,
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """同步发起一次多模态 chat completion。

    Args:
        user_id: 有效用户 id，会通过 X-Vitoom-Effective-User-Id 头注入后端。
        messages: OpenAI content-parts 格式的消息数组。例如::

            [
                {"role": "user", "content": [
                    {"type": "text", "text": "这张图片里有什么?"},
                    {"type": "image_url", "image_url": {"url": "https://.../cat.png"}},
                ]},
            ]

        model: 可选显式模型名，缺省取 agent 默认模型。
        temperature / max_tokens: OpenAI 兼容参数。
        timeout_seconds: HTTP 超时，缺省取 ``agents.llm.timeout_seconds``。

    Returns:
        ``{"content", "finish_reason", "usage", "model"}`` 字典。

    Raises:
        MultimodalCompletionError: 请求失败或响应解析异常。
    """

    user_id_str = str(user_id or "").strip()
    if not user_id_str:
        raise MultimodalCompletionError("user_id is required to run multimodal completion")
    if not messages:
        raise MultimodalCompletionError("messages are required to run multimodal completion")

    resolved_model = resolve_agent_llm_model_name(model)
    base_url = get_agent_llm_base_url().rstrip("/")
    url = f"{base_url}/chat/completions"

    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    headers = {
        "Authorization": f"Bearer {get_agent_internal_auth_token()}",
        get_agent_effective_user_header_name(): user_id_str,
        "Content-Type": "application/json",
    }

    timeout = float(timeout_seconds) if timeout_seconds else get_agent_llm_timeout_seconds()

    logger.info(
        "multimodal completion -> %s model=%s user=%s msgs=%d",
        url,
        resolved_model,
        user_id_str,
        len(messages),
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise MultimodalCompletionError(f"multimodal completion request failed: {exc}") from exc

    if response.status_code >= 400:
        snippet = response.text[:400]
        raise MultimodalCompletionError(
            f"multimodal completion returned {response.status_code}: {snippet}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise MultimodalCompletionError(f"multimodal completion response not JSON: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise MultimodalCompletionError("multimodal completion returned no choices")

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = str(message.get("content") or "").strip()
    finish_reason = str(first.get("finish_reason") or "stop")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    returned_model = str(data.get("model") or resolved_model)

    logger.info(
        "multimodal completion <- finish_reason=%s chars=%d usage=%s",
        finish_reason,
        len(content),
        bool(usage),
    )

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "model": returned_model,
    }
