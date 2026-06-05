"""Shared planner primitives for business query tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx  # type: ignore[import-not-found]

from backend.services.agent.llm import resolve_agent_llm_model_name
from backend.services.agent.settings import (
    get_agent_effective_user_header_name,
    get_agent_internal_auth_token,
    get_agent_llm_base_url,
    get_agent_llm_timeout_seconds,
)


def run_agent_planner_completion(
    messages: List[Dict[str, str]],
    *,
    user_id: str = "agent-system",
    timeout_seconds: Optional[float] = None,
    model_name: str = "",
    error_label: str = "business query planner",
) -> str:
    resolved_model = resolve_agent_llm_model_name(str(model_name or "").strip() or None)
    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {get_agent_internal_auth_token()}",
        "Content-Type": "application/json",
    }
    if user_id:
        headers[get_agent_effective_user_header_name()] = user_id
    timeout = timeout_seconds if timeout_seconds and timeout_seconds > 0 else get_agent_llm_timeout_seconds()
    url = get_agent_llm_base_url().rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
    if response.status_code >= 400:
        raise RuntimeError(f"{error_label} LLM returned {response.status_code}: {response.text[:400]}")
    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = str(message.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"{error_label} LLM returned empty content")
    return content


def parse_llm_json_object(raw: str, *, error_message: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(error_message)
    return parsed
