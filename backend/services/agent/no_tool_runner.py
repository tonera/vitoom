from __future__ import annotations

import json
from dataclasses import dataclass, field
from string import Formatter
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx

from backend.services.agent.llm import resolve_agent_llm_model_name
from backend.services.agent.settings import (
    get_agent_effective_user_header_name,
    get_agent_internal_auth_token,
    get_agent_llm_base_url,
    get_agent_llm_timeout_seconds,
)
from backend.services.agent.specs import AgentSpec, TaskSpec
from backend.services.agent.types import AgentCommand


@dataclass
class NoToolLLMResult:
    """CrewOutput-like result for the no-tool fast path."""

    raw: str
    token_usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_template(template: str, inputs: Dict[str, Any]) -> str:
    if not template:
        return ""
    try:
        return template.format_map(_SafeFormatDict(inputs))
    except Exception:
        return template


def _collect_input_keys(text: str) -> set[str]:
    keys: set[str] = set()
    if not text:
        return keys
    formatter = Formatter()
    try:
        for _literal, field_name, _format_spec, _conversion in formatter.parse(text):
            if field_name:
                keys.add(str(field_name).split(".", 1)[0].split("[", 1)[0])
    except Exception:
        return set()
    return keys


def _build_inputs(command: AgentCommand) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {
        "message": command.message,
        "user_id": command.user_id,
        "agent_id": command.agent_id,
        "source_type": command.source_type,
    }
    inputs.update(dict(command.context or {}))
    return inputs


def _agent_system_block(spec: AgentSpec) -> str:
    lines = [
        f"Agent name: {spec.name}",
        f"Role: {spec.role}",
        f"Goal: {spec.goal}",
        f"Backstory: {spec.backstory}",
    ]
    if spec.memory:
        lines.append("Memory: enabled by the agent configuration.")
    return "\n".join(line for line in lines if line.strip())


def build_no_tool_messages(
    *,
    agent_specs: List[AgentSpec],
    task_specs: List[TaskSpec],
    command: AgentCommand,
) -> List[Dict[str, str]]:
    """Build an OpenAI-compatible prompt that preserves the Crew agent/task intent."""

    inputs = _build_inputs(command)
    resolved_agents = list(agent_specs or [])
    resolved_tasks = list(task_specs or [])

    system_parts: List[str] = [
        "No external tools are available for this run.",
        "Answer directly with the available conversation context. Do not mention tool availability or claim that you used tools.",
        # 锁定回复语种：防止纯英文 system prompt 让中文/日文 query 拿到英文回答。
        "Always respond in the same language as the latest user message.",
    ]
    if resolved_agents:
        system_parts.append("Agent configuration:")
        system_parts.extend(_agent_system_block(spec) for spec in resolved_agents)

    user_parts: List[str] = []
    for idx, task in enumerate(resolved_tasks, start=1):
        agent_name = str(task.agent_name or "").strip()
        title = f"Task {idx}"
        if agent_name:
            title += f" assigned to {agent_name}"
        user_parts.append(title + ":")
        description = _format_template(task.description, inputs).strip()
        if description:
            user_parts.append(description)
        expected_output = _format_template(task.expected_output, inputs).strip()
        if expected_output:
            user_parts.append("Expected output:\n" + expected_output)

    used_keys = set()
    for task in resolved_tasks:
        used_keys.update(_collect_input_keys(task.description))
        used_keys.update(_collect_input_keys(task.expected_output))
    if "message" not in used_keys and command.message:
        user_parts.append("User message:\n" + command.message)

    if not user_parts:
        user_parts.append(command.message)

    return [
        {"role": "system", "content": "\n\n".join(part for part in system_parts if part.strip())},
        {"role": "user", "content": "\n\n".join(part for part in user_parts if part.strip())},
    ]


def _completion_payload(
    *,
    messages: List[Dict[str, str]],
    command: AgentCommand,
    stream: bool,
) -> tuple[str, Dict[str, Any]]:
    runtime_config = dict(command.runtime_config or {})
    preferred_model_name = str(runtime_config.get("load_name") or "").strip() or None
    model_name = resolve_agent_llm_model_name(preferred_model_name)
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": bool(stream),
        # vLLM / vitoom OpenAI-compatible 后端识别的顶级扩展字段。
        # `extra_body` 是 OpenAI Python SDK 客户端层语法，
        # 直接 POST raw JSON 时不会被服务器解析——必须把字段平铺到顶级，
        # 否则 ``enable_thinking`` 不会生效，部分模型会把 ``<think>...</think>``
        # 链路推理直接当作普通 chunk 流给前端 / TTS。
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    for key in (
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = runtime_config.get(key)
        if value is not None:
            payload[key] = value
    return model_name, payload


def _headers(command: AgentCommand) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {get_agent_internal_auth_token()}",
        "Content-Type": "application/json",
    }
    user_id = str(command.user_id or "").strip()
    if user_id:
        headers[get_agent_effective_user_header_name()] = user_id
    return headers


def _iter_sse_payloads(lines: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for line in lines:
        text = str(line or "").strip()
        if not text or not text.startswith("data:"):
            continue
        data = text[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def run_no_tool_completion(
    *,
    agent_specs: List[AgentSpec],
    task_specs: List[TaskSpec],
    command: AgentCommand,
    stream: bool = False,
    on_llm_started: Optional[Callable[[], None]] = None,
    on_llm_chunk: Optional[Callable[[str], None]] = None,
) -> NoToolLLMResult:
    """Run a direct LLM completion for agent turns that have no selected tools."""

    messages = build_no_tool_messages(
        agent_specs=agent_specs,
        task_specs=task_specs,
        command=command,
    )
    model_name, payload = _completion_payload(messages=messages, command=command, stream=stream)
    url = get_agent_llm_base_url().rstrip("/") + "/chat/completions"
    timeout = get_agent_llm_timeout_seconds()

    if callable(on_llm_started):
        on_llm_started()

    if stream:
        parts: List[str] = []
        token_usage: Optional[Dict[str, Any]] = None
        finish_reason: Optional[str] = None
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload, headers=_headers(command)) as response:
                if response.status_code >= 400:
                    error_body = response.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"no-tool llm completion returned {response.status_code}: {error_body[:400]}"
                    )
                for event in _iter_sse_payloads(response.iter_lines()):
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        token_usage = dict(usage)
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    finish = choice.get("finish_reason")
                    if finish:
                        finish_reason = str(finish)
                    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    content = str(delta.get("content") or "")
                    if not content:
                        continue
                    parts.append(content)
                    if callable(on_llm_chunk):
                        on_llm_chunk(content)
        return NoToolLLMResult(
            raw="".join(parts).strip(),
            token_usage=token_usage,
            model=model_name,
            finish_reason=finish_reason,
            metadata={"mode": "no_tool_fast_path", "stream": True},
        )

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=_headers(command))
    if response.status_code >= 400:
        raise RuntimeError(f"no-tool llm completion returned {response.status_code}: {response.text[:400]}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"no-tool llm completion response is not JSON: {exc}") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = str(message.get("content") or "")
    usage = data.get("usage") if isinstance(data, dict) and isinstance(data.get("usage"), dict) else None
    finish_reason = str(first.get("finish_reason") or "") or None
    return NoToolLLMResult(
        raw=content.strip(),
        token_usage=dict(usage) if usage else None,
        model=str(data.get("model") or model_name) if isinstance(data, dict) else model_name,
        finish_reason=finish_reason,
        metadata={"mode": "no_tool_fast_path", "stream": False},
    )
