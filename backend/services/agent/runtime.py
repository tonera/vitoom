from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.database import Agent, AgentRun, Task
from backend.queue import get_task_queue
from backend.utils import generate_uuid

from pydantic import ValidationError

from .events import record_run_cancelled
from .presets import ensure_default_agent_presets
from .settings import is_agents_enabled, is_openclaw_enabled
from .types import AgentCommand, RuntimeConfigSchema


class AgentRuntimeError(RuntimeError):
    """Agent 运行时异常。"""


class AgentValidationError(AgentRuntimeError):
    """Agent 请求校验异常。"""


def _sanitize_runtime_config(runtime_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """使用 RuntimeConfigSchema 做 Pydantic 校验 + 兼容额外字段。

    - 已知字段走 schema：priority / process / max_tools_per_run /
      tool_allowlist / require_approval_for
    - 未知字段保持透传（extra="allow"），例如 preset 自带的扩展键。
    - 非法 priority / process 等按 Pydantic 校验错误抛给上层。
    """
    raw = dict(runtime_config or {})
    try:
        schema = RuntimeConfigSchema.model_validate(raw)
    except ValidationError as e:
        raise AgentValidationError(f"Invalid runtime_config: {e.errors()}") from e

    merged: Dict[str, Any] = dict(raw)
    merged.update(schema.model_dump(exclude_none=False))
    if merged.get("tool_allowlist") is None:
        merged.pop("tool_allowlist", None)
    if merged.get("max_tools_per_run") is None:
        merged.pop("max_tools_per_run", None)
    return merged


def _get_active_agent(agent_id: str) -> Dict[str, Any]:
    agent = Agent.get_by_id(agent_id)
    if not agent:
        raise AgentValidationError("Agent not found")
    if str(agent.get("status") or "").strip().lower() != "active":
        raise AgentValidationError("Agent is not active")
    agent_type = str(agent.get("type") or "").strip().lower()
    if agent_type == "openclaw" and not is_openclaw_enabled():
        raise AgentValidationError("OpenClaw integration is disabled")
    return agent


def _get_runtime_defaults(agent_record: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(agent_record.get("config") or {})
    runtime_defaults = config.get("runtime_defaults")
    if isinstance(runtime_defaults, dict):
        return dict(runtime_defaults)
    return {}


async def create_agent_run(command: AgentCommand) -> Dict[str, Any]:
    """创建 AgentRun 并投递到现有任务队列。"""
    if not is_agents_enabled():
        raise AgentRuntimeError("Agent feature is disabled")
    if not command.user_id:
        raise AgentValidationError("user_id is required")
    if not command.agent_id:
        raise AgentValidationError("agent_id is required")
    if not command.message:
        raise AgentValidationError("message is required")

    ensure_default_agent_presets()
    agent_record = _get_active_agent(command.agent_id)

    run_id = generate_uuid()
    task_id = generate_uuid()
    runtime_config = _sanitize_runtime_config(
        {
            **_get_runtime_defaults(agent_record),
            **dict(command.runtime_config or {}),
        }
    )

    queue = get_task_queue()
    try:
        await queue.add_task(
            task_id=task_id,
            user_id=command.user_id,
            task_type="agent",
            prompt=command.message,
            params={
                "agent_run_id": run_id,
                "agent_id": command.agent_id,
                "source_type": command.source_type,
                "source_ref": command.source_ref,
                "runtime_config": runtime_config,
                "attachments": command.attachments,
                "context": command.context,
            },
            priority=int(runtime_config.get("priority", 5)),
            model_key=None,
        )
    except Exception as e:
        raise AgentRuntimeError(f"Failed to enqueue agent task: {e}") from e

    agent_run = AgentRun.create(
        id=run_id,
        user_id=command.user_id,
        agent_id=command.agent_id,
        task_id=task_id,
        source_type=command.source_type or "web",
        source_ref=command.source_ref,
        status="queued",
        input_payload=command.to_dict(),
        runtime_config=runtime_config,
        conversation_id=command.conversation_id,
        parent_run_id=command.parent_run_id,
        crew_tool_name=command.crew_tool_name,
    )
    if not agent_run:
        try:
            await queue.cancel_task(task_id)
        except Exception:
            Task.update(
                task_id,
                status="failed",
                error="Failed to create corresponding agent run",
                completed_at=datetime.utcnow(),
            )
        raise AgentRuntimeError("Failed to create agent run")

    return agent_run


def get_agent_run_for_user(run_id: str, user_id: str) -> Dict[str, Any]:
    agent_run = AgentRun.get_by_id(run_id)
    if not agent_run:
        raise AgentValidationError("Agent run not found")
    if agent_run["user_id"] != user_id:
        raise AgentValidationError("Permission denied")
    return agent_run


def list_agent_runs_for_user(
    user_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return AgentRun.list_by_user(user_id, limit=limit, offset=offset, status=status)


async def cancel_agent_run(run_id: str, user_id: str) -> Dict[str, Any]:
    agent_run = get_agent_run_for_user(run_id, user_id)
    if agent_run["status"] in {"completed", "failed", "cancelled"}:
        return agent_run

    queue = get_task_queue()
    cancelled = await queue.cancel_task(agent_run["task_id"])
    if cancelled:
        cancelled_at = datetime.utcnow()
        updated_run = AgentRun.update(
            run_id,
            status="cancelled",
            completed_at=cancelled_at,
            error_message=None,
        )
        if updated_run:
            record_run_cancelled(run_id, completed_at=cancelled_at)
            return updated_run
    return AgentRun.get_by_id(run_id) or agent_run
