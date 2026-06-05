from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.auth import get_current_user_id
from backend.core.response import ok
from backend.services.agent import (
    AgentCommand,
    AgentRuntimeError,
    AgentValidationError,
    cancel_agent_run,
    create_agent_run,
    ensure_default_agent_presets,
    get_agent_or_raise,
    get_agent_run_for_user,
    list_agent_run_events,
    list_agent_runs_for_user,
    list_agents,
)

router = APIRouter(prefix="/v1/agents", tags=["Agents"])


class AgentRunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent_id: str = Field(..., description="Agent template ID")
    message: str = Field(..., description="User task instruction")
    source_type: str = Field(default="web", description="Task source")
    source_ref: Optional[str] = Field(default=None, description="Source-side reference ID")
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    runtime_config: Dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def list_agents_endpoint(
    status: Optional[str] = None,
    is_preset: Optional[bool] = None,
    _user_id: str = Depends(get_current_user_id),
):
    ensure_default_agent_presets()
    agents = list_agents(status=status, is_preset=is_preset)
    return ok(data={"agents": agents, "total": len(agents)}, msg="ok")


def _to_http_exception(exc: Exception) -> HTTPException:
    detail = str(exc) or exc.__class__.__name__
    if isinstance(exc, AgentValidationError):
        status_code = 400
        if "not found" in detail.lower():
            status_code = 404
        elif "permission denied" in detail.lower():
            status_code = 403
        return HTTPException(status_code=status_code, detail=detail)
    return HTTPException(status_code=500, detail=detail)


@router.post("/runs", status_code=201)
async def create_agent_run_endpoint(
    request: AgentRunCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    try:
        agent_run = await create_agent_run(
            AgentCommand(
                user_id=user_id,
                agent_id=request.agent_id,
                message=request.message,
                source_type=request.source_type,
                source_ref=request.source_ref,
                attachments=request.attachments,
                context=request.context,
                runtime_config=request.runtime_config,
            )
        )
    except (AgentValidationError, AgentRuntimeError) as e:
        raise _to_http_exception(e) from e

    return ok(
        data={
            "run_id": agent_run["id"],
            "task_id": agent_run["task_id"],
            "status": agent_run["status"],
        },
        msg="created",
    )


@router.get("/runs/{run_id}")
async def get_agent_run_endpoint(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
):
    try:
        agent_run = get_agent_run_for_user(run_id, user_id)
    except AgentValidationError as e:
        raise _to_http_exception(e) from e
    return ok(data=agent_run, msg="ok")


@router.get("/runs/{run_id}/events")
async def list_agent_run_events_endpoint(
    run_id: str,
    limit: int = 500,
    offset: int = 0,
    order: str = "asc",
    after_sequence: Optional[int] = None,
    user_id: str = Depends(get_current_user_id),
):
    """按时间顺序返回一次 AgentRun 的步骤事件流。

    权限校验：先过 `get_agent_run_for_user` 确认调用方拥有该 run。

    Query 参数：
      - ``order``：``asc`` / ``desc``，默认 asc
      - ``after_sequence``：仅返回 ``sequence > after_sequence`` 的事件，前端
        做增量轮询时传入上一批的最大 sequence 即可（配合 ``order=asc`` 使用）
      - ``limit`` / ``offset``：分页；``limit`` 上限 2000

    响应 ``data`` 里除 ``events`` 外会同时返回 ``last_sequence``（本批最后一条
    事件的 sequence，用作下次 ``after_sequence`` 的输入）。
    """
    try:
        get_agent_run_for_user(run_id, user_id)
    except AgentValidationError as e:
        raise _to_http_exception(e) from e

    safe_limit = max(1, min(int(limit or 500), 2000))
    safe_offset = max(0, int(offset or 0))
    ascending = str(order or "asc").strip().lower() != "desc"
    safe_after = None
    if after_sequence is not None:
        try:
            safe_after = max(0, int(after_sequence))
        except (TypeError, ValueError):
            safe_after = None
    events = list_agent_run_events(
        run_id,
        limit=safe_limit,
        offset=safe_offset,
        ascending=ascending,
        after_sequence=safe_after,
    )
    last_sequence: Optional[int] = None
    if events:
        # 不依赖 ascending：取本批里 sequence 最大值，作为下次 after_sequence 输入
        try:
            last_sequence = max(int(e.get("sequence") or 0) for e in events)
        except (TypeError, ValueError):
            last_sequence = None
    return ok(
        data={
            "run_id": run_id,
            "events": events,
            "total": len(events),
            "last_sequence": last_sequence,
        },
        msg="ok",
    )


@router.get("/runs")
async def list_agent_runs_endpoint(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id),
):
    runs = list_agent_runs_for_user(user_id, limit=limit, offset=offset, status=status)
    return ok(data={"runs": runs, "total": len(runs)}, msg="ok")


@router.post("/runs/{run_id}/cancel")
async def cancel_agent_run_endpoint(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
):
    try:
        agent_run = await cancel_agent_run(run_id, user_id)
    except (AgentValidationError, AgentRuntimeError) as e:
        raise _to_http_exception(e) from e
    return ok(
        data={
            "run_id": agent_run["id"],
            "task_id": agent_run["task_id"],
            "status": agent_run["status"],
        },
        msg="cancelled",
    )


@router.get("/{agent_id}")
async def get_agent_endpoint(
    agent_id: str,
    _user_id: str = Depends(get_current_user_id),
):
    ensure_default_agent_presets()
    try:
        agent = get_agent_or_raise(agent_id)
    except AgentValidationError as e:
        raise _to_http_exception(e) from e
    return ok(data=agent, msg="ok")
