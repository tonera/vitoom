"""AgentRunEvent 埋点辅助层。

所有埋点统一走 `record_event`，职责：
  - 把 payload 序列化友好化（截断长字符串、剔除敏感字段）
  - `try/except` 吞所有异常，埋点失败**绝不影响**主流程
  - 提供细粒度的便捷函数（run_started / tool_call_started 等），避免调用方
    到处手写字面量字符串

设计原则：
  1. append-only：不做更新；"工具调用耗时"用 started→completed 两条事件表达。
  2. payload 大字段做 preview 截断，原始大对象走 logger 就好。
  3. `agent_run_id` 无效时（比如 Crew-as-Tool 里 parent 为空）直接静默丢弃。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.core.logger import get_app_logger
from backend.database import AgentRunEvent

logger = get_app_logger(__name__)

# 单个字符串字段 preview 上限：避免单条 event 行过大撑爆 DB / 前端
_PREVIEW_MAX_CHARS = 1024


def _preview(value: Any, *, limit: int = _PREVIEW_MAX_CHARS) -> Any:
    """对字符串/容器做 preview 截断，非字符串类型尽量保留结构。"""
    if value is None:
        return None
    if isinstance(value, str):
        text = value
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
    if isinstance(value, (list, tuple)):
        # list 本身保留，但元素里的字符串做截断
        return [_preview(item, limit=limit) for item in value]
    if isinstance(value, dict):
        return {str(k): _preview(v, limit=limit) for k, v in value.items()}
    # 其他标量类型原样返回（int/float/bool）
    return value


def _sanitize_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    try:
        return {k: _preview(v) for k, v in payload.items()}
    except Exception:
        try:
            return json.loads(json.dumps(payload, default=str))
        except Exception:
            return {"_error": "failed to sanitize payload"}


def record_event(
    agent_run_id: Optional[str],
    event_type: str,
    *,
    tool_name: Optional[str] = None,
    content: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> None:
    """通用埋点入口；所有异常被吞，失败只 log。"""
    run_id = str(agent_run_id or "").strip()
    if not run_id:
        return
    try:
        AgentRunEvent.append(
            agent_run_id=run_id,
            event_type=str(event_type or "").strip() or "unknown",
            tool_name=str(tool_name).strip() if tool_name else None,
            content=_preview(content) if content is not None else None,
            payload=_sanitize_payload(payload),
            started_at=started_at,
            completed_at=completed_at,
        )
    except Exception:
        logger.exception(
            "Failed to record agent_run_event: run=%s type=%s tool=%s",
            run_id,
            event_type,
            tool_name,
        )


# ---------------------------------------------------------------------------
# 便捷函数：按事件语义封装 payload 结构，调用方少关心字段
# ---------------------------------------------------------------------------


def record_run_started(
    agent_run_id: str,
    *,
    agent_id: str,
    source_type: Optional[str],
    conversation_id: Optional[str],
    message: Optional[str],
    runtime_config: Optional[Dict[str, Any]] = None,
    started_at: Optional[datetime] = None,
) -> None:
    record_event(
        agent_run_id,
        "run_started",
        payload={
            "agent_id": agent_id,
            "source_type": source_type,
            "conversation_id": conversation_id,
            "message": message,
            "runtime_config": runtime_config,
        },
        started_at=started_at or datetime.utcnow(),
    )


def record_tool_selected(
    agent_run_id: str,
    *,
    declared: List[str],
    selected: List[str],
    pool: str,
    preferred: Optional[List[str]] = None,
    max_tools: Optional[int] = None,
) -> None:
    record_event(
        agent_run_id,
        "tool_selected",
        payload={
            "declared": list(declared or []),
            "selected": list(selected or []),
            "pool": pool,
            "preferred": list(preferred or []) if preferred else [],
            "max_tools": max_tools,
        },
    )


def record_tool_call_started(
    agent_run_id: str,
    *,
    exposed_name: str,
    provider: str,
    target_tool_name: Optional[str] = None,
    args: Any = None,
    started_at: Optional[datetime] = None,
) -> None:
    record_event(
        agent_run_id,
        "tool_call_started",
        tool_name=exposed_name,
        payload={
            "provider": provider,
            "target_tool_name": target_tool_name,
            "args_preview": _preview(args),
        },
        started_at=started_at or datetime.utcnow(),
    )


def record_tool_call_completed(
    agent_run_id: str,
    *,
    exposed_name: str,
    provider: str,
    output: Any = None,
    duration_ms: Optional[int] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> None:
    output_text = (
        output if isinstance(output, str) else _preview(output)
    )
    output_len: Optional[int]
    if isinstance(output, str):
        output_len = len(output)
    elif output is None:
        output_len = 0
    else:
        try:
            output_len = len(json.dumps(output, ensure_ascii=False, default=str))
        except Exception:
            output_len = None
    record_event(
        agent_run_id,
        "tool_call_completed",
        tool_name=exposed_name,
        payload={
            "provider": provider,
            "duration_ms": duration_ms,
            "output_len": output_len,
            "output_preview": output_text,
        },
        started_at=started_at,
        completed_at=completed_at or datetime.utcnow(),
    )


def record_tool_call_failed(
    agent_run_id: str,
    *,
    exposed_name: str,
    provider: str,
    error: str,
    duration_ms: Optional[int] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> None:
    record_event(
        agent_run_id,
        "tool_call_failed",
        tool_name=exposed_name,
        content=str(error)[:_PREVIEW_MAX_CHARS],
        payload={
            "provider": provider,
            "duration_ms": duration_ms,
        },
        started_at=started_at,
        completed_at=completed_at or datetime.utcnow(),
    )


def record_crew_tool_invoked(
    parent_agent_run_id: str,
    *,
    crew_tool_name: str,
    preset_id: str,
    child_run_id: str,
    query: Optional[str] = None,
) -> None:
    record_event(
        parent_agent_run_id,
        "crew_tool_invoked",
        tool_name=crew_tool_name,
        payload={
            "preset_id": preset_id,
            "child_run_id": child_run_id,
            "query": query,
        },
    )


def record_run_completed(
    agent_run_id: str,
    *,
    output_text: Optional[str],
    usage_metrics: Optional[Dict[str, Any]] = None,
    completed_at: Optional[datetime] = None,
) -> None:
    record_event(
        agent_run_id,
        "run_completed",
        payload={
            "output_len": len(output_text or ""),
            "usage_metrics": usage_metrics,
        },
        completed_at=completed_at or datetime.utcnow(),
    )


def record_run_failed(
    agent_run_id: str,
    *,
    error: str,
    completed_at: Optional[datetime] = None,
) -> None:
    record_event(
        agent_run_id,
        "run_failed",
        content=str(error)[:_PREVIEW_MAX_CHARS],
        completed_at=completed_at or datetime.utcnow(),
    )


def record_run_cancelled(
    agent_run_id: str,
    *,
    completed_at: Optional[datetime] = None,
) -> None:
    record_event(
        agent_run_id,
        "run_cancelled",
        completed_at=completed_at or datetime.utcnow(),
    )


def list_events(
    agent_run_id: str,
    *,
    limit: int = 500,
    offset: int = 0,
    ascending: bool = True,
    after_sequence: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """查询一次 run 的事件流。

    ``after_sequence``：仅返回 ``sequence > after_sequence`` 的事件；前端每次
    轮询时把上一批拉到的最大 ``sequence`` 作为下次请求的 ``after_sequence``，
    即可增量拉取，避免重复传输。
    """
    run_id = str(agent_run_id or "").strip()
    if not run_id:
        return []
    try:
        return AgentRunEvent.list_by_run(
            run_id,
            limit=limit,
            offset=offset,
            ascending=ascending,
            after_sequence=after_sequence,
        )
    except Exception:
        logger.exception("Failed to list agent_run_events: run=%s", run_id)
        return []
