"""Crew-as-Tool 的运行期桥接：把注册过的专业 Crew 打包成 CrewAI 工具。

调用时序：
  1. Master Agent LLM 决定调用某个 crew-tool，传入 query 字符串；
  2. 本桥接在**当前线程**内同步构建子 Crew 并 kickoff，不再走任务队列
     （避免 worker 并发数不足时的死锁）；
  3. 同时在数据库内创建一条子 AgentRun 记录，附带 parent_run_id 方便追踪；
  4. 执行完成后返回文本结果给 Master Agent 继续整合。
"""

from __future__ import annotations

import json
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from backend.core.logger import get_app_logger
from backend.database import AgentRun
from backend.services.agent.crews import CrewFactory
from backend.services.agent.events import (
    record_crew_tool_invoked,
    record_run_completed,
    record_run_failed,
    record_run_started,
)
from backend.services.agent.presets import get_preset_definition
from backend.services.agent.specs import AgentSpec, TaskSpec
from backend.services.agent.tool_selection import ToolSelectionService
from backend.services.agent.types import AgentCommand
from backend.utils import generate_uuid

from .registry import CrewToolMetadata, get_crew_tool_registry

logger = get_app_logger(__name__)

_NESTED_TOOL_SCOPE_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "crew_tool_nested_scope_stack", default=()
)


@contextmanager
def nested_tool_event_scope(parent_tool: str):
    """标记当前线程正在执行某个 crew-tool 的子 Crew。

    外层 `master_runtime` 监听的是同一个 CrewAI 全局事件总线；子 Crew kickoff
    期间需要让外层忽略内部 `ToolUsage*Event`，否则前端会把它们当成顶层工具再
    渲染一遍，出现重复事件。
    """

    normalized = str(parent_tool or "").strip() or "crew-tool"
    stack = _NESTED_TOOL_SCOPE_STACK.get(())
    token = _NESTED_TOOL_SCOPE_STACK.set(stack + (normalized,))
    try:
        yield
    finally:
        _NESTED_TOOL_SCOPE_STACK.reset(token)


def get_current_nested_parent_tool() -> Optional[str]:
    stack = _NESTED_TOOL_SCOPE_STACK.get(())
    return stack[-1] if stack else None


def is_nested_tool_event_scope_active() -> bool:
    return get_current_nested_parent_tool() is not None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _coerce_query(raw_input: Any) -> str:
    if raw_input is None:
        return ""
    if isinstance(raw_input, str):
        text = raw_input.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return str(parsed.get("query") or parsed.get("input") or parsed.get("message") or text).strip()
        except Exception:
            pass
        return text
    if isinstance(raw_input, dict):
        for key in ("query", "input", "message", "task"):
            value = raw_input.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return _stringify(raw_input)
    return str(raw_input).strip()


def build_crew_tool(
    metadata: CrewToolMetadata,
    *,
    parent_agent_run_id: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    source_type: str = "master-agent",
    turn_id: Optional[str] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    session_nested_tool_hooks: Optional[Dict[str, Any]] = None,
) -> Any:
    """构造一个 CrewAI 工具实例，调用时同步跑目标 Crew 并返回结果。"""

    try:
        from crewai.tools import tool as crewai_tool
    except Exception as e:
        raise RuntimeError("crewai is required to build crew-as-tool") from e

    description = metadata.description or f"调用 {metadata.name} 专业 Crew 完成复杂任务"
    docstring = (
        f"{description}\n\n"
        f"Input: 一段精炼的中文任务描述（{metadata.input_hint}）。"
        " 返回该 Crew 产出的最终文本结果。"
    )

    preset_id = metadata.preset_id
    resolved_runtime_config = dict(runtime_config or {})
    nested_hooks = session_nested_tool_hooks

    @crewai_tool(metadata.name)
    def crew_tool(arguments: str = "") -> str:
        """Placeholder; overridden below."""
        query = _coerce_query(arguments)
        if not query:
            return f"crew-tool {metadata.name} 需要一个非空的 query 字符串"

        child_run_id = generate_uuid()
        started_at = datetime.utcnow()

        try:
            AgentRun.create(
                id=child_run_id,
                user_id=user_id,
                agent_id=preset_id,
                task_id=child_run_id,
                source_type=source_type,
                source_ref=None,
                status="running",
                input_payload={
                    "user_id": user_id,
                    "agent_id": preset_id,
                    "message": query,
                    "source_type": source_type,
                    "conversation_id": conversation_id,
                    "parent_run_id": parent_agent_run_id,
                    "crew_tool_name": metadata.name,
                },
                runtime_config=resolved_runtime_config,
                conversation_id=conversation_id,
                parent_run_id=parent_agent_run_id,
                crew_tool_name=metadata.name,
            )
            AgentRun.update(child_run_id, started_at=started_at)
        except Exception:
            logger.exception(
                "[crew-tool %s] Failed to persist child AgentRun for parent=%s",
                metadata.name,
                parent_agent_run_id,
            )

        # 父 run 视角：`tool_call_started / completed / failed` 由 ToolResolver 的
        # 统一包装层负责埋点，这里只补 crew-tool 专有语义事件：child_run_id + preset_id。
        record_crew_tool_invoked(
            parent_agent_run_id,
            crew_tool_name=metadata.name,
            preset_id=preset_id,
            child_run_id=child_run_id,
            query=query,
        )
        # 子 run 视角：run_started，方便单独查子 run 的事件流
        record_run_started(
            child_run_id,
            agent_id=preset_id,
            source_type=source_type,
            conversation_id=conversation_id,
            message=query,
            runtime_config=resolved_runtime_config,
            started_at=started_at,
        )

        try:
            logger.info(
                "[crew-tool %s] parent=%s child=%s kickoff preset=%s query=%s",
                metadata.name,
                parent_agent_run_id,
                child_run_id,
                preset_id,
                query[:120],
            )
            result_text, child_usage = _run_crew_sync(
                metadata=metadata,
                query=query,
                user_id=user_id,
                child_run_id=child_run_id,
                parent_agent_run_id=parent_agent_run_id,
                conversation_id=conversation_id,
                runtime_config=resolved_runtime_config,
                source_type=source_type,
                started_at=started_at,
                turn_id=turn_id,
                task_event_callback=task_event_callback,
                session_nested_tool_hooks=nested_hooks,
            )
            # child run 的 result_summary 也存完整输出（Text 列无长度限制），
            # 64KB 上限兜底，便于 CLI `/run <child_id>` 调试时看到 crew 全量产出。
            completed_at = datetime.utcnow()
            try:
                MAX_CHILD_OUTPUT = 64 * 1024
                stored_child = (
                    (result_text or "")
                    if len(result_text or "") <= MAX_CHILD_OUTPUT
                    else (result_text or "")[: MAX_CHILD_OUTPUT - 3] + "..."
                )
                AgentRun.update(
                    child_run_id,
                    status="completed",
                    result_summary=stored_child,
                    usage_metrics=child_usage,
                    completed_at=completed_at,
                    error_message=None,
                )
            except Exception:
                logger.exception(
                    "[crew-tool %s] Failed to update child AgentRun %s",
                    metadata.name,
                    child_run_id,
                )
            # 子 run 的 run_completed；父 run 的 tool_call_completed 由 resolver 包装层负责
            record_run_completed(
                child_run_id,
                output_text=result_text,
                usage_metrics=child_usage,
                completed_at=completed_at,
            )
            logger.info(
                "[crew-tool %s] child=%s completed, output_len=%d",
                metadata.name,
                child_run_id,
                len(result_text or ""),
            )
            return result_text or ""
        except Exception as exc:
            tb = traceback.format_exc()
            failed_at = datetime.utcnow()
            logger.error(
                "[crew-tool %s] child=%s failed: %s\n%s",
                metadata.name,
                child_run_id,
                exc,
                tb,
            )
            try:
                AgentRun.update(
                    child_run_id,
                    status="failed",
                    error_message=str(exc),
                    completed_at=failed_at,
                )
            except Exception:
                pass
            record_run_failed(child_run_id, error=str(exc), completed_at=failed_at)
            # 父 run 的 tool_call_failed 由 resolver 包装层负责；这里抛出 return
            # 的字符串会被父 crew 当成"工具返回失败说明"。
            return f"调用 {metadata.name} 失败：{exc}"

    crew_tool.__doc__ = docstring
    return crew_tool


def _run_crew_sync(
    *,
    metadata: CrewToolMetadata,
    query: str,
    user_id: str,
    child_run_id: str,
    parent_agent_run_id: str,
    conversation_id: Optional[str],
    runtime_config: Dict[str, Any],
    source_type: str,
    started_at: Optional[datetime] = None,
    turn_id: Optional[str] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    session_nested_tool_hooks: Optional[Dict[str, Any]] = None,
) -> tuple[str, Dict[str, Any]]:
    definition = get_preset_definition(metadata.preset_id)
    if definition is None:
        raise RuntimeError(
            f"crew-tool {metadata.name} references unknown preset: {metadata.preset_id}"
        )

    preset_config = definition.resolved_config()
    synthetic_agent_record: Dict[str, Any] = {
        "id": metadata.preset_id,
        "name": definition.name,
        "type": definition.agent_type,
        "config": preset_config,
    }

    agent_specs: List[AgentSpec] = AgentSpec.list_from_agent_record(synthetic_agent_record)
    task_specs: List[TaskSpec] = TaskSpec.list_from_agent_record(synthetic_agent_record)
    if not agent_specs or not task_specs:
        raise RuntimeError(
            f"crew-tool {metadata.name} preset is missing agents/tasks definition"
        )

    command = AgentCommand(
        user_id=user_id,
        agent_id=metadata.preset_id,
        message=query,
        source_type=source_type,
        context={"turn_id": turn_id} if turn_id else {},
        runtime_config=dict(runtime_config or {}),
        conversation_id=conversation_id,
        parent_run_id=parent_agent_run_id,
        crew_tool_name=metadata.name,
    )

    declared_tools: List[str] = []
    seen: set = set()
    for spec in agent_specs:
        for name in spec.tools:
            if name and name not in seen:
                seen.add(name)
                declared_tools.append(name)

    selected_tool_names = ToolSelectionService().select_tool_names(
        declared_tools,
        command=command,
        task_specs=task_specs,
        runtime_allowlist=runtime_config.get("tool_allowlist") if isinstance(runtime_config, dict) else None,
        max_tools=runtime_config.get("max_tools_per_run") if isinstance(runtime_config, dict) else None,
        pool="declared",
    )

    from backend.services.agent.tool_resolver import ToolResolver

    tool_resolver = ToolResolver()
    resolved_tools = tool_resolver.resolve_tools(
        selected_tool_names,
        agent_run_id=child_run_id,
        runtime_allowlist=runtime_config.get("tool_allowlist") if isinstance(runtime_config, dict) else None,
        crew_tool_context={
            "user_id": command.user_id,
            "agent_run_id": child_run_id,
            "turn_id": turn_id,
            "task_event_callback": task_event_callback if callable(task_event_callback) else None,
            "conversation_id": command.conversation_id,
            "runtime_config": runtime_config,
            "source_type": command.source_type or "crew-tool",
            "session_nested_tool_hooks": session_nested_tool_hooks,
        },
    )
    tools_by_name = dict(zip(selected_tool_names, resolved_tools))

    crew, inputs = CrewFactory().build(
        agent_specs=agent_specs,
        task_specs=task_specs,
        command=command,
        tools_by_name=tools_by_name,
        process_name=str((runtime_config or {}).get("process") or "sequential"),
    )

    hooks = session_nested_tool_hooks if isinstance(session_nested_tool_hooks, dict) else None
    emit_start = hooks.get("emit_tool_started") if hooks else None
    emit_done = hooks.get("emit_tool_finished") if hooks else None
    emit_err = hooks.get("emit_tool_failed") if hooks else None
    parent_tool = metadata.name

    if callable(emit_start) and callable(emit_done) and callable(emit_err):
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.events.types.tool_usage_events import (
                ToolUsageErrorEvent,
                ToolUsageFinishedEvent,
                ToolUsageStartedEvent,
            )
        except Exception:
            with nested_tool_event_scope(parent_tool):
                raw_result = crew.kickoff(inputs=inputs)
        else:
            with nested_tool_event_scope(parent_tool):
                with crewai_event_bus.scoped_handlers():

                    @crewai_event_bus.on(ToolUsageStartedEvent)
                    def _nested_tool_start(source, event):  # noqa: ARG001
                        if get_current_nested_parent_tool() != parent_tool:
                            return
                        emit_start(
                            str(getattr(event, "tool_name", "") or ""),
                            getattr(event, "tool_args", None),
                            str(getattr(event, "event_id", "") or ""),
                            parent_tool,
                        )

                    @crewai_event_bus.on(ToolUsageFinishedEvent)
                    def _nested_tool_done(source, event):  # noqa: ARG001
                        if get_current_nested_parent_tool() != parent_tool:
                            return
                        emit_done(
                            str(getattr(event, "tool_name", "") or ""),
                            getattr(event, "output", None),
                            str(getattr(event, "event_id", "") or ""),
                            parent_tool,
                        )

                    @crewai_event_bus.on(ToolUsageErrorEvent)
                    def _nested_tool_error(source, event):  # noqa: ARG001
                        if get_current_nested_parent_tool() != parent_tool:
                            return
                        emit_err(
                            str(getattr(event, "tool_name", "") or ""),
                            str(getattr(event, "error", "") or ""),
                            str(getattr(event, "event_id", "") or ""),
                            parent_tool,
                        )

                    raw_result = crew.kickoff(inputs=inputs)
    else:
        with nested_tool_event_scope(parent_tool):
            raw_result = crew.kickoff(inputs=inputs)
    elapsed: Optional[float] = None
    if started_at is not None:
        elapsed = max(0.0, (datetime.utcnow() - started_at).total_seconds())
    # 复用 worker 侧的工具函数，避免重复实现
    from backend.workers.agent_worker import _extract_usage_metrics

    usage_metrics = _extract_usage_metrics(raw_result, elapsed_seconds=elapsed)
    logger.info(
        "[crew-tool %s] child=%s usage_metrics=%s",
        metadata.name,
        child_run_id,
        usage_metrics,
    )
    return _stringify(raw_result), usage_metrics


def get_master_crew_tools(
    *,
    parent_agent_run_id: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    source_type: str = "master-agent",
) -> Dict[str, Any]:
    """一次性构造所有已注册 crew-tool 的 CrewAI 工具实例（按名称返回）。"""
    tools: Dict[str, Any] = {}
    for registration in get_crew_tool_registry().all_registrations().values():
        metadata = registration.metadata
        if not metadata.enabled:
            continue
        try:
            tools[metadata.name] = build_crew_tool(
                metadata,
                parent_agent_run_id=parent_agent_run_id,
                user_id=user_id,
                conversation_id=conversation_id,
                runtime_config=runtime_config,
                source_type=source_type,
            )
        except Exception:
            logger.exception("Failed to build crew-tool %s", metadata.name)
    return tools
