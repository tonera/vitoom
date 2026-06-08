from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from backend.core.logger import get_app_logger
from backend.database import Agent, AgentRun
from backend.queue import get_task_queue
from backend.queue.worker import TaskWorker
from backend.services.agent.crews import CrewFactory
from backend.services.agent.events import (
    record_run_completed,
    record_run_failed,
    record_run_started,
    record_tool_selected,
)
from backend.services.agent.flows import FlowRunner
from backend.services.agent.no_tool_runner import run_no_tool_completion
from backend.services.agent.presets import ensure_default_agent_presets
from backend.services.agent.settings import is_agent_recovery_enabled, is_agents_enabled
from backend.services.agent.specs import AgentSpec, GLOBAL_TOOL_POOL, TaskSpec
from backend.services.agent.tool_resolver import ToolResolver
from backend.services.agent.tool_selection import ToolSelectionService
from backend.services.agent.types import AgentCommand, AgentResult

logger = get_app_logger(__name__)


def _result_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    raw = getattr(value, "raw", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    try:
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _extract_usage_metrics(raw_result: Any, *, elapsed_seconds: Optional[float]) -> Dict[str, Any]:
    """从 CrewAI 的 CrewOutput / TaskOutput 上把 token_usage 抽出来并算 tok/s。

    CrewOutput 暴露的是 UsageMetrics（pydantic），常见字段：
      - total_tokens / prompt_tokens / completion_tokens
      - successful_requests / cached_prompt_tokens
    老版本可能用 totalTokens 这种 camelCase，尽量兼容。
    """

    def _coerce_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    usage_obj = getattr(raw_result, "token_usage", None)
    if usage_obj is None and isinstance(raw_result, dict):
        usage_obj = raw_result.get("token_usage")

    if usage_obj is None:
        return {
            "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        }

    # pydantic v2 -> model_dump(); v1 -> dict(); 普通 dict 直接用
    usage: Dict[str, Any]
    if isinstance(usage_obj, dict):
        usage = dict(usage_obj)
    else:
        for method in ("model_dump", "dict"):
            dumper = getattr(usage_obj, method, None)
            if callable(dumper):
                try:
                    usage = dumper()
                    break
                except Exception:
                    continue
        else:
            usage = {
                key: getattr(usage_obj, key, None)
                for key in (
                    "total_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                    "successful_requests",
                    "cached_prompt_tokens",
                )
            }

    def _first_present(*keys: str) -> Any:
        for key in keys:
            value = usage.get(key)
            if value is not None:
                return value
        return None

    total = _coerce_int(_first_present("total_tokens", "totalTokens"))
    prompt = _coerce_int(_first_present("prompt_tokens", "promptTokens"))
    completion = _coerce_int(
        _first_present("completion_tokens", "completionTokens", "output_tokens")
    )
    requests = _coerce_int(_first_present("successful_requests", "successfulRequests"))
    cached = _coerce_int(_first_present("cached_prompt_tokens", "cachedPromptTokens"))

    metrics: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "successful_requests": requests,
        "cached_prompt_tokens": cached,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
    }

    # 平均 tok/s：用 completion_tokens / 耗时；如果没有 completion 就退回 total
    if elapsed_seconds and elapsed_seconds > 0:
        divisor_token = completion if completion else total
        if divisor_token:
            metrics["tokens_per_second"] = round(divisor_token / elapsed_seconds, 2)

    return metrics


class AgentWorker(TaskWorker):
    """Agent 类型任务的工作线程。"""

    def __init__(self):
        super().__init__("agent")

    @staticmethod
    def _persist_conversation_reply(
        *,
        agent_run_id: str,
        conversation_id: Any,
        content: str,
    ) -> None:
        """把 agent 产出的最终文本作为 assistant 消息落库，失败不影响主流程。"""
        if not conversation_id:
            return
        text = str(content or "").strip()
        if not text:
            return
        try:
            from backend.services.conversation import append_message

            append_message(
                conversation_id=str(conversation_id),
                role="assistant",
                content=text,
                agent_run_id=agent_run_id,
            )
        except Exception:
            logger.exception(
                "[agent-run %s] Failed to append assistant message to conversation %s",
                agent_run_id,
                conversation_id,
            )

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(task.get("id") or "").strip()
        params = dict(task.get("params") or {})
        agent_run_id = str(params.get("agent_run_id") or "").strip()
        if not task_id or not agent_run_id:
            raise RuntimeError("agent task requires both task_id and agent_run_id")

        agent_run = AgentRun.get_by_id(agent_run_id)
        if not agent_run:
            raise RuntimeError(f"Agent run not found: {agent_run_id}")

        agent_record = Agent.get_by_id(agent_run["agent_id"])
        if not agent_record:
            raise RuntimeError(f"Agent not found: {agent_run['agent_id']}")

        command = AgentCommand.from_dict(agent_run.get("input_payload") or {})
        runtime_config = dict(agent_run.get("runtime_config") or {})

        started_at = datetime.utcnow()
        try:
            AgentRun.update(
                agent_run_id,
                status="running",
                started_at=started_at,
                error_message=None,
            )
            record_run_started(
                agent_run_id,
                agent_id=agent_run["agent_id"],
                source_type=command.source_type,
                conversation_id=command.conversation_id,
                message=command.message,
                runtime_config=runtime_config,
                started_at=started_at,
            )
            await self.update_progress(task_id, 10, "Initializing agent runtime")

            agent_specs = AgentSpec.list_from_agent_record(agent_record)
            task_specs = TaskSpec.list_from_agent_record(agent_record)
            await self.update_progress(task_id, 30, "Building agent crew")

            unique_tool_names: list = []
            seen_tool_names: set = set()
            preferred_tool_names: list = []
            for spec in agent_specs:
                for name in spec.tools:
                    if name not in seen_tool_names:
                        seen_tool_names.add(name)
                        unique_tool_names.append(name)
                for name in spec.preferred_tool_names:
                    if name not in preferred_tool_names:
                        preferred_tool_names.append(name)

            uses_global_pool = any(spec.tool_pool == GLOBAL_TOOL_POOL for spec in agent_specs)

            logger.info(
                "[agent-run %s] declared tools: %s | per-agent: %s | pool=%s | preferred=%s",
                agent_run_id,
                unique_tool_names,
                {spec.name: list(spec.tools) for spec in agent_specs},
                "global" if uses_global_pool else "declared",
                preferred_tool_names,
            )

            selected_tool_names = ToolSelectionService().select_tool_names(
                unique_tool_names,
                command=command,
                task_specs=task_specs,
                runtime_allowlist=runtime_config.get("tool_allowlist"),
                max_tools=runtime_config.get("max_tools_per_run"),
                pool="global" if uses_global_pool else "declared",
                preferred_tool_names=preferred_tool_names,
            )
            logger.info(
                "[agent-run %s] selected tools (after ToolSelectionService filter): %s",
                agent_run_id,
                selected_tool_names,
            )
            record_tool_selected(
                agent_run_id,
                declared=unique_tool_names,
                selected=list(selected_tool_names),
                pool="global" if uses_global_pool else "declared",
                preferred=preferred_tool_names,
                max_tools=runtime_config.get("max_tools_per_run"),
            )
            if not selected_tool_names:
                logger.info(
                    "[agent-run %s] no tools selected; using direct LLM fast path",
                    agent_run_id,
                )
                await self.update_progress(task_id, 60, "Running direct LLM")
                raw_result = await asyncio.to_thread(
                    run_no_tool_completion,
                    agent_specs=agent_specs,
                    task_specs=task_specs,
                    command=command,
                    stream=False,
                )
            else:
                tool_resolver = ToolResolver()
                resolved_tools = tool_resolver.resolve_tools(
                    selected_tool_names,
                    agent_run_id=agent_run_id,
                    runtime_allowlist=runtime_config.get("tool_allowlist"),
                    crew_tool_context={
                        "user_id": command.user_id,
                        "conversation_id": command.conversation_id,
                        "runtime_config": runtime_config,
                        "source_type": command.source_type or "master-agent",
                    },
                )
                tools_by_name = {
                    name: tool
                    for name, tool in zip(selected_tool_names, resolved_tools)
                }
                logger.info(
                    "[agent-run %s] resolved %d crewai tool objects: %s",
                    agent_run_id,
                    len(resolved_tools),
                    [
                        {
                            "name": name,
                            "class": type(tool).__name__,
                            "description": (getattr(tool, "description", None) or "")[:80],
                        }
                        for name, tool in tools_by_name.items()
                    ],
                )

                # 对于使用 global pool 的 agent，把全局筛选出来的工具挂到它身上，
                # 保证 CrewFactory 能把这些工具绑定到对应的 CrewAIAgent。
                for spec in agent_specs:
                    if spec.tool_pool != GLOBAL_TOOL_POOL:
                        continue
                    merged: list = []
                    seen: set = set()
                    for name in spec.tools + list(selected_tool_names):
                        norm = str(name or "").strip()
                        if norm and norm not in seen:
                            seen.add(norm)
                            merged.append(norm)
                    spec.tools = merged

                crew, inputs = CrewFactory().build(
                    agent_specs=agent_specs,
                    task_specs=task_specs,
                    command=command,
                    tools_by_name=tools_by_name,
                    process_name=str(runtime_config.get("process") or "sequential"),
                )
                await self.update_progress(task_id, 60, "Running crew")

                raw_result = await FlowRunner().run(crew, inputs=inputs)
            completed_at = datetime.utcnow()
            elapsed_seconds = max(0.0, (completed_at - started_at).total_seconds())
            usage_metrics = _extract_usage_metrics(raw_result, elapsed_seconds=elapsed_seconds)
            logger.info("[agent-run %s] usage_metrics=%s", agent_run_id, usage_metrics)

            output_text = _result_to_text(raw_result)
            # AgentResult.summary 给调度/日志看的短摘要，保留 500 字；
            # AgentRun.result_summary 直接落完整输出（Text 列无长度限制），
            # 避免 CLI / 前端读到的内容被截成"只有第一天"。
            short_summary = (
                output_text if len(output_text) <= 500 else output_text[:497] + "..."
            )
            # 64KB 上限兜底，防止异常超长输出撑爆 DB；正常长输出远小于这个值。
            MAX_STORED_OUTPUT = 64 * 1024
            stored_output = (
                output_text
                if len(output_text) <= MAX_STORED_OUTPUT
                else output_text[: MAX_STORED_OUTPUT - 3] + "..."
            )

            result = AgentResult(
                run_id=agent_run_id,
                task_id=task_id,
                status="completed",
                summary=short_summary,
                output_text=output_text,
                artifacts=[],
                error=None,
                metrics=usage_metrics,
            )

            AgentRun.update(
                agent_run_id,
                status="completed",
                result_summary=stored_output,
                usage_metrics=usage_metrics,
                error_message=None,
                completed_at=completed_at,
            )
            record_run_completed(
                agent_run_id,
                output_text=output_text,
                usage_metrics=usage_metrics,
                completed_at=completed_at,
            )
            self._persist_conversation_reply(
                agent_run_id=agent_run_id,
                conversation_id=command.conversation_id,
                content=output_text,
            )
            await self.update_progress(task_id, 95, "Agent run completed")
            return result.to_dict()

        except Exception as e:
            failed_at = datetime.utcnow()
            AgentRun.update(
                agent_run_id,
                status="failed",
                error_message=str(e),
                completed_at=failed_at,
            )
            record_run_failed(agent_run_id, error=str(e), completed_at=failed_at)
            raise


_agent_worker = AgentWorker()


async def _recover_agent_tasks_later(queue, delay_seconds: float = 1.0):
    await asyncio.sleep(max(0.0, float(delay_seconds)))
    try:
        await queue.recover_tasks(task_types=["agent"])
        logger.info("Recovered pending agent tasks")
    except Exception:
        logger.exception("Failed to recover pending agent tasks")


def configure_agent_task_queue():
    """注册 agent 任务处理器。"""
    queue = get_task_queue()
    queue.register_handler("agent", _agent_worker.process)
    return queue


async def startup_agent_runtime():
    """启动 agent 任务处理。"""
    if not is_agents_enabled():
        logger.info("Agent runtime is disabled by config")
        return
    ensure_default_agent_presets()
    queue = configure_agent_task_queue()
    if not queue.is_running:
        await queue.start_workers()
        if is_agent_recovery_enabled():
            asyncio.create_task(_recover_agent_tasks_later(queue))
        else:
            logger.info("Agent pending task recovery is disabled by config")
        logger.info("Agent task runtime started")


async def shutdown_agent_runtime():
    """停止 agent 任务处理。"""
    queue = get_task_queue()
    if queue.is_running:
        await queue.stop_workers()
        logger.info("Agent task runtime stopped")
