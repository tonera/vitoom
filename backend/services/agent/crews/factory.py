from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.core.logger import get_app_logger
from backend.services.agent.llm import build_crewai_llm
from backend.services.agent.specs import AgentSpec, TaskSpec
from backend.services.agent.types import AgentCommand

logger = get_app_logger(__name__)


def _resolve_tool_name(tool: Any) -> str:
    for attr in ("name", "tool_name"):
        value = getattr(tool, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    func_name = getattr(tool, "__name__", None)
    if isinstance(func_name, str) and func_name.strip():
        return func_name.strip()
    return ""


def _sanitize_input_value(value: Any, *, key_path: str) -> tuple[bool, Any]:
    """过滤掉 CrewAI inputs 不支持的值类型，避免插值阶段直接报错。"""
    if isinstance(value, (str, int, float, bool)):
        return True, value
    if value is None:
        logger.warning("[crew-build] dropping unsupported input %s=None", key_path)
        return False, None
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            child_key = str(raw_key)
            ok, cleaned = _sanitize_input_value(
                raw_item,
                key_path=f"{key_path}.{child_key}",
            )
            if ok:
                sanitized[child_key] = cleaned
        return True, sanitized
    if isinstance(value, list):
        sanitized_list: List[Any] = []
        for idx, item in enumerate(value):
            ok, cleaned = _sanitize_input_value(
                item,
                key_path=f"{key_path}[{idx}]",
            )
            if ok:
                sanitized_list.append(cleaned)
        return True, sanitized_list
    logger.warning(
        "[crew-build] dropping unsupported input %s type=%s",
        key_path,
        type(value).__name__,
    )
    return False, None


def _sanitize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in dict(inputs or {}).items():
        normalized_key = str(key)
        ok, cleaned = _sanitize_input_value(value, key_path=normalized_key)
        if ok:
            sanitized[normalized_key] = cleaned
    return sanitized


class CrewFactory:
    """根据 AgentSpec / TaskSpec 组装 CrewAI Crew，支持多 Agent 协作。"""

    def build(
        self,
        *,
        agent_specs: Optional[List[AgentSpec]] = None,
        task_specs: List[TaskSpec],
        command: AgentCommand,
        tools: Optional[List[Any]] = None,
        tools_by_name: Optional[Dict[str, Any]] = None,
        process_name: str = "sequential",
        agent_spec: Optional[AgentSpec] = None,
        stream: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        try:
            from crewai import Agent as CrewAIAgent
            from crewai import Crew, Process, Task as CrewAITask
        except Exception as e:
            raise RuntimeError("crewai is required to build and run agent crews") from e

        resolved_specs: List[AgentSpec] = []
        if agent_specs:
            resolved_specs = list(agent_specs)
        elif agent_spec is not None:
            resolved_specs = [agent_spec]
        if not resolved_specs:
            raise RuntimeError("CrewFactory.build requires at least one AgentSpec")

        resolved_tools_by_name: Dict[str, Any] = {}
        if tools_by_name:
            resolved_tools_by_name.update(tools_by_name)
        for item in tools or []:
            name = _resolve_tool_name(item)
            if name and name not in resolved_tools_by_name:
                resolved_tools_by_name[name] = item

        # 会话 / Agent run 的 runtime_config 里如果带了 load_name（通常来自
        # chat session metadata：POST /v1/chat/sessions 时传入的 load_name），
        # 要透传给 CrewAI LLM，否则会静默落回 agents.default_model，出现
        # "用户要 A 模型、实际跑了 B 模型" 的串台。
        runtime_config = dict(command.runtime_config or {})
        preferred_model_name = str(runtime_config.get("load_name") or "").strip() or None
        shared_llm = build_crewai_llm(
            preferred_model_name=preferred_model_name,
            effective_user_id=command.user_id,
            stream=stream,
        )

        crewai_agents_by_name: Dict[str, Any] = {}
        for spec in resolved_specs:
            agent_tools = [
                resolved_tools_by_name[name]
                for name in spec.tools
                if name in resolved_tools_by_name
            ]
            bound_tool_names = [_resolve_tool_name(t) or type(t).__name__ for t in agent_tools]
            logger.info(
                "[crew-build] agent=%s role=%s -> bound %d tools to LLM: %s",
                spec.name,
                spec.role,
                len(agent_tools),
                bound_tool_names,
            )
            crewai_agents_by_name[spec.name] = CrewAIAgent(
                role=spec.role,
                goal=spec.goal,
                backstory=spec.backstory,
                llm=shared_llm,
                tools=agent_tools,
                verbose=spec.verbose,
                memory=spec.memory,
                allow_delegation=spec.allow_delegation,
            )

        default_agent_name = resolved_specs[0].name

        crew_tasks: List[Any] = []
        built_tasks: Dict[str, Any] = {}
        for task_spec in task_specs:
            context_tasks = [built_tasks[item] for item in task_spec.context if item in built_tasks]
            target_agent_name = (task_spec.agent_name or default_agent_name).strip() or default_agent_name
            agent_ref = crewai_agents_by_name.get(target_agent_name) or crewai_agents_by_name[default_agent_name]

            task_kwargs: Dict[str, Any] = {
                "description": task_spec.description,
                "expected_output": task_spec.expected_output,
                "agent": agent_ref,
            }
            if context_tasks:
                task_kwargs["context"] = context_tasks
            crew_task = CrewAITask(**task_kwargs)
            built_tasks[task_spec.task_id] = crew_task
            crew_tasks.append(crew_task)

        normalized_process = str(process_name or "sequential").strip().lower()
        process = Process.sequential
        if normalized_process == "hierarchical":
            process = Process.hierarchical

        verbose_flag = any(spec.verbose for spec in resolved_specs)
        crew = Crew(
            agents=list(crewai_agents_by_name.values()),
            tasks=crew_tasks,
            process=process,
            verbose=bool(verbose_flag),
        )

        inputs: Dict[str, Any] = {
            "message": command.message,
            "user_id": command.user_id,
            "agent_id": command.agent_id,
            "source_type": command.source_type,
        }
        inputs.update(dict(command.context or {}))
        return crew, _sanitize_inputs(inputs)
