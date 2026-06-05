from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_AGENT_NAME = "primary"


GLOBAL_TOOL_POOL = "global"
DECLARED_TOOL_POOL = "declared"
_AUTO_TOOL_MARKERS = {"auto", "global", "catalog", "all", "*"}


def _parse_tools_value(raw_tools: Any) -> tuple:
    """返回 (tool_names, pool_mode, preferred_names)。

    - ``tools: [...]``（列表 / 单字符串 / 字典）→ declared pool，工具清单即声明列表
    - ``tools: auto`` / ``tools: "global"``  → global pool，候选=全局目录
    - ``tools: {"mode": "auto", "preferred": [...]}`` → global pool，preferred 作为偏好项
    """
    if raw_tools is None:
        return ([], DECLARED_TOOL_POOL, [])

    if isinstance(raw_tools, str):
        normalized = raw_tools.strip().lower()
        if normalized in _AUTO_TOOL_MARKERS:
            return ([], GLOBAL_TOOL_POOL, [])
        if raw_tools.strip():
            return ([raw_tools.strip()], DECLARED_TOOL_POOL, [])
        return ([], DECLARED_TOOL_POOL, [])

    if isinstance(raw_tools, dict):
        mode = str(raw_tools.get("mode") or raw_tools.get("pool") or "").strip().lower()
        preferred = _collect_tool_names(raw_tools.get("preferred") or raw_tools.get("always_include"))
        declared = _collect_tool_names(raw_tools.get("tools") or raw_tools.get("names"))
        if mode in _AUTO_TOOL_MARKERS or (raw_tools.get("auto") is True):
            return (declared, GLOBAL_TOOL_POOL, preferred)
        return (declared, DECLARED_TOOL_POOL, preferred)

    if isinstance(raw_tools, list):
        if len(raw_tools) == 1 and isinstance(raw_tools[0], str) and raw_tools[0].strip().lower() in _AUTO_TOOL_MARKERS:
            return ([], GLOBAL_TOOL_POOL, [])
        return (_collect_tool_names(raw_tools), DECLARED_TOOL_POOL, [])

    return ([], DECLARED_TOOL_POOL, [])


def _collect_tool_names(raw_tools: Any) -> List[str]:
    if not raw_tools:
        return []
    if isinstance(raw_tools, str):
        normalized = raw_tools.strip()
        return [normalized] if normalized else []
    if not isinstance(raw_tools, list):
        return []
    tool_names: List[str] = []
    for item in raw_tools:
        if isinstance(item, str) and item.strip():
            tool_names.append(item.strip())
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("tool") or "").strip()
            if name:
                tool_names.append(name)
    return tool_names


def _normalize_tool_names(raw_tools: Any) -> List[str]:
    """向后兼容：仅抽工具名，不识别 auto 语义。"""
    names, _mode, _preferred = _parse_tools_value(raw_tools)
    return names


@dataclass
class AgentSpec:
    """CrewAI Agent 的最小配置表示。"""

    role: str
    goal: str
    backstory: str
    name: str = DEFAULT_AGENT_NAME
    tools: List[str] = field(default_factory=list)
    verbose: bool = False
    memory: bool = False
    allow_delegation: bool = False
    tool_pool: str = DECLARED_TOOL_POOL
    preferred_tool_names: List[str] = field(default_factory=list)

    @classmethod
    def from_agent_record(cls, agent_record: Dict[str, Any]) -> "AgentSpec":
        """向后兼容：返回单 Agent 配置。"""
        specs = cls.list_from_agent_record(agent_record)
        return specs[0]

    @classmethod
    def list_from_agent_record(cls, agent_record: Dict[str, Any]) -> List["AgentSpec"]:
        """支持多 Agent 配置：优先读取 config.agents(list)，回退到 config.agent(dict)。"""
        config = dict(agent_record.get("config") or {})
        record_name = str(agent_record.get("name") or "Agent").strip() or "Agent"
        record_description = str(agent_record.get("description") or "").strip()
        record_type = str(agent_record.get("type") or "general").strip() or "general"

        raw_agents = config.get("agents")
        agent_items: List[Dict[str, Any]] = []
        if isinstance(raw_agents, list) and raw_agents:
            for item in raw_agents:
                if isinstance(item, dict):
                    agent_items.append(item)

        if not agent_items:
            singular = config.get("agent") if isinstance(config.get("agent"), dict) else config
            if not isinstance(singular, dict):
                singular = {}
            agent_items = [dict(singular)]

        specs: List[AgentSpec] = []
        used_names: set = set()
        for idx, agent_cfg in enumerate(agent_items):
            raw_name = str(agent_cfg.get("name") or agent_cfg.get("id") or "").strip()
            fallback_name = DEFAULT_AGENT_NAME if idx == 0 else f"agent_{idx + 1}"
            candidate_name = raw_name or fallback_name
            unique_name = candidate_name
            suffix = 2
            while unique_name in used_names:
                unique_name = f"{candidate_name}_{suffix}"
                suffix += 1
            used_names.add(unique_name)

            role = str(
                agent_cfg.get("role")
                or (record_name if idx == 0 else unique_name)
            ).strip() or unique_name
            goal = str(
                agent_cfg.get("goal")
                or record_description
                or f"Use the available tools to complete the assigned {record_type} task."
            ).strip()
            backstory = str(
                agent_cfg.get("backstory")
                or record_description
                or f"You are {role}, a reliable agent specialized in {record_type} tasks."
            ).strip()

            tool_names, tool_pool, preferred_tool_names = _parse_tools_value(agent_cfg.get("tools"))

            specs.append(
                cls(
                    role=role,
                    goal=goal,
                    backstory=backstory,
                    name=unique_name,
                    tools=tool_names,
                    verbose=bool(agent_cfg.get("verbose", False)),
                    memory=bool(agent_cfg.get("memory", False)),
                    allow_delegation=bool(agent_cfg.get("allow_delegation", False)),
                    tool_pool=tool_pool,
                    preferred_tool_names=preferred_tool_names,
                )
            )

        if not specs:
            specs.append(
                cls(
                    role=record_name,
                    goal=f"Use the available tools to complete the assigned {record_type} task.",
                    backstory=f"You are {record_name}, a reliable agent specialized in {record_type} tasks.",
                    name=DEFAULT_AGENT_NAME,
                )
            )
        return specs


@dataclass
class TaskSpec:
    """CrewAI Task 的最小配置表示。"""

    task_id: str
    description: str
    expected_output: str
    context: List[str] = field(default_factory=list)
    agent_name: Optional[str] = None

    @classmethod
    def list_from_agent_record(cls, agent_record: Dict[str, Any]) -> List["TaskSpec"]:
        config = dict(agent_record.get("config") or {})
        raw_tasks = config.get("tasks")
        if isinstance(raw_tasks, list) and raw_tasks:
            task_specs: List[TaskSpec] = []
            for idx, item in enumerate(raw_tasks, start=1):
                if not isinstance(item, dict):
                    continue
                task_id = str(item.get("id") or f"task_{idx}").strip() or f"task_{idx}"
                description = str(
                    item.get("description")
                    or "Complete the user's request using the available context and tools. Input: {message}"
                ).strip()
                expected_output = str(
                    item.get("expected_output")
                    or "A complete, accurate, and concise result in markdown format."
                ).strip()
                context = [str(v).strip() for v in (item.get("context") or []) if str(v).strip()]
                agent_name = str(
                    item.get("agent_name")
                    or item.get("agent")
                    or item.get("agent_id")
                    or ""
                ).strip() or None
                task_specs.append(
                    cls(
                        task_id=task_id,
                        description=description,
                        expected_output=expected_output,
                        context=context,
                        agent_name=agent_name,
                    )
                )
            if task_specs:
                return task_specs

        return [
            cls(
                task_id="task_main",
                description="Complete the user's request. Input: {message}",
                expected_output="A complete, accurate, and concise result in markdown format.",
                context=[],
                agent_name=None,
            )
        ]
