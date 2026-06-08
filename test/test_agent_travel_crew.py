"""
对标 CrewAI 官方教程的「城市专家 + 旅行规划师」双 Agent 协作用例测试。

验证：
1. 旅行预置 Agent 解析出两个 AgentSpec（city_expert + travel_concierge）。
2. 任务与各自 Agent 正确绑定，plan_itinerary_task 依赖 identify_attractions_task。
3. CrewFactory 可以把规格拼装成 CrewAI Crew（无需真正调用 LLM）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _install_fake_crewai_module() -> None:
    """当运行环境未安装 crewai 时，注入一个最小可用的 stub，便于单元测试。"""
    if "crewai" in sys.modules:
        return

    class _FakeAgent:
        def __init__(self, role="", goal="", backstory="", llm=None, tools=None, verbose=False, memory=False, allow_delegation=False, **_):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.llm = llm
            self.tools = list(tools or [])
            self.verbose = verbose
            self.memory = memory
            self.allow_delegation = allow_delegation

    class _FakeTask:
        def __init__(self, description="", expected_output="", agent=None, context=None, **_):
            self.description = description
            self.expected_output = expected_output
            self.agent = agent
            self.context = list(context or [])

    class _FakeProcess:
        sequential = "sequential"
        hierarchical = "hierarchical"

    class _FakeCrew:
        def __init__(self, agents=None, tasks=None, process=None, verbose=False, **_):
            self.agents = list(agents or [])
            self.tasks = list(tasks or [])
            self.process = process
            self.verbose = verbose

        def kickoff(self, inputs=None):
            return {"inputs": inputs, "tasks": len(self.tasks), "agents": len(self.agents)}

    fake_module = types.ModuleType("crewai")
    fake_module.Agent = _FakeAgent
    fake_module.Task = _FakeTask
    fake_module.Crew = _FakeCrew
    fake_module.Process = _FakeProcess
    fake_module.LLM = lambda **kwargs: SimpleNamespace(**kwargs)

    fake_tools_module = types.ModuleType("crewai.tools")

    def _fake_tool(name=None):
        def decorator(func):
            func.name = str(name or getattr(func, "__name__", "tool"))
            return func
        return decorator

    fake_tools_module.tool = _fake_tool

    sys.modules["crewai"] = fake_module
    sys.modules["crewai.tools"] = fake_tools_module


_install_fake_crewai_module()

from backend.services.agent.crews import CrewFactory  # noqa: E402
from backend.services.agent.presets import (  # noqa: E402
    TRAVEL_PRESET_AGENT_ID,
    get_preset_definition,
)
from backend.services.agent.specs import AgentSpec, TaskSpec  # noqa: E402
from backend.services.agent.types import AgentCommand  # noqa: E402


def _fake_record() -> dict:
    definition = get_preset_definition(TRAVEL_PRESET_AGENT_ID)
    assert definition is not None, "preset-travel-planner-agent YAML 未被加载"
    return {
        "id": definition.id,
        "name": definition.name,
        "description": definition.description,
        "type": definition.agent_type,
        "config": definition.resolved_config(),
        "status": "active",
        "is_preset": True,
    }


def test_travel_preset_parses_two_agent_specs_and_binds_tasks():
    record = _fake_record()

    agent_specs = AgentSpec.list_from_agent_record(record)
    task_specs = TaskSpec.list_from_agent_record(record)

    assert [spec.name for spec in agent_specs] == ["city_expert", "travel_concierge"]
    assert all("tavily_search" in spec.tools for spec in agent_specs)

    city_expert = next(spec for spec in agent_specs if spec.name == "city_expert")
    travel_concierge = next(spec for spec in agent_specs if spec.name == "travel_concierge")
    assert "本地城市专家" in city_expert.role
    assert "旅行规划师" in travel_concierge.role

    task_ids = [task.task_id for task in task_specs]
    assert task_ids == ["identify_attractions_task", "plan_itinerary_task"]

    identify_task = task_specs[0]
    plan_task = task_specs[1]
    assert identify_task.agent_name == "city_expert"
    assert plan_task.agent_name == "travel_concierge"
    assert plan_task.context == ["identify_attractions_task"]


def test_crew_factory_builds_two_agents_with_proper_tool_binding():
    record = _fake_record()
    agent_specs = AgentSpec.list_from_agent_record(record)
    task_specs = TaskSpec.list_from_agent_record(record)

    command = AgentCommand(
        user_id="user-1",
        agent_id=TRAVEL_PRESET_AGENT_ID,
        message="帮我规划一次三日游，目的地是东京",
        context={"city": "东京"},
    )

    fake_tool = SimpleNamespace(name="tavily_search")

    fake_llm = object()
    with patch("backend.services.agent.crews.factory.build_crewai_llm", return_value=fake_llm) as mocked_build_llm:
        crew, inputs = CrewFactory().build(
            agent_specs=agent_specs,
            task_specs=task_specs,
            command=command,
            tools_by_name={"tavily_search": fake_tool},
            process_name="sequential",
        )

    mocked_build_llm.assert_called_once_with(
        preferred_model_name=None,
        effective_user_id="user-1",
        stream=True,
    )

    assert inputs["message"].startswith("帮我规划")
    assert inputs["city"] == "东京"

    crewai_agents = list(getattr(crew, "agents", []) or [])
    assert len(crewai_agents) == 2

    roles = [str(getattr(agent, "role", "")) for agent in crewai_agents]
    assert any("本地城市专家" in role for role in roles)
    assert any("旅行规划师" in role for role in roles)

    for agent in crewai_agents:
        agent_tools = list(getattr(agent, "tools", []) or [])
        assert len(agent_tools) == 1
        assert getattr(agent_tools[0], "name", "") == "tavily_search"

    crew_tasks = list(getattr(crew, "tasks", []) or [])
    assert len(crew_tasks) == 2

    identify_task, plan_task = crew_tasks
    identify_agent = getattr(identify_task, "agent", None)
    plan_agent = getattr(plan_task, "agent", None)
    assert identify_agent is not None and plan_agent is not None
    assert "本地城市专家" in str(getattr(identify_agent, "role", ""))
    assert "旅行规划师" in str(getattr(plan_agent, "role", ""))

    plan_context = list(getattr(plan_task, "context", []) or [])
    assert any(ctx is identify_task for ctx in plan_context)


def test_crew_factory_is_backward_compatible_with_single_agent_record():
    record = {
        "id": "single-agent",
        "name": "简单助手",
        "description": "测试单 Agent 向后兼容",
        "type": "general",
        "config": {
            "agent": {
                "role": "通用助手",
                "goal": "完成用户请求",
                "backstory": "一个测试用的通用助手。",
                "tools": [],
            },
            "tasks": [
                {
                    "id": "task_main",
                    "description": "回答：{message}",
                    "expected_output": "简短的 Markdown 回答。",
                }
            ],
        },
    }

    agent_specs = AgentSpec.list_from_agent_record(record)
    task_specs = TaskSpec.list_from_agent_record(record)
    assert len(agent_specs) == 1
    assert agent_specs[0].name == "primary"
    assert task_specs[0].agent_name is None

    command = AgentCommand(user_id="u1", agent_id="single-agent", message="你好")

    fake_llm = object()
    with patch("backend.services.agent.crews.factory.build_crewai_llm", return_value=fake_llm):
        crew, _ = CrewFactory().build(
            agent_specs=agent_specs,
            task_specs=task_specs,
            command=command,
            tools_by_name={},
            process_name="sequential",
        )

    crewai_agents = list(getattr(crew, "agents", []) or [])
    assert len(crewai_agents) == 1
    crew_tasks = list(getattr(crew, "tasks", []) or [])
    assert getattr(crew_tasks[0], "agent", None) is crewai_agents[0]
