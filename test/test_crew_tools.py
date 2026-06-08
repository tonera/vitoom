"""Crew-as-Tool 注册表 + 目录集成 + 运行时桥接的单元测试。"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.crew_tools import (
    get_crew_tool_registry,
    list_crew_tools,
)
from backend.services.agent.crew_tools.bridge import _coerce_query, _run_crew_sync
from backend.services.agent.crew_tools.registry import CrewToolMetadata
from backend.services.agent.tool_catalog import ToolCatalog


def test_travel_planner_crew_tool_is_registered():
    metadata_by_name = {m.name: m for m in list_crew_tools()}
    assert "travel_planner" in metadata_by_name
    meta = metadata_by_name["travel_planner"]
    assert meta.preset_id == "preset-travel-planner-agent"
    assert meta.enabled is True
    assert any("travel" == t or "旅行" == t for t in meta.tags)


def test_tool_catalog_exposes_travel_planner_as_crew_provider():
    catalog = ToolCatalog()
    entry = catalog.get("travel_planner")
    assert entry is not None
    assert entry.provider == "crew"
    assert entry.target_tool_name == "preset-travel-planner-agent"
    assert "crew" in entry.tags


def test_coerce_query_handles_plain_string_and_json_payload():
    assert _coerce_query("规划东京3日游") == "规划东京3日游"
    assert _coerce_query('{"query": "北京"}') == "北京"
    assert _coerce_query({"query": "上海"}) == "上海"
    assert _coerce_query({"input": "成都"}) == "成都"


def test_run_crew_sync_reuses_preset_and_invokes_kickoff():
    """校验 bridge 能拿到正确 preset 并最终调到 Crew.kickoff。"""
    from backend.services.agent.specs import AgentSpec, TaskSpec

    metadata = CrewToolMetadata(
        name="travel_planner",
        preset_id="preset-travel-planner-agent",
        description="test",
        tags=[],
    )

    fake_crew = MagicMock()
    fake_crew.kickoff.return_value = "OK"

    with (
        patch(
            "backend.services.agent.crew_tools.bridge.CrewFactory"
        ) as factory_cls,
        patch(
            "backend.services.agent.crew_tools.bridge.ToolSelectionService"
        ) as selector_cls,
        patch(
            "backend.services.agent.tool_resolver.ToolResolver"
        ) as registry_cls,
        patch(
            "backend.services.agent.crew_tools.bridge.get_preset_definition"
        ) as get_def,
    ):
        definition = MagicMock()
        definition.name = "Travel"
        definition.agent_type = "general"
        definition.resolved_config.return_value = {
            "agents": [
                {
                    "name": "city",
                    "role": "r",
                    "goal": "g",
                    "backstory": "b",
                    "tools": ["tavily_search"],
                }
            ],
            "tasks": [
                {
                    "id": "t1",
                    "agent": "city",
                    "description": "d",
                    "expected_output": "e",
                }
            ],
        }
        get_def.return_value = definition
        selector = selector_cls.return_value
        selector.select_tool_names.return_value = []
        registry = registry_cls.return_value
        registry.resolve_tools.return_value = []
        factory = factory_cls.return_value
        factory.build.return_value = (fake_crew, {"message": "q"})

        result, usage_metrics = _run_crew_sync(
            metadata=metadata,
            query="规划东京3日游",
            user_id="user-1",
            child_run_id="child-1",
            runtime_config={"max_tools_per_run": 3},
            source_type="master-agent",
        )

    assert result == "OK"
    assert isinstance(usage_metrics, dict)
    fake_crew.kickoff.assert_called_once()
    get_def.assert_called_once_with("preset-travel-planner-agent")


def test_crew_tool_registry_all_registrations_contains_travel_planner():
    registry = get_crew_tool_registry()
    regs = registry.all_registrations()
    assert "travel_planner" in regs
    assert regs["travel_planner"].metadata.enabled is True
