"""覆盖新插件体系：工具自动发现 + 预置 YAML/Python 聚合。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.presets import (  # noqa: E402
    TRAVEL_PRESET_AGENT_ID,
    get_preset_definition,
    list_preset_definitions,
)
from backend.services.agent.tool_catalog import ToolCatalog  # noqa: E402
from backend.services.agent.tools.registry import get_tool_plugin_registry  # noqa: E402


def test_tool_plugin_registry_auto_discovers_tavily():
    registry = get_tool_plugin_registry()
    registrations = registry.all_registrations()
    assert "tavily_search" in registrations, "tavily_search 应该通过 builtin 自动发现"
    meta = registrations["tavily_search"].metadata
    assert meta.provider == "local"
    assert "search" in meta.tags


def test_tool_catalog_includes_tavily_from_registry_metadata():
    catalog = ToolCatalog()
    entry = catalog.get("tavily_search")
    assert entry is not None, "tavily_search 应该出现在 ToolCatalog 中"
    assert entry.enabled is True
    assert any(tag in entry.tags for tag in ["search", "web", "travel"])


def test_preset_registry_loads_local_and_travel_from_yaml():
    definitions = list_preset_definitions(include_disabled=True)
    ids = {definition.id for definition in definitions}
    assert "preset-local-agent" in ids
    assert TRAVEL_PRESET_AGENT_ID in ids


def test_travel_preset_definition_has_two_agents_bound_to_tasks():
    definition = get_preset_definition(TRAVEL_PRESET_AGENT_ID)
    assert definition is not None
    config = definition.resolved_config()

    agents = config.get("agents") or []
    assert isinstance(agents, list) and len(agents) == 2
    agent_names = [agent.get("name") for agent in agents]
    assert agent_names == ["city_expert", "travel_concierge"]

    tasks = config.get("tasks") or []
    assert [task.get("agent") for task in tasks] == ["city_expert", "travel_concierge"]


def test_openclaw_preset_is_registered_via_python_builtin():
    definition = get_preset_definition("preset-openclaw-agent")
    assert definition is not None
    assert definition.agent_type == "openclaw"
    assert definition.requires_openclaw is True
    assert definition.source == "python"
