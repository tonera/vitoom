"""Master Agent 全局工具池筛选单元测试。"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tool_catalog import ToolCatalogEntry
from backend.services.agent.tool_selection import ToolSelectionService
from backend.services.agent.types import AgentCommand


class FakeCatalog:
    def __init__(self, entries):
        self._entries = entries

    def get(self, name: str):
        return self._entries.get(name)

    def all(self):
        return dict(self._entries)


def _make_catalog():
    return FakeCatalog(
        {
            "tavily_search": ToolCatalogEntry(
                name="tavily_search",
                description="使用 Tavily 联网搜索公开网页信息。",
                tags=["web", "search", "research"],
                provider="local",
                enabled=True,
            ),
            "travel_planner": ToolCatalogEntry(
                name="travel_planner",
                description="多 Agent 协作的旅行规划师，产出多日行程。",
                tags=["travel", "trip", "旅行", "行程", "规划", "crew"],
                provider="crew",
                enabled=True,
                target_tool_name="preset-travel-planner-agent",
            ),
            "analyze_media": ToolCatalogEntry(
                name="analyze_media",
                description="分析图片或视频 URL 的内容。",
                tags=["image", "video", "图片", "视频", "媒体"],
                provider="local",
                enabled=True,
            ),
            "file_reader": ToolCatalogEntry(
                name="file_reader",
                description="读取本地文件内容。",
                tags=["file", "local"],
                provider="local",
                enabled=True,
            ),
        }
    )


def test_global_pool_selects_travel_planner_for_trip_query():
    catalog = _make_catalog()
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="u1",
        agent_id="preset-master-agent",
        message="帮我规划一个东京3日游，要包含景点和美食",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=1),
    ):
        selected = selector.select_tool_names(
            [],
            command=command,
            pool="global",
        )

    assert selected == ["travel_planner"]


def test_global_pool_falls_back_to_catalog_when_declared_empty():
    catalog = _make_catalog()
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="u1",
        agent_id="preset-master-agent",
        message="帮我搜索一下 2026 年北京 GDP 数据",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=2),
    ):
        selected = selector.select_tool_names(
            [],
            command=command,
            pool="global",
        )

    assert "tavily_search" in selected
    assert len(selected) <= 2


def test_global_pool_respects_preferred_tools_when_present():
    catalog = _make_catalog()
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="u1",
        agent_id="preset-master-agent",
        message="帮我搜索东京景点并整理成 3 日行程",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=2),
    ):
        selected = selector.select_tool_names(
            [],
            command=command,
            pool="global",
            preferred_tool_names=["tavily_search"],
        )

    assert "tavily_search" in selected
    assert "travel_planner" in selected


def test_global_pool_returns_empty_when_request_is_plain_chat():
    catalog = _make_catalog()
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="u1",
        agent_id="preset-master-agent",
        message="解释一下 Python 闭包和装饰器的区别",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=2),
    ):
        selected = selector.select_tool_names(
            [],
            command=command,
            pool="global",
        )

    assert selected == []


def test_global_pool_allows_analyze_media_for_media_capability_question():
    catalog = FakeCatalog(
        {
            "tavily_search": ToolCatalogEntry(
                name="tavily_search",
                description="使用 Tavily 联网搜索公开网页信息。",
                tags=["web", "search", "research"],
                provider="local",
                enabled=True,
            ),
            "travel_planner": ToolCatalogEntry(
                name="travel_planner",
                description="多 Agent 协作的旅行规划师，产出多日行程。",
                tags=["travel", "trip", "旅行", "行程", "规划", "crew"],
                provider="crew",
                enabled=True,
                target_tool_name="preset-travel-planner-agent",
            ),
            "analyze_media": ToolCatalogEntry(
                name="analyze_media",
                description="分析图片或视频 URL 的内容。",
                tags=["image", "video", "图片", "视频", "媒体"],
                provider="local",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="u1",
        agent_id="preset-master-agent",
        message="你能识别图片和视频内容吗？",
        context={"original_user_message": "你能识别图片和视频内容吗？"},
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=3),
    ):
        selected = selector.select_tool_names(
            [],
            command=command,
            pool="global",
            preferred_tool_names=["analyze_media"],
        )

    assert selected == ["analyze_media"]
