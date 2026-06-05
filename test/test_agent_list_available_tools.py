import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tool_catalog import ToolCatalog
from backend.services.agent.tools.builtin.list_available_tools import (  # noqa: E402
    _build_capability_markdown,
)


def test_list_available_tools_is_exposed_in_catalog():
    entry = ToolCatalog().get("list_available_tools")

    assert entry is not None
    assert entry.provider == "local"
    assert entry.enabled is True
    assert "工具" in entry.description


def test_list_available_tools_markdown_lists_available_capabilities():
    with patch(
        "backend.services.agent.settings.is_openclaw_enabled",
        return_value=False,
    ):
        markdown = _build_capability_markdown(runtime_allowlist=[])

    assert "当前已上线的工具能力如下" in markdown
    assert "`list_available_tools`" in markdown
    assert "`image_generator`" in markdown
    assert "`travel_planner`" in markdown
    assert "`openclaw_browser_snapshot`" not in markdown
