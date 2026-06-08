import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tool_providers import OpenClawBridgeError, OpenClawToolBridge


def test_openclaw_bridge_allowlist():
    bridge = OpenClawToolBridge(
        base_url="http://127.0.0.1:18789",
        token="secret",
        allowed_tools=["browser_snapshot", "sessions_list"],
    )

    assert bridge.is_tool_allowed("browser_snapshot") is True
    assert bridge.is_tool_allowed("sessions_list") is True
    assert bridge.is_tool_allowed("dangerous_tool") is False
    assert bridge.is_tool_allowed("browser_snapshot", runtime_allowlist=["browser_snapshot"]) is True
    assert bridge.is_tool_allowed("browser_snapshot", runtime_allowlist=["sessions_list"]) is False


def test_openclaw_bridge_invoke_maps_to_tools_invoke():
    bridge = OpenClawToolBridge(
        base_url="http://127.0.0.1:18789",
        token="secret",
        allowed_tools=["browser_snapshot"],
    )

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.text = '{"data":{"ok":true}}'
    fake_response.json.return_value = {"data": {"ok": True}}

    with patch("httpx.Client.post", return_value=fake_response) as mock_post:
        result = bridge.invoke_tool(
            agent_run_id="run-1",
            tool_name="browser_snapshot",
            args={"tab": "current"},
            dry_run=True,
        )

    assert result["ok"] is True
    assert result["tool_name"] == "browser_snapshot"
    assert result["output"] == {"ok": True}

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["tool"] == "browser_snapshot"
    assert kwargs["json"]["args"] == {"tab": "current"}
    assert kwargs["json"]["dryRun"] is True
    assert kwargs["headers"]["Authorization"] == "Bearer secret"


def test_openclaw_bridge_rejects_non_allowlisted_tool():
    bridge = OpenClawToolBridge(
        base_url="http://127.0.0.1:18789",
        token="secret",
        allowed_tools=["browser_snapshot"],
    )

    with pytest.raises(OpenClawBridgeError, match="not allowed"):
        bridge.invoke_tool(
            agent_run_id="run-1",
            tool_name="gateway_admin",
            args={},
        )
