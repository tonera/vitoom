"""OpenClaw 工具型预置 Agent。

由于需要根据运行时 OpenClaw 工具白名单动态生成 `tools` 清单，保留在 Python 中。
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.services.agent.presets.registry import register_preset
from backend.services.agent.settings import get_openclaw_allowed_tools
from backend.services.agent.tool_catalog import ToolCatalog


def _resolve_openclaw_tool_names() -> List[str]:
    normalized_allowed = get_openclaw_allowed_tools()
    allowed_target_names = set(normalized_allowed)
    catalog = ToolCatalog()
    tool_names: List[str] = []
    for entry in catalog.all().values():
        if entry.provider != "openclaw" or not entry.enabled:
            continue
        if (
            allowed_target_names
            and "*" not in allowed_target_names
            and entry.runtime_tool_name not in allowed_target_names
        ):
            continue
        tool_names.append(entry.name)
    tool_names.sort()
    return tool_names


@register_preset(
    id="preset-openclaw-agent",
    name="OpenClaw 工具助手",
    agent_type="openclaw",
    description="Minimal runnable OpenClaw tool preset agent",
    requires_openclaw=True,
)
def build_openclaw_preset_config() -> Dict[str, Any]:
    tool_names = _resolve_openclaw_tool_names()
    return {
        "agent": {
            "role": "OpenClaw 工具助手",
            "goal": (
                "Safely use the registered OpenClaw tools to inspect sessions, browser state, "
                "and help the user finish general tasks."
            ),
            "backstory": (
                "You are a reliable operations-oriented assistant. "
                "You can use a curated set of OpenClaw tools to inspect browser state and session "
                "information, then produce concise, grounded answers."
            ),
            "tools": tool_names,
            "verbose": True,
            "memory": True,
            "allow_delegation": False,
        },
        "tasks": [
            {
                "id": "task_main",
                "description": (
                    "Understand the user's request, use the available tools only when they are helpful, "
                    "and complete the request. User input: {message}"
                ),
                "expected_output": "A concise markdown response grounded in the available context and tool results.",
            }
        ],
        "runtime_defaults": {
            "process": "sequential",
            "priority": 5,
            "tool_allowlist": get_openclaw_allowed_tools(),
        },
    }
