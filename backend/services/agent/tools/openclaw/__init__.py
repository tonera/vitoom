"""OpenClaw 工具的 CrewAI 装配层。

所有 openclaw 工具的**元数据真相**由 ``config/agent_tools.yaml`` 提供
（`provider: openclaw` + `target_tool_name: ...`），本模块只负责在运行期
根据元数据把单个 openclaw 工具组装成 CrewAI 可挂载的对象。

历史上这里还维护过一份 ``EXPLICIT_OPENCLAW_TOOLS`` 硬编码清单，与 YAML
重复；已移除，避免"两处真相"。新增 openclaw 工具只需在 YAML 里加一条
``provider: openclaw`` 条目即可。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from backend.services.agent.tool_providers.openclaw_bridge import OpenClawToolBridge


def _coerce_tool_args(raw_input: Any) -> Dict[str, Any]:
    if raw_input is None:
        return {}
    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, str):
        text = raw_input.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"input": text}
    return {"input": str(raw_input)}


def build_openclaw_tool(
    *,
    exposed_name: Optional[str] = None,
    tool_name: str,
    bridge: OpenClawToolBridge,
    agent_run_id: str,
    runtime_allowlist: Optional[List[str]] = None,
    description: Optional[str] = None,
):
    """构造一个 CrewAI 可直接挂载的 OpenClaw 工具。"""
    try:
        from crewai.tools import tool as crewai_tool
    except Exception as e:
        raise RuntimeError("crewai is required to register OpenClaw bridge tools") from e

    normalized_tool_name = str(tool_name or "").strip()
    normalized_exposed_name = str(exposed_name or normalized_tool_name).strip()
    if not normalized_tool_name:
        raise RuntimeError("tool_name is required")
    if not normalized_exposed_name:
        raise RuntimeError("exposed_name is required")

    @crewai_tool(normalized_exposed_name)
    def openclaw_tool(arguments: str = "") -> str:
        """Invoke the configured OpenClaw tool with a JSON object string as input."""
        response = bridge.invoke_tool(
            agent_run_id=agent_run_id,
            tool_name=normalized_tool_name,
            args=_coerce_tool_args(arguments),
            runtime_allowlist=runtime_allowlist,
        )
        return bridge.stringify_output(response.get("output"))

    if description:
        openclaw_tool.__doc__ = description

    return openclaw_tool
