"""列出当前 Agent 可用工具能力。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from backend.services.agent.tools.registry import register_tool

LIST_AVAILABLE_TOOLS_NAME = "list_available_tools"

LIST_AVAILABLE_TOOLS_DESCRIPTION = (
    "列出当前已上线且可用的工具能力，帮助用户了解系统现在能做什么。"
    "仅当用户询问『你有哪些工具/你能做什么/当前有哪些能力』"
    "这类 Agent 工具能力清单问题时使用；不要用于 slash command 帮助或执行具体任务。"
)

LIST_AVAILABLE_TOOLS_DOCSTRING = (
    "Return a Markdown list of currently available agent tools and capabilities. "
    "Use this only for user questions about agent tools or system capabilities, not slash commands."
)

_PROVIDER_LABELS = {
    "local": "本地工具",
    "crew": "专业 Agent",
    "openclaw": "浏览器工具",
    "mcp": "外部工具",
}


def _first_sentence(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    separators = ("。", "；", ";", ". ")
    positions = [normalized.find(sep) for sep in separators if normalized.find(sep) >= 0]
    if not positions:
        return normalized
    end = min(positions) + 1
    return normalized[:end].strip()


def _is_entry_available(entry: Any, *, runtime_allowlist: Iterable[str]) -> bool:
    if not getattr(entry, "enabled", True):
        return False

    provider = str(getattr(entry, "provider", "") or "").strip().lower()
    requires_openclaw = bool(getattr(entry, "requires_openclaw", False))
    if provider == "openclaw" or requires_openclaw:
        from backend.services.agent.settings import is_openclaw_enabled

        if not is_openclaw_enabled():
            return False
        allowlist = {str(item).strip() for item in runtime_allowlist if str(item).strip()}
        if allowlist and "*" not in allowlist:
            runtime_name = str(getattr(entry, "runtime_tool_name", "") or "").strip()
            exposed_name = str(getattr(entry, "name", "") or "").strip()
            if not ({runtime_name, exposed_name} & allowlist):
                return False
    return True


def _build_capability_markdown(*, runtime_allowlist: Iterable[str]) -> str:
    from backend.services.agent.tool_catalog import ToolCatalog

    catalog = ToolCatalog()
    entries = [
        entry
        for entry in catalog.all().values()
        if _is_entry_available(entry, runtime_allowlist=runtime_allowlist)
    ]
    if not entries:
        return "当前没有可用的已上线工具。"

    grouped: Dict[str, List[Any]] = {}
    for entry in entries:
        provider = str(getattr(entry, "provider", "") or "local").strip().lower() or "local"
        grouped.setdefault(provider, []).append(entry)

    lines: List[str] = ["当前已上线的工具能力如下："]
    provider_order = ["local", "crew", "openclaw", "mcp"]
    for provider in provider_order + sorted(set(grouped) - set(provider_order)):
        provider_entries = grouped.get(provider)
        if not provider_entries:
            continue
        lines.append("")
        lines.append(f"### {_PROVIDER_LABELS.get(provider, provider)}")
        for entry in sorted(provider_entries, key=lambda item: str(getattr(item, "name", ""))):
            name = str(getattr(entry, "name", "") or "").strip()
            description = _first_sentence(str(getattr(entry, "description", "") or "").strip())
            if description:
                lines.append(f"- `{name}`：{description}")
            else:
                lines.append(f"- `{name}`")
    return "\n".join(lines).strip()


@register_tool(
    name=LIST_AVAILABLE_TOOLS_NAME,
    description=LIST_AVAILABLE_TOOLS_DESCRIPTION,
    tags=["tools", "capabilities", "工具列表", "能力清单", "功能介绍"],
    provider="local",
    enabled=True,
)
def build_list_available_tools_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    runtime_config = dict(ctx.get("runtime_config") or {})
    runtime_allowlist = runtime_config.get("tool_allowlist") or []

    try:
        from crewai.tools import BaseTool
    except Exception as e:
        raise RuntimeError("crewai is required to register native agent tools") from e

    try:
        from pydantic import BaseModel, Field
    except Exception as e:
        raise RuntimeError("pydantic is required to build list_available_tools tool") from e

    class ListAvailableToolsArgs(BaseModel):
        include_details: bool = Field(
            default=True,
            description="Whether to return a short description for each tool. Usually keep default true.",
        )

    class ListAvailableToolsTool(BaseTool):
        name: str = LIST_AVAILABLE_TOOLS_NAME
        description: str = LIST_AVAILABLE_TOOLS_DESCRIPTION
        args_schema: type = ListAvailableToolsArgs

        def _run(self, include_details: bool = True, **_ignored: Any) -> str:
            del include_details
            return _build_capability_markdown(runtime_allowlist=runtime_allowlist)

    ListAvailableToolsTool.__doc__ = LIST_AVAILABLE_TOOLS_DOCSTRING
    return ListAvailableToolsTool()
