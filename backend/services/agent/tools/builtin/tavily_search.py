"""Tavily 联网搜索工具。"""

from __future__ import annotations

import json
from typing import Any, Dict

from backend.services.agent.settings import get_tavily_api_key
from backend.services.agent.tools.registry import register_tool

TAVILY_TOOL_NAME = "tavily_search"

TAVILY_DESCRIPTION = (
    "使用 Tavily 联网搜索公开网页信息，适合旅行规划、资料调研、事实补充和实时信息查询。"
)

TAVILY_DOCSTRING = (
    "Search the web with Tavily. Input can be plain text or a JSON object string with "
    "query/search_depth/topic/max_results."
)


def _coerce_tool_args(raw_input: Any) -> Dict[str, Any]:
    if raw_input is None:
        return {}
    if isinstance(raw_input, dict):
        return dict(raw_input)
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
        return {"query": text}
    return {"query": str(raw_input)}


def _stringify_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


@register_tool(
    name=TAVILY_TOOL_NAME,
    description=TAVILY_DESCRIPTION,
    tags=["web", "search", "travel", "research", "联网"],
    provider="local",
    enabled=True,
)
def build_tavily_search_tool():
    try:
        from crewai.tools import tool as crewai_tool
    except Exception as e:
        raise RuntimeError("crewai is required to register native agent tools") from e

    @crewai_tool(TAVILY_TOOL_NAME)
    def tavily_search(arguments: str = "") -> str:
        """Placeholder docstring; overridden below."""
        api_key = get_tavily_api_key()
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not configured. Set it in the root .env or process environment."
            )

        try:
            from tavily import TavilyClient
        except Exception as e:
            raise RuntimeError("tavily-python is required to use tavily_search") from e

        payload = _coerce_tool_args(arguments)
        query = str(payload.get("query") or payload.get("input") or "").strip()
        if not query:
            raise RuntimeError("tavily_search requires a non-empty query")

        search_depth = str(payload.get("search_depth") or "advanced").strip() or "advanced"
        topic = str(payload.get("topic") or "general").strip() or "general"

        try:
            max_results = max(1, min(10, int(payload.get("max_results") or 5)))
        except (TypeError, ValueError):
            max_results = 5

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            include_answer=bool(payload.get("include_answer", False)),
            include_images=bool(payload.get("include_images", False)),
            include_raw_content=bool(payload.get("include_raw_content", False)),
        )
        return _stringify_output(response)

    tavily_search.__doc__ = TAVILY_DOCSTRING
    return tavily_search
