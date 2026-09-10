"""local_search 工具：调用自托管 SearXNG 做联网搜索，无需第三方付费 API Key。

背景见 https://github.com/tonera/vitoom/issues/3 —— 此前唯一的联网搜索工具
tavily_search 依赖付费的 Tavily Key，未配置时整条联网核实事实的链路会失效。
本工具指向 docker-compose.yml 中的 searxng 服务（同一 vitoom-net 网络内，
无需 Key），是当前唯一内置的联网搜索工具，取代了原先的 tavily_search。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from backend.services.agent.settings import get_local_search_base_url
from backend.services.agent.tools.registry import register_tool

LOCAL_SEARCH_TOOL_NAME = "local_search"

LOCAL_SEARCH_DESCRIPTION = (
    "使用本地自托管的 SearXNG 联网搜索公开网页内容，无需任何第三方付费 API Key。"
    "适合旅行规划、新闻和实时网页检索，也适合模型自身知识库之外或拿不准的公开事实、"
    "人物、影视、历史、科普、百科类问题——凡是需要核实事实、查证具体人名/年份/数据，"
    "或用户明确要求『上网查/搜一下/别瞎编』时都应优先使用，不要仅凭记忆编造答案。"
)

LOCAL_SEARCH_DOCSTRING = (
    "Search the web via a self-hosted SearXNG instance (no API key required). "
    "Input can be plain text or a JSON object string with query/max_results/language."
)

DEFAULT_MAX_RESULTS = 5
MIN_MAX_RESULTS = 1
MAX_MAX_RESULTS = 10
REQUEST_TIMEOUT_SECONDS = 15.0


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


def _coerce_max_results(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_RESULTS
    return max(MIN_MAX_RESULTS, min(MAX_MAX_RESULTS, parsed))


def _format_results(query: str, raw_results: List[Dict[str, Any]], max_results: int) -> str:
    trimmed = raw_results[:max_results]
    if not trimmed:
        return json.dumps({"query": query, "results": []}, ensure_ascii=False, indent=2)
    formatted = [
        {
            "title": str(item.get("title") or "").strip(),
            "url": str(item.get("url") or "").strip(),
            "content": str(item.get("content") or "").strip(),
        }
        for item in trimmed
    ]
    return json.dumps({"query": query, "results": formatted}, ensure_ascii=False, indent=2)


@register_tool(
    name=LOCAL_SEARCH_TOOL_NAME,
    description=LOCAL_SEARCH_DESCRIPTION,
    tags=["web", "search", "trivia", "fact-check", "local", "联网", "事实核查", "本地搜索"],
    provider="local",
    enabled=True,
)
def build_local_search_tool():
    try:
        from crewai.tools import tool as crewai_tool
    except Exception as e:
        raise RuntimeError("crewai is required to register native agent tools") from e

    @crewai_tool(LOCAL_SEARCH_TOOL_NAME)
    def local_search(arguments: str = "") -> str:
        """Placeholder docstring; overridden below."""
        payload = _coerce_tool_args(arguments)
        query = str(payload.get("query") or payload.get("input") or "").strip()
        if not query:
            raise RuntimeError("local_search requires a non-empty query")

        max_results = _coerce_max_results(payload.get("max_results"))
        language = str(payload.get("language") or "").strip()

        base_url = get_local_search_base_url()
        if not base_url:
            raise RuntimeError("LOCAL_SEARCH_BASE_URL is not configured")

        params: Dict[str, Any] = {"q": query, "format": "json"}
        if language:
            params["language"] = language

        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = client.get(f"{base_url}/search", params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"local_search request to {base_url} failed: {e}") from e

        return _format_results(query, data.get("results") or [], max_results)

    local_search.__doc__ = LOCAL_SEARCH_DOCSTRING
    return local_search
