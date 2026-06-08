"""把 preset-travel-planner-agent 暴露为 Master Agent 可调用的 crew-tool。"""

from __future__ import annotations

from backend.services.agent.crew_tools.registry import register_crew_tool


@register_crew_tool(
    name="travel_planner",
    preset_id="preset-travel-planner-agent",
    description=(
        "Multi-agent travel planner: searches destination attractions, dining, and practical info online, "
        "and produces a multi-day itinerary in Markdown. Best for explicit requests like "
        "'plan a 3-day trip to Tokyo' or 'help me arrange travel'."
    ),
    tags=["travel", "trip", "itinerary", "旅行", "行程", "规划"],
    enabled=True,
    input_hint=(
        "One-line summary: destination + days + preferences (optional, e.g. family trip/food-focused/budget). "
        "Be specific, e.g. 'Plan a 3-day Tokyo family trip in April focused on sights and food'"
    ),
    max_wait_seconds=600,
)
def _register_travel_planner() -> None:
    """装饰器调用即完成注册，此函数体保留为空。"""
    return None
