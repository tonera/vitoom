"""Crew-as-Tool 注册中心：把专业 Crew 打包成 Master Agent 可以调用的工具。

典型用法：

    @register_crew_tool(
        name="travel_planner",
        preset_id="preset-travel-planner-agent",
        description="Plan multi-day itineraries, attractions, and dining",
        tags=["旅行", "行程", "planner"],
    )
    def _register_travel_planner() -> None:
        # 仅作标记，真正的配置由装饰器参数给出；本函数体通常为空。
        return None
"""

from __future__ import annotations

from .registry import (
    CrewToolMetadata,
    CrewToolRegistration,
    get_crew_tool_registry,
    list_crew_tools,
    register_crew_tool,
)

__all__ = [
    "CrewToolMetadata",
    "CrewToolRegistration",
    "build_crew_tool",
    "get_crew_tool_registry",
    "get_master_crew_tools",
    "list_crew_tools",
    "register_crew_tool",
]


def __getattr__(name):  # pragma: no cover - thin lazy import facade
    if name in {"build_crew_tool", "get_master_crew_tools"}:
        from . import bridge

        return getattr(bridge, name)
    raise AttributeError(name)
