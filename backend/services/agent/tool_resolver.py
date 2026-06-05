from __future__ import annotations

import functools
import inspect
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.core.logger import get_app_logger
from backend.services.agent.crew_tools.bridge import build_crew_tool
from backend.services.agent.crew_tools.registry import get_crew_tool_registry
from backend.services.agent.events import (
    record_tool_call_completed,
    record_tool_call_failed,
    record_tool_call_started,
)
from backend.services.agent.settings import is_openclaw_enabled
from backend.services.agent.tool_catalog import ToolCatalog
from backend.services.agent.tool_providers.openclaw_bridge import OpenClawToolBridge
from backend.services.agent.tools.openclaw import build_openclaw_tool
from backend.services.agent.tools.registry import (
    ToolRegistration,
    get_tool_plugin_registry,
)

logger = get_app_logger(__name__)


def _wrap_tool_with_events(
    tool_obj: Any,
    *,
    exposed_name: str,
    provider: str,
    agent_run_id: str,
    target_tool_name: Optional[str] = None,
) -> Any:
    """给 CrewAI `BaseTool` 实例打上事件埋点。

    策略：包装 `tool.func`（CrewAI 所有调用路径 `.run()` / `._run()` 最终
    都落到 `self.func(*args, **kwargs)`）。包装不改变签名，异常原样上抛。

    失败容错：任何 `setattr` 失败都只 log 不抛，保证主流程不受埋点影响。
    """
    if not agent_run_id:
        return tool_obj
    original_func = getattr(tool_obj, "func", None)
    if not callable(original_func):
        return tool_obj

    @functools.wraps(original_func)
    def instrumented(*args, **kwargs):
        started_at = datetime.utcnow()
        started_perf = time.perf_counter()
        raw_args: Any
        if args and not kwargs:
            raw_args = args[0] if len(args) == 1 else list(args)
        elif kwargs and not args:
            raw_args = dict(kwargs)
        else:
            raw_args = {"args": list(args), "kwargs": dict(kwargs)} if (args or kwargs) else None
        record_tool_call_started(
            agent_run_id,
            exposed_name=exposed_name,
            provider=provider,
            target_tool_name=target_tool_name,
            args=raw_args,
            started_at=started_at,
        )
        try:
            result = original_func(*args, **kwargs)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_perf) * 1000)
            record_tool_call_failed(
                agent_run_id,
                exposed_name=exposed_name,
                provider=provider,
                error=str(exc),
                duration_ms=duration_ms,
                started_at=started_at,
            )
            raise
        duration_ms = int((time.perf_counter() - started_perf) * 1000)
        record_tool_call_completed(
            agent_run_id,
            exposed_name=exposed_name,
            provider=provider,
            output=result,
            duration_ms=duration_ms,
            started_at=started_at,
        )
        return result

    try:
        tool_obj.func = instrumented
    except Exception:
        logger.exception(
            "Failed to wrap tool.func for events: tool=%s provider=%s",
            exposed_name,
            provider,
        )
    return tool_obj


class ToolResolver:
    """把"工具名"解析为 CrewAI 可用的工具对象（解析器 / 装配器）。

    注意：这里并不负责工具的"登记注册"（登记由 `tools/registry.py` 的
    `@register_tool` 与 `crew_tools/registry.py` 的 `@register_crew_tool`
    完成）。本类只做解析：按以下优先级将一个 tool_name 组装为可执行对象：

        1. Crew-as-Tool 注册表（子 Crew 打包为工具；对应 provider=crew）
        2. 本地工具插件注册表（@register_tool；provider=local / openclaw）
        3. 工具目录里的 OpenClaw 桥接条目（provider=openclaw 且无本地装饰器注册）

    数据源真相：
      - "这个工具是谁家的" 只看 `ToolCatalogEntry.provider`（由 YAML 或装饰器
        合并而来，详见 `tool_catalog._default_entries`）。
      - 历史上存在的 `EXPLICIT_OPENCLAW_TOOLS` 影子清单已移除，openclaw 工具
        完全通过 `config/agent_tools.yaml` 声明。

    重命名原因：历史上命名为 "ToolRegistry"，但它的职责不是 registry，
    而是 resolver/assembler，容易与 tools/registry.py 混淆。
    """

    def __init__(
        self,
        *,
        openclaw_bridge: Optional[OpenClawToolBridge] = None,
        tool_catalog: Optional[ToolCatalog] = None,
        openclaw_enabled: Optional[bool] = None,
    ):
        self._plugin_registry = get_tool_plugin_registry()
        self._plugin_registry.ensure_loaded()
        self._tool_catalog = tool_catalog or ToolCatalog()
        if openclaw_enabled is None:
            openclaw_enabled = openclaw_bridge is not None or is_openclaw_enabled()
        self._openclaw_enabled = bool(openclaw_enabled)
        self._openclaw_bridge = openclaw_bridge
        if self._openclaw_enabled and self._openclaw_bridge is None:
            self._openclaw_bridge = OpenClawToolBridge()

    def resolve_tools(
        self,
        tool_names: List[str],
        *,
        agent_run_id: str,
        runtime_allowlist: Optional[List[str]] = None,
        crew_tool_context: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        resolved: List[Any] = []
        crew_registry = get_crew_tool_registry()
        for raw_name in tool_names:
            tool_name = str(raw_name or "").strip()
            if not tool_name:
                continue

            crew_registration = crew_registry.get(tool_name)
            if crew_registration is not None and crew_registration.metadata.enabled:
                ctx = dict(crew_tool_context or {})
                crew_tool = build_crew_tool(
                    crew_registration.metadata,
                    parent_agent_run_id=agent_run_id,
                    user_id=str(ctx.get("user_id") or ""),
                    conversation_id=ctx.get("conversation_id"),
                    runtime_config=ctx.get("runtime_config") or {},
                    source_type=str(ctx.get("source_type") or "master-agent"),
                    turn_id=str(ctx.get("turn_id") or "") or None,
                    task_event_callback=ctx.get("task_event_callback")
                    if callable(ctx.get("task_event_callback"))
                    else None,
                    session_nested_tool_hooks=ctx.get("session_nested_tool_hooks"),
                )
                resolved.append(
                    _wrap_tool_with_events(
                        crew_tool,
                        exposed_name=tool_name,
                        provider="crew",
                        agent_run_id=agent_run_id,
                        target_tool_name=crew_registration.metadata.preset_id,
                    )
                )
                continue

            registration: Optional[ToolRegistration] = self._plugin_registry.get(tool_name)
            if registration is not None:
                provider = str(registration.metadata.provider or "").strip().lower()
                if provider == "openclaw" and (not self._openclaw_enabled or self._openclaw_bridge is None):
                    raise RuntimeError(f"OpenClaw integration is disabled for tool: {tool_name}")
                factory = registration.factory
                try:
                    accepts_context = "context" in inspect.signature(factory).parameters
                except (TypeError, ValueError):
                    accepts_context = False
                if accepts_context:
                    local_tool = factory(context=dict(crew_tool_context or {}))
                else:
                    local_tool = factory()
                resolved.append(
                    _wrap_tool_with_events(
                        local_tool,
                        exposed_name=tool_name,
                        provider=provider or "local",
                        agent_run_id=agent_run_id,
                        target_tool_name=registration.metadata.target_tool_name,
                    )
                )
                continue

            entry = self._tool_catalog.get(tool_name)
            provider = str(getattr(entry, "provider", "") or "").strip().lower()
            if provider == "openclaw":
                if not self._openclaw_enabled or self._openclaw_bridge is None:
                    raise RuntimeError(f"OpenClaw integration is disabled for tool: {tool_name}")
                target_tool_name = (
                    str(getattr(entry, "runtime_tool_name", "") or "").strip() or tool_name
                )
                description = str(getattr(entry, "description", "") or "").strip() or None
                openclaw_tool = build_openclaw_tool(
                    exposed_name=tool_name,
                    tool_name=target_tool_name,
                    bridge=self._openclaw_bridge,
                    agent_run_id=agent_run_id,
                    runtime_allowlist=runtime_allowlist,
                    description=description,
                )
                resolved.append(
                    _wrap_tool_with_events(
                        openclaw_tool,
                        exposed_name=tool_name,
                        provider="openclaw",
                        agent_run_id=agent_run_id,
                        target_tool_name=target_tool_name,
                    )
                )
                continue

            raise RuntimeError(f"Tool not registered: {tool_name}")
        return resolved
