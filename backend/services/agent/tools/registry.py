"""Agent 工具插件注册中心。

提供 `@register_tool` 装饰器以及基于 `backend/services/agent/tools/builtin/` 目录的
自动发现机制。新增一个工具的完整成本：在 `builtin/` 下新建一个模块并用装饰器注册，
框架其余代码无需改动。
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_BUILTIN_PACKAGE = "backend.services.agent.tools.builtin"


@dataclass(frozen=True)
class ToolMetadata:
    """工具在目录/筛选层需要的元数据，与执行实现解耦。"""

    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    provider: str = "local"
    enabled: bool = True
    always_include: bool = False
    requires_openclaw: bool = False
    target_tool_name: Optional[str] = None


@dataclass(frozen=True)
class ToolRegistration:
    metadata: ToolMetadata
    factory: Callable[..., Any]


class _ToolPluginRegistry:
    def __init__(self) -> None:
        self._registrations: Dict[str, ToolRegistration] = {}
        self._lock = threading.RLock()
        self._loaded = False

    def register(self, *, metadata: ToolMetadata, factory: Callable[..., Any]) -> None:
        if not metadata.name:
            raise ValueError("tool name is required to register")
        with self._lock:
            self._registrations[metadata.name] = ToolRegistration(metadata=metadata, factory=factory)

    def ensure_loaded(self, package_name: str = _BUILTIN_PACKAGE) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                pkg = importlib.import_module(package_name)
            except ModuleNotFoundError:
                self._loaded = True
                return
            pkg_path = getattr(pkg, "__path__", None)
            if not pkg_path:
                self._loaded = True
                return
            for module_info in pkgutil.iter_modules(pkg_path, prefix=f"{package_name}."):
                short_name = module_info.name.rsplit(".", 1)[-1]
                if short_name.startswith("_"):
                    continue
                importlib.import_module(module_info.name)
            self._loaded = True

    def all_registrations(self) -> Dict[str, ToolRegistration]:
        self.ensure_loaded()
        with self._lock:
            return dict(self._registrations)

    def get(self, name: str) -> Optional[ToolRegistration]:
        self.ensure_loaded()
        with self._lock:
            return self._registrations.get(str(name or "").strip())

    def reset_for_tests(self) -> None:
        with self._lock:
            self._registrations.clear()
            self._loaded = False


_global_registry = _ToolPluginRegistry()


def get_tool_plugin_registry() -> _ToolPluginRegistry:
    return _global_registry


def register_tool(
    *,
    name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    provider: str = "local",
    enabled: bool = True,
    always_include: bool = False,
    requires_openclaw: bool = False,
    target_tool_name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰一个工具工厂函数，注册到全局工具插件注册中心。

    装饰的函数必须是可调用对象，调用后返回 CrewAI 可直接挂载的工具实例。
    """

    # 延迟导入避免循环依赖：tool_catalog 会导入本模块
    from backend.services.agent.tool_catalog import _normalize_provider

    normalized_name = str(name or "").strip()
    metadata = ToolMetadata(
        name=normalized_name,
        description=str(description or "").strip(),
        tags=[str(item).strip() for item in (tags or []) if str(item).strip()],
        provider=_normalize_provider(provider, source=f"tool={normalized_name} (@register_tool)"),
        enabled=bool(enabled),
        always_include=bool(always_include),
        requires_openclaw=bool(requires_openclaw),
        target_tool_name=str(target_tool_name).strip() if target_tool_name else None,
    )

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        _global_registry.register(metadata=metadata, factory=factory)
        return factory

    return decorator
