"""Crew-as-Tool 插件注册中心。

与 `backend/services/agent/tools/registry` 类似，通过装饰器 + 自动发现来
收集所有可被 Master Agent 调用的"专业 Crew 工具"。注册时只登记元数据，
真正构造 CrewAI 工具实例由 `bridge.build_crew_tool` 在运行期完成。
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_BUILTIN_PACKAGE = "backend.services.agent.crew_tools.builtin"


@dataclass(frozen=True)
class CrewToolMetadata:
    """Crew 工具的注册元数据。"""

    name: str
    preset_id: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    always_include: bool = False
    input_hint: str = "query"
    max_wait_seconds: int = 600


@dataclass(frozen=True)
class CrewToolRegistration:
    metadata: CrewToolMetadata
    # 保留 hook 以便未来扩展（例如按调用前预处理 query）。目前可为 None。
    hook: Optional[Callable[..., Any]] = None


class _CrewToolRegistry:
    def __init__(self) -> None:
        self._registrations: Dict[str, CrewToolRegistration] = {}
        self._lock = threading.RLock()
        self._loaded = False

    def register(
        self,
        *,
        metadata: CrewToolMetadata,
        hook: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not metadata.name:
            raise ValueError("crew-tool name is required")
        if not metadata.preset_id:
            raise ValueError(f"crew-tool {metadata.name} requires a preset_id")
        with self._lock:
            self._registrations[metadata.name] = CrewToolRegistration(
                metadata=metadata,
                hook=hook,
            )

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

    def all_registrations(self) -> Dict[str, CrewToolRegistration]:
        self.ensure_loaded()
        with self._lock:
            return dict(self._registrations)

    def get(self, name: str) -> Optional[CrewToolRegistration]:
        self.ensure_loaded()
        with self._lock:
            return self._registrations.get(str(name or "").strip())

    def reset_for_tests(self) -> None:
        with self._lock:
            self._registrations.clear()
            self._loaded = False


_global_registry = _CrewToolRegistry()


def get_crew_tool_registry() -> _CrewToolRegistry:
    return _global_registry


def list_crew_tools() -> List[CrewToolMetadata]:
    return [r.metadata for r in _global_registry.all_registrations().values()]


def register_crew_tool(
    *,
    name: str,
    preset_id: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    enabled: bool = True,
    always_include: bool = False,
    input_hint: str = "query",
    max_wait_seconds: int = 600,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器：注册一个 Crew 工具到全局注册中心。

    被装饰的函数可以为空（仅作为注册触发点），也可以返回一个可调用 hook
    供后续扩展使用（预留，当前版本不启用 hook 逻辑）。
    """

    metadata = CrewToolMetadata(
        name=str(name or "").strip(),
        preset_id=str(preset_id or "").strip(),
        description=str(description or "").strip(),
        tags=[str(item).strip() for item in (tags or []) if str(item).strip()],
        enabled=bool(enabled),
        always_include=bool(always_include),
        input_hint=str(input_hint or "query").strip() or "query",
        max_wait_seconds=int(max_wait_seconds or 600),
    )

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        hook = factory if callable(factory) else None
        _global_registry.register(metadata=metadata, hook=hook)
        return factory

    return decorator
