"""预置 Agent 插件注册中心。

分两路来源：
- YAML 目录：`config/agents/presets/*.yaml`，适合纯数据声明；
- Python `builtin/`：适合需要运行时动态生成配置的预置（例如 OpenClaw 预置的工具清单）。

通过 `register_preset` 装饰器将 Python 来源的预置注册进来，再由 `PresetRegistry.all()`
将两路来源聚合成统一的 `PresetDefinition` 列表。
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

BUILTIN_PRESET_PACKAGE = "backend.services.agent.presets.builtin"


@dataclass
class PresetDefinition:
    """一个预置 Agent 的运行时视图。"""

    id: str
    name: str
    agent_type: str = "general"
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    requires_openclaw: bool = False
    config_resolver: Optional[Callable[[], Dict[str, Any]]] = None
    source: str = "python"

    def resolved_config(self) -> Dict[str, Any]:
        if self.config_resolver is not None:
            try:
                resolved = self.config_resolver() or {}
                if isinstance(resolved, dict):
                    return resolved
            except Exception:
                pass
        return dict(self.config or {})


class _PresetPluginRegistry:
    def __init__(self) -> None:
        self._definitions: Dict[str, PresetDefinition] = {}
        self._lock = threading.RLock()
        self._loaded = False

    def register(self, definition: PresetDefinition) -> None:
        if not definition.id:
            raise ValueError("preset id is required to register")
        with self._lock:
            self._definitions[definition.id] = definition

    def ensure_loaded(self, package_name: str = BUILTIN_PRESET_PACKAGE) -> None:
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

    def all_definitions(self) -> List[PresetDefinition]:
        self.ensure_loaded()
        with self._lock:
            return list(self._definitions.values())

    def get(self, preset_id: str) -> Optional[PresetDefinition]:
        self.ensure_loaded()
        with self._lock:
            return self._definitions.get(str(preset_id or "").strip())

    def reset_for_tests(self) -> None:
        with self._lock:
            self._definitions.clear()
            self._loaded = False


_global_registry = _PresetPluginRegistry()


def get_preset_plugin_registry() -> _PresetPluginRegistry:
    return _global_registry


def register_preset(
    *,
    id: str,
    name: str,
    agent_type: str = "general",
    description: str = "",
    enabled: bool = True,
    requires_openclaw: bool = False,
) -> Callable[[Callable[[], Dict[str, Any]]], Callable[[], Dict[str, Any]]]:
    """将一个返回 config dict 的函数注册为预置 Agent。"""

    preset_id = str(id or "").strip()
    preset_name = str(name or "").strip() or preset_id
    preset_type = str(agent_type or "general").strip() or "general"
    preset_description = str(description or "").strip()

    def decorator(config_resolver: Callable[[], Dict[str, Any]]) -> Callable[[], Dict[str, Any]]:
        definition = PresetDefinition(
            id=preset_id,
            name=preset_name,
            agent_type=preset_type,
            description=preset_description,
            enabled=bool(enabled),
            requires_openclaw=bool(requires_openclaw),
            config_resolver=config_resolver,
            source="python",
        )
        _global_registry.register(definition)
        return config_resolver

    return decorator
