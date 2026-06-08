"""Agent 预置模板：插件化注册 + 自动发现。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.database import Agent

from ..settings import (
    get_default_preset_agent_id,
    get_openclaw_preset_agent_id,
    is_agents_enabled,
    is_openclaw_enabled,
)
from .registry import (
    PresetDefinition,
    get_preset_plugin_registry,
    register_preset,
)
from .yaml_loader import load_yaml_preset_definitions

TRAVEL_PRESET_AGENT_ID = "preset-travel-planner-agent"


__all__ = [
    "PresetDefinition",
    "TRAVEL_PRESET_AGENT_ID",
    "register_preset",
    "list_preset_definitions",
    "get_preset_definition",
    "ensure_default_agent_presets",
]


def _collect_definitions() -> List[PresetDefinition]:
    """聚合 YAML + Python 两路预置，Python 同 id 覆盖 YAML（便于代码托底）。"""
    by_id: Dict[str, PresetDefinition] = {}
    for definition in load_yaml_preset_definitions():
        by_id[definition.id] = definition
    for definition in get_preset_plugin_registry().all_definitions():
        by_id[definition.id] = definition
    return list(by_id.values())


def list_preset_definitions(*, include_disabled: bool = False) -> List[PresetDefinition]:
    definitions = _collect_definitions()
    if include_disabled:
        return definitions

    openclaw_on = is_openclaw_enabled()
    results: List[PresetDefinition] = []
    for definition in definitions:
        if not definition.enabled:
            continue
        if definition.requires_openclaw and not openclaw_on:
            continue
        results.append(definition)
    return results


def get_preset_definition(preset_id: str) -> Optional[PresetDefinition]:
    normalized = str(preset_id or "").strip()
    if not normalized:
        return None
    for definition in _collect_definitions():
        if definition.id == normalized:
            return definition
    return None


def _upsert_preset_agent(definition: PresetDefinition) -> Optional[Dict[str, Any]]:
    config = definition.resolved_config()
    existing = Agent.get_by_id(definition.id)
    if existing and bool(existing.get("is_preset")):
        needs_update = (
            dict(existing.get("config") or {}) != config
            or str(existing.get("name") or "") != definition.name
            or str(existing.get("type") or "") != definition.agent_type
            or str(existing.get("description") or "") != definition.description
        )
        if needs_update:
            updated = Agent.update(
                definition.id,
                name=definition.name,
                description=definition.description,
                type=definition.agent_type,
                config=config,
            )
            return updated or existing
        return existing
    if existing:
        return existing

    created = Agent.create(
        id=definition.id,
        name=definition.name,
        agent_type=definition.agent_type,
        config=config,
        description=definition.description,
        status="active",
        is_preset=True,
    )
    return created


def ensure_default_agent_presets() -> List[Dict[str, Any]]:
    """确保声明的预置 Agent 都存在数据库中，并在代码配置更新时刷新 DB 行。

    注意：为兼容老配置中的 `agents.default_preset_agent_id` / `agents.openclaw.default_preset_agent_id`
    两个别名字段，此处在聚合阶段做一次 id 重写。
    """
    if not is_agents_enabled():
        return []

    alias_map: Dict[str, str] = {}
    configured_local_id = get_default_preset_agent_id()
    if configured_local_id and configured_local_id != "preset-local-agent":
        alias_map["preset-local-agent"] = configured_local_id
    configured_openclaw_id = get_openclaw_preset_agent_id()
    if configured_openclaw_id and configured_openclaw_id != "preset-openclaw-agent":
        alias_map["preset-openclaw-agent"] = configured_openclaw_id

    definitions = list_preset_definitions()
    saved: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for definition in definitions:
        real_id = alias_map.get(definition.id, definition.id)
        if real_id in seen_ids:
            continue
        seen_ids.add(real_id)
        if real_id != definition.id:
            definition = PresetDefinition(
                id=real_id,
                name=definition.name,
                agent_type=definition.agent_type,
                description=definition.description,
                config=definition.config,
                enabled=definition.enabled,
                requires_openclaw=definition.requires_openclaw,
                config_resolver=definition.config_resolver,
                source=definition.source,
            )
        row = _upsert_preset_agent(definition)
        if row:
            saved.append(row)
    return saved
