"""YAML 预置加载器：扫描 `config/agents/presets/*.yaml` 并产出 `PresetDefinition`。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from backend.core.logger import get_app_logger

from ..settings import get_agents_presets_dir
from .registry import PresetDefinition

logger = get_app_logger(__name__)


def _normalize_yaml_definition(raw: Dict[str, Any], *, source_file: Path) -> PresetDefinition:
    preset_id = str(raw.get("id") or "").strip()
    if not preset_id:
        raise ValueError(f"preset YAML missing `id`: {source_file}")

    config_section = raw.get("config")
    if isinstance(config_section, dict):
        config = dict(config_section)
    else:
        # 兼容「扁平写法」：把 agents/tasks/runtime_defaults 直接写在顶层
        config = {
            key: value
            for key, value in raw.items()
            if key in {"agent", "agents", "tasks", "runtime_defaults"}
        }

    return PresetDefinition(
        id=preset_id,
        name=str(raw.get("name") or preset_id).strip() or preset_id,
        agent_type=str(raw.get("type") or raw.get("agent_type") or "general").strip() or "general",
        description=str(raw.get("description") or "").strip(),
        config=config,
        enabled=bool(raw.get("enabled", True)),
        requires_openclaw=bool(raw.get("requires_openclaw", False)),
        source=f"yaml:{source_file.name}",
    )


def load_yaml_preset_definitions(directory: Path = None) -> List[PresetDefinition]:
    """扫描预置目录，返回 YAML 定义的预置列表。"""
    presets_dir = directory or get_agents_presets_dir()
    if not presets_dir.exists() or not presets_dir.is_dir():
        logger.info(f"Agent presets directory not found, skipping YAML scan: {presets_dir}")
        return []

    definitions: List[PresetDefinition] = []
    for path in sorted(presets_dir.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except Exception as exc:
            logger.warning(f"Failed to parse preset YAML {path}: {exc}")
            continue
        if not isinstance(raw, dict):
            logger.warning(f"Preset YAML root must be a mapping, got {type(raw).__name__}: {path}")
            continue
        try:
            definitions.append(_normalize_yaml_definition(raw, source_file=path))
        except Exception as exc:
            logger.warning(f"Failed to load preset YAML {path}: {exc}")
            continue
    return definitions
