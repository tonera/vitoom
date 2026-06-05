from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .supervisorctl import validate_program_name

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
GLOBAL_CONFIG_FILENAME = "inference.yaml"
RESERVED_CONFIG_STEMS = frozenset({"inference", "ex_inference"})


def config_directory() -> Path:
    return CONFIG_DIR


def global_config_path() -> Path:
    return CONFIG_DIR / GLOBAL_CONFIG_FILENAME


def service_config_path(service_id: str) -> Path:
    normalized = validate_service_id(service_id)
    return CONFIG_DIR / f"{normalized}.yaml"


def validate_service_id(service_id: str) -> str:
    normalized = validate_program_name(service_id.strip())
    if normalized in RESERVED_CONFIG_STEMS:
        raise ValueError(f"service_id '{normalized}' is reserved.")
    return normalized


def _is_service_config_candidate(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".yaml":
        return False
    stem = path.stem
    if stem in RESERVED_CONFIG_STEMS or stem.startswith("ex_"):
        return False
    return True


def _iter_service_config_paths() -> list[Path]:
    return sorted(path for path in CONFIG_DIR.glob("*.yaml") if _is_service_config_candidate(path))


def _read_yaml_dict_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _read_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = _read_yaml_dict_optional(path)
    if not data:
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def _write_yaml_dict(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if value is None:
            continue
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_service_config_path(service_id: str) -> Path | None:
    """
    定位服务配置文件：
    1. 优先 {service_id}.yaml
    2. 再扫描 config 目录内 YAML 的 service_id 字段
    """
    normalized = validate_service_id(service_id)
    direct = service_config_path(normalized)
    if direct.exists():
        return direct

    for path in _iter_service_config_paths():
        data = _read_yaml_dict_optional(path)
        configured_id = str(data.get("service_id") or "").strip()
        if configured_id == normalized:
            return path
    return None


def read_global_config() -> dict[str, Any]:
    return _read_yaml_dict(global_config_path())


def write_global_config(patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("config must be a mapping")
    current = read_global_config()
    merged = _deep_merge(current, patch)
    _write_yaml_dict(global_config_path(), merged)
    return merged


def read_service_config(service_id: str) -> tuple[dict[str, Any], str]:
    normalized = validate_service_id(service_id)
    path = resolve_service_config_path(normalized)
    if path is None:
        return {"service_id": normalized}, f"{normalized}.yaml"

    data = _read_yaml_dict(path)
    return data, path.name


def write_service_config(service_id: str, patch: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(patch, dict):
        raise ValueError("config must be a mapping")
    normalized = validate_service_id(service_id)
    path = resolve_service_config_path(normalized) or service_config_path(normalized)
    current = _read_yaml_dict(path) if path.exists() else {"service_id": normalized}
    merged = _deep_merge(current, patch)
    merged["service_id"] = normalized
    _write_yaml_dict(path, merged)
    return merged, path.name
