"""YAML-backed domain metadata loading for business query tools."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml  # type: ignore[import-untyped]


_BASE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=16)
def load_domain_metadata(domain: str) -> Dict[str, Any]:
    name = str(domain or "").strip().lower()
    if not name:
        raise ValueError("domain is required")
    path = _BASE_DIR / name / "metadata.yaml"
    if not path.exists():
        path = _BASE_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"business query domain metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"domain metadata must be a YAML object: {path}")
    return dict(data)


def field_aliases(fields: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    return {
        alias: field
        for field, meta in fields.items()
        for alias in [field, *list(meta.get("aliases") or [])]
    }


def value_hint_pairs(value_hints: Dict[str, Any]) -> Dict[str, List[tuple[str, Any]]]:
    normalized: Dict[str, List[tuple[str, Any]]] = {}
    for field, hints in value_hints.items():
        pairs: List[tuple[str, Any]] = []
        for item in hints or []:
            if isinstance(item, dict):
                pairs.append((str(item.get("alias") or ""), item.get("value")))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), item[1]))
        normalized[str(field)] = [(alias, value) for alias, value in pairs if alias]
    return normalized
