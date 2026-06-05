"""
模型目录元数据（modality / 管理端 family 清单）。

SSOT：config/model_catalog_meta.yaml
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_META_FILE = _REPO_ROOT / "config" / "model_catalog_meta.yaml"


def _norm_key(s: object) -> str:
    return (
        str(s or "")
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("_", "-")
        .replace(".", "-")
        .replace("--", "-")
        .strip("-")
    )


def _norm_family_key(s: object) -> str:
    return (
        str(s or "")
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("_", "-")
        .replace(".", "-")
        .replace("--", "-")
        .strip("-")
    )


@lru_cache(maxsize=1)
def _load_raw() -> Dict[str, Any]:
    if not _META_FILE.exists():
        raise FileNotFoundError(f"Missing model catalog meta: {_META_FILE}")
    with open(_META_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model catalog meta (expected mapping): {_META_FILE}")
    return data


def _parse_option_rows(rows: Any, *, field_name: str) -> List[Dict[str, str]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"model_catalog_meta.yaml missing non-empty '{field_name}' list")
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("id") or row.get("value") or "").strip()
        if not value:
            continue
        label = str(row.get("label") or value).strip() or value
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"value": value, "label": label})
    if not out:
        raise ValueError(f"model_catalog_meta.yaml '{field_name}' has no valid entries")
    return out


@lru_cache(maxsize=1)
def list_modality_options() -> tuple[Dict[str, str], ...]:
    raw = _load_raw()
    return tuple(_parse_option_rows(raw.get("modalities"), field_name="modalities"))


@lru_cache(maxsize=1)
def list_family_options() -> tuple[Dict[str, str], ...]:
    raw = _load_raw()
    return tuple(_parse_option_rows(raw.get("families"), field_name="families"))


def list_modality_ids() -> List[str]:
    return [str(x["value"]) for x in list_modality_options()]


def list_family_ids() -> List[str]:
    return [str(x["value"]) for x in list_family_options()]


def is_valid_modality(value: object) -> bool:
    s = str(value or "").strip().lower()
    if not s:
        return False
    return s in {str(x["value"]).strip().lower() for x in list_modality_options()}


def normalize_display_family(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    for opt in list_family_options():
        if opt["value"] == raw:
            return opt["value"]

    key = _norm_family_key(raw)
    for opt in list_family_options():
        if _norm_family_key(opt["value"]) == key:
            return opt["value"]

    return raw


def get_catalog_meta_payload() -> Dict[str, Any]:
    modalities = [dict(x) for x in list_modality_options()]
    families = [dict(x) for x in list_family_options()]
    return {
        "modalities": modalities,
        "families": families,
        "modality_ids": [x["value"] for x in modalities],
        "family_ids": [x["value"] for x in families],
    }


def reload_catalog_meta_cache() -> None:
    """测试或热重载时清缓存。"""
    _load_raw.cache_clear()
    list_modality_options.cache_clear()
    list_family_options.cache_clear()


def modality_ids_description() -> str:
    return ", ".join(list_modality_ids())
