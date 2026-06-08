from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_CONFIG_DIR = PROJECT_ROOT / "backend" / "models" / "config"
VIDEO_TASK_PROFILES_PATH = VIDEO_CONFIG_DIR / "video_task_profiles.json"
VIDEO_MODEL_PROFILES_PATH = VIDEO_CONFIG_DIR / "video_model_profiles.json"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_video_task_profiles() -> Dict[str, Any]:
    return _read_json(VIDEO_TASK_PROFILES_PATH)


@lru_cache(maxsize=1)
def load_video_model_profiles() -> Dict[str, Any]:
    return _read_json(VIDEO_MODEL_PROFILES_PATH)


def _normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text


def _basename(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return Path(raw.rstrip("/\\")).name
    except Exception:
        return raw.rstrip("/\\").split("/")[-1].split("\\")[-1]


def _resource_tail(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.rstrip("/").split("/")[-1]


def _candidate_names(model_dict: Dict[str, Any]) -> List[str]:
    source = model_dict.get("source") if isinstance(model_dict.get("source"), dict) else {}
    raw_candidates = [
        model_dict.get("id"),
        model_dict.get("model_key"),
        model_dict.get("name"),
        model_dict.get("load_name"),
        model_dict.get("family"),
        _resource_tail(source.get("repo_id")),
    ]
    out: List[str] = []
    seen = set()
    for item in raw_candidates:
        normalized = _normalize_name(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_resolutions(values: Any) -> List[int]:
    out: List[int] = []
    seen = set()
    for item in values or []:
        try:
            value = int(item)
        except Exception:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return sorted(out)


def _normalize_mode(mode_key: str, raw_mode: Dict[str, Any]) -> Dict[str, Any]:
    mode = deepcopy(raw_mode)
    mode["key"] = mode_key
    mode["supported_resolutions"] = _normalize_resolutions(mode.get("supported_resolutions"))
    try:
        default_resolution = int(mode.get("default_resolution"))
    except Exception:
        default_resolution = None
    if default_resolution not in mode["supported_resolutions"]:
        default_resolution = mode["supported_resolutions"][0] if mode["supported_resolutions"] else None
    mode["default_resolution"] = default_resolution

    controls = mode.get("controls") or {}
    mode["controls"] = {
        "lora": bool(controls.get("lora")),
        "aspect_ratio": bool(controls.get("aspect_ratio", True)),
        "resolution": bool(controls.get("resolution", True)),
        "num_images": bool(controls.get("num_images", True)),
        "advanced": bool(controls.get("advanced", True)),
    }

    inputs = mode.get("inputs") or {}
    normalized_inputs: Dict[str, Any] = {}
    for key, value in inputs.items():
        cfg = value if isinstance(value, dict) else {}
        item = deepcopy(cfg)
        item["visible"] = bool(item.get("visible"))
        item["required"] = bool(item.get("required"))
        normalized_inputs[key] = item
    mode["inputs"] = normalized_inputs
    return mode


def list_video_task_modes() -> List[Dict[str, Any]]:
    task_profiles = load_video_task_profiles()
    profiles = task_profiles.get("profiles") or {}
    order = task_profiles.get("order") or list(profiles.keys())

    items: List[Dict[str, Any]] = []
    for key in order:
        profile = profiles.get(key)
        if not isinstance(profile, dict):
            continue
        items.append(_normalize_mode(str(key), profile))
    return items


def _match_model_entry(model_dict: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    profiles = load_video_model_profiles().get("models") or {}
    candidates = _candidate_names(model_dict)
    if not candidates:
        return None, None, None

    for entry_key, entry in profiles.items():
        if not isinstance(entry, dict):
            continue
        match = entry.get("match") or {}
        exact = {_normalize_name(item) for item in match.get("exact") or [] if _normalize_name(item)}
        if exact:
            for candidate in candidates:
                if candidate in exact:
                    return str(entry_key), entry, f"exact:{candidate}"

    for entry_key, entry in profiles.items():
        if not isinstance(entry, dict):
            continue
        match = entry.get("match") or {}
        contains = [_normalize_name(item) for item in match.get("contains") or [] if _normalize_name(item)]
        if contains:
            for candidate in candidates:
                if any(token in candidate or candidate in token for token in contains):
                    return str(entry_key), entry, f"contains:{candidate}"

    return None, None, None


def resolve_video_profile(model_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    modality = _normalize_name(model_dict.get("modality"))
    if modality != "video":
        return None

    entry_key, entry, matched_by = _match_model_entry(model_dict)
    if not entry:
        return None

    profiles = load_video_task_profiles().get("profiles") or {}
    extends = entry.get("extends") or []
    shared_overrides = entry.get("overrides") or {}
    mode_overrides = entry.get("mode_overrides") or {}

    task_modes: List[Dict[str, Any]] = []
    union_resolutions: List[int] = []
    resolution_seen = set()
    labels: List[str] = []

    for mode_key in extends:
        base = profiles.get(mode_key)
        if not isinstance(base, dict):
            continue
        merged = _deep_merge(base, shared_overrides if isinstance(shared_overrides, dict) else {})
        specific = mode_overrides.get(mode_key) if isinstance(mode_overrides, dict) else None
        if isinstance(specific, dict):
            merged = _deep_merge(merged, specific)
        normalized = _normalize_mode(str(mode_key), merged)
        task_modes.append(normalized)

        label = str(normalized.get("label") or "").strip()
        if label:
            labels.append(label)
        for resolution in normalized.get("supported_resolutions") or []:
            if resolution not in resolution_seen:
                resolution_seen.add(resolution)
                union_resolutions.append(resolution)

    if not task_modes:
        return None

    default_resolution = None
    try:
        default_resolution = int(shared_overrides.get("default_resolution"))
    except Exception:
        default_resolution = task_modes[0].get("default_resolution")
    if default_resolution not in union_resolutions:
        default_resolution = union_resolutions[0] if union_resolutions else None

    resolution_badges = [f"{resolution}P" for resolution in union_resolutions]

    return {
        "config_key": entry_key,
        "matched_by": matched_by,
        "task_modes": task_modes,
        "supported_resolutions": union_resolutions,
        "default_resolution": default_resolution,
        "labels": labels,
        "resolution_badges": resolution_badges,
        "source": "backend/models/config",
    }


def augment_model_with_video_profile(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    resolved = resolve_video_profile(model_dict)
    if not resolved:
        return model_dict

    out = deepcopy(model_dict)
    runtime_config = out.get("runtime_config")
    if not isinstance(runtime_config, dict):
        runtime_config = {}
    runtime_config["video_profile"] = resolved
    out["runtime_config"] = runtime_config
    out["video_profile"] = resolved
    return out
