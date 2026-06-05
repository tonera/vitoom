from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
TTS_SPEAKERS_PATH = REPO_ROOT / "config" / "tts_speakers.json"


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


@lru_cache(maxsize=1)
def load_tts_speakers() -> Dict[str, Any]:
    with TTS_SPEAKERS_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("tts_speakers.json must contain a JSON object")
    families = raw.get("families")
    if not isinstance(families, dict):
        raise ValueError("tts_speakers.json missing families")
    return raw


def list_speaker_options(family: str) -> List[Dict[str, Any]]:
    catalog = load_tts_speakers()
    families = catalog.get("families") if isinstance(catalog.get("families"), dict) else {}
    family_cfg = families.get(_clean_str(family).lower())
    if not isinstance(family_cfg, dict):
        return []
    speakers = family_cfg.get("speakers")
    if not isinstance(speakers, list):
        return []
    return [dict(item) for item in speakers if isinstance(item, dict) and _clean_str(item.get("name"))]


def get_default_speaker(family: str) -> str:
    catalog = load_tts_speakers()
    families = catalog.get("families") if isinstance(catalog.get("families"), dict) else {}
    family_cfg = families.get(_clean_str(family).lower())
    if not isinstance(family_cfg, dict):
        return ""
    return _clean_str(family_cfg.get("default_speaker"))


def voxcpm_speaker_presets() -> Dict[str, str]:
    presets: Dict[str, str] = {}
    for item in list_speaker_options("voxcpm"):
        name = _clean_str(item.get("name")).lower()
        reference = _clean_str(item.get("reference_audio"))
        if name and reference:
            presets[name] = reference
    return presets
