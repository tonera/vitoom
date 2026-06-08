"""Locale and network region (China vs international)."""

from __future__ import annotations

from vitoom_setup.constants import CN_MIRROR

SUPPORTED_LOCALES = frozenset({"zh-CN", "en-US", "ja-JP"})
SUPPORTED_REGIONS = frozenset({"cn", "intl"})


def locale_from_env_value(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    raw = value.strip()
    if raw.lower().startswith("zh"):
        return "zh-CN"
    if raw.lower().startswith("ja"):
        return "ja-JP"
    if raw.lower().startswith("en"):
        return "en-US"
    return None


def region_from_env_value(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    raw = value.strip().lower()
    if raw in {"cn", "china"}:
        return "cn"
    if raw in {"intl", "global", "international"}:
        return "intl"
    return None


def region_from_locale(locale: str) -> str:
    """Backward-compatible fallback when ``VITOOM_REGION`` is not set."""
    return "cn" if locale == "zh-CN" else "intl"


def region_env_updates(region: str) -> dict[str, str]:
    if region == "cn":
        return {
            **CN_MIRROR,
            "VITOOM_REGION": "cn",
        }
    return {
        "APT_MIRROR": "",
        "PIP_INDEX_URL": "",
        "VITOOM_REGION": "intl",
    }


def locale_region_env_updates(region: str) -> dict[str, str]:
    """Deprecated alias; use :func:`region_env_updates`."""
    return region_env_updates(region)
