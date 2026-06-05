"""Locale detection and normalization."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import Request

SUPPORTED_LOCALES = {"zh-CN", "en-US", "ja-JP"}
DEFAULT_LOCALE = "zh-CN"
CLI_DEFAULT_LOCALE = "en-US"


def normalize_locale(value: Optional[str]) -> str:
    if not value:
        return DEFAULT_LOCALE
    raw = value.strip().lower()
    if raw.startswith("zh"):
        return "zh-CN"
    if raw.startswith("en"):
        return "en-US"
    if raw.startswith("ja"):
        return "ja-JP"
    return DEFAULT_LOCALE


def normalize_cli_locale(value: Optional[str]) -> str:
    if not value:
        return CLI_DEFAULT_LOCALE
    raw = value.strip().lower()
    if raw.startswith("zh"):
        return "zh-CN"
    if raw.startswith("en"):
        return "en-US"
    if raw.startswith("ja"):
        return "ja-JP"
    return CLI_DEFAULT_LOCALE


def detect_cli_locale(*, lang: Optional[str] = None) -> str:
    """Resolve locale for CLI tools.

    Priority: explicit ``lang`` > ``VITOOM_LOCALE`` > ``LC_ALL`` / ``LANG`` > ``CLI_DEFAULT_LOCALE``.
    """
    if lang:
        return normalize_cli_locale(lang)
    env_locale = os.environ.get("VITOOM_LOCALE")
    if env_locale:
        return normalize_cli_locale(env_locale)
    for env_key in ("LC_ALL", "LANG"):
        value = os.environ.get(env_key)
        if not value:
            continue
        candidate = value.split(".", 1)[0]
        if candidate.lower() in {"c", "posix"}:
            continue
        normalized = normalize_cli_locale(candidate)
        if normalized in SUPPORTED_LOCALES:
            return normalized
    return CLI_DEFAULT_LOCALE


def get_locale_from_request(request: Optional["Request"]) -> str:
    if request is None:
        return DEFAULT_LOCALE
    query_locale = request.query_params.get("locale")
    if query_locale:
        return normalize_locale(query_locale)
    return normalize_locale(request.headers.get("accept-language"))
