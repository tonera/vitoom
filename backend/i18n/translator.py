"""Translation helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .locale import DEFAULT_LOCALE
from .messages.en_US import MESSAGES as EN_MESSAGES
from .messages.ja_JP import MESSAGES as JA_MESSAGES
from .messages.zh_CN import MESSAGES as ZH_MESSAGES

_CATALOGS: Dict[str, Dict[str, str]] = {
    "zh-CN": ZH_MESSAGES,
    "en-US": EN_MESSAGES,
    "ja-JP": JA_MESSAGES,
}


def get_messages(locale: str) -> Dict[str, str]:
    return _CATALOGS.get(locale, _CATALOGS[DEFAULT_LOCALE])


def t(key: str, locale: str, **params: Any) -> str:
    messages = get_messages(locale)
    template = messages.get(key)
    if not template:
        template = get_messages(DEFAULT_LOCALE).get(key, key)
    if not params:
        return template
    try:
        return template.format(**params)
    except Exception:
        return template


def build_message_params(details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not details:
        return {}
    params: Dict[str, Any] = {}
    for key in ("user_id", "email", "model", "model_key", "task_id", "parameter", "seconds", "max_size", "progress"):
        if key in details and details[key] is not None:
            params[key] = details[key]
    if "model_key" in params and "model" not in params:
        params["model"] = params["model_key"]
    return params
