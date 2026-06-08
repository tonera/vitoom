"""Summarize master/chat run failures for logs and WebSocket errors."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional, Tuple

from backend.i18n.locale import DEFAULT_LOCALE
from backend.i18n.translator import t

LogUserMessage = Tuple[str, str, Optional[str], Dict[str, Any]]


def _api_error_payload(exc: BaseException) -> Dict[str, Any]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _find_api_error(exc: BaseException) -> Optional[Any]:
    try:
        from openai import APIStatusError
    except ImportError:
        APIStatusError = ()  # type: ignore[misc, assignment]

    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if APIStatusError and isinstance(cur, APIStatusError):
            return cur
        cur = cur.__cause__  # type: ignore[assignment]
    return None


def _invalid_model_from_text(text: str) -> Optional[str]:
    match = re.search(r"Invalid model:\s*(\S+)", text, re.I)
    return match.group(1) if match else None


def summarize_chat_run_error(exc: BaseException, *, locale: str = DEFAULT_LOCALE) -> LogUserMessage:
    """
    Returns (log_message, user_message, message_code, message_params).
    log_message is a single line — callers should not pass exc_info=True.
    """
    if isinstance(exc, asyncio.TimeoutError):
        user_message = t("task.timeout", locale, seconds="")
        return "master agent run timed out", user_message, "task.timeout", {}

    api_exc = _find_api_error(exc)
    if api_exc is not None:
        payload = _api_error_payload(api_exc)
        msg = str(payload.get("msg") or payload.get("message") or "").strip()
        message_code = str(payload.get("message_code") or "").strip() or None
        status = getattr(api_exc, "status_code", None)

        model = _invalid_model_from_text(msg) or _invalid_model_from_text(str(exc))
        if model or (
            message_code == "system.invalidRequest" and re.search(r"model", msg, re.I)
        ):
            params = {"model": model} if model else {}
            user_message = t("model.notFound", locale, **params) if model else t(
                "network.apiCallFailed", locale
            )
            log_message = (
                f"LLM request failed: model not available ({model})"
                if model
                else f"LLM request failed ({status or 'bad request'}): {msg or 'invalid request'}"
            )
            return log_message, user_message, "model.notFound", params

        if msg:
            user_message = msg
            log_message = f"LLM request failed ({status}): {msg}"
            return log_message, user_message, message_code, {}

    text = str(exc).strip().split("\n", 1)[0]
    model = _invalid_model_from_text(text)
    if model:
        params = {"model": model}
        return (
            f"LLM request failed: model not available ({model})",
            t("model.notFound", locale, **params),
            "model.notFound",
            params,
        )

    if len(text) > 300:
        text = text[:297] + "..."
    log_message = f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__
    return log_message, t("system.internalError", locale), None, {}
