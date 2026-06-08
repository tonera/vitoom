"""User-facing chat messages.

Keep user-visible fallback text centralized so future i18n only needs one place.
"""

from __future__ import annotations

TOOL_OR_SERVICE_UNAVAILABLE_MESSAGE = (
    "This tool or service is currently unavailable. Please try again later."
)


def unavailable_message() -> str:
    return TOOL_OR_SERVICE_UNAVAILABLE_MESSAGE


__all__ = ["TOOL_OR_SERVICE_UNAVAILABLE_MESSAGE", "unavailable_message"]
