"""Shared answer composition helpers for business query tools."""

from __future__ import annotations

import json
from typing import Any


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"
