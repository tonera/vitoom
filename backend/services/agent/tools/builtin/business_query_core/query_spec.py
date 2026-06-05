"""Shared QuerySpec primitives for business query tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


DEFAULT_LIMIT = 20
MAX_LIMIT = 100
FORBIDDEN_KEYS = {"query", "aggs", "aggregations", "script", "runtime_mappings", "knn"}


class QuerySpecValidationError(ValueError):
    """Raised when a QuerySpec violates the controlled business query schema."""


def normalize_query_spec_base(
    spec: Mapping[str, Any],
    *,
    domain: str,
    default_entity: str,
    max_limit: int = MAX_LIMIT,
) -> Dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise QuerySpecValidationError("QuerySpec must be a JSON object")
    normalized = deepcopy(dict(spec))
    normalized.setdefault("domain", domain)
    normalized.setdefault("entity", default_entity)
    normalized.setdefault("intent", "list")
    normalized["filters"] = list(normalized.get("filters") or [])
    try:
        limit = int(normalized.get("limit") or DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    normalized["limit"] = max(1, min(limit, max(1, int(max_limit or MAX_LIMIT))))
    return normalized


def reject_forbidden_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_KEYS:
                raise QuerySpecValidationError(f"forbidden ES DSL key at {path}.{key}")
            reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, path=f"{path}[{index}]")
