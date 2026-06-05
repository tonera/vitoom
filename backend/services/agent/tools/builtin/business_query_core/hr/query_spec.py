"""Core QuerySpec validation primitives for business query tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional

from backend.services.agent.tools.builtin.business_query_core.hr.metadata import (
    BUSINESS_RULES,
    ENUM_ALIASES,
    FIELD_ALIASES,
    FIELD_TYPE_OPERATORS,
    HR_FIELDS,
    METRIC_ALIASES,
    QUALITY_RULES,
    SUPPORTED_ENTITIES,
    SUPPORTED_INTENTS,
    SUPPORTED_METRICS,
    SUPPORTED_SORT_DIRECTIONS,
)

"""Controlled QuerySpec schema and validation for business queries."""





DEFAULT_LIMIT = 20
MAX_LIMIT = 100
FORBIDDEN_KEYS = {"query", "aggs", "aggregations", "script", "runtime_mappings", "knn"}


class QuerySpecValidationError(ValueError):
    """Raised when a QuerySpec violates the controlled business query schema."""


def normalize_query_spec(spec: Mapping[str, Any], *, max_limit: int = MAX_LIMIT) -> Dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise QuerySpecValidationError("QuerySpec must be a JSON object")

    normalized = deepcopy(dict(spec))
    normalized.setdefault("domain", "hr")
    normalized.setdefault("entity", "employee")
    normalized.setdefault("intent", "list")
    normalized["filters"] = list(normalized.get("filters") or [])

    try:
        limit = int(normalized.get("limit") or DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    normalized["limit"] = max(1, min(limit, max(1, int(max_limit or MAX_LIMIT))))

    if normalized.get("intent") == "quality_check" and "quality_check" in normalized and "check" not in normalized:
        quality_check = normalized.get("quality_check")
        if isinstance(quality_check, Mapping):
            normalized["check"] = quality_check.get("type")

    return normalized


def validate_query_spec(
    spec: Mapping[str, Any],
    *,
    fields: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_limit: int = MAX_LIMIT,
) -> Dict[str, Any]:
    normalized = normalize_query_spec(spec, max_limit=max_limit)
    field_metadata = fields or HR_FIELDS

    _reject_forbidden_keys(normalized)

    if normalized.get("domain") != "hr":
        raise QuerySpecValidationError("domain must be 'hr'")
    if normalized.get("entity") not in SUPPORTED_ENTITIES:
        raise QuerySpecValidationError("entity must be one of: employee, resume")

    intent = str(normalized.get("intent") or "")
    if intent not in SUPPORTED_INTENTS:
        raise QuerySpecValidationError(f"unsupported intent: {intent}")
    if intent == "clarify":
        return normalized

    _validate_filters(normalized.get("filters") or [], field_metadata)
    _validate_metrics(normalized.get("metrics") or [], field_metadata)
    _validate_group_by(normalized.get("group_by") or [], field_metadata)
    _validate_sort(normalized.get("sort") or [], field_metadata)
    _validate_select_fields(normalized.get("select_fields") or [], field_metadata)
    _validate_subjects(normalized.get("subjects") or [])

    if intent == "quality_check":
        check = str(normalized.get("check") or "all_quality_checks")
        if check not in QUALITY_RULES:
            raise QuerySpecValidationError(f"unsupported quality check: {check}")
        normalized["check"] = check

    return normalized


def _reject_forbidden_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_KEYS:
                raise QuerySpecValidationError(f"forbidden ES DSL key at {path}.{key}")
            _reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def _validate_filters(filters: Iterable[Any], fields: Mapping[str, Mapping[str, Any]]) -> None:
    for item in filters:
        if not isinstance(item, Mapping):
            raise QuerySpecValidationError("each filter must be an object")
        field = str(item.get("field") or "")
        meta = fields.get(field)
        if not meta:
            raise QuerySpecValidationError(f"unknown filter field: {field}")
        if not meta.get("filterable", False):
            raise QuerySpecValidationError(f"field is not filterable: {field}")
        op = str(item.get("op") or "=")
        allowed_ops = FIELD_TYPE_OPERATORS.get(str(meta.get("type") or ""), ["="])
        if op not in allowed_ops:
            raise QuerySpecValidationError(f"operator {op!r} is not supported for field {field!r}")
        if "value" not in item:
            raise QuerySpecValidationError(f"filter {field!r} must include value")


def _validate_metrics(metrics: Iterable[Any], fields: Mapping[str, Mapping[str, Any]]) -> None:
    for item in metrics:
        if not isinstance(item, Mapping):
            raise QuerySpecValidationError("each metric must be an object")
        metric_type = str(item.get("type") or "")
        if metric_type not in SUPPORTED_METRICS:
            raise QuerySpecValidationError(f"unsupported metric type: {metric_type}")
        field = str(item.get("field") or "")
        if field and field not in fields:
            raise QuerySpecValidationError(f"unknown metric field: {field}")


def _validate_group_by(group_by: Iterable[Any], fields: Mapping[str, Mapping[str, Any]]) -> None:
    for field_value in group_by:
        field = str(field_value or "")
        meta = fields.get(field)
        if not meta:
            raise QuerySpecValidationError(f"unknown group_by field: {field}")
        if not meta.get("aggregatable", False):
            raise QuerySpecValidationError(f"field is not aggregatable: {field}")


def _validate_sort(sort: Iterable[Any], fields: Mapping[str, Mapping[str, Any]]) -> None:
    for item in sort:
        if not isinstance(item, Mapping):
            raise QuerySpecValidationError("each sort item must be an object")
        field = str(item.get("field") or "")
        if field not in fields:
            raise QuerySpecValidationError(f"unknown sort field: {field}")
        direction = str(item.get("direction") or "asc").lower()
        if direction not in SUPPORTED_SORT_DIRECTIONS:
            raise QuerySpecValidationError(f"unsupported sort direction: {direction}")


def _validate_select_fields(select_fields: Iterable[Any], fields: Mapping[str, Mapping[str, Any]]) -> None:
    for field_value in select_fields:
        field = str(field_value or "")
        if field not in fields:
            raise QuerySpecValidationError(f"unknown select field: {field}")


def _validate_subjects(subjects: Iterable[Any]) -> None:
    for item in subjects:
        if not isinstance(item, Mapping):
            raise QuerySpecValidationError("each subject must be an object")
        subject_type = str(item.get("type") or "employee")
        if subject_type not in {"employee", "employee_name", "employee_id"}:
            raise QuerySpecValidationError(f"unsupported subject type: {subject_type}")
