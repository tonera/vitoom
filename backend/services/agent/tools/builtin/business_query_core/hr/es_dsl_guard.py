"""Safety guard for LLM-generated HR Elasticsearch search bodies."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping

from backend.services.agent.tools.builtin.business_query_core.hr.query_spec import HR_FIELDS


DEFAULT_SOURCE_FIELDS = [
    "employee_id",
    "name",
    "department_code",
    "department_name",
    "job_title",
    "grade",
    "office_city",
    "employment_status",
    "manager_id",
    "hire_date",
    "email",
]

FORBIDDEN_KEYS = {
    "script",
    "runtime_mappings",
    "script_fields",
    "query_string",
    "simple_query_string",
    "wildcard",
    "regexp",
    "prefix",
    "fuzzy",
    "knn",
    "collapse",
    "highlight",
    "rescore",
    "suggest",
    "pit",
    "search_after",
}

ALLOWED_TOP_LEVEL_KEYS = {"query", "size", "_source", "sort", "aggs", "aggregations", "track_total_hits"}
ALLOWED_QUERY_KEYS = {"bool", "term", "terms", "match", "match_phrase", "range", "exists", "match_all"}
ALLOWED_BOOL_KEYS = {"must", "filter", "should", "must_not", "minimum_should_match"}
ALLOWED_AGG_KEYS = {"terms", "avg", "sum", "min", "max", "value_count", "cardinality", "aggs", "aggregations"}
METRIC_AGG_KEYS = {"avg", "sum", "min", "max", "value_count", "cardinality"}


class HRDslValidationError(ValueError):
    """Raised when an ES DSL body is outside the HR-safe query subset."""


def validate_hr_es_search_body(
    body: Mapping[str, Any],
    *,
    max_limit: int = 100,
    max_buckets: int = 100,
) -> Dict[str, Any]:
    if not isinstance(body, Mapping):
        raise HRDslValidationError("ES search body must be a JSON object")

    sanitized = deepcopy(dict(body))
    _reject_forbidden_keys(sanitized)
    _validate_top_level(sanitized)

    if "aggregations" in sanitized and "aggs" not in sanitized:
        sanitized["aggs"] = sanitized.pop("aggregations")

    sanitized["size"] = _coerce_size(sanitized.get("size"), max_limit=max_limit)
    sanitized["track_total_hits"] = bool(sanitized.get("track_total_hits", True))
    sanitized["_source"] = _sanitize_source(sanitized.get("_source", DEFAULT_SOURCE_FIELDS))

    query = _normalize_query_clause(sanitized.get("query", {"match_all": {}}))
    _validate_query_clause(query, path="$.query")
    sanitized["query"] = query

    if "sort" in sanitized:
        _validate_sort(sanitized["sort"])

    if "aggs" in sanitized:
        _validate_aggs(sanitized["aggs"], max_buckets=max_buckets, path="$.aggs")

    return sanitized


def _reject_forbidden_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_KEYS:
                raise HRDslValidationError(f"forbidden ES DSL key at {path}.{key_text}")
            _reject_forbidden_keys(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def _normalize_query_clause(value: Any) -> Any:
    if not isinstance(value, Mapping) or len(value) != 1:
        return value
    operator, payload = next(iter(value.items()))
    if operator == "bool" and isinstance(payload, Mapping):
        normalized_bool: Dict[str, Any] = {}
        for key, child in payload.items():
            if key == "minimum_should_match":
                normalized_bool[key] = child
                continue
            if isinstance(child, list):
                normalized_bool[key] = [_normalize_query_clause(item) for item in child]
            else:
                normalized_bool[key] = _normalize_query_clause(child)
        return {"bool": normalized_bool}
    if operator == "term" and isinstance(payload, Mapping) and len(payload) == 1:
        field, term_value = next(iter(payload.items()))
        field_name = _normalize_field(str(field))
        if str(field).endswith(".keyword"):
            return value
        if HR_FIELDS.get(field_name, {}).get("type") == "text":
            return {"match_phrase": {field_name: term_value}}
    return value


def _validate_top_level(body: Mapping[str, Any]) -> None:
    for key in body:
        if str(key) not in ALLOWED_TOP_LEVEL_KEYS:
            raise HRDslValidationError(f"unsupported top-level ES DSL key: {key}")


def _validate_query_clause(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise HRDslValidationError(f"query clause at {path} must contain exactly one query operator")
    operator = next(iter(value))
    if operator not in ALLOWED_QUERY_KEYS:
        raise HRDslValidationError(f"unsupported query operator at {path}: {operator}")
    payload = value[operator]
    if operator == "bool":
        _validate_bool_query(payload, path=f"{path}.bool")
    elif operator in {"term", "match", "match_phrase"}:
        _validate_field_value_query(payload, path=f"{path}.{operator}")
    elif operator == "terms":
        _validate_terms_query(payload, path=f"{path}.terms")
    elif operator == "range":
        _validate_range_query(payload, path=f"{path}.range")
    elif operator == "exists":
        field = str(payload.get("field") if isinstance(payload, Mapping) else "")
        _require_allowed_field(field, path=f"{path}.exists.field")
    elif operator == "match_all":
        if payload not in ({}, None):
            raise HRDslValidationError(f"match_all at {path} must be empty")


def _validate_bool_query(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping):
        raise HRDslValidationError(f"bool query at {path} must be an object")
    for key, child in value.items():
        if key not in ALLOWED_BOOL_KEYS:
            raise HRDslValidationError(f"unsupported bool key at {path}: {key}")
        if key == "minimum_should_match":
            continue
        clauses = child if isinstance(child, list) else [child]
        for index, clause in enumerate(clauses):
            _validate_query_clause(clause, path=f"{path}.{key}[{index}]")


def _validate_field_value_query(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise HRDslValidationError(f"field query at {path} must contain exactly one field")
    field = str(next(iter(value)))
    _require_allowed_field(field, path=path)


def _validate_terms_query(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise HRDslValidationError(f"terms query at {path} must contain exactly one field")
    field, terms = next(iter(value.items()))
    _require_allowed_field(str(field), path=path)
    if not isinstance(terms, list):
        raise HRDslValidationError(f"terms query at {path} must use a list")


def _validate_range_query(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise HRDslValidationError(f"range query at {path} must contain exactly one field")
    field, operators = next(iter(value.items()))
    _require_allowed_field(str(field), path=path)
    if not isinstance(operators, Mapping):
        raise HRDslValidationError(f"range operators at {path} must be an object")
    for op in operators:
        if str(op) not in {"gt", "gte", "lt", "lte"}:
            raise HRDslValidationError(f"unsupported range operator at {path}: {op}")


def _validate_sort(value: Any) -> None:
    items = value if isinstance(value, list) else [value]
    for item in items:
        if isinstance(item, str):
            _require_allowed_field(item, path="$.sort")
            continue
        if not isinstance(item, Mapping) or len(item) != 1:
            raise HRDslValidationError("each sort item must be a field or single-field object")
        field, sort_config = next(iter(item.items()))
        _require_allowed_field(str(field), path="$.sort")
        if isinstance(sort_config, Mapping):
            order = str(sort_config.get("order") or "asc").lower()
        else:
            order = str(sort_config or "asc").lower()
        if order not in {"asc", "desc"}:
            raise HRDslValidationError(f"unsupported sort order for {field}: {order}")


def _validate_aggs(value: Any, *, max_buckets: int, path: str) -> None:
    if not isinstance(value, Mapping):
        raise HRDslValidationError(f"aggs at {path} must be an object")
    for agg_name, agg_body in value.items():
        if not isinstance(agg_body, Mapping):
            raise HRDslValidationError(f"aggregation {agg_name} at {path} must be an object")
        for key, payload in agg_body.items():
            if key not in ALLOWED_AGG_KEYS:
                raise HRDslValidationError(f"unsupported aggregation key at {path}.{agg_name}: {key}")
            if key == "terms":
                _validate_terms_agg(payload, max_buckets=max_buckets, path=f"{path}.{agg_name}.terms")
            elif key in METRIC_AGG_KEYS:
                _validate_metric_agg(payload, path=f"{path}.{agg_name}.{key}")
            elif key in {"aggs", "aggregations"}:
                _validate_aggs(payload, max_buckets=max_buckets, path=f"{path}.{agg_name}.{key}")


def _validate_terms_agg(value: Any, *, max_buckets: int, path: str) -> None:
    if not isinstance(value, Mapping):
        raise HRDslValidationError(f"terms aggregation at {path} must be an object")
    _require_allowed_field(str(value.get("field") or ""), path=f"{path}.field")
    size = _coerce_size(value.get("size"), max_limit=max_buckets)
    value["size"] = size


def _validate_metric_agg(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping):
        raise HRDslValidationError(f"metric aggregation at {path} must be an object")
    _require_allowed_field(str(value.get("field") or ""), path=f"{path}.field")


def _sanitize_source(value: Any) -> list[str] | bool:
    if value is False:
        return False
    if value in (None, True):
        return list(DEFAULT_SOURCE_FIELDS)
    if not isinstance(value, list):
        raise HRDslValidationError("_source must be false or a list of allowed fields")
    fields = []
    for field in value:
        normalized = _normalize_field(str(field))
        _require_allowed_field(normalized, path="$._source")
        fields.append(normalized)
    return list(dict.fromkeys(fields))


def _coerce_size(value: Any, *, max_limit: int) -> int:
    try:
        size = int(value if value is not None else max_limit)
    except (TypeError, ValueError):
        size = max_limit
    return max(0, min(size, max(0, int(max_limit or 0))))


def _require_allowed_field(field: str, *, path: str) -> None:
    normalized = _normalize_field(field)
    if not normalized or normalized not in HR_FIELDS:
        raise HRDslValidationError(f"field is not allowed at {path}: {field}")


def _normalize_field(field: str) -> str:
    text = str(field or "").strip()
    if text.endswith(".keyword"):
        text = text[: -len(".keyword")]
    return text
