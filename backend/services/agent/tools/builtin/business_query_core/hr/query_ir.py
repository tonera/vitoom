"""QueryIR validation and Elasticsearch compilation for HR queries."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping

from backend.services.agent.tools.builtin.business_query_core.hr.metadata import (
    ENUM_ALIASES,
    FIELD_ALIASES,
    FIELD_TYPE_OPERATORS,
    HR_FIELDS,
    SUPPORTED_METRICS,
    SUPPORTED_SORT_DIRECTIONS,
    VALUE_HINTS,
)


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
    "attrition_risk",
]
DEFAULT_LIMIT = 20
MAX_LIMIT = 100


class QueryIRValidationError(ValueError):
    """Raised when HR QueryIR is invalid or cannot be compiled."""


def query_ir_field_metadata() -> Dict[str, Dict[str, Any]]:
    fields: Dict[str, Dict[str, Any]] = {}
    for field, meta in HR_FIELDS.items():
        fields[field] = {
            "type": meta.get("type"),
            "aliases": meta.get("aliases", []),
            "filterable": meta.get("filterable"),
            "aggregatable": meta.get("aggregatable"),
            "computed": meta.get("computed", False),
            "computed_from": meta.get("computed_from"),
            "compile": meta.get("compile"),
        }
    return fields


def normalize_query_ir(ir: Mapping[str, Any], *, max_limit: int = MAX_LIMIT) -> Dict[str, Any]:
    if not isinstance(ir, Mapping):
        raise QueryIRValidationError("QueryIR must be a JSON object")

    normalized = deepcopy(dict(ir))
    normalized.setdefault("domain", "hr")
    normalized.setdefault("entity", "employee")
    normalized.setdefault("intent", "list")
    normalized["filters"] = _normalize_filters(normalized.get("filters") or [])
    normalized["select"] = _normalize_field_list(normalized.get("select") or normalized.get("select_fields") or [])
    normalized["group_by"] = _normalize_field_list(normalized.get("group_by") or [])
    normalized["metrics"] = _normalize_metrics(normalized.get("metrics") or [])
    normalized["sort"] = _normalize_sort(normalized.get("sort") or [])

    try:
        limit = int(normalized.get("limit") or DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    normalized["limit"] = max(1, min(limit, max(1, int(max_limit or MAX_LIMIT))))
    return normalized


def validate_query_ir(ir: Mapping[str, Any], *, max_limit: int = MAX_LIMIT) -> Dict[str, Any]:
    normalized = normalize_query_ir(ir, max_limit=max_limit)
    if normalized.get("domain") != "hr":
        raise QueryIRValidationError("QueryIR domain must be 'hr'")
    if normalized.get("entity") != "employee":
        raise QueryIRValidationError("QueryIR entity must be 'employee'")
    if str(normalized.get("intent") or "") not in {"list", "aggregation", "attribute_lookup"}:
        raise QueryIRValidationError("QueryIR intent must be one of: list, aggregation, attribute_lookup")

    for item in normalized["filters"]:
        field = str(item.get("field") or "")
        meta = HR_FIELDS.get(field)
        if not meta:
            raise QueryIRValidationError(f"unknown filter field: {field}")
        if not meta.get("filterable", False):
            raise QueryIRValidationError(f"field is not filterable: {field}")
        op = str(item.get("op") or "=")
        allowed_ops = FIELD_TYPE_OPERATORS.get(str(meta.get("type") or ""), ["="])
        if op not in allowed_ops and op != "range_year":
            raise QueryIRValidationError(f"operator {op!r} is not supported for field {field!r}")
        if "value" not in item:
            raise QueryIRValidationError(f"filter {field!r} must include value")

    for field in normalized["select"]:
        if field not in HR_FIELDS:
            raise QueryIRValidationError(f"unknown select field: {field}")
    for field in normalized["group_by"]:
        meta = HR_FIELDS.get(field)
        if not meta:
            raise QueryIRValidationError(f"unknown group_by field: {field}")
        if not meta.get("aggregatable", False):
            raise QueryIRValidationError(f"field is not aggregatable: {field}")
    for item in normalized["metrics"]:
        metric_type = str(item.get("type") or "")
        if metric_type not in SUPPORTED_METRICS:
            raise QueryIRValidationError(f"unsupported metric type: {metric_type}")
        field = str(item.get("field") or "")
        if field and field not in HR_FIELDS:
            raise QueryIRValidationError(f"unknown metric field: {field}")
    for item in normalized["sort"]:
        field = str(item.get("field") or "")
        if field not in HR_FIELDS:
            raise QueryIRValidationError(f"unknown sort field: {field}")
        direction = str(item.get("direction") or "asc").lower()
        if direction not in SUPPORTED_SORT_DIRECTIONS:
            raise QueryIRValidationError(f"unsupported sort direction: {direction}")

    return normalized


def compile_query_ir_to_es_dsl(ir: Mapping[str, Any], *, max_limit: int = MAX_LIMIT) -> Dict[str, Any]:
    normalized = validate_query_ir(ir, max_limit=max_limit)
    intent = str(normalized.get("intent") or "list")
    filters, must_not = _compile_filters(normalized["filters"])
    query = _bool_query(filters=filters, must_not=must_not)
    body: Dict[str, Any] = {
        "size": 0 if intent == "aggregation" else int(normalized.get("limit") or DEFAULT_LIMIT),
        "track_total_hits": True,
        "query": query,
    }

    aggs = _compile_aggs(normalized)
    if aggs:
        body["aggs"] = aggs
    if body["size"] > 0:
        body["_source"] = _compile_source(normalized["select"])
        sort = _compile_sort(normalized["sort"])
        if sort:
            body["sort"] = sort
        elif intent == "list":
            body["sort"] = [{"grade_level": {"order": "desc"}}, {"employee_id": {"order": "asc"}}]
    return body


def _normalize_field(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return FIELD_ALIASES.get(text, text if text in HR_FIELDS else "")


def _normalize_field_list(values: Iterable[Any]) -> List[str]:
    fields: List[str] = []
    for value in values:
        field = _normalize_field(value)
        if field:
            fields.append(field)
    return list(dict.fromkeys(fields))


def _normalize_filters(filters: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in filters:
        if not isinstance(item, Mapping):
            continue
        field = _normalize_field(item.get("field"))
        if not field:
            continue
        fixed = dict(item)
        fixed["field"] = field
        fixed["op"] = str(fixed.get("op") or "=")
        if fixed["op"] == "exists":
            continue
        if "value" in fixed:
            fixed["value"] = _normalize_filter_value(field, fixed.get("value"))
        normalized.append(fixed)
    return _dedupe_filters(normalized)


def _normalize_filter_value(field: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_filter_value(field, item) for item in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    for alias, normalized in VALUE_HINTS.get(field, ()):
        if text.lower() == str(alias).strip().lower():
            return normalized
    return ENUM_ALIASES.get(text, value)


def _normalize_metrics(metrics: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in metrics:
        if not isinstance(item, Mapping):
            continue
        metric_type = str(item.get("type") or "").strip()
        field = _normalize_field(item.get("field")) or str(item.get("field") or "")
        normalized.append({"type": metric_type, "field": field})
    return normalized


def _normalize_sort(sort: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in sort:
        if not isinstance(item, Mapping):
            continue
        field = _normalize_field(item.get("field"))
        if not field:
            continue
        direction = str(item.get("direction") or item.get("order") or "asc").lower()
        normalized.append({"field": field, "direction": direction})
    return normalized


def _dedupe_filters(filters: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in filters:
        field = str(item.get("field") or "")
        op = str(item.get("op") or "=")
        value = item.get("value")
        key = (field, op, _value_key(value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"field": field, "op": op, "value": value})
    return deduped


def _value_key(value: Any) -> str:
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    return repr(value)


def _compile_filters(filters: Iterable[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clauses: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []
    for item in filters:
        field = str(item.get("field") or "")
        op = str(item.get("op") or "=")
        value = item.get("value")
        if op in {"!=", "not_in"}:
            positive = dict(item)
            positive["op"] = "in" if op == "not_in" else "="
            must_not.append(_compile_filter_clause(positive))
            continue
        clauses.append(_compile_filter_clause({"field": field, "op": op, "value": value}))
    return clauses, must_not


def _compile_filter_clause(item: Mapping[str, Any]) -> Dict[str, Any]:
    field = str(item.get("field") or "")
    op = str(item.get("op") or "=")
    value = item.get("value")
    meta = HR_FIELDS.get(field) or {}
    if meta.get("computed"):
        return _compile_computed_filter(field, op, value, meta)
    if op == "=":
        if str(meta.get("type") or "") == "text":
            return {"match_phrase": {field: value}}
        return {"term": {field: value}}
    if op == "contains" or op == "match":
        return {"match_phrase": {field: value}}
    if op == "in":
        values = value if isinstance(value, list) else [value]
        return {"terms": {field: values}}
    if op in {">", ">=", "<", "<="}:
        if field == "grade":
            return {"range": {"grade_level": {_range_operator(op): _grade_value(str(value))}}}
        return {"range": {field: {_range_operator(op): value}}}
    if op == "range_year":
        return {"range": {field: {"gte": f"{value}-01-01", "lte": f"{value}-12-31"}}}
    raise QueryIRValidationError(f"cannot compile filter: {field} {op}")


def _compile_computed_filter(field: str, op: str, value: Any, meta: Mapping[str, Any]) -> Dict[str, Any]:
    compile_spec = meta.get("compile")
    if not isinstance(compile_spec, Mapping) or compile_spec.get("kind") != "years_since_date":
        raise QueryIRValidationError(f"computed field is not compilable: {field}")
    source_field = str(compile_spec.get("source_field") or meta.get("computed_from") or "")
    if not source_field:
        raise QueryIRValidationError(f"computed field missing source_field: {field}")
    cutoff = _years_ago_iso(float(value or 0))
    if op == ">":
        return {"range": {source_field: {"lt": cutoff}}}
    if op == ">=":
        return {"range": {source_field: {"lte": cutoff}}}
    if op == "<":
        return {"range": {source_field: {"gt": cutoff}}}
    if op == "<=":
        return {"range": {source_field: {"gte": cutoff}}}
    raise QueryIRValidationError(f"unsupported computed filter operator for {field}: {op}")


def _range_operator(op: str) -> str:
    return {">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}[op]


def _years_ago_iso(years: float) -> str:
    today = date.today()
    if years.is_integer():
        try:
            return today.replace(year=today.year - int(years)).isoformat()
        except ValueError:
            return today.replace(month=2, day=28, year=today.year - int(years)).isoformat()
    return (today - timedelta(days=round(years * 365.2425))).isoformat()


def _grade_value(value: str) -> int:
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits or 0)


def _bool_query(*, filters: List[Dict[str, Any]], must_not: List[Dict[str, Any]]) -> Dict[str, Any]:
    bool_body: Dict[str, Any] = {}
    if filters:
        bool_body["filter"] = filters
    if must_not:
        bool_body["must_not"] = must_not
    return {"bool": bool_body} if bool_body else {"match_all": {}}


def _compile_aggs(ir: Mapping[str, Any]) -> Dict[str, Any]:
    group_by = list(ir.get("group_by") or [])
    metrics = list(ir.get("metrics") or [])
    aggs: Dict[str, Any] = {}
    if group_by:
        field = str(group_by[0])
        aggs[f"by_{field}"] = {"terms": {"field": field, "size": 100}}
    for item in metrics:
        metric_type = str(item.get("type") or "")
        field = str(item.get("field") or "")
        if metric_type == "count":
            continue
        if metric_type in {"avg", "sum", "min", "max"} and field:
            aggs[f"{metric_type}_{field}"] = {metric_type: {"field": field}}
    return aggs


def _compile_source(select_fields: Iterable[str]) -> List[str]:
    source = list(DEFAULT_SOURCE_FIELDS)
    for field in select_fields:
        meta = HR_FIELDS.get(field) or {}
        if meta.get("computed"):
            source_field = _computed_source_field(meta)
            if source_field:
                source.append(source_field)
            continue
        source.append(field)
    return list(dict.fromkeys(source))


def _compile_sort(sort: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in sort:
        field = str(item.get("field") or "")
        direction = str(item.get("direction") or "asc").lower()
        meta = HR_FIELDS.get(field) or {}
        if meta.get("computed"):
            compiled = _compile_computed_sort(field, direction, meta)
            items.extend(compiled)
            continue
        items.append({field: {"order": direction}})
    return items


def _compile_computed_sort(field: str, direction: str, meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    compile_spec = meta.get("compile")
    if not isinstance(compile_spec, Mapping) or compile_spec.get("kind") != "years_since_date":
        raise QueryIRValidationError(f"computed sort is not compilable: {field}")
    source_field = _computed_source_field(meta)
    if not source_field:
        raise QueryIRValidationError(f"computed sort missing source field: {field}")
    source_direction = "desc" if direction == "asc" else "asc"
    return [{source_field: {"order": source_direction}}, {"employee_id": {"order": "asc"}}]


def _computed_source_field(meta: Mapping[str, Any]) -> str:
    compile_spec = meta.get("compile")
    if isinstance(compile_spec, Mapping):
        return str(compile_spec.get("source_field") or meta.get("computed_from") or "")
    return str(meta.get("computed_from") or "")
