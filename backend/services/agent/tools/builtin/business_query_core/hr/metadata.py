"""HR domain metadata loaded from YAML."""

from __future__ import annotations

from typing import Any, Dict, List

from backend.services.agent.tools.builtin.business_query_core.domain_metadata import (
    field_aliases,
    load_domain_metadata,
    value_hint_pairs,
)


HR_METADATA = load_domain_metadata("hr")
HR_FIELDS: Dict[str, Dict[str, Any]] = dict(HR_METADATA.get("fields") or {})
FIELD_ALIASES: Dict[str, str] = field_aliases(HR_FIELDS)
ENUM_ALIASES: Dict[str, Any] = dict(HR_METADATA.get("enum_aliases") or {})
METRIC_ALIASES: Dict[str, str] = dict(HR_METADATA.get("metric_aliases") or {})
BUSINESS_RULES: Dict[str, Dict[str, Any]] = dict(HR_METADATA.get("business_rules") or {})
QUALITY_RULES: Dict[str, str] = dict(HR_METADATA.get("quality_rules") or {})
FIELD_TYPE_OPERATORS: Dict[str, List[str]] = {
    str(key): list(value or [])
    for key, value in dict(HR_METADATA.get("field_type_operators") or {}).items()
}
SUPPORTED_ENTITIES = set(HR_METADATA.get("entities") or [])
SUPPORTED_INTENTS = set(HR_METADATA.get("supported_intents") or [])
SUPPORTED_METRICS = set(HR_METADATA.get("supported_metrics") or [])
SUPPORTED_SORT_DIRECTIONS = set(HR_METADATA.get("supported_sort_directions") or [])

_SEMANTIC = dict(HR_METADATA.get("semantic") or {})
BUSINESS_RULE_ALIASES = {
    str(key): tuple(value or [])
    for key, value in dict(_SEMANTIC.get("business_rule_aliases") or {}).items()
}
QUALITY_RULE_ALIASES = {
    str(key): tuple(value or [])
    for key, value in dict(_SEMANTIC.get("quality_rule_aliases") or {}).items()
}
INTENT_ALIASES = {
    str(key): tuple(value or [])
    for key, value in dict(_SEMANTIC.get("intent_aliases") or {}).items()
}
VALUE_HINTS = value_hint_pairs(dict(_SEMANTIC.get("value_hints") or {}))
EXAMPLE_LIBRARY = list(_SEMANTIC.get("examples") or [])
