"""Planner primitives for domain-specific business query tools."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from backend.services.agent.tools.builtin.business_query_core.planner_base import (
    parse_llm_json_object,
    run_agent_planner_completion,
)
from backend.services.agent.tools.builtin.business_query_core.hr import executor as sample_backend
from backend.services.agent.tools.builtin.business_query_core.hr.es_dsl_guard import validate_hr_es_search_body
from backend.services.agent.tools.builtin.business_query_core.hr.metadata import (
    BUSINESS_RULE_ALIASES,
    BUSINESS_RULES,
    ENUM_ALIASES,
    EXAMPLE_LIBRARY,
    FIELD_ALIASES,
    HR_FIELDS,
    INTENT_ALIASES,
    QUALITY_RULE_ALIASES,
    QUALITY_RULES,
    VALUE_HINTS,
)
from backend.services.agent.tools.builtin.business_query_core.hr.query_ir import (
    compile_query_ir_to_es_dsl,
    query_ir_field_metadata,
    validate_query_ir,
)
from backend.services.agent.tools.builtin.business_query_core.hr.query_spec import QuerySpecValidationError, validate_query_spec

@dataclass(frozen=True)
class SemanticCandidate:
    key: str
    score: int
    evidence: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticContext:
    intents: List[SemanticCandidate]
    fields: List[SemanticCandidate]
    business_rules: List[SemanticCandidate]
    quality_rules: List[SemanticCandidate]
    filters: List[Dict[str, Any]]
    select_fields: List[str]
    group_by: List[str]
    metrics: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]

    def as_prompt_payload(self) -> Dict[str, Any]:
        return {
            "intent_candidates": [_candidate_payload(item) for item in self.intents],
            "field_candidates": [_candidate_payload(item) for item in self.fields],
            "business_rule_candidates": [_candidate_payload(item) for item in self.business_rules],
            "quality_rule_candidates": [_candidate_payload(item) for item in self.quality_rules],
            "deterministic_filter_hints": self.filters,
            "select_field_hints": self.select_fields,
            "group_by_hints": self.group_by,
            "metric_hints": self.metrics,
            "retrieved_examples": self.examples,
        }


def build_semantic_context(query: str, *, max_examples: int = 6) -> SemanticContext:
    text = str(query or "")
    intents = _rank_aliases(text, INTENT_ALIASES)
    fields = _rank_field_candidates(text)
    business_rules = _rank_aliases(text, BUSINESS_RULE_ALIASES, descriptions=BUSINESS_RULES)
    quality_rules = _rank_aliases(text, QUALITY_RULE_ALIASES, descriptions={key: {"description": value} for key, value in QUALITY_RULES.items()})
    filters = infer_legacy_filter_hints(text)
    select_fields = infer_legacy_select_field_hints(text, fields)
    group_by = infer_legacy_group_by_hints(text, fields)
    metrics = infer_legacy_metric_hints(text)
    examples = retrieve_examples(text, intents=intents, fields=fields, business_rules=business_rules, quality_rules=quality_rules, max_examples=max_examples)
    return SemanticContext(
        intents=intents,
        fields=fields,
        business_rules=business_rules,
        quality_rules=quality_rules,
        filters=filters,
        select_fields=select_fields,
        group_by=group_by,
        metrics=metrics,
        examples=examples,
    )


def build_query_ir_semantic_context(query: str, *, max_examples: int = 6) -> SemanticContext:
    text = str(query or "")
    intents = _rank_aliases(text, INTENT_ALIASES)
    fields = _rank_field_candidates(text)
    business_rules = _rank_aliases(text, BUSINESS_RULE_ALIASES, descriptions=BUSINESS_RULES)
    quality_rules = _rank_aliases(text, QUALITY_RULE_ALIASES, descriptions={key: {"description": value} for key, value in QUALITY_RULES.items()})
    examples = retrieve_examples(text, intents=intents, fields=fields, business_rules=business_rules, quality_rules=quality_rules, max_examples=max_examples)
    return SemanticContext(
        intents=intents,
        fields=fields,
        business_rules=business_rules,
        quality_rules=quality_rules,
        filters=infer_metadata_value_filter_hints(text),
        select_fields=infer_metadata_select_field_hints(fields),
        group_by=infer_metadata_group_by_hints(fields),
        metrics=[],
        examples=examples,
    )


def normalize_spec_with_semantics(spec: Mapping[str, Any], query: str) -> Dict[str, Any]:
    normalized = deepcopy(dict(spec))
    context = build_semantic_context(query)

    normalized["filters"] = _normalize_filters(normalized.get("filters") or [])
    if normalized.get("intent") in {"list", "aggregation"}:
        normalized["filters"] = _merge_filters(normalized.get("filters") or [], context.filters)

    business_rule = _first_key(context.business_rules)
    if business_rule:
        normalized["business_rule"] = str(normalized.get("business_rule") or business_rule)
        normalized["filters"] = _merge_filters(normalized.get("filters") or [], BUSINESS_RULES.get(business_rule, {}).get("filters") or [])

    check = _first_key(context.quality_rules)
    if normalized.get("intent") == "quality_check" and check and not normalized.get("check"):
        normalized["check"] = check

    if normalized.get("intent") == "attribute_lookup":
        fields = [normalize_field_name(field) for field in normalized.get("select_fields") or [] if normalize_field_name(field)]
        normalized["select_fields"] = list(dict.fromkeys([*fields, *context.select_fields]))

    if normalized.get("intent") == "aggregation":
        group_by = [normalize_field_name(field) for field in normalized.get("group_by") or [] if normalize_field_name(field)]
        normalized["group_by"] = list(dict.fromkeys([*group_by, *context.group_by]))
        if not normalized.get("metrics") and context.metrics:
            normalized["metrics"] = context.metrics

    return normalized


def normalize_field_name(value: Any) -> str:
    key = str(value or "").strip()
    if not key:
        return ""
    return FIELD_ALIASES.get(key, key if key in HR_FIELDS else "")


def normalize_enum_value(value: Any) -> Any:
    if isinstance(value, str):
        return ENUM_ALIASES.get(value.strip(), value)
    return value


def infer_metadata_value_filter_hints(text: str) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    for field, hints in VALUE_HINTS.items():
        for alias, value in hints:
            if _contains(text, alias):
                filters.append({"field": field, "op": "=", "value": value})
    return _merge_filters([], filters)


def infer_metadata_select_field_hints(field_candidates: Sequence[SemanticCandidate]) -> List[str]:
    return [candidate.key for candidate in field_candidates if candidate.key in HR_FIELDS]


def infer_metadata_group_by_hints(field_candidates: Sequence[SemanticCandidate]) -> List[str]:
    return [candidate.key for candidate in field_candidates if HR_FIELDS.get(candidate.key, {}).get("aggregatable")]


def infer_legacy_filter_hints(text: str) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = infer_metadata_value_filter_hints(text)
    if _contains_any(text, ("2023 年后", "2023年后")) and _contains_any(text, ("入职", "加入")):
        filters.append({"field": "hire_date", "op": ">=", "value": "2023-01-01"})
    elif "2023" in text and _contains_any(text, ("入职", "加入")):
        filters.append({"field": "hire_date", "op": "range_year", "value": 2023})
    tenure_match = re.search(r"(?:入职|司龄)(?:超过|大于|多于|小于|少于|不足|未满|不满)\s*(\d+(?:\.\d+)?)\s*年", text)
    if tenure_match:
        op_text = tenure_match.group(0)
        op = "<" if _contains_any(op_text, ("小于", "少于", "不足", "未满", "不满")) else ">"
        filters.append({"field": "tenure_years", "op": op, "value": float(tenure_match.group(1))})
    if _contains_any(text, ("GR-08", "GR08")) and _contains_any(text, ("以上", "及以上")):
        filters.append({"field": "grade", "op": ">=", "value": "GR-08"})
    if _contains_any(text, ("GR-10", "GR10", "经理级", "高管", "VP")) and _contains_any(text, ("以上", "高管", "VP", "经理级")):
        filters.append({"field": "grade", "op": ">=", "value": "GR-10"})
    if _contains_any(text, ("绩效中", "绩效为中", "中绩效", "绩效评级中", "绩效评级为中")):
        filters.append({"field": "performance_rating", "op": "=", "value": "中"})
    if _contains_any(text, ("绩效高", "绩效为高", "高绩效", "绩效评级高", "绩效评级为高")):
        filters.append({"field": "performance_rating", "op": "=", "value": "高"})
    name_filter = _infer_name_filter(text)
    if name_filter:
        filters.append(name_filter)
    return _merge_filters([], filters)


def _infer_name_filter(text: str) -> Dict[str, Any]:
    for pattern in (
        r"(?:名字|姓名|员工姓名)(?:中)?(?:包含|含有|带有)[“\"']?([\u4e00-\u9fff]{2,4}\d{0,4})[”\"']?",
        r"[“\"']([\u4e00-\u9fff]{2,4}\d{0,4})[”\"'](?:的)?(?:人员|员工)?(?:名单|清单|列表)",
        r"(?:名为|叫|姓名(?:是|为)?)([\u4e00-\u9fff]{2,4}\d{0,4})",
        r"(?:有几个|有多少个|多少个|几个)([\u4e00-\u9fff]{2,4}\d{0,4})",
        r"([\u4e00-\u9fff]{2,4}\d{0,4})(?:的)?(?:清单|名单|列表)",
    ):
        match = re.search(pattern, text)
        if not match:
            continue
        name = match.group(1).strip()
        if name and name not in {"公司", "员工", "人员", "清单", "名单", "列表", "中的员工", "年的员工"}:
            return {"field": "name", "op": "contains", "value": name}
    return {}


def infer_legacy_select_field_hints(text: str, field_candidates: Sequence[SemanticCandidate]) -> List[str]:
    fields: List[str] = []
    if _contains_any(text, ("在哪里办公", "在哪办公", "办公地点", "办公室在哪", "工作地")):
        fields.append("office_city")
    if _contains_any(text, ("基本信息", "员工信息", "个人信息", "详细信息", "资料")):
        fields.extend(["name", "employee_id", "department_name", "job_title", "office_city", "manager_id", "hire_date", "email"])
    for candidate in field_candidates:
        if candidate.key in {"name", "employee_id", "email", "department_code", "department_name", "job_title", "manager_id", "timezone", "hire_date", "tenure_years"}:
            fields.append(candidate.key)
    if "部门" in text and "department_name" not in fields:
        fields.extend(["department_code", "department_name"])
    if _contains_any(text, ("经理是谁", "上级是谁", "主管是谁", "汇报给谁", "直属经理")):
        fields.append("manager_id")
    if _contains_any(text, ("这个员工是谁", "该员工是谁", "员工是谁")):
        fields.extend(["name", "employee_id", "department_name", "job_title"])
    return list(dict.fromkeys(fields))


def infer_legacy_group_by_hints(text: str, field_candidates: Sequence[SemanticCandidate]) -> List[str]:
    if not _contains_any(text, ("分布", "比例", "占比", "按", "by ")):
        return []
    groupable = [item.key for item in field_candidates if HR_FIELDS.get(item.key, {}).get("aggregatable")]
    if groupable:
        return [groupable[0]]
    if "部门" in text:
        return ["department_code"]
    if "性别" in text:
        return ["gender"]
    if "职级" in text:
        return ["grade"]
    if "时区" in text:
        return ["timezone"]
    return []


def infer_legacy_metric_hints(text: str) -> List[Dict[str, Any]]:
    if _contains_any(text, ("平均", "avg")) and "司龄" in text:
        return [{"type": "avg", "field": "tenure_years"}]
    if _contains_any(text, ("多少", "几个人", "人数", "员工数", "headcount", "分布", "比例", "占比")):
        return [{"type": "count", "field": "employee_id"}]
    return []


def retrieve_examples(
    text: str,
    *,
    intents: Sequence[SemanticCandidate],
    fields: Sequence[SemanticCandidate],
    business_rules: Sequence[SemanticCandidate],
    quality_rules: Sequence[SemanticCandidate],
    max_examples: int = 6,
) -> List[Dict[str, Any]]:
    tags = {item.key for item in [*intents, *fields, *business_rules, *quality_rules]}
    ranked: List[tuple[int, Dict[str, Any]]] = []
    for example in EXAMPLE_LIBRARY:
        example_tags = set(example.get("tags") or [])
        score = len(tags & example_tags) * 5
        if str(example.get("query") or "") in text or text in str(example.get("query") or ""):
            score += 3
        if score > 0:
            ranked.append((score, example))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [dict(item[1]) for item in ranked[:max_examples]]
    if selected:
        return selected
    return [dict(item) for item in EXAMPLE_LIBRARY[:max_examples]]


def _rank_field_candidates(text: str) -> List[SemanticCandidate]:
    candidates: List[SemanticCandidate] = []
    for field, meta in HR_FIELDS.items():
        aliases = [field, *meta.get("aliases", [])]
        evidence = [alias for alias in aliases if _contains(text, alias)]
        if evidence:
            candidates.append(SemanticCandidate(key=field, score=len(evidence) * 10, evidence=evidence))
    return sorted(candidates, key=lambda item: (-item.score, item.key))


def _rank_aliases(
    text: str,
    aliases_by_key: Mapping[str, Sequence[str]],
    *,
    descriptions: Mapping[str, Any] | None = None,
) -> List[SemanticCandidate]:
    candidates: List[SemanticCandidate] = []
    for key, aliases in aliases_by_key.items():
        evidence = [alias for alias in aliases if _contains(text, alias)]
        description = descriptions.get(key) if descriptions else None
        description_text = ""
        if isinstance(description, Mapping):
            description_text = str(description.get("description") or "")
        elif description:
            description_text = str(description)
        if description_text and _contains_any(text, description_text.replace("/", " ").split()):
            evidence.append(description_text)
        if evidence:
            candidates.append(SemanticCandidate(key=key, score=len(evidence) * 10, evidence=list(dict.fromkeys(evidence))))
    return sorted(candidates, key=lambda item: (-item.score, item.key))


def _normalize_filters(filters: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in filters:
        if not isinstance(item, Mapping):
            continue
        field = normalize_field_name(item.get("field"))
        if not field:
            continue
        fixed = dict(item)
        fixed["field"] = field
        fixed["op"] = str(fixed.get("op") or "=")
        if fixed["op"] == "exists":
            continue
        if "value" in fixed:
            fixed["value"] = normalize_enum_value(fixed.get("value"))
        normalized.append(fixed)
    return normalized


def _merge_filters(primary: Iterable[Mapping[str, Any]], hints: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*primary, *hints]:
        field = normalize_field_name(item.get("field"))
        if not field:
            continue
        op = str(item.get("op") or "=")
        value = normalize_enum_value(item.get("value"))
        key = (field, op, _filter_value_key(value))
        if key in seen:
            continue
        seen.add(key)
        merged.append({"field": field, "op": op, "value": value})
    return merged


def _filter_value_key(value: Any) -> str:
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    return repr(value)


def _candidate_payload(candidate: SemanticCandidate) -> Dict[str, Any]:
    return {"key": candidate.key, "score": candidate.score, "evidence": candidate.evidence}


def _first_key(candidates: Sequence[SemanticCandidate]) -> str:
    return candidates[0].key if candidates else ""


def _contains(text: str, needle: str) -> bool:
    if not needle:
        return False
    return needle in text or needle.lower() in text.lower()


def _contains_any(text: str, words: Iterable[str]) -> bool:
    return any(_contains(text, word) for word in words)

"""LLM planner that emits guarded HR Elasticsearch search bodies."""






@dataclass
class HRDslPlannerResult:
    search_body: Dict[str, Any]
    debug: Dict[str, Any] = field(default_factory=dict)


class HRDslPlanner:
    """OpenAI-compatible planner for safe ES search bodies."""

    def __init__(self, *, timeout_seconds: Optional[float] = None, model_name: str = "", max_limit: int = 100, max_buckets: int = 100):
        self.timeout_seconds = timeout_seconds
        self.model_name = model_name
        self.max_limit = int(max_limit or 100)
        self.max_buckets = int(max_buckets or 100)

    def plan(self, query: str, *, user_id: str = "agent-system") -> HRDslPlannerResult:
        messages = build_hr_query_ir_planner_messages(query, max_limit=self.max_limit, max_buckets=self.max_buckets)
        raw = run_hr_es_dsl_completion(messages, user_id=user_id, timeout_seconds=self.timeout_seconds, model_name=self.model_name)
        query_ir = parse_llm_query_ir(raw)
        query_ir = repair_query_ir_from_query(query_ir, query, max_limit=self.max_limit)
        search_body = compile_query_ir_to_es_dsl(query_ir, max_limit=self.max_limit)
        search_body = validate_hr_es_search_body(search_body, max_limit=self.max_limit, max_buckets=self.max_buckets)
        return HRDslPlannerResult(search_body=search_body, debug={"planner": "query_ir", "raw": raw, "query_ir": query_ir})


def build_hr_query_ir_planner_messages(query: str, *, max_limit: int = 100, max_buckets: int = 100) -> List[Dict[str, str]]:
    semantic_context = build_query_ir_semantic_context(query)
    examples = [
        {
            "query": "在北京办公，且司龄小于1年的员工名单发给我。",
            "query_ir": {
                "domain": "hr",
                "entity": "employee",
                "intent": "list",
                "filters": [
                    {"field": "office_city", "op": "=", "value": "北京"},
                    {"field": "tenure_years", "op": "<", "value": 1},
                    {"field": "employment_status", "op": "=", "value": "active"},
                ],
                "select": ["employee_id", "name", "department_code", "job_title", "grade", "office_city", "hire_date", "tenure_years"],
                "limit": max_limit,
            },
        },
        {
            "query": "Beijing employees with tenure under one year",
            "query_ir": {
                "domain": "hr",
                "entity": "employee",
                "intent": "list",
                "filters": [
                    {"field": "office_city", "op": "=", "value": "北京"},
                    {"field": "tenure_years", "op": "<", "value": 1},
                    {"field": "employment_status", "op": "=", "value": "active"},
                ],
                "select": ["employee_id", "name", "office_city", "hire_date", "tenure_years"],
                "limit": max_limit,
            },
        },
        {
            "query": "所有员工中绩效为高的有多少？",
            "query_ir": {
                "domain": "hr",
                "entity": "employee",
                "intent": "aggregation",
                "filters": [
                    {"field": "performance_rating", "op": "=", "value": "高"},
                    {"field": "employment_status", "op": "=", "value": "active"},
                ],
                "metrics": [{"type": "count", "field": "employee_id"}],
                "limit": 1,
            },
        },
        {
            "query": "绩效为高的员工中入职年限最短的人是谁",
            "query_ir": {
                "domain": "hr",
                "entity": "employee",
                "intent": "list",
                "filters": [
                    {"field": "performance_rating", "op": "=", "value": "高"},
                    {"field": "employment_status", "op": "=", "value": "active"},
                ],
                "select": ["employee_id", "name", "hire_date", "tenure_years", "performance_rating"],
                "sort": [{"field": "tenure_years", "direction": "asc"}],
                "limit": 1,
            },
        },
    ]
    system = (
        "You are an HR QueryIR planner. Return exactly one JSON object named QueryIR; do not return Markdown or prose. "
        "Do not produce Elasticsearch DSL. QueryIR is a safe logical query with fields: domain, entity, intent, filters, select, "
        "group_by, metrics, sort, limit. Use only fields from metadata. Normalize multilingual phrasing to metadata fields, "
        "for example tenure / length of service / 司龄 -> tenure_years. Preserve comparison operators exactly: under/less than -> <, "
        "over/greater than -> >, at least/no less than -> >=, at most/no more than -> <=. "
        "For list questions, use intent=list and include useful select fields. For count/distribution questions, use intent=aggregation. "
        f"Use limit <= {max_limit}; aggregation buckets are capped later at {max_buckets}."
    )
    user = json.dumps(
        {
            "natural_language_query": query,
            "fields": query_ir_field_metadata(),
            "semantic_hints": semantic_context.as_prompt_payload(),
            "examples": examples,
            "max_limit": max_limit,
            "max_aggregation_buckets": max_buckets,
        },
        ensure_ascii=False,
        indent=2,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_hr_es_dsl_planner_messages(query: str, *, max_limit: int = 100, max_buckets: int = 100) -> List[Dict[str, str]]:
    fields = {
        field: {
            "type": meta.get("type"),
            "aliases": meta.get("aliases", []),
            "filterable": meta.get("filterable"),
            "aggregatable": meta.get("aggregatable"),
        }
        for field, meta in HR_FIELDS.items()
    }
    examples = [
        {
            "query": "把公司所有名字包含“赵九”的人员名单给我",
            "search_body": {
                "size": max_limit,
                "track_total_hits": True,
                "_source": ["employee_id", "name", "department_code", "department_name", "job_title", "grade", "office_city", "employment_status"],
                "query": {"bool": {"filter": [{"match_phrase": {"name": "赵九"}}]}},
                "sort": [{"grade_level": {"order": "desc"}}, {"employee_id": {"order": "asc"}}],
            },
        },
        {
            "query": "公司有几个赵九？",
            "search_body": {
                "size": 0,
                "track_total_hits": True,
                "query": {"bool": {"filter": [{"match_phrase": {"name": "赵九"}}]}},
            },
        },
        {
            "query": "北京办公室有多少员工？",
            "search_body": {
                "size": 0,
                "track_total_hits": True,
                "query": {"bool": {"filter": [{"term": {"office_city": "北京"}}, {"term": {"employment_status": "active"}}]}},
            },
        },
        {
            "query": "各部门人数分布",
            "search_body": {
                "size": 0,
                "track_total_hits": True,
                "query": {"match_all": {}},
                "aggs": {"by_department": {"terms": {"field": "department_code", "size": min(max_buckets, 100)}}},
            },
        },
        {
            "query": "300001 这个员工是谁？",
            "search_body": {
                "size": 1,
                "track_total_hits": True,
                "_source": ["employee_id", "name", "department_name", "job_title", "office_city", "employment_status"],
                "query": {"bool": {"filter": [{"term": {"employee_id": "300001"}}]}},
            },
        },
        {
            "query": "请给我查一下陈五100这个员工的资料",
            "search_body": {
                "size": 1,
                "track_total_hits": True,
                "_source": ["employee_id", "name", "department_name", "job_title", "office_city", "employment_status", "manager_id", "hire_date", "email"],
                "query": {"bool": {"filter": [{"term": {"name.keyword": "陈五100"}}]}},
            },
        },
    ]
    system = (
        "You are an HR Elasticsearch search-body planner. Return exactly one JSON object: the body for Elasticsearch _search. "
        "Do not return Markdown or prose. Never include index names, URLs, HTTP methods, scripts, runtime_mappings, query_string, wildcard, regexp, prefix, fuzzy, knn, highlight, collapse, or rescore. "
        "Use only this safe query subset: bool, term, terms, match, match_phrase, range, exists, match_all, terms aggs and simple metric aggs. "
        f"Use size <= {max_limit}; use aggregation terms size <= {max_buckets}. "
        "For Chinese name substring requests such as 名字包含赵九, use match_phrase on field name. "
        "Chinese names may include numeric suffixes, e.g. 陈五100 or 何九177. Preserve the full token as name.keyword or name match_phrase; do not treat the suffix as employee_id unless the user explicitly says 工号 or 员工ID. "
        "For headcount/count questions, use size 0 and track_total_hits true. "
        "For list questions, return a useful _source field list and sort by grade_level desc then employee_id asc when appropriate."
    )
    user = json.dumps(
        {
            "natural_language_query": query,
            "fields": fields,
            "max_limit": max_limit,
            "max_aggregation_buckets": max_buckets,
            "examples": examples,
        },
        ensure_ascii=False,
        indent=2,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair_hr_es_dsl_from_query(search_body: Dict[str, Any], query: str) -> Dict[str, Any]:
    explicit_name = _extract_legacy_generated_employee_name(query)
    if not explicit_name:
        return _apply_default_active_scope(search_body, query)
    if any(keyword in query for keyword in ("工号", "员工ID", "员工 ID", "employee_id")):
        return _apply_default_active_scope(search_body, query)
    repaired = dict(search_body)
    repaired["query"] = {"bool": {"filter": [{"term": {"name.keyword": explicit_name}}]}}
    if int(repaired.get("size") or 0) == 0:
        repaired["size"] = 1
    required_source = list(sample_backend.EMPLOYEE_SUMMARY_FIELDS)  # noqa: SLF001 - shared HR ES source baseline
    if any(keyword in query for keyword in ("离职风险", "流失风险", "风险")):
        required_source.append("attrition_risk")
    existing_source = repaired.get("_source")
    if existing_source is False:
        repaired["_source"] = required_source
    elif isinstance(existing_source, list):
        repaired["_source"] = list(dict.fromkeys([*existing_source, *required_source]))
    else:
        repaired["_source"] = required_source
    repaired["track_total_hits"] = True
    return repaired


def _apply_default_active_scope(search_body: Dict[str, Any], query: str) -> Dict[str, Any]:
    text = str(query or "")
    if _explicitly_requests_inactive_or_history(text):
        return search_body
    if not _looks_like_broad_employee_query(text):
        return search_body
    repaired = dict(search_body)
    query_body = repaired.get("query")
    if _query_has_field_filter(query_body, "employment_status"):
        return repaired
    active_filter = {"term": {"employment_status": "active"}}
    if not isinstance(query_body, dict) or not query_body:
        repaired["query"] = {"bool": {"filter": [active_filter]}}
        return repaired
    if "bool" in query_body and isinstance(query_body.get("bool"), dict):
        bool_query = dict(query_body["bool"])
        filters = bool_query.get("filter") or []
        if not isinstance(filters, list):
            filters = [filters]
        bool_query["filter"] = [*filters, active_filter]
        repaired["query"] = {"bool": bool_query}
        return repaired
    repaired["query"] = {"bool": {"must": [query_body], "filter": [active_filter]}}
    return repaired


def _looks_like_broad_employee_query(text: str) -> bool:
    return any(keyword in text for keyword in ("公司", "员工", "人员", "经理", "名单", "清单", "列出", "几个", "多少", "有哪些"))


def _explicitly_requests_inactive_or_history(text: str) -> bool:
    normalized = str(text or "")
    risk_text = normalized.replace("离职风险", "").replace("流失风险", "").replace("离职风险等级", "")
    return any(keyword in risk_text for keyword in ("离职", "已离职", "退休", "已退休", "非在职", "历史", "所有记录", "含离职", "包括离职"))


def _query_has_field_filter(value: Any, field: str) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"term", "terms", "match", "match_phrase", "range"} and isinstance(child, dict):
                if any(str(candidate).removesuffix(".keyword") == field for candidate in child):
                    return True
            if key == "exists" and isinstance(child, dict) and str(child.get("field") or "").removesuffix(".keyword") == field:
                return True
            if _query_has_field_filter(child, field):
                return True
    elif isinstance(value, list):
        return any(_query_has_field_filter(item, field) for item in value)
    return False


def _extract_legacy_generated_employee_name(query: str) -> str:
    match = re.search(r"[\u4e00-\u9fff]{2}\d{1,4}", str(query or ""))
    if not match:
        return ""
    name = match.group(0)
    if name[:2] in {"小于", "大于", "超过", "少于", "低于", "高于", "不足", "未满", "不满", "至多"}:
        return ""
    return name


def run_hr_es_dsl_completion(
    messages: List[Dict[str, str]],
    *,
    user_id: str = "agent-system",
    timeout_seconds: Optional[float] = None,
    model_name: str = "",
) -> str:
    return run_agent_planner_completion(
        messages,
        user_id=user_id,
        timeout_seconds=timeout_seconds,
        model_name=model_name,
        error_label="HR DSL planner",
    )


def parse_llm_search_body(raw: str) -> Dict[str, Any]:
    return parse_llm_json_object(raw, error_message="HR DSL planner output must be a JSON object")


def parse_llm_query_ir(raw: str) -> Dict[str, Any]:
    return parse_llm_json_object(raw, error_message="HR QueryIR planner output must be a JSON object")


def repair_query_ir_from_query(query_ir: Dict[str, Any], query: str, *, max_limit: int = 100) -> Dict[str, Any]:
    repaired = deepcopy(dict(query_ir or {}))
    for forbidden_key in ("query", "aggs", "aggregations", "search_body", "script", "runtime_mappings"):
        repaired.pop(forbidden_key, None)
    if "select_fields" in repaired and "select" not in repaired:
        repaired["select"] = repaired.pop("select_fields")
    repaired.setdefault("domain", "hr")
    repaired.setdefault("entity", "employee")
    repaired.setdefault("intent", "list")

    text = str(query or "")
    context = build_query_ir_semantic_context(text)
    if repaired.get("intent") in {"list", "aggregation", "attribute_lookup"}:
        repaired["filters"] = _merge_filters(repaired.get("filters") or [], context.filters)
    if repaired.get("intent") == "aggregation":
        group_by = [normalize_field_name(field) for field in repaired.get("group_by") or [] if normalize_field_name(field)]
        repaired["group_by"] = list(dict.fromkeys([*group_by, *context.group_by]))
        if not repaired.get("metrics") and context.metrics:
            repaired["metrics"] = context.metrics
    if repaired.get("intent") in {"list", "attribute_lookup"}:
        select = [normalize_field_name(field) for field in repaired.get("select") or [] if normalize_field_name(field)]
        repaired["select"] = list(dict.fromkeys([*select, *context.select_fields]))
    _apply_default_query_ir_active_scope(repaired, text)
    _apply_query_ir_full_list_limit(repaired, text, max_limit=max_limit)
    return validate_query_ir(repaired, max_limit=max_limit)


def _apply_default_query_ir_active_scope(query_ir: Dict[str, Any], text: str) -> None:
    if query_ir.get("intent") not in {"list", "aggregation"}:
        return
    if _explicitly_requests_inactive_or_history(text):
        return
    filters = list(query_ir.get("filters") or [])
    if any(isinstance(item, Mapping) and item.get("field") == "employment_status" for item in filters):
        return
    filters.append({"field": "employment_status", "op": "=", "value": "active"})
    query_ir["filters"] = filters


def _apply_query_ir_full_list_limit(query_ir: Dict[str, Any], text: str, *, max_limit: int) -> None:
    if query_ir.get("intent") != "list":
        return
    if not any(keyword in text for keyword in ("列出", "名单", "清单", "列表", "有哪些", "都有谁", "都是谁")):
        return
    query_ir["limit"] = max(int(query_ir.get("limit") or 20), int(max_limit or 100))

"""Planner layer for HR production QuerySpec generation."""






PLANNER_LLM = "llm"


@dataclass
class PlannerResult:
    query_spec: Dict[str, Any]
    debug: Dict[str, Any] = field(default_factory=dict)


class LLMQuerySpecPlanner:
    """OpenAI-compatible planner that returns only controlled QuerySpec JSON."""

    def __init__(self, *, timeout_seconds: Optional[float] = None, model_name: str = ""):
        self.timeout_seconds = timeout_seconds
        self.model_name = model_name

    def plan(self, query: str, *, user_id: str = "agent-system") -> PlannerResult:
        messages = build_hr_planner_messages(query)
        raw = run_hr_planner_completion(messages, user_id=user_id, timeout_seconds=self.timeout_seconds, model_name=self.model_name)
        query_spec = parse_llm_query_spec(raw)
        return PlannerResult(query_spec=query_spec, debug={"planner": PLANNER_LLM, "raw": raw})

def plan_hr_query_spec(
    query: str,
    *,
    max_limit: int = 100,
    user_id: str = "agent-system",
    llm_planner: Optional[LLMQuerySpecPlanner] = None,
) -> PlannerResult:
    """Plan a validated HR QuerySpec through the single internal LLM path."""
    result = (llm_planner or LLMQuerySpecPlanner()).plan(query, user_id=user_id)
    result.query_spec = repair_query_spec_from_query(result.query_spec, query)
    _apply_business_rule_filters(result.query_spec, query)
    _apply_default_headcount_scope(result.query_spec, query)
    _apply_default_active_list_scope(result.query_spec, query)
    _apply_legacy_requested_full_list_limit(result.query_spec, query)
    result.query_spec = validate_query_spec(result.query_spec, max_limit=max_limit)
    result.debug.setdefault("planner", PLANNER_LLM)
    return result


def parse_query_spec_input(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or value == "":
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise QuerySpecValidationError("query_spec must be an object or JSON object string")


def repair_query_spec_from_query(query_spec: Dict[str, Any], query: str) -> Dict[str, Any]:
    repaired = normalize_spec_with_semantics(query_spec, query)
    subject_name = _extract_legacy_explicit_employee_name(query)
    if subject_name:
        subjects = list(repaired.get("subjects") or [])
        if not subjects:
            subjects = [{"type": "employee_name", "value": subject_name}]
        else:
            fixed_subjects: List[Dict[str, Any]] = []
            for item in subjects:
                subject = dict(item) if isinstance(item, dict) else {}
                value = str(subject.get("value") or subject.get("name") or "")
                if not value or subject_name.startswith(value) or value.startswith(subject_name.rstrip("0123456789")):
                    subject["type"] = "employee_name"
                    subject["value"] = subject_name
                    subject.pop("name", None)
                fixed_subjects.append(subject)
            subjects = fixed_subjects
        repaired["subjects"] = subjects

    text = str(query or "")
    if repaired.get("intent") == "attribute_lookup" and any(keyword in text for keyword in ("办公", "办", "工作地")):
        fields = list(repaired.get("select_fields") or [])
        if "office_city" not in fields:
            fields.append("office_city")
        repaired["select_fields"] = fields
    return normalize_spec_with_semantics(repaired, query)


def _extract_legacy_explicit_employee_name(query: str) -> str:
    text = str(query or "")
    match = re.search(r"[\u4e00-\u9fff]{2,4}\d{1,4}", text)
    if match:
        name = match.group(0)
        if name[:2] not in {"小于", "大于", "超过", "少于", "低于", "高于", "不足", "未满", "不满", "至多"}:
            return name
    match = re.search(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s*\d{1,4}\b", text)
    if match:
        return " ".join(match.group(0).split())
    return ""


def build_hr_planner_messages(query: str, *, validation_error: str = "", previous_output: str = "") -> List[Dict[str, str]]:
    semantic_context = build_semantic_context(query)
    field_summary = {
        field: {
            "type": meta.get("type"),
            "aliases": meta.get("aliases", [])[:4],
            "filterable": meta.get("filterable"),
            "aggregatable": meta.get("aggregatable"),
        }
        for field, meta in HR_FIELDS.items()
    }
    system = (
        "You are an HR QuerySpec planner. Return one JSON object only. "
        "Do not return Markdown, prose, Elasticsearch DSL, script, query, aggs, or aggregations. "
        "Names may contain numeric suffixes, e.g. 'Helen Wang 161' or '陈明19'; preserve the full name as employee_name "
        "and do not treat the suffix as employee_id unless the user explicitly says employee ID or 工号. "
        "Use semantic_hints first, then the full field metadata. Prefer business_rule keys when a business concept matches. "
        "If deterministic_filter_hints are relevant, include equivalent filters in the QuerySpec. "
        "Use only fields from the provided metadata. If the request is ambiguous, return "
        '{"domain":"hr","entity":"employee","intent":"clarify","needs_clarification":true,"clarifying_question":"..."}'
    )
    user = json.dumps(
        {
            "natural_language_query": query,
            "supported_intents": [
                "aggregation",
                "list",
                "relationship",
                "quality_check",
                "document_fetch",
                "hybrid_search",
                "attribute_lookup",
                "clarify",
            ],
            "fields": field_summary,
            "semantic_hints": semantic_context.as_prompt_payload(),
            "business_rules": BUSINESS_RULES,
            "quality_rules": QUALITY_RULES,
            "examples": semantic_context.examples,
            "previous_output": previous_output,
            "validation_error_to_fix": validation_error,
        },
        ensure_ascii=False,
        indent=2,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_attribute_lookup_spec(text: str) -> Dict[str, Any]:
    if not _looks_like_legacy_attribute_lookup(text):
        return {}

    fields = _infer_legacy_attribute_select_fields(text)
    if not fields:
        return {}

    people = sample_backend._find_people_by_query(text)  # noqa: SLF001 - migration baseline
    if not people and any(pronoun in text for pronoun in ("他", "她", "其", "这个人", "该员工")):
        return {
            "domain": "hr",
            "entity": "employee",
            "intent": "clarify",
            "needs_clarification": True,
            "clarifying_question": "你想查询哪位员工？请提供姓名或员工 ID。",
            "limit": 1,
        }
    subjects = [
        {"type": "employee", "employee_id": person.employee_id, "name": person.name}
        for person in people[:3]
    ]
    if not subjects:
        subject_name = _extract_legacy_explicit_employee_name(text)
        if subject_name:
            subjects = [{"type": "employee_name", "value": subject_name}]
    return {
        "domain": "hr",
        "entity": "employee",
        "intent": "attribute_lookup",
        "subjects": subjects,
        "select_fields": fields,
        "limit": max(1, len(subjects) or 1),
    }


def _looks_like_legacy_attribute_lookup(text: str) -> bool:
    if any(keyword in text for keyword in ("格式不对", "格式不规范", "异常", "重复", "出现多次", "流失风险", "关键岗位员工")):
        return False
    return any(
        keyword in text
        for keyword in (
            "在哪里办公",
            "在哪办公",
            "办公地点",
            "办公室在哪",
            "工号",
            "员工 ID",
            "员工ID",
            "编号",
            "邮箱",
            "邮件",
            "基本信息",
            "员工信息",
            "个人信息",
            "详细信息",
            "资料",
            "哪个部门",
            "什么部门",
            "什么职位",
            "是什么职位",
            "什么岗位",
            "是什么岗位",
            "经理是谁",
            "上级是谁",
            "主管是谁",
            "汇报给谁",
            "这个员工是谁",
            "该员工是谁",
            "员工是谁",
            "入职多久",
            "入职几年",
            "入职多少年",
            "入职年限",
            "司龄",
            "哪一年入职",
            "什么时候入职",
            "入职日期",
        )
    )


def _infer_legacy_attribute_select_fields(text: str) -> List[str]:
    fields: List[str] = []
    if any(keyword in text for keyword in ("在哪里办公", "在哪办公", "办公地点", "办公室在哪", "工作地")):
        fields.append("office_city")
    if any(keyword in text for keyword in ("工号", "员工 ID", "员工ID", "编号")):
        fields.append("employee_id")
    if any(keyword in text for keyword in ("邮箱", "邮件")):
        fields.append("email")
    if any(keyword in text for keyword in ("基本信息", "员工信息", "个人信息", "详细信息", "资料")):
        fields.extend(["name", "employee_id", "department_name", "job_title", "office_city", "manager_id", "hire_date", "email"])
    if any(keyword in text for keyword in ("哪个部门", "什么部门", "部门")):
        fields.extend(["department_code", "department_name"])
    if any(keyword in text for keyword in ("职位", "岗位", "职务")):
        fields.append("job_title")
    if any(keyword in text for keyword in ("经理是谁", "上级是谁", "主管是谁", "汇报给谁", "直属经理")):
        fields.append("manager_id")
    if any(keyword in text for keyword in ("这个员工是谁", "该员工是谁", "员工是谁")):
        fields.extend(["name", "employee_id", "department_name", "job_title"])
    if any(keyword in text for keyword in ("入职多久", "入职几年", "入职多少年", "入职年限", "司龄")):
        fields.append("tenure_years")
    if any(keyword in text for keyword in ("哪一年入职", "什么时候入职", "入职日期", "入职时间")):
        fields.append("hire_date")
    semantic_fields = build_semantic_context(text, max_examples=0).select_fields
    return list(dict.fromkeys([*fields, *semantic_fields]))


def _apply_default_headcount_scope(query_spec: Dict[str, Any], text: str) -> None:
    if query_spec.get("intent") != "aggregation":
        return
    metrics = query_spec.get("metrics") or []
    is_count = any(item.get("type") == "count" and item.get("field") == "employee_id" for item in metrics)
    if not is_count:
        return
    if _explicitly_requests_inactive_or_history(text):
        return
    filters = list(query_spec.get("filters") or [])
    if any(item.get("field") == "employment_status" for item in filters):
        return
    filters.append({"field": "employment_status", "op": "=", "value": "active"})
    query_spec["filters"] = filters
    query_spec["business_scope"] = "active_headcount"


def _apply_default_active_list_scope(query_spec: Dict[str, Any], text: str) -> None:
    if query_spec.get("intent") != "list":
        return
    if _explicitly_requests_inactive_or_history(text):
        return
    filters = list(query_spec.get("filters") or [])
    if any(item.get("field") == "employment_status" for item in filters if isinstance(item, dict)):
        if any(
            item.get("field") == "employment_status" and item.get("value") == "active"
            for item in filters
            if isinstance(item, dict)
        ):
            query_spec["business_scope"] = "active_employee_list"
        return
    filters.append({"field": "employment_status", "op": "=", "value": "active"})
    query_spec["filters"] = filters
    query_spec["business_scope"] = "active_employee_list"


def _apply_legacy_requested_full_list_limit(query_spec: Dict[str, Any], text: str) -> None:
    if query_spec.get("intent") != "list":
        return
    if not any(
        keyword in text
        for keyword in (
            "列出所有",
            "列出全部",
            "全部列出",
            "完整名单",
            "完整列表",
            "清单",
            "名单",
            "列表",
            "所有在",
            "都有哪些",
            "有哪些人",
            "有哪些员工",
            "都是谁",
            "都有谁",
        )
    ):
        return
    query_spec["limit"] = max(int(query_spec.get("limit") or 20), 100)


def _explicitly_requests_inactive_or_history(text: str) -> bool:
    return any(
        keyword in text
        for keyword in ("已解雇", "退休", "非在职", "历史", "所有记录", "全部记录", "含离职", "包括离职", "离职员工", "已离职")
    )


def _build_quality_check_spec(text: str) -> Dict[str, Any]:
    check = ""
    if any(keyword in text for keyword in ("经理 ID 不存在", "经理ID不存在", "经理不存在", "没有合法经理", "上级编号无效", "manager 不存在")):
        check = "invalid_manager_id"
    elif "邮箱" in text and any(keyword in text for keyword in ("格式不对", "格式不规范", "不规范", "异常", "不合法")):
        check = "invalid_email"
    elif any(keyword in text for keyword in ("重复", "出现多次")) and any(keyword in text for keyword in ("员工 ID", "员工ID", "员工编号", "工号")):
        check = "duplicate_employee_id"
    elif "FTE" in text and any(keyword in text for keyword in ("标准工时", "40 小时", "40小时")):
        check = "fte_hours_conflict"
    elif any(keyword in text for keyword in ("缺失关键字段", "缺少关键字段", "部门职级缺失")):
        check = "missing_critical_fields"
    if not check:
        return {}
    return {
        "domain": "hr",
        "entity": "employee",
        "intent": "quality_check",
        "filters": [],
        "check": check,
        "limit": 20,
    }


def _apply_paraphrase_intent(query_spec: Dict[str, Any], text: str) -> None:
    if any(keyword in text for keyword in ("底下有几个人", "底下有多少人", "带几个人", "带多少人", "汇报给他的人")):
        query_spec["intent"] = "relationship"
        return
    if query_spec.get("intent") == "list" and any(keyword in text for keyword in ("几个", "几个人", "多少人", "总人数", "headcount")):
        query_spec["intent"] = "aggregation"
        query_spec["metrics"] = [{"type": "count", "field": "employee_id"}]


def _apply_business_rule_filters(query_spec: Dict[str, Any], text: str) -> None:
    if (
        query_spec.get("case") == "high_risk_key_talent"
        or any(keyword in text for keyword in ("离职风险", "流失风险"))
        and any(keyword in text for keyword in ("影响度高", "关键人才", "关键岗位"))
    ):
        query_spec["intent"] = "list"
        query_spec["filters"] = [
            {"field": "attrition_risk", "op": "=", "value": "高"},
            {"field": "key_position", "op": "=", "value": True},
            {"field": "employment_status", "op": "=", "value": "active"},
        ]
        query_spec["business_rule"] = "high_risk_key_talent"


def run_hr_planner_completion(
    messages: List[Dict[str, str]],
    *,
    user_id: str = "agent-system",
    timeout_seconds: Optional[float] = None,
    model_name: str = "",
) -> str:
    return run_agent_planner_completion(
        messages,
        user_id=user_id,
        timeout_seconds=timeout_seconds,
        model_name=model_name,
        error_label="HR planner",
    )


def parse_llm_query_spec(raw: str) -> Dict[str, Any]:
    try:
        return parse_llm_json_object(raw, error_message="LLM planner output must be a JSON object")
    except ValueError as exc:
        raise QuerySpecValidationError("LLM planner output must be a JSON object") from exc
