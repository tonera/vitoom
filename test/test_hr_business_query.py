from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tool_catalog import ToolCatalog  # noqa: E402
from backend.services.agent.tools.builtin import hr_business_query as hr_query  # noqa: E402
from backend.services.agent.tools.builtin.business_query_core.hr.es_dsl_guard import (  # noqa: E402
    HRDslValidationError,
    validate_hr_es_search_body,
)
from backend.services.agent.tools.builtin.business_query_core.hr.planner import (  # noqa: E402
    LLMQuerySpecPlanner,
    PlannerResult,
    build_hr_query_ir_planner_messages,
    plan_hr_query_spec,
    repair_query_spec_from_query,
)
from backend.services.agent.tools.builtin.business_query_core.hr.query_ir import compile_query_ir_to_es_dsl, validate_query_ir  # noqa: E402
from backend.services.agent.tools.builtin.business_query_core.hr.query_spec import QuerySpecValidationError, validate_query_spec  # noqa: E402
from backend.services.agent.tools.builtin.list_available_tools import _build_capability_markdown  # noqa: E402
from backend.services.agent.tools.registry import get_tool_plugin_registry  # noqa: E402


def _execution_result(*, total: int = 1, rows: list[dict] | None = None) -> dict:
    selected_rows = rows if rows is not None else [{"employee_id": "100001", "name": "张三"}]
    return {
        "intent": "es_dsl",
        "backend": "es",
        "lines": [f"匹配员工共 {total} 人。"],
        "rows": selected_rows,
        "summary": {"row_count": len(selected_rows), "total": total},
        "debug": {"es_query": {"query": {"match_all": {}}}},
    }


def test_hr_business_query_registered_in_plugin_registry_and_catalog():
    registrations = get_tool_plugin_registry().all_registrations()
    assert hr_query.HR_BUSINESS_QUERY_TOOL_NAME in registrations

    entry = ToolCatalog().get(hr_query.HR_BUSINESS_QUERY_TOOL_NAME)
    assert entry is not None
    assert entry.enabled is True
    assert "QuerySpec" in entry.description


def test_available_tools_list_includes_hr_business_query():
    output = _build_capability_markdown(runtime_allowlist=[])

    assert "`hr_business_query`" in output


def test_hr_business_query_tool_schema_does_not_expose_planner_or_backend():
    tool = hr_query.build_hr_business_query_tool(context={"user_id": "test-user"})
    fields = set(getattr(tool.args_schema, "model_fields", getattr(tool.args_schema, "__fields__", {})))

    assert "query" in fields
    assert "planner" not in fields
    assert "backend" not in fields


def test_hr_business_query_uses_unified_dsl_path(monkeypatch):
    def fake_run_dsl_path(query: str, *, configured: dict, user_id: str):
        assert query == "北京办公室有多少员工？"
        return (
            {"domain": "hr", "entity": "employee", "intent": "es_dsl", "search_body": {"query": {"match_all": {}}}},
            {"planner": "query_ir"},
            _execution_result(total=3, rows=[]),
        )

    monkeypatch.setattr(hr_query, "_run_dsl_path", fake_run_dsl_path)
    output = hr_query.run_hr_business_query("北京办公室有多少员工？", include_debug=True)

    assert "匹配员工共 3 人" in output
    assert '"intent": "es_dsl"' in output
    assert '"planner": "unified"' in output
    assert '"fallback_used": false' in output


def test_hr_business_query_zero_result_does_not_fallback(monkeypatch):
    calls = {"queryspec": 0}

    def fake_run_dsl_path(query: str, *, configured: dict, user_id: str):
        return (
            {"domain": "hr", "entity": "employee", "intent": "es_dsl", "search_body": {"query": {"match_all": {}}}},
            {"planner": "query_ir"},
            _execution_result(total=0, rows=[]),
        )

    def fake_plan_queryspec(*_args, **_kwargs):
        calls["queryspec"] += 1
        raise AssertionError("0-result DSL response must not fallback to QuerySpec")

    monkeypatch.setattr(hr_query, "_run_dsl_path", fake_run_dsl_path)
    monkeypatch.setattr(hr_query, "_plan_queryspec", fake_plan_queryspec)

    output = hr_query.run_hr_business_query("不存在的城市有多少员工？")

    assert "匹配员工共 0 人" in output
    assert calls["queryspec"] == 0


def test_hr_business_query_falls_back_to_queryspec_when_dsl_fails(monkeypatch):
    def fake_run_dsl_path(query: str, *, configured: dict, user_id: str):
        raise HRDslValidationError("bad dsl")

    def fake_plan_queryspec(query: str, *, configured: dict, user_id: str):
        return PlannerResult(
            query_spec={
                "domain": "hr",
                "entity": "employee",
                "intent": "aggregation",
                "filters": [{"field": "office_city", "op": "=", "value": "北京"}],
                "metrics": [{"type": "count", "field": "employee_id"}],
                "limit": 20,
            },
            debug={"planner": "llm"},
        )

    def fake_execute_hr_query_spec(query_spec: dict, *, query: str = ""):
        assert query_spec["intent"] == "aggregation"
        return {
            "intent": "aggregation",
            "backend": "es",
            "lines": ["匹配员工共 3 人。"],
            "rows": [],
            "summary": {"row_count": 0},
            "debug": {"es_total": 3},
        }

    monkeypatch.setattr(hr_query, "_run_dsl_path", fake_run_dsl_path)
    monkeypatch.setattr(hr_query, "_plan_queryspec", fake_plan_queryspec)
    monkeypatch.setattr(hr_query, "execute_hr_query_spec", fake_execute_hr_query_spec)

    output = hr_query.run_hr_business_query("北京办公室有多少员工？", include_debug=True)

    assert "匹配员工共 3 人" in output
    assert '"fallback_used": true' in output
    assert "HRDslValidationError: bad dsl" in output


def test_hr_business_query_routes_resume_request_to_queryspec(monkeypatch):
    calls = {"dsl": 0}

    def fake_run_dsl_path(*_args, **_kwargs):
        calls["dsl"] += 1
        raise AssertionError("resume requests should route directly to QuerySpec")

    def fake_plan_queryspec(query: str, *, configured: dict, user_id: str):
        assert "简历" in query
        return PlannerResult(
            query_spec={
                "domain": "hr",
                "entity": "employee",
                "intent": "document_fetch",
                "subjects": [{"type": "employee_name", "value": "张一84"}],
                "limit": 1,
            },
            debug={"planner": "llm"},
        )

    def fake_execute_hr_query_spec(query_spec: dict, *, query: str = ""):
        return {
            "intent": "document_fetch",
            "backend": "es",
            "lines": ["找到 1 份简历资料：", "- 张一84（100178）：简历摘要；资产：mock://resume.pdf"],
            "rows": [{"employee_id": "100178", "name": "张一84"}],
            "summary": {"row_count": 1},
            "debug": {"employee_hits": 1},
        }

    monkeypatch.setattr(hr_query, "_run_dsl_path", fake_run_dsl_path)
    monkeypatch.setattr(hr_query, "_plan_queryspec", fake_plan_queryspec)
    monkeypatch.setattr(hr_query, "execute_hr_query_spec", fake_execute_hr_query_spec)

    output = hr_query.run_hr_business_query("张一84的简历给我")

    assert calls["dsl"] == 0
    assert "找到 1 份简历资料" in output
    assert '"intent": "document_fetch"' in output


def test_hr_business_query_accepts_direct_query_spec(monkeypatch):
    def fake_execute_hr_query_spec(query_spec: dict, *, query: str = ""):
        assert query_spec["filters"] == [{"field": "office_city", "op": "=", "value": "北京"}]
        return {
            "intent": "aggregation",
            "backend": "es",
            "lines": ["匹配员工共 4 人。"],
            "rows": [],
            "summary": {"row_count": 0},
            "debug": {"es_total": 4},
        }

    monkeypatch.setattr(hr_query, "execute_hr_query_spec", fake_execute_hr_query_spec)

    output = hr_query.run_hr_business_query(
        query_spec={
            "domain": "hr",
            "entity": "employee",
            "intent": "aggregation",
            "filters": [{"field": "office_city", "op": "=", "value": "北京"}],
            "metrics": [{"type": "count", "field": "employee_id"}],
            "limit": 20,
        },
        include_debug=True,
    )

    assert "匹配员工共 4 人" in output
    assert '"planner": "direct_query_spec"' in output


@pytest.mark.parametrize(
    "query",
    [
        "哪些人没有合法经理？",
        "谁的邮箱格式不对？",
        "同一员工编号重复的有哪些？",
    ],
)
def test_hr_business_query_blocks_data_quality_checks(query: str):
    output = hr_query.run_hr_business_query(query)

    assert "HR 数据质量/数据治理检查问题" in output
    assert "不属于员工业务查询入口" in output
    assert "QuerySpec 校验失败" not in output


def test_hr_business_query_blocks_direct_quality_check_spec():
    output = hr_query.run_hr_business_query(
        query_spec={
            "domain": "hr",
            "entity": "employee",
            "intent": "quality_check",
            "check": "invalid_email",
        },
    )

    assert "HR 数据质量/数据治理检查问题" in output
    assert "invalid_email" not in output


def test_plan_hr_query_spec_uses_semantic_repair_before_validation():
    class AliasLLMPlanner(LLMQuerySpecPlanner):
        def plan(self, query: str, *, user_id: str = "agent-system"):
            return PlannerResult(
                query_spec={
                    "domain": "hr",
                    "entity": "employee",
                    "intent": "list",
                    "filters": [{"field": "办公室", "op": "=", "value": "旧金山"}],
                    "limit": 20,
                },
                debug={"planner": "llm", "raw": "{}"},
            )

    result = plan_hr_query_spec(
        "旧金山当前在岗员工名单",
        max_limit=100,
        llm_planner=AliasLLMPlanner(),
    )

    assert {"field": "office_city", "op": "=", "value": "旧金山"} in result.query_spec["filters"]
    assert {"field": "employment_status", "op": "=", "value": "active"} in result.query_spec["filters"]


def test_repair_query_spec_preserves_english_name_suffix():
    repaired = repair_query_spec_from_query(
        {
            "domain": "hr",
            "entity": "employee",
            "intent": "document_fetch",
            "subjects": [
                {"type": "employee_name", "value": "Helen Wang"},
                {"type": "employee_id", "value": "161"},
            ],
            "limit": 1,
        },
        "把 Helen Wang 161 的简历发给我",
    )

    assert repaired["subjects"][0] == {"type": "employee_name", "value": "Helen Wang 161"}


def test_hr_query_spec_validator_rejects_raw_es_dsl():
    with pytest.raises(QuerySpecValidationError, match="forbidden ES DSL key"):
        validate_query_spec(
            {
                "domain": "hr",
                "entity": "employee",
                "intent": "list",
                "query": {"match_all": {}},
            }
        )


def test_hr_es_dsl_guard_accepts_safe_name_contains_query_and_caps_size():
    guarded = validate_hr_es_search_body(
        {
            "size": 500,
            "_source": ["employee_id", "name", "department_name"],
            "query": {"bool": {"filter": [{"match_phrase": {"name": "赵九"}}]}},
            "sort": [{"grade_level": {"order": "desc"}}, {"employee_id": {"order": "asc"}}],
        },
        max_limit=100,
    )

    assert guarded["size"] == 100
    assert guarded["query"]["bool"]["filter"][0] == {"match_phrase": {"name": "赵九"}}
    assert guarded["track_total_hits"] is True


def test_hr_es_dsl_guard_rejects_dangerous_or_unknown_dsl():
    with pytest.raises(HRDslValidationError):
        validate_hr_es_search_body({"query": {"wildcard": {"name": "*赵九*"}}})

    with pytest.raises(HRDslValidationError):
        validate_hr_es_search_body({"query": {"term": {"salary": 100}}})

    with pytest.raises(HRDslValidationError):
        validate_hr_es_search_body({"script_fields": {"x": {"script": "doc['x'].value"}}})


def test_hr_query_ir_compiles_computed_tenure_filter_to_hire_date():
    query_ir = validate_query_ir(
        {
            "domain": "hr",
            "entity": "employee",
            "intent": "list",
            "filters": [
                {"field": "office_city", "op": "=", "value": "Beijing"},
                {"field": "tenure_years", "op": "<", "value": 1},
                {"field": "employment_status", "op": "=", "value": "active"},
            ],
            "select": ["employee_id", "name", "tenure_years"],
            "limit": 100,
        }
    )
    search_body = compile_query_ir_to_es_dsl(query_ir)
    search_text = json.dumps(search_body, ensure_ascii=False)

    assert query_ir["filters"][0]["value"] == "北京"
    assert {"term": {"office_city": "北京"}} in search_body["query"]["bool"]["filter"]
    assert '"hire_date"' in search_text
    assert '"gt"' in search_text
    assert "tenure_years" not in search_text
    assert "hire_date" in search_body["_source"]


def test_hr_query_ir_prompt_exposes_computed_field_metadata_and_english_example():
    messages = build_hr_query_ir_planner_messages("Beijing employees with tenure under one year")
    payload = json.loads(messages[1]["content"])

    tenure_meta = payload["fields"]["tenure_years"]
    assert tenure_meta["computed"] is True
    assert tenure_meta["compile"]["kind"] == "years_since_date"
    assert tenure_meta["compile"]["source_field"] == "hire_date"
    assert any(example["query"] == "Beijing employees with tenure under one year" for example in payload["examples"])
