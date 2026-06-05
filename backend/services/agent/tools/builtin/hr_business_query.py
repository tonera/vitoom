"""Production HR business query tool.

This domain-specific tool keeps LLM tool selection explicit while routing HR traffic
through Planner -> Validator -> Executor -> Answer Composer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from backend.services.agent.tools.builtin.business_query_core.hr.answer import compose_hr_answer, compose_hr_error
from backend.services.agent.tools.builtin.business_query_core.hr.planner import HRDslPlanner, plan_hr_query_spec
from backend.services.agent.tools.builtin.business_query_core.hr.executor import (
    execute_guarded_hr_es_dsl,
    execute_hr_query_spec,
    placeholder_cost_check,
    placeholder_permission_check,
)
from backend.services.agent.tools.builtin.business_query_core.hr.planner import parse_query_spec_input
from backend.services.agent.tools.builtin.business_query_core.hr.query_spec import QuerySpecValidationError, validate_query_spec
from backend.services.agent.tools.registry import register_tool

HR_BUSINESS_QUERY_TOOL_NAME = "hr_business_query"

HR_BUSINESS_QUERY_DESCRIPTION = (
    "HR 业务域智能查询生产版：将自然语言 HR 问题规划为 QueryIR 并编译为受限 ES DSL，或路由到受控 QuerySpec，"
    "经校验后执行员工问数、人员筛选、组织关系和简历资料查询。适合正式 HR/员工/简历/组织/职级/绩效/"
    "入职/时区/经理关系等查询。多轮对话中遇到他/她/这个人等指代时，调用工具前应将"
    "上下文中的员工姓名或工号补全到 query；遇到“把清单给我/列出来/名单呢”等追问时，必须把"
    "上一轮的过滤条件一起补全，例如改写为“列出姓名包含赵九的员工清单”。"
    "遇到“是谁/都有谁/对，请查一下/查一下/详细信息发给我”等追问时，也必须结合上一轮统计口径、"
    "城市、部门、在职状态、退休/离职状态或上一轮返回的人名/工号补全 query 后再调用本工具；"
    "回答员工姓名、工号、状态、风险等级时只能使用本工具返回结果，不得凭上下文或常识编造人名。"
    "如果工具结果只返回总数，没有返回某字段的分布或明细，不得自行补充“均小于/均大于/全部属于”等结论；"
    "用户明确说“不管/忽略/无论”某个条件时，应去掉历史中的该条件重新查询。"
    "数据质量/数据治理问题（如邮箱格式异常、重复员工 ID、经理 ID 不存在、缺失关键字段）不属于本业务查询工具。"
    "不要用于旅游、旅行规划、天气交通、新闻搜索、闲聊或非 HR 问题。"
)

HR_BUSINESS_QUERY_DOCSTRING = "Run a production HR business query via guarded ES DSL or QuerySpec planning."


def _coerce_tool_args(raw_input: Any = None, **kwargs: Any) -> Dict[str, Any]:
    if kwargs:
        return dict(kwargs)
    if raw_input is None:
        return {}
    if isinstance(raw_input, dict):
        return dict(raw_input)
    if isinstance(raw_input, str):
        text = raw_input.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"query": text}
    return {"query": str(raw_input)}

def run_hr_business_query(
    query: str = "",
    *,
    query_spec: Any = None,
    include_query_spec: bool = True,
    include_sample_rows: bool = True,
    include_debug: bool = False,
    user_id: str = "agent-system",
) -> str:
    text = str(query or "").strip()
    try:
        configured = _query_config()
        direct_spec = parse_query_spec_input(query_spec)
        route = _classify_hr_query_route(text, direct_spec)
        if route == "data_quality":
            return _compose_data_quality_boundary_response(include_debug=include_debug, query=text)
        if direct_spec is not None:
            try:
                planned_spec = validate_query_spec(direct_spec, max_limit=configured["max_limit"])
                if planned_spec.get("intent") == "quality_check":
                    return _compose_data_quality_boundary_response(include_debug=include_debug, query=text)
                planner_debug = {"planner": "direct_query_spec"}
            except QuerySpecValidationError as exc:
                if not text:
                    raise
                plan = _plan_queryspec(text, configured=configured, user_id=user_id)
                planned_spec = plan.query_spec
                planner_debug = {
                    **plan.debug,
                    "direct_query_spec_rejected": True,
                    "direct_query_spec_rejection_reason": str(exc),
                }
        else:
            if not text:
                return "请提供一个 HR 业务问题，或传入 query_spec JSON。"
            if route == "queryspec":
                plan = _plan_queryspec(text, configured=configured, user_id=user_id)
                planned_spec = plan.query_spec
                planner_debug = {**plan.debug, "planner": "unified", "route": route}
            else:
                try:
                    planned_spec, planner_debug, execution_result = _run_dsl_path(text, configured=configured, user_id=user_id)
                    permission_check = placeholder_permission_check(planned_spec)
                    cost_check = placeholder_cost_check(planned_spec)
                    answer = compose_hr_answer(
                        query_spec=planned_spec,
                        execution_result=execution_result,
                        planner_debug={**planner_debug, "planner": "unified", "fallback_used": False},
                        permission_check=permission_check,
                        cost_check=cost_check,
                        include_query_spec=include_query_spec,
                        include_sample_rows=include_sample_rows,
                        include_debug=include_debug,
                    )
                    return _append_unsupported_field_notes(answer, text)
                except Exception as exc:
                    plan = _plan_queryspec(text, configured=configured, user_id=user_id)
                    planned_spec = plan.query_spec
                    planner_debug = {
                        **plan.debug,
                        "planner": "unified",
                        "primary_planner": "dsl",
                        "fallback_used": True,
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                    }

        permission_check = placeholder_permission_check(planned_spec)
        cost_check = placeholder_cost_check(planned_spec)
        execution_result = execute_hr_query_spec(planned_spec, query=text)
        answer = compose_hr_answer(
            query_spec=planned_spec,
            execution_result=execution_result,
            planner_debug=planner_debug,
            permission_check=permission_check,
            cost_check=cost_check,
            include_query_spec=include_query_spec,
            include_sample_rows=include_sample_rows,
            include_debug=include_debug,
        )
        return _append_unsupported_field_notes(answer, text)
    except QuerySpecValidationError as exc:
        return compose_hr_error(f"QuerySpec 校验失败：{exc}", include_debug=include_debug, debug={"query": text})
    except Exception as exc:
        return compose_hr_error(f"HR 查询执行失败：{type(exc).__name__}: {exc}", include_debug=include_debug, debug={"query": text})


def _classify_hr_query_route(query: str, query_spec: Any = None) -> str:
    if isinstance(query_spec, dict) and query_spec.get("intent") == "quality_check":
        return "data_quality"
    text = str(query or "")
    if _looks_like_data_quality_query(text):
        return "data_quality"
    if _looks_like_queryspec_business_query(text):
        return "queryspec"
    return "dsl"


def _looks_like_data_quality_query(text: str) -> bool:
    return any(
        keyword in text
        for keyword in (
            "数据质量",
            "邮箱格式",
            "邮箱不规范",
            "邮箱异常",
            "邮箱不合法",
            "重复员工",
            "员工 ID 重复",
            "员工ID重复",
            "员工编号重复",
            "同一员工 ID",
            "同一员工ID",
            "经理 ID 不存在",
            "经理ID不存在",
            "上级编号无效",
            "没有合法经理",
            "缺失关键字段",
            "缺少关键字段",
            "字段缺失",
        )
    )


def _looks_like_queryspec_business_query(text: str) -> bool:
    return any(
        keyword in text
        for keyword in (
            "直接下属",
            "汇报",
            "下属最多",
            "跨时区",
            "经理和员工",
            "简历",
            "履历",
            "附件",
            "下载",
            "详细信息",
            "详细资料",
            "基本信息",
            "个人信息",
            "员工信息",
            "资料给我",
            "FTE 为 0.5 但标准工时",
        )
    )


def _compose_data_quality_boundary_response(*, include_debug: bool = False, query: str = "") -> str:
    return compose_hr_error(
        "这是 HR 数据质量/数据治理检查问题，不属于员工业务查询入口。请使用管理员或数据治理工具处理，"
        "例如独立的 `hr_data_quality_query`、数据巡检任务或数据治理后台。",
        include_debug=include_debug,
        debug={"query": query, "blocked_reason": "data_quality_boundary"},
    )


def _append_unsupported_field_notes(answer: str, query: str) -> str:
    notes = []
    text = str(query or "")
    if any(keyword in text for keyword in ("学历", "学位", "毕业院校", "学校")):
        notes.append("当前 HR 员工索引未提供学历/学位/毕业院校字段，因此无法从本系统直接查询该信息。")
    if not notes:
        return answer
    return answer.rstrip() + "\n\n### 未覆盖字段\n" + "\n".join(f"- {note}" for note in notes)


def _plan_queryspec(query: str, *, configured: Dict[str, Any], user_id: str):
    return plan_hr_query_spec(
        query,
        max_limit=configured["max_limit"],
        user_id=user_id,
    )


def _run_dsl_path(query: str, *, configured: Dict[str, Any], user_id: str):
    dsl_plan = HRDslPlanner(
        timeout_seconds=configured["planner_timeout_seconds"],
        max_limit=configured["dsl_max_limit"],
        max_buckets=configured["dsl_max_aggregation_buckets"],
    ).plan(query, user_id=user_id)
    execution_result = execute_guarded_hr_es_dsl(
        dsl_plan.search_body,
        query=query,
        max_limit=configured["dsl_max_limit"],
        max_buckets=configured["dsl_max_aggregation_buckets"],
    )
    planned_spec = {
        "domain": "hr",
        "entity": "employee",
        "intent": "es_dsl",
        "search_body": dsl_plan.search_body,
    }
    return planned_spec, dsl_plan.debug, execution_result


def _query_config() -> Dict[str, Any]:
    try:
        from backend.services.agent import settings

        max_limit = settings.get_hr_business_query_max_limit()
        planner_timeout_seconds = settings.get_hr_business_query_planner_timeout_seconds()
        dsl_max_limit = settings.get_hr_business_query_dsl_max_limit()
        dsl_max_aggregation_buckets = settings.get_hr_business_query_dsl_max_aggregation_buckets()
    except Exception:
        max_limit = 100
        planner_timeout_seconds = 30
        dsl_max_limit = 100
        dsl_max_aggregation_buckets = 100
    return {
        "max_limit": max_limit,
        "planner_timeout_seconds": planner_timeout_seconds,
        "dsl_max_limit": dsl_max_limit,
        "dsl_max_aggregation_buckets": dsl_max_aggregation_buckets,
    }


@register_tool(
    name=HR_BUSINESS_QUERY_TOOL_NAME,
    description=HR_BUSINESS_QUERY_DESCRIPTION,
    tags=["hr", "employee", "resume", "analytics", "queryspec", "问数", "简历", "员工", "组织关系"],
    provider="local",
    enabled=True,
)
def build_hr_business_query_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()

    try:
        from crewai.tools import BaseTool  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError("crewai is required to register native agent tools") from e

    try:
        from pydantic import BaseModel, Field  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError("pydantic is required to build hr_business_query tool") from e

    class HRBusinessQueryArgs(BaseModel):
        query: str = Field(
            default="",
            description=(
                "User's original HR business question, e.g. How many employees are in the Beijing office? "
                "Must be a natural-language question; do not generate QuerySpec, field names, or ES DSL yourself. "
                "If the user says 'he/she/this person', rewrite using conversation context with an explicit "
                "employee name or ID, e.g. 'Send me Helen Wang 161's resume'. Note that suffixes like "
                "Helen Wang 161 or Chen Ming 19 are part of the name, not employee IDs, unless the user "
                "explicitly says employee ID/employee number. "
                "If the user asks for a list ('give me the list/names'), preserve filters from the previous "
                "turn such as name, city, or department. "
                "Do not use this tool for non-HR questions such as travel planning, sightseeing, weather, "
                "news search, or casual chat."
            ),
        )
        include_query_spec: bool = Field(default=True, description="Whether to show QuerySpec and validation info.")
        include_sample_rows: bool = Field(default=True, description="Whether to show a few sample detail rows.")
        include_debug: bool = Field(default=False, description="Whether to show planning/execution debug info.")

    class HRBusinessQueryTool(BaseTool):
        name: str = HR_BUSINESS_QUERY_TOOL_NAME
        description: str = HR_BUSINESS_QUERY_DESCRIPTION
        args_schema: type = HRBusinessQueryArgs

        def _run(
            self,
            query: str = "",
            include_query_spec: bool = True,
            include_sample_rows: bool = True,
            include_debug: bool = False,
            **_ignored: Any,
        ) -> str:
            payload = _coerce_tool_args(
                query=query,
                include_query_spec=include_query_spec,
                include_sample_rows=include_sample_rows,
                include_debug=include_debug,
            )
            return run_hr_business_query(
                str(payload.get("query") or ""),
                include_query_spec=bool(payload.get("include_query_spec", True)),
                include_sample_rows=bool(payload.get("include_sample_rows", True)),
                include_debug=bool(payload.get("include_debug", False)),
                user_id=bound_user_id or "agent-system",
            )

    HRBusinessQueryTool.__doc__ = HR_BUSINESS_QUERY_DOCSTRING
    return HRBusinessQueryTool()
