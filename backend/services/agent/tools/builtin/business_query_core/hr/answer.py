"""Markdown answer composition for HR QuerySpec execution results."""

from __future__ import annotations

import json

from typing import Any, Dict, Iterable, List

from backend.services.agent.tools.builtin.business_query_core.hr import executor as sample_backend


def compose_hr_answer(
    *,
    query_spec: Dict[str, Any],
    execution_result: Dict[str, Any],
    planner_debug: Dict[str, Any],
    permission_check: Dict[str, Any],
    cost_check: Dict[str, Any],
    include_query_spec: bool = True,
    include_sample_rows: bool = True,
    include_debug: bool = False,
) -> str:
    rows = list(execution_result.get("rows") or [])
    output: List[str] = ["### HR 查询结果", ""]
    output.extend(str(line) for line in execution_result.get("lines") or [])
    output.append("")
    output.append("### 说明")
    output.append("- 当前生产入口使用 Elasticsearch HR 索引执行受控查询计划。")
    output.append("- 统一规划器生成 QueryIR、ES DSL 或内部 QuerySpec；执行器只消费通过校验的受控结构。")
    output.append("- 权限校验与查询成本控制仍为第一阶段占位。")

    if include_sample_rows and _should_show_sample_rows(query_spec, execution_result, rows):
        output.append("")
        output.append("### 匹配人员")
        output.extend(_format_people(rows[:5]))
        if len(rows) > 5:
            output.append(f"- 另有 {len(rows) - 5} 条未展示。")

    if include_query_spec or include_debug:
        debug_payload: Dict[str, Any] = {
            "query_spec": query_spec,
            "permission_check": permission_check,
            "cost_check": cost_check,
            "backend": execution_result.get("backend"),
        }
        if include_debug:
            debug_payload["planner"] = planner_debug
            if execution_result.get("debug"):
                debug_payload["execution_context"] = execution_result.get("debug")
        output.append("")
        output.append("### 查询计划与校验")
        output.append(json_block(debug_payload))

    return "\n".join(output).strip()


def compose_hr_error(message: str, *, include_debug: bool = False, debug: Dict[str, Any] | None = None) -> str:
    output = ["### HR 查询无法执行", "", message]
    if include_debug and debug:
        output.extend(["", "### Debug", json_block(debug)])
    return "\n".join(output).strip()


def _format_people(rows: Iterable[Any]) -> List[str]:
    return sample_backend._format_any_people(rows)  # noqa: SLF001 - domain backend


def _should_show_sample_rows(query_spec: Dict[str, Any], execution_result: Dict[str, Any], rows: List[Any]) -> bool:
    if not rows or query_spec.get("intent") in {"document_fetch", "attribute_lookup"}:
        return False
    if query_spec.get("intent") != "aggregation":
        return True

    debug = execution_result.get("debug") if isinstance(execution_result.get("debug"), dict) else {}
    total = debug.get("es_total")
    try:
        total_count = int(total)
    except (TypeError, ValueError):
        total_count = len(rows)
    return 0 < total_count <= 5 and len(rows) >= total_count



def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"
