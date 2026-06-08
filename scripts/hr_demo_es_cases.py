#!/usr/bin/env python
"""Run HR business query smoke cases against the configured ES backend.

Prerequisites:
  1. HR employee/resume indices exist in Elasticsearch.
  2. config/default.yaml or user config points agents.business_queries.hr to that ES.

Usage:
  python scripts/hr_demo_es_cases.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.agent.tools.builtin.hr_business_query import run_hr_business_query  # noqa: E402

DEFAULT_USER_ID = "test_user_252ddf69126e47f08450d24e57eda853"


@dataclass(frozen=True)
class Case:
    case_id: str
    query: str
    expected: tuple[str, ...]
    query_spec: Mapping[str, Any] | None = None
    include_query_spec: bool = False
    include_debug: bool = False


def aggregation_case(
    *,
    filters: list[dict[str, Any]] | None = None,
    group_by: list[str] | None = None,
    metrics: list[dict[str, Any]] | None = None,
    case: str = "",
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "domain": "hr",
        "entity": "employee",
        "intent": "aggregation",
        "filters": filters or [],
        "metrics": metrics or [{"type": "count", "field": "employee_id"}],
        "limit": 20,
    }
    if group_by:
        spec["group_by"] = group_by
    if case:
        spec["case"] = case
    return spec


def list_case(
    *,
    filters: list[dict[str, Any]] | None = None,
    case: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "domain": "hr",
        "entity": "employee",
        "intent": "list",
        "filters": filters or [],
        "limit": limit,
    }
    if case:
        spec["case"] = case
    return spec


def relationship_case(*, case: str = "") -> dict[str, Any]:
    spec: dict[str, Any] = {
        "domain": "hr",
        "entity": "employee",
        "intent": "relationship",
        "filters": [],
        "limit": 20,
    }
    if case:
        spec["case"] = case
    return spec


CASES: tuple[Case, ...] = (
    # 场景一：人员统计与分布分析
    Case("HR-001", "北京办公室有多少员工？", ("匹配员工共",)),
    Case("HR-002", "各部门人数分布情况", ("by_department_code",)),
    Case("HR-003", "男女比例是多少？", ("by_gender",)),
    Case("HR-004", "已婚员工占比多少？", ("已婚",), query_spec=aggregation_case(group_by=["marital_status"], case="married_ratio")),
    Case(
        "HR-005",
        "各职级 (GR-03 到 GR-16) 的人数分布",
        ("按 `grade` 分布",),
        query_spec=aggregation_case(
            filters=[
                {"field": "grade_level", "op": ">=", "value": 3},
                {"field": "grade_level", "op": "<=", "value": 16},
            ],
            group_by=["grade"],
        ),
    ),
    # 场景二：组织与汇报关系分析
    Case("HR-006", "James Kwok 有多少直接下属？", ("直接下属",)),
    Case("HR-007", "找出所有向 107009 汇报的员工", ("直接下属",), query_spec=relationship_case()),
    Case("HR-008", "哪个经理的直接下属最多？", ("直接下属最多",), query_spec=relationship_case()),
    Case("HR-009", "北京和上海两地的人员汇报关系对比", ("北京和上海两地人员汇报关系对比",), query_spec=relationship_case(case="reporting_compare_beijing_shanghai")),
    Case("HR-010", "没有指定经理的员工有哪些？", ("没有指定经理",), query_spec=relationship_case()),
    # 场景三：雇佣与入职分析
    Case("HR-011", "2023 年入职的员工有多少？", ("匹配员工共",)),
    Case(
        "HR-012",
        "平均司龄是多少年？",
        ("平均司龄",),
        query_spec=aggregation_case(metrics=[{"type": "avg", "field": "tenure_years"}]),
    ),
    Case("HR-013", "全职 (FTE=1) 和兼职员工分布", ("按 `fte` 分布",), query_spec=aggregation_case(group_by=["fte"], case="fte_distribution")),
    Case("HR-014", "计时工和固定职工的比例", ("按 `employment_type` 分布",), query_spec=aggregation_case(group_by=["employment_type"], case="employment_type_distribution")),
    Case("HR-015", "已退休或已解雇的人员列表", ("已退休或已解雇/离职人员",), query_spec=list_case(case="inactive_employee_list")),
    # 场景四：薪酬与职级分析
    Case("HR-016", "GR-08 及以上职级的人员名单", ("找到",), query_spec=list_case(filters=[{"field": "grade_level", "op": ">=", "value": 8}], limit=100)),
    Case("HR-017", "各薪酬等级 (Pay Scale Group) 的平均 FTE", ("各薪酬等级的平均 FTE",), query_spec=aggregation_case(case="avg_fte_by_pay_scale_group")),
    Case("HR-018", "时薪制和月薪制员工分别有多少？", ("按 `employment_type` 分布",), query_spec=aggregation_case(group_by=["employment_type"], case="employment_type_distribution")),
    Case("HR-019", "VP 级别以上的高管有哪些？", ("找到",), query_spec=list_case(filters=[{"field": "grade_level", "op": ">=", "value": 10}], limit=100)),
    Case("HR-020", "标准工时为 40 小时的员工占比", ("标准工时为 40 小时",), query_spec=aggregation_case(case="standard_hours_40_ratio")),
    # 场景五：人才与绩效管理
    Case("HR-021", "高潜力员工有多少？", ("匹配员工共",), query_spec=aggregation_case(filters=[{"field": "high_potential", "op": "=", "value": True}])),
    Case("HR-022", "绩效为低且流失风险高的员工", ("匹配人员",), query_spec=list_case(case="low_perf_high_risk")),
    Case("HR-023", "关键岗位 (keyPosition=1) 的人员", ("匹配人员",), query_spec=list_case(filters=[{"field": "key_position", "op": "=", "value": True}], limit=100)),
    Case("HR-024", "标记为未来领袖 (futureLeader) 的员工", ("匹配人员",), query_spec=list_case(filters=[{"field": "future_leader", "op": "=", "value": True}], limit=100)),
    Case("HR-025", "很可能升职的员工列表", ("匹配人员",), query_spec=list_case(filters=[{"field": "promotion_likelihood", "op": "=", "value": "很可能"}], limit=100)),
    # 场景六：地域与时区分析
    Case("HR-026", "使用 Asia/Shanghai 时区的员工", ("匹配人员",), query_spec=list_case(filters=[{"field": "timezone", "op": "=", "value": "Asia/Shanghai"}], limit=100)),
    Case("HR-027", "深圳办公室的人员分布", ("按 `department_code` 分布",), query_spec=aggregation_case(filters=[{"field": "office_city", "op": "=", "value": "深圳"}], group_by=["department_code"], case="office_department_distribution")),
    Case("HR-028", "跨国汇报的情况（经理和员工时区不同，跨时区）", ("跨时区汇报",), query_spec=relationship_case(case="cross_timezone_reporting")),
    Case("HR-029", "各时区的人员数量分布", ("按 `timezone` 分布",), query_spec=aggregation_case(group_by=["timezone"])),
    Case("HR-030", "美国时区的员工有哪些？", ("匹配人员",), query_spec=list_case(filters=[{"field": "country_region", "op": "=", "value": "美国"}], limit=100)),
    # 场景七：复杂组合查询
    Case(
        "HR-031",
        "北京制造部门 (MANU) 在职工程师",
        ("匹配人员",),
        query_spec=list_case(
            filters=[
                {"field": "office_city", "op": "=", "value": "北京"},
                {"field": "department_code", "op": "=", "value": "MANU"},
                {"field": "employment_status", "op": "=", "value": "active"},
            ],
            limit=100,
        ),
    ),
    Case(
        "HR-032",
        "入职超过 5 年且绩效高的员工",
        ("入职超过 5 年且满足条件",),
        query_spec=list_case(
            filters=[
                {"field": "tenure_years", "op": ">", "value": 5},
                {"field": "performance_rating", "op": "=", "value": "高"},
            ],
            limit=100,
        ),
    ),
    Case("HR-033", "经理级 (GR-10 及以上) 且直接下属 > 3 人的名单", ("经理级且直接下属 > 3 人",), query_spec=list_case(case="manager_grade_reports_gt3")),
    Case(
        "HR-034",
        "2023 年后入职的女性高管",
        ("匹配人员",),
        query_spec=list_case(
            filters=[
                {"field": "hire_date", "op": ">=", "value": "2023-01-01"},
                {"field": "gender", "op": "=", "value": "女"},
                {"field": "grade_level", "op": ">=", "value": 10},
            ],
            limit=100,
        ),
    ),
    Case("HR-035", "有离职风险且影响度高的关键人才", ("匹配人员",), query_spec=list_case(case="high_risk_key_talent")),
)


def _run_case(case: Case) -> tuple[bool, str]:
    output = run_hr_business_query(
        case.query,
        query_spec=case.query_spec,
        include_query_spec=case.include_query_spec,
        include_debug=case.include_debug,
        user_id=os.getenv("HR_DEMO_USER_ID", DEFAULT_USER_ID),
    )
    missing = [item for item in case.expected if item not in output]
    if missing:
        return False, f"missing expected text {missing!r}\n{output}"
    if "HR 查询执行失败" in output or "QuerySpec 校验失败" in output:
        return False, output
    return True, output


def run_cases(cases: Iterable[Case] = CASES) -> int:
    passed = 0
    total = 0
    for case in cases:
        total += 1
        ok, output = _run_case(case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.case_id} {case.query}")
        if not ok:
            print(output)
        else:
            passed += 1
    print(f"HR business query ES cases passed: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(run_cases())
