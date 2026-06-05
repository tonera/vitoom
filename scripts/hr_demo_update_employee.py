#!/usr/bin/env python3
"""定点更新 HR demo 员工事实字段。

示例：
  python scripts/hr_demo_update_employee.py 100001 --set manager_id=300001 --set office_city=深圳 --refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.agent import settings as agent_settings  # noqa: E402
from backend.services.agent.tools.builtin.business_query_core.hr import executor as hr_es  # noqa: E402

DEFAULT_EMPLOYEE_INDEX = agent_settings.get_hr_business_query_employee_index()
DEFAULT_ES_URL = agent_settings.get_hr_business_query_es_url() or hr_es.DEFAULT_ES_URL
es_request = hr_es.es_request

ALLOWED_UPDATE_FIELDS = {
    "email",
    "gender",
    "marital_status",
    "office_city",
    "office_location",
    "country_region",
    "timezone",
    "department_code",
    "department_name",
    "job_title",
    "job_family",
    "grade",
    "grade_level",
    "pay_scale_group",
    "fte",
    "standard_hours",
    "employment_type",
    "employment_status",
    "hire_date",
    "termination_date",
    "manager_id",
    "performance_rating",
    "attrition_risk",
    "high_potential",
    "key_position",
    "future_leader",
    "promotion_likelihood",
    "skills",
    "project_keywords",
    "resume_summary",
    "updated_at",
}


def coerce_value(raw: str) -> Any:
    text = str(raw).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"null", "none"}:
        return None
    if text.startswith("[") or text.startswith("{"):
        return json.loads(text)
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_updates(items: list[str]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        field = key.strip()
        if field not in ALLOWED_UPDATE_FIELDS:
            raise ValueError(f"Field {field!r} is not allowed for demo updates")
        updates[field] = coerce_value(raw_value)
    if not updates:
        raise ValueError("At least one --set key=value is required")
    return updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update one HR demo employee document")
    parser.add_argument("employee_id", help="员工 ID")
    parser.add_argument("--set", action="append", default=[], help="字段更新，格式 key=value，可重复")
    parser.add_argument("--url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--username", default="", help="ES Basic Auth 用户名")
    parser.add_argument("--password", default="", help="ES Basic Auth 密码")
    parser.add_argument("--index", default=DEFAULT_EMPLOYEE_INDEX, help="员工索引名")
    parser.add_argument("--refresh", action="store_true", help="更新后立即 refresh")
    parser.add_argument("--timeout", type=float, default=30.0, help="请求超时秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    updates = parse_updates(args.set)
    body = {"doc": updates}
    if args.refresh:
        body["refresh"] = True
    response = es_request(
        "POST",
        args.url,
        f"/{args.index}/_update/{args.employee_id}",
        body=body,
        username=args.username,
        password=args.password,
        timeout=args.timeout,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
