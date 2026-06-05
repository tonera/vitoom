#!/usr/bin/env python3
"""HR demo ES 固定查询 smoke test。"""

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
DEFAULT_RESUME_ASSET_INDEX = agent_settings.get_hr_business_query_resume_asset_index()
DEFAULT_RESUME_CHUNK_INDEX = agent_settings.get_hr_business_query_resume_chunk_index()
es_request = hr_es.es_request


def search(args: argparse.Namespace, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    return es_request(
        "POST",
        args.url,
        f"/{index}/_search",
        body=body,
        username=args.username,
        password=args.password,
        timeout=args.timeout,
    )


def total_hits(response: Dict[str, Any]) -> int:
    total = response.get("hits", {}).get("total", {})
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    return int(total or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test HR demo Elasticsearch indices")
    parser.add_argument("--url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--username", default="", help="ES Basic Auth 用户名")
    parser.add_argument("--password", default="", help="ES Basic Auth 密码")
    parser.add_argument("--timeout", type=float, default=30.0, help="请求超时秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    employee_count = es_request(
        "GET",
        args.url,
        f"/{DEFAULT_EMPLOYEE_INDEX}/_count",
        username=args.username,
        password=args.password,
        timeout=args.timeout,
    )
    beijing = search(
        args,
        DEFAULT_EMPLOYEE_INDEX,
        {"size": 0, "query": {"term": {"office_city": "北京"}}},
    )
    department_distribution = search(
        args,
        DEFAULT_EMPLOYEE_INDEX,
        {"size": 0, "aggs": {"by_department": {"terms": {"field": "department_code", "size": 20}}}},
    )
    reports = search(
        args,
        DEFAULT_EMPLOYEE_INDEX,
        {"size": 10, "query": {"bool": {"filter": [{"term": {"manager_id": "107009"}}, {"term": {"employment_status": "active"}}]}}},
    )
    chunks = search(args, DEFAULT_RESUME_CHUNK_INDEX, {"size": 1, "query": {"match": {"text": "Python"}}})
    assets = search(args, DEFAULT_RESUME_ASSET_INDEX, {"size": 1, "query": {"term": {"employee_id": "100001"}}})

    report = {
        "employee_count": employee_count.get("count"),
        "beijing_count": total_hits(beijing),
        "department_buckets": department_distribution.get("aggregations", {}).get("by_department", {}).get("buckets", []),
        "manager_107009_direct_reports": total_hits(reports),
        "python_resume_chunk_hits": total_hits(chunks),
        "employee_100001_asset_hits": total_hits(assets),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert report["employee_count"] and report["employee_count"] > 0
    assert report["beijing_count"] > 0
    assert report["department_buckets"]
    assert report["manager_107009_direct_reports"] >= 2
    assert report["employee_100001_asset_hits"] >= 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
