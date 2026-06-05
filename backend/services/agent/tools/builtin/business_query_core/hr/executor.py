"""Execution primitives for business query tools.

This module contains the first HR-backed executor implementation and the generic
permission/cost placeholders used by domain tools. Finance can add a sibling
executor implementation while reusing the same result contract.
"""

from __future__ import annotations

import base64
import json
import re
import ssl
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from backend.services.agent.tools.builtin.business_query_core.executor_base import (
    placeholder_cost_check as base_placeholder_cost_check,
    placeholder_permission_check as base_placeholder_permission_check,
)
from backend.services.agent.tools.builtin.business_query_core.hr.es_dsl_guard import validate_hr_es_search_body

sample_backend = sys.modules[__name__]

"""HR Elasticsearch support helpers for the business query domain."""



DEFAULT_ES_URL = "http://127.0.0.1:9200"
DEFAULT_EMPLOYEE_INDEX = "hr_employee_v1"
DEFAULT_RESUME_CHUNK_INDEX = "hr_resume_chunk_v1"
DEFAULT_RESUME_ASSET_INDEX = "hr_resume_asset_v1"
DEFAULT_RAW_QUALITY_FILE = "raw_employees_with_quality_cases.jsonl"
DEFAULT_EMPLOYEE_FILE = "employees.jsonl"
DEFAULT_RESUME_CHUNK_FILE = "resume_chunks.jsonl"
DEFAULT_RESUME_ASSET_FILE = "resume_assets.jsonl"
DEFAULT_VECTOR_DIMS = 32
EMPLOYEE_SUMMARY_FIELDS = [
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


def employee_mapping() -> Dict[str, Any]:
    keyword_with_text = {
        "type": "text",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
    }
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "employee_id": {"type": "keyword"},
                "name": keyword_with_text,
                "name_pinyin": {"type": "keyword"},
                "email": {"type": "keyword"},
                "gender": {"type": "keyword"},
                "marital_status": {"type": "keyword"},
                "office_city": {"type": "keyword"},
                "office_location": {"type": "keyword"},
                "country_region": {"type": "keyword"},
                "timezone": {"type": "keyword"},
                "department_code": {"type": "keyword"},
                "department_name": keyword_with_text,
                "job_title": keyword_with_text,
                "job_family": {"type": "keyword"},
                "grade": {"type": "keyword"},
                "grade_level": {"type": "integer"},
                "pay_scale_group": {"type": "keyword"},
                "fte": {"type": "float"},
                "standard_hours": {"type": "integer"},
                "employment_type": {"type": "keyword"},
                "employment_status": {"type": "keyword"},
                "hire_date": {"type": "date"},
                "termination_date": {"type": "date"},
                "manager_id": {"type": "keyword"},
                "performance_rating": {"type": "keyword"},
                "attrition_risk": {"type": "keyword"},
                "high_potential": {"type": "boolean"},
                "key_position": {"type": "boolean"},
                "future_leader": {"type": "boolean"},
                "promotion_likelihood": {"type": "keyword"},
                "skills": {"type": "keyword"},
                "project_keywords": {"type": "keyword"},
                "resume_summary": keyword_with_text,
                "updated_at": {"type": "date"},
                "source_system": {"type": "keyword"},
                "assignments": {
                    "type": "nested",
                    "properties": {
                        "assignment_id": {"type": "keyword"},
                        "department_code": {"type": "keyword"},
                        "department_name": keyword_with_text,
                        "office_city": {"type": "keyword"},
                        "office_location": {"type": "keyword"},
                        "job_title": keyword_with_text,
                        "job_family": {"type": "keyword"},
                        "grade": {"type": "keyword"},
                        "grade_level": {"type": "integer"},
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        "is_primary": {"type": "boolean"},
                        "assignment_status": {"type": "keyword"},
                    },
                },
            },
        },
    }


def resume_chunk_mapping(*, vector_dims: int = DEFAULT_VECTOR_DIMS) -> Dict[str, Any]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "employee_id": {"type": "keyword"},
                "resume_id": {"type": "keyword"},
                "chunk_type": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "text": {"type": "text"},
                "skills": {"type": "keyword"},
                "years_of_experience": {"type": "float"},
                "language": {"type": "keyword"},
                "source_version": {"type": "keyword"},
                "source_updated_at": {"type": "date"},
                "embedding": {"type": "dense_vector", "dims": vector_dims},
            },
        },
    }


def resume_asset_mapping() -> Dict[str, Any]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "asset_id": {"type": "keyword"},
                "employee_id": {"type": "keyword"},
                "resume_id": {"type": "keyword"},
                "resume_version": {"type": "keyword"},
                "file_name": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "file_type": {"type": "keyword"},
                "storage_uri": {"type": "keyword", "ignore_above": 2048},
                "asset_type": {"type": "keyword"},
                "language": {"type": "keyword"},
                "updated_at": {"type": "date"},
                "access_level": {"type": "keyword"},
                "source_system": {"type": "keyword"},
            },
        },
    }


def index_mappings(*, vector_dims: int = DEFAULT_VECTOR_DIMS) -> Dict[str, Dict[str, Any]]:
    return {
        DEFAULT_EMPLOYEE_INDEX: employee_mapping(),
        DEFAULT_RESUME_CHUNK_INDEX: resume_chunk_mapping(vector_dims=vector_dims),
        DEFAULT_RESUME_ASSET_INDEX: resume_asset_mapping(),
    }


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            yield parsed


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


class EsHttpError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Elasticsearch HTTP {status}: {message}")
        self.status = status
        self.message = message


def es_request(
    method: str,
    url: str,
    path: str,
    *,
    body: Optional[Any] = None,
    username: str = "",
    password: str = "",
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Any:
    base_url = str(url or DEFAULT_ES_URL).rstrip("/") + "/"
    full_url = urljoin(base_url, str(path or "").lstrip("/"))
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    data: Optional[bytes] = None
    if body is not None:
        if isinstance(body, bytes):
            data = body
        elif isinstance(body, str):
            data = body.encode("utf-8")
        else:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
    if username:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        request_headers["Authorization"] = f"Basic {token}"

    request = Request(full_url, data=data, headers=request_headers, method=method.upper())
    context = ssl.create_default_context()
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            raw = response.read()
    except HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        raise EsHttpError(exc.code, raw_error) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to connect Elasticsearch at {base_url}: {exc}") from exc

    if not raw:
        return {}
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_bulk_body(index_name: str, rows: Iterable[Dict[str, Any]], *, id_field: str) -> Tuple[bytes, int]:
    lines: List[str] = []
    count = 0
    for row in rows:
        doc_id = str(row.get(id_field) or "").strip()
        if not doc_id:
            raise ValueError(f"Missing id field {id_field!r} for index {index_name}")
        lines.append(json.dumps({"index": {"_index": index_name, "_id": doc_id}}, ensure_ascii=False))
        lines.append(json.dumps(row, ensure_ascii=False))
        count += 1
    body = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    return body, count


def bulk_import(
    *,
    url: str,
    index_name: str,
    rows: Iterable[Dict[str, Any]],
    id_field: str,
    username: str = "",
    password: str = "",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    body, count = build_bulk_body(index_name, rows, id_field=id_field)
    if count == 0:
        return {"errors": False, "items": [], "imported": 0}
    response = es_request(
        "POST",
        url,
        "/_bulk",
        body=body,
        username=username,
        password=password,
        headers={"Content-Type": "application/x-ndjson"},
        timeout=timeout,
    )
    if isinstance(response, dict):
        response["imported"] = count
    return response

"""HR sample backend and Elasticsearch QuerySpec execution helpers."""




_TODAY = date(2026, 4, 26)


@dataclass(frozen=True)
class EmployeeRecord:
    employee_id: str
    name: str
    gender: str
    marital_status: str
    office_city: str
    department_code: str
    department_name: str
    job_title: str
    grade: str
    pay_scale_group: str
    fte: float
    standard_hours: int
    employment_type: str
    employment_status: str
    hire_date: str
    timezone: str
    manager_id: str
    email: str
    performance_rating: str
    attrition_risk: str
    high_potential: bool
    key_position: bool
    future_leader: bool
    promotion_likelihood: str
    resume_summary: str


_EMPLOYEES: Tuple[EmployeeRecord, ...] = (
    EmployeeRecord(
        employee_id="100001",
        name="张三",
        gender="男",
        marital_status="已婚",
        office_city="北京",
        department_code="MANU",
        department_name="制造部",
        job_title="高级工程师",
        grade="GR-08",
        pay_scale_group="PSG-3",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="active",
        hire_date="2018-03-12",
        timezone="Asia/Shanghai",
        manager_id="107009",
        email="zhangsan@example.com",
        performance_rating="高",
        attrition_risk="低",
        high_potential=True,
        key_position=True,
        future_leader=False,
        promotion_likelihood="很可能",
        resume_summary="8 年制造系统后端经验，熟悉 MES、Python、数据集成和跨部门项目推进。",
    ),
    EmployeeRecord(
        employee_id="100002",
        name="李四",
        gender="女",
        marital_status="未婚",
        office_city="北京",
        department_code="MANU",
        department_name="制造部",
        job_title="工程师",
        grade="GR-07",
        pay_scale_group="PSG-2",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="active",
        hire_date="2023-07-01",
        timezone="Asia/Shanghai",
        manager_id="107009",
        email="lisi@example.com",
        performance_rating="中",
        attrition_risk="中",
        high_potential=False,
        key_position=False,
        future_leader=False,
        promotion_likelihood="可能",
        resume_summary="3 年自动化测试与生产数据分析经验，熟悉 SQL、报表和质量追踪。",
    ),
    EmployeeRecord(
        employee_id="100003",
        name="王五",
        gender="男",
        marital_status="已婚",
        office_city="上海",
        department_code="SALES",
        department_name="销售部",
        job_title="销售经理",
        grade="GR-10",
        pay_scale_group="PSG-4",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="active",
        hire_date="2016-05-20",
        timezone="Asia/Shanghai",
        manager_id="200001",
        email="wangwu@example.com",
        performance_rating="高",
        attrition_risk="低",
        high_potential=True,
        key_position=True,
        future_leader=True,
        promotion_likelihood="很可能",
        resume_summary="销售团队管理经验丰富，擅长大客户拓展、渠道建设和目标拆解。",
    ),
    EmployeeRecord(
        employee_id="100004",
        name="赵六",
        gender="女",
        marital_status="已婚",
        office_city="深圳",
        department_code="RND",
        department_name="研发部",
        job_title="AI 工程师",
        grade="GR-09",
        pay_scale_group="PSG-3",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="active",
        hire_date="2020-11-18",
        timezone="Asia/Shanghai",
        manager_id="300001",
        email="zhaoliu@example.com",
        performance_rating="高",
        attrition_risk="高",
        high_potential=True,
        key_position=True,
        future_leader=True,
        promotion_likelihood="很可能",
        resume_summary="5 年机器学习和推荐系统经验，主导过模型服务化、特征平台和人才画像项目。",
    ),
    EmployeeRecord(
        employee_id="100005",
        name="Alice Chen",
        gender="女",
        marital_status="未婚",
        office_city="纽约",
        department_code="FIN",
        department_name="财务部",
        job_title="Finance Analyst",
        grade="GR-08",
        pay_scale_group="PSG-3",
        fte=0.5,
        standard_hours=40,
        employment_type="计时工",
        employment_status="active",
        hire_date="2022-02-14",
        timezone="America/New_York",
        manager_id="200001",
        email="alice.chen@example.com",
        performance_rating="中",
        attrition_risk="低",
        high_potential=False,
        key_position=False,
        future_leader=False,
        promotion_likelihood="一般",
        resume_summary="财务分析、预算建模和跨国费用核算经验，熟悉 Excel、SQL 和 BI 报表。",
    ),
    EmployeeRecord(
        employee_id="100006",
        name="Bob Smith",
        gender="男",
        marital_status="未婚",
        office_city="旧金山",
        department_code="RND",
        department_name="研发部",
        job_title="Staff Engineer",
        grade="GR-11",
        pay_scale_group="PSG-5",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="active",
        hire_date="2019-09-09",
        timezone="America/Los_Angeles",
        manager_id="300001",
        email="bob.smith@example.com",
        performance_rating="高",
        attrition_risk="中",
        high_potential=True,
        key_position=True,
        future_leader=False,
        promotion_likelihood="可能",
        resume_summary="分布式系统和平台工程专家，具备跨时区远程协作、架构设计和技术带教经验。",
    ),
    EmployeeRecord(
        employee_id="107009",
        name="James Kwok",
        gender="男",
        marital_status="已婚",
        office_city="北京",
        department_code="MANU",
        department_name="制造部",
        job_title="制造总监",
        grade="GR-12",
        pay_scale_group="PSG-6",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="active",
        hire_date="2014-01-06",
        timezone="Asia/Shanghai",
        manager_id="200001",
        email="james.kwok@example.com",
        performance_rating="高",
        attrition_risk="低",
        high_potential=False,
        key_position=True,
        future_leader=False,
        promotion_likelihood="可能",
        resume_summary="制造运营和组织管理负责人，长期负责多工厂交付、成本优化和团队梯队建设。",
    ),
    EmployeeRecord(
        employee_id="200001",
        name="Grace Lee",
        gender="女",
        marital_status="已婚",
        office_city="上海",
        department_code="EXEC",
        department_name="管理层",
        job_title="VP Operations",
        grade="GR-15",
        pay_scale_group="PSG-8",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="active",
        hire_date="2010-08-01",
        timezone="Asia/Shanghai",
        manager_id="",
        email="grace.lee@example.com",
        performance_rating="高",
        attrition_risk="低",
        high_potential=False,
        key_position=True,
        future_leader=False,
        promotion_likelihood="一般",
        resume_summary="运营 VP，负责亚太制造、销售运营和跨区域组织协同。",
    ),
    EmployeeRecord(
        employee_id="300001",
        name="陈七",
        gender="男",
        marital_status="已婚",
        office_city="深圳",
        department_code="RND",
        department_name="研发部",
        job_title="研发总监",
        grade="GR-13",
        pay_scale_group="PSG-7",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="active",
        hire_date="2012-04-15",
        timezone="Asia/Shanghai",
        manager_id="200001",
        email="chenqi@example.com",
        performance_rating="高",
        attrition_risk="低",
        high_potential=False,
        key_position=True,
        future_leader=False,
        promotion_likelihood="一般",
        resume_summary="研发组织负责人，覆盖平台工程、AI 应用和全球研发协作。",
    ),
    EmployeeRecord(
        employee_id="100007",
        name="周八",
        gender="女",
        marital_status="已婚",
        office_city="北京",
        department_code="HR",
        department_name="人力资源部",
        job_title="HRBP",
        grade="GR-08",
        pay_scale_group="PSG-3",
        fte=1.0,
        standard_hours=40,
        employment_type="固定职工",
        employment_status="terminated",
        hire_date="2021-03-01",
        timezone="Asia/Shanghai",
        manager_id="999999",
        email="invalid-email",
        performance_rating="低",
        attrition_risk="高",
        high_potential=False,
        key_position=False,
        future_leader=False,
        promotion_likelihood="一般",
        resume_summary="曾负责招聘运营、员工关系和组织数据维护。",
    ),
)


_GRADE_ORDER = {f"GR-{level:02d}": level for level in range(3, 17)}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _record_to_dict(record: EmployeeRecord) -> Dict[str, Any]:
    return {
        "employee_id": record.employee_id,
        "name": record.name,
        "gender": record.gender,
        "marital_status": record.marital_status,
        "office_city": record.office_city,
        "department_code": record.department_code,
        "department_name": record.department_name,
        "job_title": record.job_title,
        "grade": record.grade,
        "pay_scale_group": record.pay_scale_group,
        "fte": record.fte,
        "standard_hours": record.standard_hours,
        "employment_type": record.employment_type,
        "employment_status": record.employment_status,
        "hire_date": record.hire_date,
        "timezone": record.timezone,
        "manager_id": record.manager_id,
        "email": record.email,
        "performance_rating": record.performance_rating,
        "attrition_risk": record.attrition_risk,
        "high_potential": record.high_potential,
        "key_position": record.key_position,
        "future_leader": record.future_leader,
        "promotion_likelihood": record.promotion_likelihood,
        "resume_summary": record.resume_summary,
    }


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


def _grade_value(grade: str) -> int:
    return _GRADE_ORDER.get(str(grade or "").strip().upper(), 0)


def _is_active(record: EmployeeRecord) -> bool:
    return record.employment_status == "active"


def _tenure_years(record: EmployeeRecord) -> float:
    try:
        year, month, day = [int(part) for part in record.hire_date.split("-")]
        hired = date(year, month, day)
    except Exception:
        return 0.0
    return round((_TODAY - hired).days / 365.25, 1)


def _format_person(record: EmployeeRecord) -> str:
    status = "在职" if _is_active(record) else "非在职"
    return (
        f"{record.name}（{record.employee_id}，{record.department_code}，{record.grade}，"
        f"{record.job_title}，{record.office_city}，{status}）"
    )


def _format_people(records: Iterable[EmployeeRecord], *, empty_text: str = "未找到匹配人员。") -> List[str]:
    rows = [f"- {_format_person(record)}" for record in records]
    return rows or [empty_text]


def _doc_get(record: Any, field: str, default: Any = "") -> Any:
    if isinstance(record, dict):
        return record.get(field, default)
    return getattr(record, field, default)


def _format_any_person(record: Any) -> str:
    employment_status = _doc_get(record, "employment_status", None)
    if employment_status == "active":
        status = "在职"
    elif employment_status in {"terminated", "retired"}:
        status = "非在职"
    else:
        status = "状态未返回"
    return (
        f"{_doc_get(record, 'name')}（{_doc_get(record, 'employee_id')}，"
        f"{_doc_get(record, 'department_code')}，{_doc_get(record, 'grade')}，"
        f"{_doc_get(record, 'job_title')}，{_doc_get(record, 'office_city')}，{status}）"
    )


def _format_any_people(records: Iterable[Any], *, empty_text: str = "未找到匹配人员。") -> List[str]:
    rows = [f"- {_format_any_person(record)}" for record in records]
    return rows or [empty_text]


def _contains_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def _find_people_by_query(text: str) -> List[EmployeeRecord]:
    normalized = text.lower()
    matched: List[EmployeeRecord] = []
    for record in _EMPLOYEES:
        if record.name in text or record.employee_id in text or record.name.lower() in normalized:
            matched.append(record)
    return matched


def _infer_filters(text: str) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    for city in ("北京", "上海", "深圳", "纽约", "旧金山"):
        if city in text:
            filters.append({"field": "office_city", "op": "=", "value": city})
    for department_code, department_name in (
        ("MANU", "制造"),
        ("RND", "研发"),
        ("SALES", "销售"),
        ("FIN", "财务"),
        ("HR", "人力"),
        ("EXEC", "管理"),
    ):
        if department_code in text.upper() or department_name in text:
            filters.append({"field": "department_code", "op": "=", "value": department_code})
    if _contains_any(text, ("在职", "当前员工", "现有员工")):
        filters.append({"field": "employment_status", "op": "=", "value": "active"})
    if _contains_any(text, ("女性", "女员工")):
        filters.append({"field": "gender", "op": "=", "value": "女"})
    if _contains_any(text, ("男性", "男员工")):
        filters.append({"field": "gender", "op": "=", "value": "男"})
    if "2023" in text and _contains_any(text, ("入职", "加入")) and not _contains_any(text, ("2023 年后", "2023年后")):
        filters.append({"field": "hire_date", "op": "range_year", "value": 2023})
    if _contains_any(text, ("2023 年后", "2023年后")) and _contains_any(text, ("入职", "加入")):
        filters.append({"field": "hire_date", "op": ">=", "value": "2023-01-01"})
    tenure_match = re.search(r"(?:入职|司龄)(?:超过|大于)\s*(\d+(?:\.\d+)?)\s*年", text)
    if tenure_match:
        filters.append({"field": "tenure_years", "op": ">", "value": float(tenure_match.group(1))})
    if _contains_any(text, ("GR-08", "GR08")) and _contains_any(text, ("以上", "及以上")):
        filters.append({"field": "grade", "op": ">=", "value": "GR-08"})
    if _contains_any(text, ("GR-10", "GR10", "经理级", "高管", "VP")) and _contains_any(text, ("以上", "高管", "VP", "经理级")):
        filters.append({"field": "grade", "op": ">=", "value": "GR-10"})
    if _contains_any(text, ("绩效高", "绩效为高", "高绩效")):
        filters.append({"field": "performance_rating", "op": "=", "value": "高"})
    if _contains_any(text, ("绩效中", "绩效为中", "中绩效", "绩效评级中", "绩效评级为中")):
        filters.append({"field": "performance_rating", "op": "=", "value": "中"})
    if _contains_any(text, ("流失风险高", "离职风险高")):
        filters.append({"field": "attrition_risk", "op": "=", "value": "高"})
    if _contains_any(text, ("高潜", "高潜力")):
        filters.append({"field": "high_potential", "op": "=", "value": True})
    if _contains_any(text, ("关键岗位", "keyPosition")):
        filters.append({"field": "key_position", "op": "=", "value": True})
    if _contains_any(text, ("未来领袖", "futureLeader")):
        filters.append({"field": "future_leader", "op": "=", "value": True})
    if _contains_any(text, ("很可能升职", "很可能晋升")):
        filters.append({"field": "promotion_likelihood", "op": "=", "value": "很可能"})
    if _contains_any(text, ("Asia/Shanghai", "上海时区")):
        filters.append({"field": "timezone", "op": "=", "value": "Asia/Shanghai"})
    if _contains_any(text, ("美国时区", "美国")) and "跨国" not in text:
        filters.append({"field": "country_region", "op": "=", "value": "美国"})
    if _contains_any(text, ("FTE=1", "FTE = 1", "全职")):
        filters.append({"field": "fte", "op": "=", "value": 1.0})
    if _contains_any(text, ("FTE 为 0.5", "FTE=0.5", "FTE = 0.5")):
        filters.append({"field": "fte", "op": "=", "value": 0.5})
    if _contains_any(text, ("标准工时为 40", "标准工时 40", "40 小时", "40小时")):
        filters.append({"field": "standard_hours", "op": "=", "value": 40})
    if "计时工" in text:
        filters.append({"field": "employment_type", "op": "=", "value": "计时工"})
    if "固定职工" in text:
        filters.append({"field": "employment_type", "op": "=", "value": "固定职工"})
    if _contains_any(text, ("已退休", "退休")):
        filters.append({"field": "employment_status", "op": "=", "value": "retired"})
    if _contains_any(text, ("已解雇", "离职", "解雇")):
        filters.append({"field": "employment_status", "op": "=", "value": "terminated"})
    return filters


def _infer_case(text: str) -> str:
    if "已婚" in text and _contains_any(text, ("占比", "比例")):
        return "married_ratio"
    if _contains_any(text, ("FTE=1", "FTE = 1", "全职")) and _contains_any(text, ("兼职", "分布")):
        return "fte_distribution"
    if "计时工" in text and "固定职工" in text and _contains_any(text, ("比例", "分布")):
        return "employment_type_distribution"
    if "Pay Scale Group" in text or "薪酬等级" in text:
        return "avg_fte_by_pay_scale_group"
    if _contains_any(text, ("时薪制", "月薪制")):
        return "employment_type_distribution"
    if _contains_any(text, ("标准工时为 40", "标准工时 40", "40 小时", "40小时")) and _contains_any(text, ("占比", "比例")):
        return "standard_hours_40_ratio"
    if "深圳" in text and "分布" in text:
        return "office_department_distribution"
    if "北京" in text and "上海" in text and "汇报关系对比" in text:
        return "reporting_compare_beijing_shanghai"
    if _contains_any(text, ("跨国汇报", "经理和员工时区不同")):
        return "cross_timezone_reporting"
    if _contains_any(text, ("退休", "解雇", "离职")) and _contains_any(text, ("列表", "人员")):
        return "inactive_employee_list"
    if _contains_any(text, ("经理级", "直接下属 > 3", "直接下属>3")):
        return "manager_grade_reports_gt3"
    if _contains_any(text, ("绩效为低", "绩效低")) and _contains_any(text, ("流失风险高", "离职风险高")):
        return "low_perf_high_risk"
    if _contains_any(text, ("有离职风险", "流失风险")) and _contains_any(text, ("影响度高", "关键人才")):
        return "high_risk_key_talent"
    if _contains_any(text, ("FTE 为 0.5", "FTE=0.5", "FTE = 0.5")) and _contains_any(text, ("标准工时为 40", "标准工时 40", "40 小时", "40小时")):
        return "fte_hours_conflict"
    if "邮箱" in text and _contains_any(text, ("不规范", "异常", "格式")):
        return "invalid_email"
    if ("重复" in text or "出现多次" in text) and "员工 ID" in text:
        return "duplicate_employee_id"
    if _contains_any(text, ("缺失关键字段", "缺失关键")):
        return "missing_critical_fields"
    return ""


def _matches_filter(record: EmployeeRecord, condition: Dict[str, Any]) -> bool:
    field = str(condition.get("field") or "")
    op = str(condition.get("op") or "=")
    value = condition.get("value")
    actual = getattr(record, field, None)
    if op == "=":
        return actual == value
    if op in {"contains", "match"}:
        return str(value or "").lower() in str(actual or "").lower()
    if op == ">=" and field == "grade":
        return _grade_value(str(actual)) >= _grade_value(str(value))
    if op == ">=" and field == "hire_date":
        return str(actual) >= str(value)
    if op == ">" and field == "tenure_years":
        return _tenure_years(record) > float(value)
    if op == "range_year" and field == "hire_date":
        return str(actual).startswith(f"{value}-")
    return True


def _apply_filters(records: Iterable[EmployeeRecord], filters: Iterable[Dict[str, Any]]) -> List[EmployeeRecord]:
    conditions = list(filters)
    return [record for record in records if all(_matches_filter(record, condition) for condition in conditions)]


def _detect_intent(text: str) -> str:
    if _contains_any(text, ("简历", "履历", "附件", "下载")):
        return "document_fetch"
    if _contains_any(text, ("经理 ID 不存在", "经理ID不存在", "邮箱格式", "重复", "出现多次", "缺失关键字段", "数据质量", "异常", "FTE 为 0.5")):
        return "quality_check"
    if _contains_any(text, ("直接下属", "汇报", "经理", "下属最多", "跨时区")):
        return "relationship"
    if _contains_any(text, ("分布", "比例", "占比", "平均", "多少", "人数", "有多少")):
        return "aggregation"
    return "list"


def _build_query_spec(text: str) -> Dict[str, Any]:
    intent = _detect_intent(text)
    spec: Dict[str, Any] = {
        "domain": "hr",
        "entity": "employee" if intent != "document_fetch" else "resume",
        "intent": intent,
        "filters": _infer_filters(text),
        "limit": 20,
    }
    case = _infer_case(text)
    if case:
        spec["case"] = case
    if intent == "aggregation":
        if "部门" in text and "分布" in text:
            spec["group_by"] = ["department_code"]
            spec["metrics"] = [{"type": "count", "field": "employee_id"}]
        elif _contains_any(text, ("男女比例", "性别比例")):
            spec["group_by"] = ["gender"]
            spec["metrics"] = [{"type": "count", "field": "employee_id"}]
        elif "职级" in text and "分布" in text:
            spec["group_by"] = ["grade"]
            spec["metrics"] = [{"type": "count", "field": "employee_id"}]
        elif "时区" in text and "分布" in text:
            spec["group_by"] = ["timezone"]
            spec["metrics"] = [{"type": "count", "field": "employee_id"}]
        elif "FTE" in text and "分布" in text:
            spec["group_by"] = ["fte"]
            spec["metrics"] = [{"type": "count", "field": "employee_id"}]
        elif "雇佣类型" in text or "用工" in text:
            spec["group_by"] = ["employment_type"]
            spec["metrics"] = [{"type": "count", "field": "employee_id"}]
        elif "司龄" in text:
            spec["metrics"] = [{"type": "avg", "field": "tenure_years"}]
        else:
            spec["metrics"] = [{"type": "count", "field": "employee_id"}]
    if intent == "document_fetch":
        subjects = [
            {"type": "employee", "employee_id": record.employee_id, "name": record.name}
            for record in _find_people_by_query(text)
        ]
        spec["subjects"] = subjects
        spec["output"] = {"format": "resume_summary", "delivery": "chat"}
    if intent == "quality_check":
        if _contains_any(text, ("经理 ID 不存在", "经理ID不存在")):
            spec["check"] = "invalid_manager_id"
        elif "邮箱" in text:
            spec["check"] = "invalid_email"
        elif "重复" in text or "出现多次" in text:
            spec["check"] = "duplicate_employee_id"
        elif "FTE" in text:
            spec["check"] = "fte_hours_conflict"
        elif _contains_any(text, ("缺失关键字段", "缺失关键")):
            spec["check"] = "missing_critical_fields"
        else:
            spec["check"] = "all_quality_checks"
    return spec


def _placeholder_permission_check(query_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": False,
        "status": "placeholder_passed",
        "note": "版本不实现真实权限校验；生产版本应在这里注入行级、字段级、操作级权限。",
        "query_spec_intent": query_spec.get("intent"),
    }


def _placeholder_cost_check(query_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": False,
        "status": "placeholder_passed",
        "note": "版本不实现真实查询成本控制；生产版本应限制返回行数、聚合 bucket、向量候选数等。",
        "limit": query_spec.get("limit"),
    }


def _execute_aggregation(query_spec: Dict[str, Any]) -> Tuple[List[str], List[EmployeeRecord]]:
    records = _apply_filters(_EMPLOYEES, query_spec.get("filters") or [])
    group_by = list(query_spec.get("group_by") or [])
    lines: List[str] = []
    if group_by:
        field = group_by[0]
        counter = Counter(str(getattr(record, field, "") or "未填") for record in records)
        lines.append(f"匹配员工共 {len(records)} 人，按 `{field}` 分布如下：")
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            ratio = (count / len(records) * 100) if records else 0.0
            lines.append(f"- {key}：{count} 人（{ratio:.1f}%）")
        return lines, records

    metrics = list(query_spec.get("metrics") or [])
    if metrics and metrics[0].get("type") == "avg" and metrics[0].get("field") == "tenure_years":
        active_records = [record for record in records if _is_active(record)]
        avg = sum(_tenure_years(record) for record in active_records) / len(active_records) if active_records else 0.0
        lines.append(f"匹配在职员工 {len(active_records)} 人，平均司龄约 {avg:.1f} 年。")
        return lines, active_records

    lines.append(f"匹配员工共 {len(records)} 人。")
    if query_spec.get("filters"):
        lines.append("已按 QuerySpec 中的过滤条件统计。")
    return lines, records


def _execute_list(query_spec: Dict[str, Any], text: str) -> Tuple[List[str], List[EmployeeRecord]]:
    filters = list(query_spec.get("filters") or [])
    if _contains_any(text, ("工程师", "engineer")):
        records = [
            record for record in _apply_filters(_EMPLOYEES, filters) if "工程师" in record.job_title or "Engineer" in record.job_title
        ]
    elif _contains_any(text, ("VP", "高管")):
        records = [
            record
            for record in _apply_filters(_EMPLOYEES, filters)
            if _grade_value(record.grade) >= _grade_value("GR-10") or "VP" in record.job_title
        ]
    else:
        records = _apply_filters(_EMPLOYEES, filters)
    lines = [f"找到 {len(records)} 名匹配人员："]
    lines.extend(_format_people(records))
    return lines, records


def _execute_relationship(text: str) -> Tuple[List[str], List[EmployeeRecord], Dict[str, Any]]:
    by_id = {record.employee_id: record for record in _EMPLOYEES}
    active_records = [record for record in _EMPLOYEES if _is_active(record)]
    if "下属最多" in text:
        counter = Counter(record.manager_id for record in active_records if record.manager_id)
        manager_id, count = counter.most_common(1)[0]
        manager = by_id.get(manager_id)
        manager_name = manager.name if manager else manager_id
        reports = [record for record in active_records if record.manager_id == manager_id]
        lines = [f"直接下属最多的是 {manager_name}（{manager_id}），共有 {count} 名在职直接下属。"]
        lines.extend(_format_people(reports))
        return lines, reports, {"manager_id": manager_id}

    if "跨时区" in text:
        records: List[EmployeeRecord] = []
        lines = ["跨时区汇报情况如下："]
        for record in active_records:
            manager = by_id.get(record.manager_id)
            if manager and manager.timezone != record.timezone:
                records.append(record)
                lines.append(
                    f"- {record.name}（{record.timezone}） -> {manager.name}（{manager.timezone}）"
                )
        if not records:
            lines.append("未发现跨时区汇报。")
        return lines, records, {"relationship": "cross_timezone_reporting"}

    matched_people = _find_people_by_query(text)
    target = matched_people[0] if matched_people else None
    manager_id_match = re.search(r"\b\d{6}\b", text)
    manager_id = manager_id_match.group(0) if manager_id_match else (target.employee_id if target else "")
    if manager_id:
        reports = [record for record in active_records if record.manager_id == manager_id]
        manager = by_id.get(manager_id)
        label = manager.name if manager else manager_id
        lines = [f"{label}（{manager_id}）共有 {len(reports)} 名在职直接下属。"]
        lines.extend(_format_people(reports))
        return lines, reports, {"manager_id": manager_id}

    records = [record for record in _EMPLOYEES if not record.manager_id]
    lines = [f"没有指定经理的员工共有 {len(records)} 名："]
    lines.extend(_format_people(records))
    return lines, records, {"relationship": "missing_manager"}


def _execute_quality_check(query_spec: Dict[str, Any]) -> Tuple[List[str], List[EmployeeRecord]]:
    check = str(query_spec.get("check") or "all_quality_checks")
    by_id = {record.employee_id: record for record in _EMPLOYEES}
    records: List[EmployeeRecord] = []
    lines: List[str] = []
    if check in {"invalid_manager_id", "all_quality_checks"}:
        invalid_manager = [
            record for record in _EMPLOYEES if record.manager_id and record.manager_id not in by_id
        ]
        records.extend(invalid_manager)
        lines.append(f"经理 ID 不存在的记录：{len(invalid_manager)} 条。")
        lines.extend(_format_people(invalid_manager, empty_text="- 无"))
    if check in {"invalid_email", "all_quality_checks"}:
        invalid_email = [record for record in _EMPLOYEES if not _EMAIL_RE.match(record.email)]
        records.extend(invalid_email)
        lines.append(f"邮箱格式异常的记录：{len(invalid_email)} 条。")
        lines.extend(_format_people(invalid_email, empty_text="- 无"))
    if check in {"duplicate_employee_id", "all_quality_checks"}:
        id_counter = Counter(record.employee_id for record in _EMPLOYEES)
        duplicate_ids = sorted(employee_id for employee_id, count in id_counter.items() if count > 1)
        lines.append(f"重复员工 ID：{len(duplicate_ids)} 个。")
        if duplicate_ids:
            lines.extend(f"- {employee_id}" for employee_id in duplicate_ids)
        else:
            lines.append("- 样例主索引中未发现重复员工 ID；真实场景应在原始数据层检测。")
    deduped = list({record.employee_id: record for record in records}.values())
    return lines, deduped


def _execute_document_fetch(query_spec: Dict[str, Any], text: str) -> Tuple[List[str], List[EmployeeRecord]]:
    records = _find_people_by_query(text)
    if not records:
        records = _apply_filters(_EMPLOYEES, query_spec.get("filters") or [])[:3]
    lines = [f"找到 {len(records)} 份简历摘要："]
    for record in records:
        lines.append(f"- {record.name}（{record.employee_id}）：{record.resume_summary}")
    if not records:
        lines.append("- 未识别到明确人员；可尝试输入姓名、员工 ID，或先筛选人员列表。")
    return lines, records


def _execute_query_spec(query_spec: Dict[str, Any], text: str) -> Tuple[List[str], List[EmployeeRecord], Dict[str, Any]]:
    intent = str(query_spec.get("intent") or "list")
    if intent == "aggregation":
        lines, records = _execute_aggregation(query_spec)
        return lines, records, {}
    if intent == "relationship":
        return _execute_relationship(text)
    if intent == "quality_check":
        lines, records = _execute_quality_check(query_spec)
        return lines, records, {}
    if intent == "document_fetch":
        lines, records = _execute_document_fetch(query_spec, text)
        return lines, records, {}
    lines, records = _execute_list(query_spec, text)
    return lines, records, {}


def _hr_es_config() -> Dict[str, Any]:
    from backend.services.agent import settings

    return {
        "url": settings.get_hr_business_query_es_url(),
        "username": settings.get_hr_business_query_es_username(),
        "password": settings.get_hr_business_query_es_password(),
        "employee_index": settings.get_hr_business_query_employee_index(),
        "resume_chunk_index": settings.get_hr_business_query_resume_chunk_index(),
        "resume_asset_index": settings.get_hr_business_query_resume_asset_index(),
        "timeout": settings.get_hr_business_query_request_timeout_seconds(),
    }


def _es_call(config: Dict[str, Any], method: str, path: str, *, body: Optional[Dict[str, Any]] = None) -> Any:
    return es_request(
        method,
        str(config.get("url") or ""),
        path,
        body=body,
        username=str(config.get("username") or ""),
        password=str(config.get("password") or ""),
        timeout=float(config.get("timeout") or 30),
    )


def _es_search(config: Dict[str, Any], index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    response = _es_call(config, "POST", f"/{index}/_search", body=body)
    return response if isinstance(response, dict) else {}


def _es_hits_to_docs(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits = response.get("hits", {}).get("hits", [])
    docs: List[Dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source")
        if isinstance(source, dict):
            docs.append(source)
    return docs


def _es_total(response: Dict[str, Any]) -> int:
    total = response.get("hits", {}).get("total", {})
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    try:
        return int(total or 0)
    except (TypeError, ValueError):
        return 0


def _es_filter_clause(condition: Dict[str, Any]) -> Dict[str, Any]:
    field = str(condition.get("field") or "")
    op = str(condition.get("op") or "=")
    value = condition.get("value")
    if op == "=":
        if field in {"name", "department_name", "job_title"}:
            return {"term": {f"{field}.keyword": value}}
        return {"term": {field: value}}
    if op in {"contains", "match"}:
        if field in {"name", "department_name", "job_title", "resume_summary"}:
            return {"match_phrase": {field: value}}
        return {"match": {field: value}}
    if op == ">=" and field == "grade":
        return {"range": {"grade_level": {"gte": _grade_value(str(value))}}}
    if op == ">=" and field == "hire_date":
        return {"range": {"hire_date": {"gte": str(value)}}}
    if op == ">" and field == "tenure_years":
        return {"match_all": {}}
    if op == "range_year" and field == "hire_date":
        return {"range": {"hire_date": {"gte": f"{value}-01-01", "lte": f"{value}-12-31"}}}
    return {"match_all": {}}


def _es_bool_query(filters: Iterable[Dict[str, Any]], *, must: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    clauses = [
        clause
        for clause in (_es_filter_clause(condition) for condition in filters)
        if clause != {"match_all": {}}
    ]
    bool_query: Dict[str, Any] = {}
    if clauses:
        bool_query["filter"] = clauses
    if must:
        bool_query["must"] = must
    return {"bool": bool_query} if bool_query else {"match_all": {}}


def _es_fetch_all_employees(config: Dict[str, Any], *, size: int = 5000) -> List[Dict[str, Any]]:
    employee_index = str(config.get("employee_index") or "")
    response = _es_search(config, employee_index, {"size": size, "query": {"match_all": {}}})
    return _es_hits_to_docs(response)


def _doc_tenure_years(doc: Dict[str, Any]) -> float:
    try:
        year, month, day = [int(part) for part in str(doc.get("hire_date")).split("-")]
        return round((_TODAY - date(year, month, day)).days / 365.25, 1)
    except Exception:
        return 0.0


def _execute_es_aggregation(
    config: Dict[str, Any],
    query_spec: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    employee_index = str(config.get("employee_index") or "")
    filters = list(query_spec.get("filters") or [])
    case = str(query_spec.get("case") or "")
    if case == "married_ratio":
        response = _es_search(
            config,
            employee_index,
            {"size": 0, "aggs": {"by_marital_status": {"terms": {"field": "marital_status", "size": 10}}}},
        )
        buckets = response.get("aggregations", {}).get("by_marital_status", {}).get("buckets", [])
        total = sum(int(bucket.get("doc_count") or 0) for bucket in buckets)
        married = sum(int(bucket.get("doc_count") or 0) for bucket in buckets if bucket.get("key") == "已婚")
        ratio = (married / total * 100) if total else 0.0
        return [f"已婚员工 {married} 人，占全部员工 {ratio:.1f}%。"], [], {"buckets": buckets}
    if case in {"fte_distribution", "employment_type_distribution"}:
        field = "fte" if case == "fte_distribution" else "employment_type"
        query_spec = dict(query_spec)
        filters = []
        query_spec["group_by"] = [field]
    if case == "avg_fte_by_pay_scale_group":
        body = {
            "size": 0,
            "query": _es_bool_query(filters),
            "aggs": {
                "by_pay_scale_group": {
                    "terms": {"field": "pay_scale_group", "size": 30},
                    "aggs": {"avg_fte": {"avg": {"field": "fte"}}},
                }
            },
        }
        response = _es_search(config, employee_index, body)
        buckets = response.get("aggregations", {}).get("by_pay_scale_group", {}).get("buckets", [])
        lines = ["各薪酬等级的平均 FTE 如下："]
        for bucket in buckets:
            avg_fte = bucket.get("avg_fte", {}).get("value") or 0
            lines.append(f"- {bucket.get('key')}：平均 FTE {avg_fte:.2f}，样本 {bucket.get('doc_count')} 人")
        return lines, [], {"es_query": body}
    if case == "standard_hours_40_ratio":
        response = _es_search(config, employee_index, {"size": 0, "query": {"match_all": {}}})
        total = _es_total(response)
        matched = _es_search(config, employee_index, {"size": 0, "query": {"term": {"standard_hours": 40}}})
        count = _es_total(matched)
        ratio = (count / total * 100) if total else 0.0
        return [f"标准工时为 40 小时的员工 {count} 人，占全部员工 {ratio:.1f}%。"], [], {"total": total}
    if case == "office_department_distribution":
        query_spec = dict(query_spec)
        query_spec["group_by"] = ["department_code"]
    group_by = list(query_spec.get("group_by") or [])
    if group_by:
        field = str(group_by[0])
        body = {
            "size": 0,
            "query": _es_bool_query(filters),
            "aggs": {"distribution": {"terms": {"field": field, "size": 50}}},
        }
        response = _es_search(config, employee_index, body)
        total = _es_total(response)
        buckets = response.get("aggregations", {}).get("distribution", {}).get("buckets", [])
        lines = [f"匹配员工共 {total} 人，按 `{field}` 分布如下："]
        for bucket in buckets:
            count = int(bucket.get("doc_count") or 0)
            ratio = (count / total * 100) if total else 0.0
            lines.append(f"- {bucket.get('key')}：{count} 人（{ratio:.1f}%）")
        return lines, [], {"es_query": body}

    metrics = list(query_spec.get("metrics") or [])
    if metrics and metrics[0].get("type") == "avg" and metrics[0].get("field") == "tenure_years":
        response = _es_search(
            config,
            employee_index,
            {"size": 1000, "query": _es_bool_query(filters + [{"field": "employment_status", "op": "=", "value": "active"}])},
        )
        docs = _es_hits_to_docs(response)
        tenures = []
        for doc in docs:
            try:
                year, month, day = [int(part) for part in str(doc.get("hire_date")).split("-")]
                tenures.append(round((_TODAY - date(year, month, day)).days / 365.25, 1))
            except Exception:
                continue
        avg = sum(tenures) / len(tenures) if tenures else 0.0
        return [f"匹配在职员工 {len(tenures)} 人，平均司龄约 {avg:.1f} 年。"], docs[:5], {"es_query": response}

    response = _es_search(config, employee_index, {"size": 5, "query": _es_bool_query(filters)})
    docs = _es_hits_to_docs(response)
    total = _es_total(response)
    lines = [f"匹配员工共 {total} 人。"]
    if filters:
        lines.append("已按 QuerySpec 中的过滤条件统计。")
    return lines, docs, {"es_total": total}


def _execute_es_list(
    config: Dict[str, Any],
    query_spec: Dict[str, Any],
    text: str,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    employee_index = str(config.get("employee_index") or "")
    case = str(query_spec.get("case") or "")
    if case == "inactive_employee_list":
        body = {
            "size": int(query_spec.get("limit") or 20),
            "query": {"bool": {"filter": [{"terms": {"employment_status": ["terminated", "retired"]}}]}},
            "sort": [{"employee_id": {"order": "asc"}}],
        }
        response = _es_search(config, employee_index, body)
        docs = _es_hits_to_docs(response)
        lines = [f"已退休或已解雇/离职人员共 {_es_total(response)} 人："]
        lines.extend(_format_any_people(docs))
        return lines, docs, {"es_query": body}
    if case == "low_perf_high_risk":
        query_spec = dict(query_spec)
        query_spec["filters"] = [
            {"field": "performance_rating", "op": "=", "value": "低"},
            {"field": "attrition_risk", "op": "=", "value": "高"},
        ]
    if case == "high_risk_key_talent":
        query_spec = dict(query_spec)
        query_spec["filters"] = [
            {"field": "attrition_risk", "op": "=", "value": "高"},
            {"field": "key_position", "op": "=", "value": True},
        ]
    if case == "manager_grade_reports_gt3":
        docs = _es_fetch_all_employees(config)
        active = [doc for doc in docs if doc.get("employment_status") == "active"]
        report_counter = Counter(str(doc.get("manager_id") or "") for doc in active if doc.get("manager_id"))
        managers = [
            doc
            for doc in docs
            if int(doc.get("grade_level") or 0) >= 10 and report_counter.get(str(doc.get("employee_id")), 0) > 3
        ]
        lines = [f"经理级且直接下属 > 3 人的员工共 {len(managers)} 人："]
        for doc in managers:
            lines.append(f"- {_format_any_person(doc)}，直接下属 {report_counter.get(str(doc.get('employee_id')), 0)} 人")
        return lines, managers, {"computed_from_docs": len(docs)}
    sort_items = list(query_spec.get("sort") or [])
    if any(str(item.get("field")) == "tenure_years" for item in sort_items if isinstance(item, dict)):
        docs = _es_fetch_all_employees(config)
        filters = list(query_spec.get("filters") or [])
        docs = [
            doc
            for doc in docs
            if all(
                _doc_get(doc, str(cond.get("field"))) == cond.get("value")
                for cond in filters
                if cond.get("field") != "tenure_years"
            )
        ]
        direction = "asc"
        for item in sort_items:
            if isinstance(item, dict) and str(item.get("field")) == "tenure_years":
                direction = str(item.get("direction") or item.get("order") or "asc").lower()
                break
        docs.sort(key=_doc_tenure_years, reverse=direction == "desc")
        limit = int(query_spec.get("limit") or 20)
        selected = docs[:limit]
        label = "最长" if direction == "desc" else "最短"
        lines = [f"按入职年限{label}筛选，匹配员工共 {len(docs)} 人，返回前 {len(selected)} 人："]
        for doc in selected:
            lines.append(f"- {_format_any_person(doc)}，入职年限约 {_doc_tenure_years(doc):.1f} 年")
        return lines, selected, {"computed_from_docs": len(docs), "computed_sort": "tenure_years"}
    if any(str(item.get("field")) == "tenure_years" for item in query_spec.get("filters") or []):
        docs = _es_fetch_all_employees(config)
        threshold = 0.0
        for cond in query_spec.get("filters") or []:
            if cond.get("field") == "tenure_years" and cond.get("op") == ">":
                try:
                    threshold = float(cond.get("value") or 0)
                except (TypeError, ValueError):
                    threshold = 0.0
        docs = [
            doc
            for doc in docs
            if _doc_tenure_years(doc) > threshold
            and all(
                _doc_get(doc, str(cond.get("field"))) == cond.get("value")
                for cond in query_spec.get("filters") or []
                if cond.get("field") not in {"tenure_years"}
            )
        ]
        lines = [f"入职超过 {threshold:g} 年且满足条件的员工共 {len(docs)} 人："]
        lines.extend(_format_any_people(docs))
        return lines, docs, {"computed_from_docs": len(docs)}
    must: List[Dict[str, Any]] = []
    if _contains_any(text, ("工程师", "engineer")):
        must.append(
            {
                "multi_match": {
                    "query": "工程师 Engineer",
                    "fields": ["job_title^3", "resume_summary", "skills"],
                }
            }
        )
    body = {
        "size": int(query_spec.get("limit") or 20),
        "query": _es_bool_query(query_spec.get("filters") or [], must=must),
        "sort": [{"grade_level": {"order": "desc"}}, {"employee_id": {"order": "asc"}}],
    }
    response = _es_search(config, employee_index, body)
    docs = _es_hits_to_docs(response)
    lines = [f"找到 {_es_total(response)} 名匹配人员："]
    lines.extend(_format_any_people(docs))
    return lines, docs, {"es_query": body}


def _execute_es_relationship(config: Dict[str, Any], text: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    employee_index = str(config.get("employee_index") or "")
    if _contains_any(text, ("经理级", "直接下属 > 3", "直接下属>3")):
        docs = _es_fetch_all_employees(config)
        active = [doc for doc in docs if doc.get("employment_status") == "active"]
        report_counter = Counter(str(doc.get("manager_id") or "") for doc in active if doc.get("manager_id"))
        managers = [
            doc
            for doc in docs
            if int(doc.get("grade_level") or 0) >= 10 and report_counter.get(str(doc.get("employee_id")), 0) > 3
        ]
        lines = [f"经理级且直接下属 > 3 人的员工共 {len(managers)} 人："]
        for doc in managers:
            lines.append(f"- {_format_any_person(doc)}，直接下属 {report_counter.get(str(doc.get('employee_id')), 0)} 人")
        return lines, managers, {"computed_from_docs": len(docs)}
    if "北京" in text and "上海" in text and "汇报关系对比" in text:
        docs = _es_fetch_all_employees(config)
        by_id = {str(doc.get("employee_id")): doc for doc in docs}
        lines = ["北京和上海两地人员汇报关系对比："]
        rows: List[Dict[str, Any]] = []
        for city in ("北京", "上海"):
            city_docs = [doc for doc in docs if doc.get("office_city") == city]
            managers = {str(doc.get("manager_id")) for doc in city_docs if doc.get("manager_id")}
            cross_city = [
                doc for doc in city_docs
                if doc.get("manager_id") and by_id.get(str(doc.get("manager_id")), {}).get("office_city") not in {"", None, city}
            ]
            rows.extend(cross_city)
            lines.append(f"- {city}：员工 {len(city_docs)} 人，涉及直属经理 {len(managers)} 位，跨城市汇报 {len(cross_city)} 人。")
        return lines, rows, {"relationship": "beijing_shanghai_reporting_compare"}
    if "下属最多" in text:
        body = {
            "size": 0,
            "query": {"term": {"employment_status": "active"}},
            "aggs": {"managers": {"terms": {"field": "manager_id", "size": 1, "order": {"_count": "desc"}}}},
        }
        response = _es_search(config, employee_index, body)
        buckets = response.get("aggregations", {}).get("managers", {}).get("buckets", [])
        manager_id = str(buckets[0].get("key") or "") if buckets else ""
        count = int(buckets[0].get("doc_count") or 0) if buckets else 0
        reports_response = _es_search(
            config,
            employee_index,
            {"size": 20, "query": _es_bool_query([{"field": "manager_id", "op": "=", "value": manager_id}])},
        )
        manager_response = _es_search(config, employee_index, {"size": 1, "query": {"term": {"employee_id": manager_id}}})
        manager_docs = _es_hits_to_docs(manager_response)
        manager_name = manager_docs[0].get("name") if manager_docs else manager_id
        reports = _es_hits_to_docs(reports_response)
        lines = [f"直接下属最多的是 {manager_name}（{manager_id}），共有 {count} 名在职直接下属。"]
        lines.extend(_format_any_people(reports))
        return lines, reports, {"manager_id": manager_id}

    if "跨时区" in text:
        response = _es_search(config, employee_index, {"size": 1000, "query": {"term": {"employment_status": "active"}}})
        docs = _es_hits_to_docs(response)
        by_id = {str(doc.get("employee_id")): doc for doc in docs}
        rows: List[Dict[str, Any]] = []
        lines = ["跨时区汇报情况如下："]
        for doc in docs:
            manager = by_id.get(str(doc.get("manager_id") or ""))
            if manager and manager.get("timezone") != doc.get("timezone"):
                rows.append(doc)
                lines.append(f"- {doc.get('name')}（{doc.get('timezone')}） -> {manager.get('name')}（{manager.get('timezone')}）")
        if not rows:
            lines.append("未发现跨时区汇报。")
        return lines, rows, {"relationship": "cross_timezone_reporting"}

    matched_people = _find_people_by_query(text)
    manager_id_match = re.search(r"\b\d{6}\b", text)
    manager_id = manager_id_match.group(0) if manager_id_match else (matched_people[0].employee_id if matched_people else "")
    if manager_id:
        reports_response = _es_search(
            config,
            employee_index,
            {"size": 50, "query": _es_bool_query([{"field": "manager_id", "op": "=", "value": manager_id}, {"field": "employment_status", "op": "=", "value": "active"}])},
        )
        reports = _es_hits_to_docs(reports_response)
        manager_response = _es_search(config, employee_index, {"size": 1, "query": {"term": {"employee_id": manager_id}}})
        manager_docs = _es_hits_to_docs(manager_response)
        label = manager_docs[0].get("name") if manager_docs else manager_id
        lines = [f"{label}（{manager_id}）共有 {_es_total(reports_response)} 名在职直接下属。"]
        lines.extend(_format_any_people(reports))
        return lines, reports, {"manager_id": manager_id}

    response = _es_search(
        config,
        employee_index,
        {"size": 20, "query": {"bool": {"should": [{"term": {"manager_id": ""}}], "minimum_should_match": 1}}},
    )
    docs = _es_hits_to_docs(response)
    lines = [f"没有指定经理的员工共有 {_es_total(response)} 名："]
    lines.extend(_format_any_people(docs))
    return lines, docs, {"relationship": "missing_manager"}


def _execute_es_quality_check(config: Dict[str, Any], query_spec: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    employee_index = str(config.get("employee_index") or "")
    check = str(query_spec.get("check") or "all_quality_checks")
    response = _es_search(config, employee_index, {"size": 1000, "query": {"match_all": {}}})
    docs = _es_hits_to_docs(response)
    by_id = {str(doc.get("employee_id")): doc for doc in docs}
    rows: List[Dict[str, Any]] = []
    lines: List[str] = []
    if check in {"invalid_manager_id", "all_quality_checks"}:
        invalid = [doc for doc in docs if doc.get("manager_id") and str(doc.get("manager_id")) not in by_id]
        rows.extend(invalid)
        lines.append(f"经理 ID 不存在的记录：{len(invalid)} 条。")
        lines.extend(_format_any_people(invalid, empty_text="- 无"))
    if check in {"invalid_email", "all_quality_checks"}:
        invalid_email = [doc for doc in docs if not _EMAIL_RE.match(str(doc.get("email") or ""))]
        rows.extend(invalid_email)
        lines.append(f"邮箱格式异常的记录：{len(invalid_email)} 条。")
        lines.extend(_format_any_people(invalid_email, empty_text="- 无"))
    if check in {"fte_hours_conflict", "all_quality_checks"}:
        conflicts = [
            doc for doc in docs
            if float(doc.get("fte") or 0) == 0.5 and int(doc.get("standard_hours") or 0) == 40
        ]
        rows.extend(conflicts)
        lines.append(f"FTE 为 0.5 但标准工时为 40 的记录：{len(conflicts)} 条。")
        lines.extend(_format_any_people(conflicts, empty_text="- 无"))
    if check in {"missing_critical_fields", "all_quality_checks"}:
        missing = [
            doc for doc in docs
            if not doc.get("department_code") or not doc.get("department_name") or not doc.get("grade")
        ]
        rows.extend(missing)
        lines.append(f"缺失关键字段（部门、职级）的记录：{len(missing)} 条。")
        lines.extend(_format_any_people(missing, empty_text="- 无"))
    if check in {"duplicate_employee_id", "all_quality_checks"}:
        lines.append("重复员工 ID：主员工索引用 `_id=employee_id` 会覆盖重复；请在 raw JSONL 或原始数据层检测。")
    deduped = list({str(doc.get("employee_id")): doc for doc in rows}.values())
    return lines, deduped, {"checked_docs": len(docs)}


def _execute_es_document_fetch(
    config: Dict[str, Any],
    query_spec: Dict[str, Any],
    text: str,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    employee_index = str(config.get("employee_index") or "")
    asset_index = str(config.get("resume_asset_index") or "")
    chunk_index = str(config.get("resume_chunk_index") or "")
    subjects = list(query_spec.get("subjects") or [])
    subject_ids = [str(item.get("employee_id")) for item in subjects if item.get("employee_id")]
    if subject_ids:
        employee_query: Dict[str, Any] = {"ids": {"values": subject_ids}}
    else:
        employee_query = _es_bool_query(query_spec.get("filters") or [])
    employee_response = _es_search(config, employee_index, {"size": 10, "query": employee_query})
    employees = _es_hits_to_docs(employee_response)
    employee_ids = [str(doc.get("employee_id")) for doc in employees]
    lines = [f"找到 {len(employees)} 份简历资料："]
    if not employee_ids:
        lines.append("- 未识别到明确人员；可尝试输入姓名、员工 ID，或先筛选人员列表。")
        return lines, [], {"employee_query": employee_query}

    asset_response = _es_search(config, asset_index, {"size": 20, "query": {"terms": {"employee_id": employee_ids}}})
    chunk_response = _es_search(
        config,
        chunk_index,
        {"size": 20, "query": {"bool": {"filter": [{"terms": {"employee_id": employee_ids}}, {"term": {"chunk_type": "profile"}}]}}},
    )
    assets = _es_hits_to_docs(asset_response)
    chunks = _es_hits_to_docs(chunk_response)
    assets_by_employee = {str(asset.get("employee_id")): asset for asset in assets}
    chunks_by_employee = {str(chunk.get("employee_id")): chunk for chunk in chunks}
    for employee in employees:
        employee_id = str(employee.get("employee_id"))
        asset = assets_by_employee.get(employee_id, {})
        chunk = chunks_by_employee.get(employee_id, {})
        summary = chunk.get("text") or employee.get("resume_summary") or "无简历摘要"
        storage_uri = asset.get("storage_uri") or "无模拟链接"
        lines.append(f"- {employee.get('name')}（{employee_id}）：{summary}；资产：{storage_uri}")
    return lines, employees, {"asset_hits": len(assets), "chunk_hits": len(chunks)}


def _execute_es_query_spec(
    query_spec: Dict[str, Any],
    text: str,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    config = _hr_es_config()
    intent = str(query_spec.get("intent") or "list")
    if intent == "aggregation":
        return _execute_es_aggregation(config, query_spec)
    if intent == "relationship":
        return _execute_es_relationship(config, text)
    if intent == "quality_check":
        return _execute_es_quality_check(config, query_spec)
    if intent == "document_fetch":
        return _execute_es_document_fetch(config, query_spec, text)
    return _execute_es_list(config, query_spec, text)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"

"""Execution layer for validated HR QuerySpec objects."""





def execute_hr_query_spec(query_spec: Dict[str, Any], *, query: str = "") -> Dict[str, Any]:
    if query_spec.get("intent") == "clarify":
        return {
            "intent": "clarify",
            "backend": "es",
            "lines": [str(query_spec.get("clarifying_question") or "请补充查询条件。")],
            "rows": [],
            "summary": {"row_count": 0},
            "debug": {},
        }

    query_spec = _prepare_query_spec_for_execution(query_spec)
    if query_spec.get("intent") == "attribute_lookup":
        lines, rows, extra = _execute_attribute_lookup(query_spec)
    elif query_spec.get("intent") == "document_fetch":
        lines, rows, extra = _execute_es_document_fetch(query_spec)
    else:
        lines, rows, extra = sample_backend._execute_es_query_spec(query_spec, query)  # noqa: SLF001 - domain backend

    return {
        "intent": query_spec.get("intent"),
        "backend": "es",
        "lines": list(lines or []),
        "rows": list(rows or []),
        "summary": {"row_count": len(rows or [])},
        "debug": dict(extra or {}),
    }


def execute_guarded_hr_es_dsl(
    search_body: Dict[str, Any],
    *,
    query: str = "",
    max_limit: int = 100,
    max_buckets: int = 100,
) -> Dict[str, Any]:
    guarded_body = validate_hr_es_search_body(search_body, max_limit=max_limit, max_buckets=max_buckets)
    config = sample_backend._hr_es_config()  # noqa: SLF001 - production wrapper for HR ES config
    employee_index = str(config.get("employee_index") or "")
    response = sample_backend._es_search(config, employee_index, guarded_body)  # noqa: SLF001
    rows = sample_backend._es_hits_to_docs(response)  # noqa: SLF001
    total = sample_backend._es_total(response)  # noqa: SLF001
    lines = _format_dsl_result_lines(guarded_body, response, rows, total, query=query)
    return {
        "intent": "es_dsl",
        "backend": "es",
        "lines": lines,
        "rows": rows,
        "summary": {"row_count": len(rows), "total": total},
        "debug": {"es_query": guarded_body, "original_query": query},
    }


def _format_dsl_result_lines(search_body: Dict[str, Any], response: Dict[str, Any], rows: List[Dict[str, Any]], total: int, *, query: str = "") -> List[str]:
    aggregations = response.get("aggregations") if isinstance(response, dict) else None
    if isinstance(aggregations, dict) and aggregations:
        lines = [f"匹配员工共 {total} 人，聚合结果如下："]
        lines.extend(_format_aggregation_lines(aggregations))
        return lines
    if int(search_body.get("size") or 0) == 0:
        return [f"匹配员工共 {total} 人。"]
    attribute_fields = _requested_attribute_fields(query)
    if rows and attribute_fields and not _looks_like_list_request(query):
        lines = []
        for row in rows:
            subject = row.get("name") or row.get("employee_id") or "该员工"
            for field in attribute_fields:
                value = _attribute_value(row, field)
                lines.append(f"{subject}的{_field_label(field)}是{_format_attribute_value(field, value)}。")
        return lines
    if rows and _looks_like_detail_request(query):
        lines = [f"找到 {total} 名匹配人员："]
        for row in rows:
            lines.extend(_format_employee_detail(row))
        return lines
    if rows and len(rows) < total:
        lines = [f"匹配员工共 {total} 人，返回前 {len(rows)} 人："]
    else:
        lines = [f"找到 {total} 名匹配人员："]
    if "tenure_years" in attribute_fields:
        for row in rows:
            tenure = _attribute_value(row, "tenure_years")
            lines.append(f"- {sample_backend._format_any_person(row)}，入职年限{_format_attribute_value('tenure_years', tenure)}")  # noqa: SLF001
    else:
        lines.extend(sample_backend._format_any_people(rows))  # noqa: SLF001
    return lines


def _looks_like_detail_request(query: str) -> bool:
    return any(keyword in str(query or "") for keyword in ("详细资料", "详细信息", "基本信息", "员工资料", "这个员工的资料", "资料给我"))


def _looks_like_list_request(query: str) -> bool:
    return any(keyword in str(query or "") for keyword in ("名单", "清单", "列表", "列出", "发给我", "有哪些", "都有谁"))


def _requested_attribute_fields(query: str) -> List[str]:
    text = str(query or "")
    lowered = text.lower()
    fields: List[str] = []
    if any(keyword in text for keyword in ("入职年限", "入职几年", "入职多久", "司龄")) or any(
        keyword in lowered for keyword in ("tenure", "years of service", "length of service")
    ):
        fields.append("tenure_years")
    elif any(keyword in text for keyword in ("哪一年入职", "什么时候入职", "入职日期", "入职时间")):
        fields.append("hire_date")
    if any(keyword in text for keyword in ("经理是谁", "他的经理", "她的经理", "汇报给谁")):
        fields.append("manager_id")
    if any(keyword in text for keyword in ("离职风险", "流失风险", "风险等级")):
        fields.append("attrition_risk")
    return list(dict.fromkeys(fields))


def _format_employee_detail(row: Dict[str, Any]) -> List[str]:
    status = "在职" if row.get("employment_status") == "active" else "非在职"
    manager_id = str(row.get("manager_id") or "").strip() or "未填"
    return [
        f"- {row.get('name')}（{row.get('employee_id')}）：",
        f"  - 状态：{status}",
        f"  - 部门：{row.get('department_name') or row.get('department_code') or '未填'}",
        f"  - 职位：{row.get('job_title') or '未填'}",
        f"  - 职级：{row.get('grade') or '未填'}",
        f"  - 办公城市：{row.get('office_city') or '未填'}",
        f"  - 入职日期：{row.get('hire_date') or '未填'}",
        f"  - 经理 ID：{manager_id}",
        f"  - 邮箱：{row.get('email') or '未填'}",
    ]


def _format_aggregation_lines(aggregations: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for name, value in aggregations.items():
        if isinstance(value, dict) and isinstance(value.get("buckets"), list):
            lines.append(f"- {name}：")
            for bucket in value.get("buckets") or []:
                lines.append(f"  - {bucket.get('key')}：{bucket.get('doc_count')} 人")
        elif isinstance(value, dict) and "value" in value:
            lines.append(f"- {name}：{value.get('value')}")
        else:
            lines.append(f"- {name}：{value}")
    return lines


def _prepare_query_spec_for_execution(query_spec: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(query_spec)
    filters = list(prepared.get("filters") or [])
    has_employment_status = any(item.get("field") == "employment_status" for item in filters if isinstance(item, dict))
    if prepared.get("business_rule") == "high_risk_key_talent" and has_employment_status:
        # The sample backend has a hard-coded high_risk_key_talent case;
        # production QuerySpec filters should keep the explicit active scope.
        prepared.pop("case", None)
    return prepared


def _execute_attribute_lookup(query_spec: Dict[str, Any]) -> tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    fields = [str(field) for field in query_spec.get("select_fields") or [] if str(field)]
    if not fields:
        return ["请说明要查询员工的哪个字段，例如办公地点、工号、邮箱或部门。"], [], {}

    rows = _fetch_attribute_subjects_from_es(query_spec)

    if not rows:
        return ["未找到匹配员工；请提供更明确的姓名或员工 ID。"], [], {"backend": "es"}

    lines: List[str] = []
    for row in rows:
        name = str(row.get("name") or row.get("employee_id") or "该员工")
        values = [_format_attribute_value(field, _attribute_value(row, field)) for field in fields]
        if len(fields) == 1:
            lines.append(f"{name}的{_field_label(fields[0])}是{values[0]}。")
        else:
            subject = name
            employee_id = str(row.get("employee_id") or "").strip()
            if employee_id:
                subject = f"{name}（{employee_id}）"
            labels = [_attribute_label(field) for field in fields]
            lines.append(f"{subject}的信息：{'；'.join(f'{label}：{value}' for label, value in zip(labels, values))}。")
    return lines, rows, {"backend": "es", "select_fields": fields}


def _fetch_employee_by_id(employee_id: str) -> Dict[str, Any]:
    if not employee_id:
        return {}
    config = sample_backend._hr_es_config()  # noqa: SLF001 - domain backend
    response = sample_backend._es_search(  # noqa: SLF001 - domain backend
        config,
        str(config.get("employee_index") or ""),
        {"size": 1, "query": {"term": {"employee_id": employee_id}}},
    )
    docs = sample_backend._es_hits_to_docs(response)  # noqa: SLF001 - domain backend
    return dict(docs[0]) if docs else {}


def _fetch_attribute_subjects_from_es(query_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    subjects = list(query_spec.get("subjects") or [])
    employee_ids = [str(item.get("employee_id") or item.get("value") or "") for item in subjects if item.get("employee_id") or item.get("type") == "employee_id"]
    names = [str(item.get("name") or item.get("value") or "") for item in subjects if item.get("name") or item.get("type") in {"employee", "employee_name"}]
    if not employee_ids and not names:
        return []

    should: List[Dict[str, Any]] = []
    if employee_ids:
        should.append({"terms": {"employee_id": employee_ids}})
    for name in names:
        should.append({"term": {"name.keyword": name}})
    config = sample_backend._hr_es_config()  # noqa: SLF001 - domain backend
    response = sample_backend._es_search(  # noqa: SLF001 - domain backend
        config,
        str(config.get("employee_index") or ""),
        {
            "size": int(query_spec.get("limit") or 5),
            "query": {"bool": {"should": should, "minimum_should_match": 1}},
        },
    )
    return sample_backend._es_hits_to_docs(response)  # noqa: SLF001 - domain backend


def _execute_es_document_fetch(query_spec: Dict[str, Any]) -> tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    employees = _fetch_attribute_subjects_from_es(query_spec)
    if not employees:
        return ["未找到匹配员工；请提供更明确的姓名或员工 ID。"], [], {"employee_hits": 0}

    config = sample_backend._hr_es_config()  # noqa: SLF001 - domain backend
    asset_index = str(config.get("resume_asset_index") or "")
    chunk_index = str(config.get("resume_chunk_index") or "")
    employee_ids = [str(doc.get("employee_id")) for doc in employees if doc.get("employee_id")]
    asset_response = sample_backend._es_search(  # noqa: SLF001
        config,
        asset_index,
        {"size": 20, "query": {"terms": {"employee_id": employee_ids}}},
    )
    chunk_response = sample_backend._es_search(  # noqa: SLF001
        config,
        chunk_index,
        {
            "size": 20,
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"employee_id": employee_ids}},
                        {"term": {"chunk_type": "profile"}},
                    ]
                }
            },
        },
    )
    assets = sample_backend._es_hits_to_docs(asset_response)  # noqa: SLF001
    chunks = sample_backend._es_hits_to_docs(chunk_response)  # noqa: SLF001
    assets_by_employee = {str(asset.get("employee_id")): asset for asset in assets}
    chunks_by_employee = {str(chunk.get("employee_id")): chunk for chunk in chunks}

    lines = [f"找到 {len(employees)} 份简历资料："]
    for employee in employees:
        employee_id = str(employee.get("employee_id") or "")
        asset = assets_by_employee.get(employee_id, {})
        chunk = chunks_by_employee.get(employee_id, {})
        summary = chunk.get("text") or employee.get("resume_summary") or "无简历摘要"
        storage_uri = asset.get("storage_uri") or "无模拟链接"
        lines.append(f"- {employee.get('name')}（{employee_id}）：{summary}；资产：{storage_uri}")
    return lines, employees, {"employee_hits": len(employees), "asset_hits": len(assets), "chunk_hits": len(chunks)}


def _field_label(field: str) -> str:
    return {
        "employee_id": "工号",
        "name": "姓名",
        "office_city": "办公地点",
        "email": "邮箱",
        "department_code": "部门编码",
        "department_name": "部门",
        "job_title": "职位",
        "manager_id": "经理",
        "hire_date": "入职日期",
        "tenure_years": "入职年限",
        "employment_status": "状态",
        "attrition_risk": "离职风险",
    }.get(field, field)


def _attribute_label(field: str) -> str:
    if field == "office_city":
        return "办公城市"
    return _field_label(field)


def _format_attribute_value(field: str, value: Any) -> str:
    if _is_empty_attribute_value(value):
        return "未填"
    if isinstance(value, list):
        return "、".join(str(item) for item in value if not _is_empty_attribute_value(item)) or "未填"
    if field == "tenure_years":
        try:
            return f"约 {float(value):.1f} 年"
        except (TypeError, ValueError):
            return str(value)
    if field == "employment_status":
        if value == "active":
            return "在职"
        if value in {"terminated", "retired"}:
            return "非在职"
    return str(value)


def _attribute_value(row: Dict[str, Any], field: str) -> Any:
    if field == "manager_id":
        manager_id = str(_first_attribute_value(row.get("manager_id")) or "").strip()
        if not manager_id:
            return None
        manager = _fetch_employee_by_id(manager_id)
        manager_name = str(manager.get("name") or "").strip()
        return f"{manager_name}（{manager_id}）" if manager_name else manager_id
    if field != "tenure_years":
        return row.get(field)
    value = row.get(field)
    if not _is_empty_attribute_value(value):
        return value
    return _calculate_tenure_years(row.get("hire_date"))


def _is_empty_attribute_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _first_attribute_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        for item in value:
            if not _is_empty_attribute_value(item):
                return item
        return None
    return value


def _calculate_tenure_years(hire_date: Any, *, today: date | None = None) -> Any:
    text = str(hire_date or "").strip()
    if not text:
        return None
    try:
        year, month, day = [int(part) for part in text[:10].split("-")]
        hired_at = date(year, month, day)
    except Exception:
        return None
    current = today or getattr(sample_backend, "_TODAY", date.today())
    days = max(0, (current - hired_at).days)
    return round(days / 365.25, 1)


def placeholder_permission_check(query_spec: Dict[str, Any]) -> Dict[str, Any]:
    return base_placeholder_permission_check(query_spec, domain="hr")


def placeholder_cost_check(query_spec: Dict[str, Any]) -> Dict[str, Any]:
    return base_placeholder_cost_check(query_spec)
