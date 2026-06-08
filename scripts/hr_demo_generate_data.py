#!/usr/bin/env python3
"""生成 HR ES demo JSONL 数据。

示例：
  python scripts/hr_demo_generate_data.py --count 200 --out-dir data/demo/hr
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.agent.tools.builtin.business_query_core.hr import executor as hr_es  # noqa: E402

DEFAULT_EMPLOYEE_FILE = hr_es.DEFAULT_EMPLOYEE_FILE
DEFAULT_RAW_QUALITY_FILE = hr_es.DEFAULT_RAW_QUALITY_FILE
DEFAULT_RESUME_ASSET_FILE = hr_es.DEFAULT_RESUME_ASSET_FILE
DEFAULT_RESUME_CHUNK_FILE = hr_es.DEFAULT_RESUME_CHUNK_FILE
DEMO_VECTOR_DIMS = hr_es.DEFAULT_VECTOR_DIMS
write_jsonl = hr_es.write_jsonl

TODAY = date(2026, 4, 26)

DEPARTMENTS: Tuple[Tuple[str, str, str, List[str]], ...] = (
    ("MANU", "制造部", "Operations", ["MES", "生产计划", "质量追踪", "自动化"]),
    ("RND", "研发部", "Engineering", ["Python", "平台工程", "AI", "分布式系统"]),
    ("SALES", "销售部", "Sales", ["大客户", "渠道管理", "CRM", "销售预测"]),
    ("FIN", "财务部", "Finance", ["预算", "财务分析", "SQL", "BI"]),
    ("HR", "人力资源部", "HumanResources", ["招聘", "员工关系", "组织发展", "HRIS"]),
    ("EXEC", "管理层", "Executive", ["战略", "组织管理", "经营分析", "跨区域协同"]),
)

CITIES: Tuple[Tuple[str, str, str, str], ...] = (
    ("北京", "北京总部", "中国", "Asia/Shanghai"),
    ("上海", "上海张江", "中国", "Asia/Shanghai"),
    ("深圳", "深圳南山", "中国", "Asia/Shanghai"),
    ("纽约", "New York Office", "美国", "America/New_York"),
    ("旧金山", "San Francisco Office", "美国", "America/Los_Angeles"),
)

FIRST_NAMES_CN = ["张", "李", "王", "赵", "陈", "周", "吴", "郑", "孙", "钱", "林", "何"]
LAST_NAMES_CN = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "明", "华", "宁"]
EN_NAMES = ["Alice Chen", "Bob Smith", "Grace Lee", "Helen Wang", "Kevin Liu", "Nina Zhao"]


def grade_level_to_code(level: int) -> str:
    return f"GR-{level:02d}"


def mock_embedding(seed: int, dims: int = DEMO_VECTOR_DIMS) -> List[float]:
    rng = random.Random(seed)
    return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(dims)]


def employee_doc(
    *,
    employee_id: str,
    name: str,
    gender: str,
    marital_status: str,
    city: Tuple[str, str, str, str],
    department: Tuple[str, str, str, List[str]],
    job_title: str,
    grade_level: int,
    fte: float,
    standard_hours: int,
    employment_type: str,
    employment_status: str,
    hire_date: str,
    manager_id: str,
    email: str,
    performance_rating: str,
    attrition_risk: str,
    high_potential: bool,
    key_position: bool,
    future_leader: bool,
    promotion_likelihood: str,
) -> Dict[str, Any]:
    office_city, office_location, country_region, timezone = city
    department_code, department_name, job_family, skills = department
    grade = grade_level_to_code(grade_level)
    summary = (
        f"{name} 具备 {department_name} {job_title} 经验，熟悉"
        f"{'、'.join(skills[:3])}，绩效{performance_rating}，流失风险{attrition_risk}。"
    )
    return {
        "employee_id": employee_id,
        "name": name,
        "name_pinyin": name.lower().replace(" ", "_"),
        "email": email,
        "gender": gender,
        "marital_status": marital_status,
        "office_city": office_city,
        "office_location": office_location,
        "country_region": country_region,
        "timezone": timezone,
        "department_code": department_code,
        "department_name": department_name,
        "job_title": job_title,
        "job_family": job_family,
        "grade": grade,
        "grade_level": grade_level,
        "pay_scale_group": f"PSG-{max(1, min(8, grade_level - 5))}",
        "fte": fte,
        "standard_hours": standard_hours,
        "employment_type": employment_type,
        "employment_status": employment_status,
        "hire_date": hire_date,
        "termination_date": None if employment_status == "active" else "2025-12-31",
        "manager_id": manager_id,
        "performance_rating": performance_rating,
        "attrition_risk": attrition_risk,
        "high_potential": high_potential,
        "key_position": key_position,
        "future_leader": future_leader,
        "promotion_likelihood": promotion_likelihood,
        "skills": skills,
        "project_keywords": [skills[0], "流程优化", "数据分析"],
        "resume_summary": summary,
        "updated_at": TODAY.isoformat(),
        "source_system": "demo_generator",
        "assignments": [
            {
                "assignment_id": f"A-{employee_id}-001",
                "department_code": department_code,
                "department_name": department_name,
                "office_city": office_city,
                "office_location": office_location,
                "job_title": job_title,
                "job_family": job_family,
                "grade": grade,
                "grade_level": grade_level,
                "start_date": hire_date,
                "end_date": None,
                "is_primary": True,
                "assignment_status": "current" if employment_status == "active" else "ended",
            }
        ],
    }


def fixed_seed_employees() -> List[Dict[str, Any]]:
    return [
        employee_doc(
            employee_id="200001",
            name="Grace Lee",
            gender="女",
            marital_status="已婚",
            city=CITIES[1],
            department=DEPARTMENTS[5],
            job_title="VP Operations",
            grade_level=15,
            fte=1.0,
            standard_hours=40,
            employment_type="固定职工",
            employment_status="active",
            hire_date="2010-08-01",
            manager_id="",
            email="grace.lee@example.com",
            performance_rating="高",
            attrition_risk="低",
            high_potential=False,
            key_position=True,
            future_leader=False,
            promotion_likelihood="一般",
        ),
        employee_doc(
            employee_id="107009",
            name="James Kwok",
            gender="男",
            marital_status="已婚",
            city=CITIES[0],
            department=DEPARTMENTS[0],
            job_title="制造总监",
            grade_level=12,
            fte=1.0,
            standard_hours=40,
            employment_type="固定职工",
            employment_status="active",
            hire_date="2014-01-06",
            manager_id="200001",
            email="james.kwok@example.com",
            performance_rating="高",
            attrition_risk="低",
            high_potential=False,
            key_position=True,
            future_leader=False,
            promotion_likelihood="可能",
        ),
        employee_doc(
            employee_id="300001",
            name="陈七",
            gender="男",
            marital_status="已婚",
            city=CITIES[2],
            department=DEPARTMENTS[1],
            job_title="研发总监",
            grade_level=13,
            fte=1.0,
            standard_hours=40,
            employment_type="固定职工",
            employment_status="active",
            hire_date="2012-04-15",
            manager_id="200001",
            email="chenqi@example.com",
            performance_rating="高",
            attrition_risk="低",
            high_potential=False,
            key_position=True,
            future_leader=False,
            promotion_likelihood="一般",
        ),
        employee_doc(
            employee_id="100001",
            name="张三",
            gender="男",
            marital_status="已婚",
            city=CITIES[0],
            department=DEPARTMENTS[0],
            job_title="高级工程师",
            grade_level=8,
            fte=1.0,
            standard_hours=40,
            employment_type="固定职工",
            employment_status="active",
            hire_date="2018-03-12",
            manager_id="107009",
            email="zhangsan@example.com",
            performance_rating="高",
            attrition_risk="低",
            high_potential=True,
            key_position=True,
            future_leader=False,
            promotion_likelihood="很可能",
        ),
        employee_doc(
            employee_id="100002",
            name="李四",
            gender="女",
            marital_status="未婚",
            city=CITIES[0],
            department=DEPARTMENTS[0],
            job_title="工程师",
            grade_level=7,
            fte=1.0,
            standard_hours=40,
            employment_type="固定职工",
            employment_status="active",
            hire_date="2023-07-01",
            manager_id="107009",
            email="lisi@example.com",
            performance_rating="中",
            attrition_risk="中",
            high_potential=False,
            key_position=False,
            future_leader=False,
            promotion_likelihood="可能",
        ),
        employee_doc(
            employee_id="100007",
            name="周八",
            gender="女",
            marital_status="已婚",
            city=CITIES[0],
            department=DEPARTMENTS[4],
            job_title="HRBP",
            grade_level=8,
            fte=1.0,
            standard_hours=40,
            employment_type="固定职工",
            employment_status="terminated",
            hire_date="2021-03-01",
            manager_id="999999",
            email="invalid-email",
            performance_rating="低",
            attrition_risk="高",
            high_potential=False,
            key_position=False,
            future_leader=False,
            promotion_likelihood="一般",
        ),
    ]


def random_name(rng: random.Random, index: int) -> str:
    if rng.random() < 0.78:
        return f"{rng.choice(FIRST_NAMES_CN)}{rng.choice(LAST_NAMES_CN)}{index}"
    return f"{rng.choice(EN_NAMES)} {index}"


def random_hire_date(rng: random.Random) -> str:
    days = rng.randint(30, 15 * 365)
    return (TODAY - timedelta(days=days)).isoformat()


def generate_employees(count: int, *, seed: int, with_quality_cases: bool) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    employees = fixed_seed_employees()
    manager_ids = ["200001", "107009", "300001"]
    next_id = 100100
    while len(employees) < count:
        department = rng.choice(DEPARTMENTS[:-1])
        city = rng.choice(CITIES)
        grade_level = rng.randint(3, 11)
        if rng.random() < 0.12:
            grade_level = rng.randint(10, 14)
        employee_id = str(next_id)
        next_id += 1
        status = rng.choices(["active", "terminated", "retired"], weights=[86, 10, 4], k=1)[0]
        gender = rng.choice(["男", "女"])
        job_title = {
            "MANU": rng.choice(["工程师", "高级工程师", "生产主管", "质量工程师"]),
            "RND": rng.choice(["软件工程师", "AI 工程师", "平台工程师", "架构师"]),
            "SALES": rng.choice(["销售代表", "客户经理", "销售经理"]),
            "FIN": rng.choice(["财务分析师", "会计", "预算专员"]),
            "HR": rng.choice(["HRBP", "招聘专员", "组织发展顾问"]),
        }.get(department[0], "员工")
        manager_id = rng.choice(manager_ids)
        if with_quality_cases and len(employees) == count - 1:
            manager_id = "999998"
        email = f"user{employee_id}@example.com"
        if with_quality_cases and len(employees) == count - 2:
            email = f"bad-email-{employee_id}"
        employees.append(
            employee_doc(
                employee_id=employee_id,
                name=random_name(rng, len(employees)),
                gender=gender,
                marital_status=rng.choice(["已婚", "未婚"]),
                city=city,
                department=department,
                job_title=job_title,
                grade_level=grade_level,
                fte=rng.choice([1.0, 1.0, 1.0, 0.5]),
                standard_hours=40,
                employment_type=rng.choice(["固定职工", "固定职工", "计时工", "合同工"]),
                employment_status=status,
                hire_date=random_hire_date(rng),
                manager_id=manager_id,
                email=email,
                performance_rating=rng.choice(["高", "中", "低"]),
                attrition_risk=rng.choice(["低", "中", "高"]),
                high_potential=rng.random() < 0.18,
                key_position=rng.random() < 0.22,
                future_leader=rng.random() < 0.12,
                promotion_likelihood=rng.choice(["很可能", "可能", "一般"]),
            )
        )
    return employees[:count]


def generate_chunks(employees: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, employee in enumerate(employees):
        resume_id = f"R-{employee['employee_id']}-v1"
        chunks = [
            ("profile", "个人概览", employee["resume_summary"]),
            ("skill", "技能摘要", f"熟悉 {'、'.join(employee['skills'])}，可承担 {employee['job_family']} 相关工作。"),
            ("project", "项目经历", f"参与 {employee['project_keywords'][0]} 项目，负责流程优化、数据分析和跨团队协作。"),
        ]
        for chunk_index, (chunk_type, title, text) in enumerate(chunks, start=1):
            rows.append(
                {
                    "chunk_id": f"{employee['employee_id']}-{chunk_type}-{chunk_index}",
                    "employee_id": employee["employee_id"],
                    "resume_id": resume_id,
                    "chunk_type": chunk_type,
                    "title": title,
                    "text": text,
                    "skills": employee["skills"],
                    "years_of_experience": round(max(0.5, (TODAY - date.fromisoformat(employee["hire_date"])).days / 365.25), 1),
                    "language": "en" if any(ord(ch) < 128 and ch.isalpha() for ch in employee["name"]) else "zh",
                    "source_version": "v1",
                    "source_updated_at": employee["updated_at"],
                    "embedding": mock_embedding(idx * 17 + chunk_index),
                }
            )
    return rows


def generate_assets(employees: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for employee in employees:
        resume_id = f"R-{employee['employee_id']}-v1"
        rows.append(
            {
                "asset_id": f"ASSET-{employee['employee_id']}-resume-v1",
                "employee_id": employee["employee_id"],
                "resume_id": resume_id,
                "resume_version": "v1",
                "file_name": f"{employee['name']}_{employee['employee_id']}_简历.pdf",
                "file_type": "pdf",
                "storage_uri": f"mock://hr-demo/resumes/{employee['employee_id']}.pdf",
                "asset_type": "redacted_resume",
                "language": "zh",
                "updated_at": employee["updated_at"],
                "access_level": "redacted",
                "source_system": "demo_generator",
            }
        )
    return rows


def generate_raw_quality_rows(employees: List[Dict[str, Any]], *, with_quality_cases: bool) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in employees]
    if with_quality_cases and employees:
        duplicate = dict(employees[0])
        duplicate["raw_record_id"] = "RAW-DUP-001"
        rows.append(duplicate)
    for idx, row in enumerate(rows):
        row.setdefault("raw_record_id", f"RAW-{idx + 1:06d}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HR ES demo JSONL data")
    parser.add_argument("--count", type=int, default=200, help="员工数量，建议 100-1000")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out-dir", default="data/demo/hr", help="输出目录")
    parser.add_argument("--with-quality-cases", action="store_true", default=True, help="生成少量数据质量异常")
    parser.add_argument("--no-quality-cases", dest="with_quality_cases", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    employees = generate_employees(max(1, args.count), seed=args.seed, with_quality_cases=args.with_quality_cases)
    chunks = generate_chunks(employees)
    assets = generate_assets(employees)
    raw_rows = generate_raw_quality_rows(employees, with_quality_cases=args.with_quality_cases)

    counts = {
        DEFAULT_EMPLOYEE_FILE: write_jsonl(out_dir / DEFAULT_EMPLOYEE_FILE, employees),
        DEFAULT_RESUME_CHUNK_FILE: write_jsonl(out_dir / DEFAULT_RESUME_CHUNK_FILE, chunks),
        DEFAULT_RESUME_ASSET_FILE: write_jsonl(out_dir / DEFAULT_RESUME_ASSET_FILE, assets),
        DEFAULT_RAW_QUALITY_FILE: write_jsonl(out_dir / DEFAULT_RAW_QUALITY_FILE, raw_rows),
    }
    for file_name, count in counts.items():
        print(f"{out_dir / file_name}: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
