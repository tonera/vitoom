#!/usr/bin/env python3
"""HR demo Elasticsearch 索引管理脚本。

示例：
  python scripts/hr_demo_es_manage.py reset --data-dir data/demo/hr
  python scripts/hr_demo_es_manage.py import --url http://127.0.0.1:9200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.agent import settings as agent_settings  # noqa: E402
from backend.services.agent.tools.builtin.business_query_core.hr import executor as hr_es  # noqa: E402

DEFAULT_EMPLOYEE_FILE = hr_es.DEFAULT_EMPLOYEE_FILE
DEFAULT_EMPLOYEE_INDEX = agent_settings.get_hr_business_query_employee_index()
DEFAULT_ES_URL = agent_settings.get_hr_business_query_es_url() or hr_es.DEFAULT_ES_URL
DEFAULT_RESUME_ASSET_FILE = hr_es.DEFAULT_RESUME_ASSET_FILE
DEFAULT_RESUME_ASSET_INDEX = agent_settings.get_hr_business_query_resume_asset_index()
DEFAULT_RESUME_CHUNK_FILE = hr_es.DEFAULT_RESUME_CHUNK_FILE
DEFAULT_RESUME_CHUNK_INDEX = agent_settings.get_hr_business_query_resume_chunk_index()
STALE_DEMO_INDEX_NAMES = (
    "hr_employee_demo_v1",
    "hr_resume_chunk_demo_v1",
    "hr_resume_asset_demo_v1",
)
bulk_import = hr_es.bulk_import
es_request = hr_es.es_request
read_jsonl = hr_es.read_jsonl


def index_mappings() -> Dict[str, Dict[str, Any]]:
    mappings = hr_es.index_mappings()
    return {
        DEFAULT_EMPLOYEE_INDEX: mappings[hr_es.DEFAULT_EMPLOYEE_INDEX],
        DEFAULT_RESUME_CHUNK_INDEX: mappings[hr_es.DEFAULT_RESUME_CHUNK_INDEX],
        DEFAULT_RESUME_ASSET_INDEX: mappings[hr_es.DEFAULT_RESUME_ASSET_INDEX],
    }


def index_files(data_dir: Path) -> Tuple[Tuple[str, Path, str], ...]:
    return (
        (DEFAULT_EMPLOYEE_INDEX, data_dir / DEFAULT_EMPLOYEE_FILE, "employee_id"),
        (DEFAULT_RESUME_CHUNK_INDEX, data_dir / DEFAULT_RESUME_CHUNK_FILE, "chunk_id"),
        (DEFAULT_RESUME_ASSET_INDEX, data_dir / DEFAULT_RESUME_ASSET_FILE, "asset_id"),
    )


def create_indices(args: argparse.Namespace) -> None:
    for index_name, mapping in index_mappings().items():
        es_request(
            "PUT",
            args.url,
            f"/{index_name}",
            body=mapping,
            username=args.username,
            password=args.password,
            timeout=args.timeout,
        )
        print(f"created {index_name}")


def delete_indices(args: argparse.Namespace) -> None:
    for index_name in index_mappings().keys():
        delete_index(args, index_name)


def delete_stale_demo_indices(args: argparse.Namespace) -> None:
    for index_name in STALE_DEMO_INDEX_NAMES:
        delete_index(args, index_name)


def delete_index(args: argparse.Namespace, index_name: str) -> None:
    try:
        es_request(
            "DELETE",
            args.url,
            f"/{index_name}",
            username=args.username,
            password=args.password,
            timeout=args.timeout,
        )
        print(f"deleted {index_name}")
    except Exception as exc:
        if "404" in str(exc) or "index_not_found_exception" in str(exc):
            print(f"skip missing {index_name}")
            return
        raise


def delete_all_demo_indices(args: argparse.Namespace) -> None:
    delete_indices(args)
    delete_stale_demo_indices(args)


def assert_no_stale_demo_indices(args: argparse.Namespace) -> None:
    stale = []
    for index_name in STALE_DEMO_INDEX_NAMES:
        try:
            es_request(
                "HEAD",
                args.url,
                f"/{index_name}",
                username=args.username,
                password=args.password,
                timeout=args.timeout,
            )
            stale.append(index_name)
        except Exception as exc:
            if "404" in str(exc) or "index_not_found_exception" in str(exc):
                continue
            raise
    if stale:
        raise RuntimeError(f"stale HR demo indices still exist: {', '.join(stale)}")


def import_indices(args: argparse.Namespace, *, hot_update: bool = False) -> None:
    data_dir = Path(args.data_dir)
    for index_name, path, id_field in index_files(data_dir):
        rows = list(read_jsonl(path))
        if not rows:
            print(f"skip empty {path}")
            continue
        response = bulk_import(
            url=args.url,
            index_name=index_name,
            rows=rows,
            id_field=id_field,
            username=args.username,
            password=args.password,
            timeout=args.timeout,
        )
        errors = bool(response.get("errors")) if isinstance(response, dict) else False
        mode = "hot-updated" if hot_update else "imported"
        print(f"{mode} {len(rows)} rows into {index_name}; errors={errors}")
        if errors:
            raise RuntimeError(f"bulk import reported errors for {index_name}")


def refresh_indices(args: argparse.Namespace) -> None:
    names = ",".join(index_mappings().keys())
    es_request(
        "POST",
        args.url,
        f"/{names}/_refresh",
        username=args.username,
        password=args.password,
        timeout=args.timeout,
    )
    print(f"refreshed {names}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage HR demo Elasticsearch indices")
    parser.add_argument(
        "command",
        choices=["create", "reset", "import", "refresh", "hot-update", "delete-demo", "delete-old"],
        help="操作类型",
    )
    parser.add_argument("--url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--username", default="", help="ES Basic Auth 用户名")
    parser.add_argument("--password", default="", help="ES Basic Auth 密码")
    parser.add_argument("--timeout", type=float, default=60.0, help="请求超时秒数")
    parser.add_argument("--data-dir", default="data/demo/hr", help="JSONL 数据目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create":
        create_indices(args)
    elif args.command == "reset":
        delete_all_demo_indices(args)
        create_indices(args)
        import_indices(args)
        refresh_indices(args)
        assert_no_stale_demo_indices(args)
    elif args.command == "import":
        import_indices(args)
    elif args.command == "refresh":
        refresh_indices(args)
    elif args.command == "hot-update":
        import_indices(args, hot_update=True)
        refresh_indices(args)
    elif args.command == "delete-demo":
        delete_all_demo_indices(args)
    elif args.command == "delete-old":
        delete_stale_demo_indices(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
