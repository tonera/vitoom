#!/usr/bin/env python3
"""Knowledge base management CLI.

Examples:
  python scripts/knowledge_base_manage.py organize --scan-root docs --dry-run
  python scripts/knowledge_base_manage.py create
  python scripts/knowledge_base_manage.py clear --yes
  python scripts/knowledge_base_manage.py ingest --manifest resources/knowledge_sources/manifest.jsonl
  python scripts/knowledge_base_manage.py refresh
  python scripts/knowledge_base_manage.py smoke --smoke-query Vitoom
  python scripts/knowledge_base_manage.py query "知识库里有没有 Vitoom 架构设计？" --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=r"urllib3 .* doesn't match a supported version!")
for _logger_name in ("pdfminer", "pdfminer.pdffont", "fontTools"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.agent import settings as agent_settings  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_query import run_knowledge_base_query  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.es_client import KnowledgeBaseEsClient  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.ingest import ensure_indices, ingest_manifest  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.source_organizer import organize_sources, stderr_progress  # noqa: E402


def _quiet_noisy_dependencies() -> None:
    try:
        from requests.exceptions import RequestsDependencyWarning  # type: ignore[import-not-found]

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except Exception:
        pass
    for logger_name in ("pdfminer", "pdfminer.pdffont", "fontTools"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _json_print(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _default_client(args: argparse.Namespace) -> KnowledgeBaseEsClient:
    return KnowledgeBaseEsClient(url=args.url, username=args.username, password=args.password, timeout=args.timeout)


def _scan_roots(args: argparse.Namespace) -> List[Path]:
    configured = [Path(item) for item in agent_settings.get_knowledge_base_source_scan_roots()]
    explicit = [Path(item) for item in args.scan_root]
    roots = explicit or configured
    if not roots:
        raise RuntimeError("No scan roots configured. Pass --scan-root or set knowledge_base.source_organizer.scan_roots.")
    return roots


def cmd_organize(args: argparse.Namespace) -> None:
    summary = organize_sources(
        _scan_roots(args),
        canonical_root=Path(args.canonical_root),
        manifest_path=Path(args.manifest),
        scan_state_path=Path(args.scan_state),
        copy_files=not args.no_copy,
        skip_previously_scanned_dirs=not args.no_skip_state,
        classify=args.classify,
        low_confidence_threshold=args.low_confidence_threshold,
        classifier_user_id=args.classifier_user_id,
        progress_callback=None if args.quiet else stderr_progress,
        progress_every=args.progress_every,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        max_files=args.max_files,
    )
    _json_print(summary)


def cmd_create(args: argparse.Namespace) -> None:
    ensure_indices(
        _default_client(args),
        document_index=args.document_index,
        chunk_index=args.chunk_index,
        dims=args.embedding_dims,
    )
    _json_print({"created_or_exists": [args.document_index, args.chunk_index]})


def cmd_clear(args: argparse.Namespace) -> None:
    if not args.yes:
        raise RuntimeError("clear 会清空知识库 ES 数据。请显式传入 --yes 确认。")
    client = _default_client(args)
    result: Dict[str, Any] = {}
    for index_name in (args.document_index, args.chunk_index):
        response = client.request(
            "POST",
            f"/{index_name}/_delete_by_query?conflicts=proceed&refresh=true",
            body={"query": {"match_all": {}}},
        )
        if isinstance(response, dict):
            result[index_name] = {
                "deleted": response.get("deleted", 0),
                "version_conflicts": response.get("version_conflicts", 0),
                "failures": response.get("failures", []),
            }
        else:
            result[index_name] = response
    _json_print({"cleared": result})


def cmd_ingest(args: argparse.Namespace) -> None:
    if args.ensure_indices:
        cmd_create(args)
    summary = ingest_manifest(
        Path(args.manifest),
        client=_default_client(args),
        document_index=args.document_index,
        chunk_index=args.chunk_index,
        knowledge_base_id=args.knowledge_base_id,
        dry_run=args.dry_run,
        progress_callback=None if args.quiet else stderr_progress,
        progress_every=args.progress_every,
    )
    _json_print(summary)


def cmd_refresh(args: argparse.Namespace) -> None:
    client = _default_client(args)
    names = f"{args.document_index},{args.chunk_index}"
    client.request("POST", f"/{names}/_refresh")
    _json_print({"refreshed": names})


def _count(client: KnowledgeBaseEsClient, index_name: str) -> int:
    response = client.request("GET", f"/{index_name}/_count")
    if isinstance(response, dict):
        return int(response.get("count") or 0)
    return 0


def cmd_smoke(args: argparse.Namespace) -> None:
    client = _default_client(args)
    query = str(args.smoke_query or args.query or "知识库").strip()
    search_body = {
        "size": 3,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["file_name^4", "title^3", "metadata_text^2", "text"],
            }
        },
    }
    response = client.search(args.chunk_index, search_body)
    hits = (response.get("hits") or {}).get("hits") if isinstance(response, dict) else []
    samples = []
    for item in hits or []:
        source = item.get("_source") if isinstance(item, dict) else {}
        if isinstance(source, dict):
            samples.append(
                {
                    "score": item.get("_score"),
                    "chunk_id": source.get("chunk_id"),
                    "document_id": source.get("document_id"),
                    "file_name": source.get("file_name"),
                    "chunk_type": source.get("chunk_type"),
                }
            )
    _json_print(
        {
            "document_count": _count(client, args.document_index),
            "chunk_count": _count(client, args.chunk_index),
            "smoke_query": query,
            "sample_hits": samples,
        }
    )


def cmd_query(args: argparse.Namespace) -> None:
    print(
        run_knowledge_base_query(
            args.query,
            top_k=args.top_k,
            include_sources=True,
            include_debug=args.debug,
            knowledge_base_id=args.knowledge_base_id,
            user_id=args.user_id,
            tenant_id=args.tenant_id,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage local knowledge base sources and Elasticsearch indices")
    parser.add_argument("command", choices=["organize", "create", "clear", "ingest", "refresh", "smoke", "query"], help="操作类型")
    parser.add_argument("query", nargs="?", default="", help="query 命令的自然语言问题")
    parser.add_argument("--url", default=agent_settings.get_knowledge_base_es_url(), help="Elasticsearch URL")
    parser.add_argument("--username", default=agent_settings.get_knowledge_base_es_username(), help="ES Basic Auth 用户名")
    parser.add_argument("--password", default=agent_settings.get_knowledge_base_es_password(), help="ES Basic Auth 密码")
    parser.add_argument("--timeout", type=float, default=agent_settings.get_knowledge_base_request_timeout_seconds(), help="请求超时秒数")
    parser.add_argument("--document-index", default=agent_settings.get_knowledge_base_document_index(), help="文档索引名")
    parser.add_argument("--chunk-index", default=agent_settings.get_knowledge_base_chunk_index(), help="Chunk 索引名")
    parser.add_argument("--embedding-dims", type=int, default=agent_settings.get_knowledge_base_embedding_dims(), help="Embedding 维度")
    parser.add_argument("--manifest", default=str(agent_settings.get_knowledge_base_manifest_path()), help="manifest JSONL 路径")
    parser.add_argument("--scan-state", default=str(agent_settings.get_knowledge_base_scan_state_path()), help="扫描状态 JSON 路径")
    parser.add_argument("--canonical-root", default=str(agent_settings.get_knowledge_base_canonical_root()), help="规范源文件仓库根目录")
    parser.add_argument("--scan-root", action="append", default=[], help="可重复传入的源文件扫描根目录")
    parser.add_argument("--no-copy", action="store_true", help="只生成 manifest，不复制文件到 canonical root")
    parser.add_argument("--no-skip-state", action="store_true", help="忽略 scan state，强制重新扫描")
    parser.add_argument("--classify", action="store_true", help="organize 时调用内部 LLM 对源文件分类")
    parser.add_argument("--classifier-user-id", default="", help="LLM 分类使用的有效用户 ID；为空时自动选择一个 active/admin 用户")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.75, help="LLM 分类低置信阈值")
    parser.add_argument("--progress-every", type=int, default=10, help="organize 进度输出间隔，按文件数计")
    parser.add_argument("--quiet", action="store_true", help="关闭 organize 进度输出")
    parser.add_argument("--no-resume", action="store_true", help="关闭默认断点续跑，重新分类已完成文件")
    parser.add_argument("--max-files", type=int, default=0, help="最多扫描文件数，0 表示不限")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写 manifest/ES")
    parser.add_argument("--ensure-indices", action="store_true", help="ingest 前先创建缺失索引")
    parser.add_argument("--yes", action="store_true", help="确认执行危险操作，例如 clear")
    parser.add_argument("--knowledge-base-id", default="default", help="知识库 ID")
    parser.add_argument("--smoke-query", default="知识库", help="smoke 命令使用的基础检索词")
    parser.add_argument("--top-k", type=int, default=8, help="query 命令最终来源数量")
    parser.add_argument("--debug", action="store_true", help="query 命令输出 debug")
    parser.add_argument("--user-id", default="agent-system", help="query 权限上下文用户 ID")
    parser.add_argument("--tenant-id", default="default", help="query 权限上下文租户 ID")
    return parser.parse_args()


def main() -> int:
    _quiet_noisy_dependencies()
    args = parse_args()
    if args.command == "organize":
        cmd_organize(args)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "refresh":
        cmd_refresh(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "query":
        if not args.query.strip():
            raise RuntimeError("query command requires a natural-language question")
        cmd_query(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
