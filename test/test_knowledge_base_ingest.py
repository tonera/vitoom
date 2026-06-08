from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin.knowledge_base_core.chunker import chunk_document  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.document_parser import parse_document  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.ingest import ingest_manifest  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.models import ParsedDocument  # noqa: E402


class _NoopClient:
    def request(self, *_args, **_kwargs):
        raise AssertionError("dry_run should not write to Elasticsearch")


def _write_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_parse_markdown_document_and_chunk_body(tmp_path):
    source = tmp_path / "Vitoom架构设计.md"
    source.write_text("# 架构\n\n系统采用检索、规划、执行分层。", encoding="utf-8")

    parsed = parse_document(
        {
            "document_id": "doc_1",
            "canonical_path": str(source),
            "file_name": source.name,
            "file_stem": source.stem,
            "extension": ".md",
            "domain": "项目",
            "topic": "架构设计",
        }
    )
    chunks = chunk_document(parsed, chunk_size=200, overlap=20)

    assert parsed.parse_status == "completed"
    assert "检索、规划、执行" in parsed.text
    assert chunks[0].chunk_type == "body"
    assert chunks[0].metadata_text


def test_metadata_only_document_still_generates_searchable_chunk():
    parsed = ParsedDocument(
        document_id="doc_zip",
        title="项目交付资料",
        text="",
        metadata_text="文件名：项目交付资料.zip\n路径：项目/交付资料.zip",
        parse_status="metadata_only",
    )

    chunks = chunk_document(parsed)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "metadata"
    assert "项目交付资料.zip" in chunks[0].metadata_text


def test_ingest_manifest_dry_run_counts_documents_and_chunks(tmp_path):
    source = tmp_path / "报销流程.md"
    source.write_text("交通费报销需要发票、行程单和审批记录。", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        {
            "document_id": "doc_expense",
            "document_group_id": "doc_expense",
            "canonical_path": str(source),
            "file_name": source.name,
            "file_stem": source.stem,
            "extension": ".md",
            "tenant_id": "default",
            "owner_user_id": "agent-system",
            "access_level": "public",
            "active": True,
            "deleted": False,
            "archived": False,
        },
    )

    summary = ingest_manifest(
        manifest,
        client=_NoopClient(),
        document_index="kb_document_v1",
        chunk_index="kb_chunk_v1",
        dry_run=True,
    )

    assert summary == {"documents": 1, "chunks": 1, "skipped_media": 0, "dry_run": True}


def test_ingest_manifest_skips_media_rows_from_existing_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "document_id": "doc_image",
            "document_group_id": "doc_image",
            "canonical_path": str(tmp_path / "photo.png"),
            "file_name": "photo.png",
            "file_stem": "photo",
            "extension": ".png",
            "mime_type": "image/png",
        },
        {
            "document_id": "doc_doc",
            "document_group_id": "doc_doc",
            "canonical_path": str(tmp_path / "notes.md"),
            "file_name": "notes.md",
            "file_stem": "notes",
            "extension": ".md",
        },
    ]
    (tmp_path / "notes.md").write_text("非媒体文件", encoding="utf-8")
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    summary = ingest_manifest(
        manifest,
        client=_NoopClient(),
        document_index="kb_document_v1",
        chunk_index="kb_chunk_v1",
        dry_run=True,
    )

    assert summary["documents"] == 1
    assert summary["chunks"] == 1
    assert summary["skipped_media"] == 1


def test_ingest_manifest_reports_progress(tmp_path):
    source = tmp_path / "资料.md"
    source.write_text("知识库入库进度测试。", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        {
            "document_id": "doc_progress",
            "document_group_id": "doc_progress",
            "canonical_path": str(source),
            "file_name": source.name,
            "file_stem": source.stem,
            "extension": ".md",
            "tenant_id": "default",
            "owner_user_id": "agent-system",
            "access_level": "public",
            "active": True,
            "deleted": False,
            "archived": False,
        },
    )
    events = []

    ingest_manifest(
        manifest,
        client=_NoopClient(),
        document_index="kb_document_v1",
        chunk_index="kb_chunk_v1",
        dry_run=True,
        progress_callback=lambda event, payload: events.append((event, payload)),
        progress_every=1,
    )

    event_names = [event for event, _payload in events]
    assert "ingest_start" in event_names
    assert "ingest" in event_names
    assert "ingest_done" in event_names
