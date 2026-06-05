from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List

from backend.services.agent.embeddings import get_embedding_service

from .chunker import chunk_document
from .document_parser import parse_document
from .es_client import EsHttpError, KnowledgeBaseEsClient
from .mappings import kb_chunk_mapping, kb_document_mapping
from .source_manifest import read_manifest
from .source_organizer import MEDIA_EXTENSIONS, is_skipped_media_name

ProgressCallback = Callable[[str, Dict[str, Any]], None]


def build_bulk_body(index_name: str, rows: Iterable[Dict[str, Any]], *, id_field: str) -> tuple[bytes, int]:
    lines: List[str] = []
    count = 0
    for row in rows:
        doc_id = str(row.get(id_field) or "").strip()
        if not doc_id:
            raise ValueError(f"Missing id field {id_field!r} for {index_name}")
        lines.append(json.dumps({"index": {"_index": index_name, "_id": doc_id}}, ensure_ascii=False))
        lines.append(json.dumps(row, ensure_ascii=False))
        count += 1
    return (("\n".join(lines) + "\n").encode("utf-8") if lines else b"", count)


def _index_exists(client: KnowledgeBaseEsClient, index_name: str) -> bool:
    try:
        client.request("HEAD", f"/{index_name}")
        return True
    except EsHttpError as exc:
        if exc.status == 404:
            return False
        raise


def ensure_indices(client: KnowledgeBaseEsClient, *, document_index: str, chunk_index: str, dims: int = 384) -> None:
    for index_name, mapping in ((document_index, kb_document_mapping()), (chunk_index, kb_chunk_mapping(dims=dims))):
        if not _index_exists(client, index_name):
            client.request("PUT", f"/{index_name}", body=mapping)


def is_media_manifest_row(row: Dict[str, Any]) -> bool:
    file_name = str(row.get("file_name") or "")
    source_path = str(row.get("source_path") or "")
    canonical_path = str(row.get("canonical_path") or "")
    extension = str(row.get("extension") or "").lower()
    mime_type = str(row.get("mime_type") or "").lower()
    if extension in MEDIA_EXTENSIONS:
        return True
    if mime_type.startswith(("image/", "audio/", "video/")):
        return True
    return any(is_skipped_media_name(value) for value in (file_name, source_path, canonical_path))


def ingest_manifest(
    manifest_path: Path,
    *,
    client: KnowledgeBaseEsClient,
    document_index: str,
    chunk_index: str,
    knowledge_base_id: str = "default",
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 25,
) -> Dict[str, Any]:
    return ingest_rows(
        list(read_manifest(manifest_path)),
        client=client,
        document_index=document_index,
        chunk_index=chunk_index,
        knowledge_base_id=knowledge_base_id,
        dry_run=dry_run,
        progress_callback=progress_callback,
        progress_every=progress_every,
    )


def ingest_rows(
    manifest_rows: List[Dict[str, Any]],
    *,
    client: KnowledgeBaseEsClient,
    document_index: str,
    chunk_index: str,
    knowledge_base_id: str = "default",
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 25,
) -> Dict[str, Any]:
    started = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    documents: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    embedding_service = get_embedding_service()
    skipped_media = sum(1 for row in manifest_rows if not row.get("is_duplicate") and is_media_manifest_row(row))
    total = sum(1 for row in manifest_rows if not row.get("is_duplicate") and not is_media_manifest_row(row))
    if progress_callback:
        progress_callback("ingest_start", {"total": total, "skipped_media": skipped_media, "dry_run": dry_run, "elapsed_seconds": 0.0})
    processed = 0
    for row in manifest_rows:
        if row.get("is_duplicate"):
            continue
        if is_media_manifest_row(row):
            continue
        processed += 1
        parsed = parse_document(row)
        document = dict(row)
        document.update(
            {
                "title": parsed.title,
                "knowledge_base_id": knowledge_base_id,
                "ingested_at": now,
                "parse_status": parsed.parse_status,
                "parse_error": parsed.parse_error,
                "active": bool(document.get("active", True)),
                "deleted": bool(document.get("deleted", False)),
            }
        )
        documents.append(document)
        chunk_records = chunk_document(parsed)
        embedding_inputs = [
            f"标题：{item.title}\n文件名：{document.get('file_name') or ''}\n章节：{item.section_path}\n正文：{item.text}\n元数据：{item.metadata_text}"
            for item in chunk_records
        ]
        vectors = embedding_service.embed_documents(embedding_inputs) if embedding_service.is_ready() else [None for _ in chunk_records]
        for item, vector in zip(chunk_records, vectors):
            chunk = asdict(item)
            chunk.update(
                {
                    "document_group_id": document.get("document_group_id") or document.get("document_id"),
                    "knowledge_base_id": knowledge_base_id,
                    "file_name": document.get("file_name") or "",
                    "file_stem": document.get("file_stem") or "",
                    "domain": document.get("domain") or "",
                    "topic": document.get("topic") or "",
                    "tags": document.get("tags") or [],
                    "source_uri": document.get("source_uri") or "",
                    "canonical_path": document.get("canonical_path") or "",
                    "owner_user_id": document.get("owner_user_id") or "agent-system",
                    "tenant_id": document.get("tenant_id") or "default",
                    "access_level": document.get("access_level") or "public",
                    "allowed_user_ids": document.get("allowed_user_ids") or [],
                    "allowed_group_ids": document.get("allowed_group_ids") or [],
                    "archived": bool(document.get("archived", False)),
                    "active": bool(document.get("active", True)),
                    "deleted": bool(document.get("deleted", False)),
                    "embedding_model": "multilingual-e5-small-onnx" if vector else "",
                    "embedding": vector,
                    "ingested_at": now,
                    "source_modified_at": document.get("modified_at"),
                }
            )
            if vector is None:
                chunk.pop("embedding", None)
            chunks.append(chunk)
        if progress_callback and (processed == 1 or processed % max(1, progress_every) == 0 or processed == total):
            progress_callback(
                "ingest",
                {
                    "processed": processed,
                    "total": total,
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "parse_status": parsed.parse_status,
                    "elapsed_seconds": time.perf_counter() - started,
                    "current": document.get("file_name") or "",
                },
            )
    if dry_run:
        if progress_callback:
            progress_callback("ingest_done", {"documents": len(documents), "chunks": len(chunks), "skipped_media": skipped_media, "dry_run": True, "elapsed_seconds": time.perf_counter() - started})
        return {"documents": len(documents), "chunks": len(chunks), "skipped_media": skipped_media, "dry_run": True}
    if progress_callback:
        progress_callback("bulk_prepare", {"documents": len(documents), "chunks": len(chunks), "elapsed_seconds": time.perf_counter() - started})
    doc_body, doc_count = build_bulk_body(document_index, documents, id_field="document_id")
    chunk_body, chunk_count = build_bulk_body(chunk_index, chunks, id_field="chunk_id")
    if doc_count:
        if progress_callback:
            progress_callback("bulk_write", {"index": document_index, "documents": doc_count, "elapsed_seconds": time.perf_counter() - started})
        client.request("POST", "/_bulk", body=doc_body, headers={"Content-Type": "application/x-ndjson"})
    if chunk_count:
        if progress_callback:
            progress_callback("bulk_write", {"index": chunk_index, "chunks": chunk_count, "elapsed_seconds": time.perf_counter() - started})
        client.request("POST", "/_bulk", body=chunk_body, headers={"Content-Type": "application/x-ndjson"})
    if progress_callback:
        progress_callback("ingest_done", {"documents": doc_count, "chunks": chunk_count, "skipped_media": skipped_media, "dry_run": False, "elapsed_seconds": time.perf_counter() - started})
    return {"documents": doc_count, "chunks": chunk_count, "skipped_media": skipped_media, "dry_run": False}
