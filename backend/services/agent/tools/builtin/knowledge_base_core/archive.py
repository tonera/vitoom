from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import httpx

from backend.services.agent import settings as agent_settings
from backend.services.agent.tools.builtin.business_query_core.planner_base import run_agent_planner_completion
from backend.utils import safe_filename

from .classifier import classify_source_row
from .es_client import KnowledgeBaseEsClient
from .ingest import ensure_indices, ingest_rows, is_media_manifest_row
from .source_manifest import upsert_manifest_rows
from .source_organizer import is_skipped_media_file

ARCHIVE_SUBDIR = "用户归档"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _content_disposition_filename(header: str) -> str:
    if not header:
        return ""
    message = Message()
    message["content-disposition"] = header
    filename = message.get_param("filename", header="content-disposition") or ""
    return str(filename or "").strip()


def _guess_url_filename(url: str, *, content_type: str = "", disposition: str = "") -> str:
    from_header = _content_disposition_filename(disposition)
    if from_header:
        return from_header
    parsed = urlparse(url)
    name = Path(unquote(parsed.path or "")).name
    if name:
        return name
    extension = mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else ""
    return f"archived-file{extension or ''}"


def _archive_dir(*, user_id: str, source_kind: str) -> Path:
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = agent_settings.get_knowledge_base_canonical_root()
    safe_user = safe_filename(user_id or "anonymous") or "anonymous"
    return base / ARCHIVE_SUBDIR / safe_user / source_kind / date_part


def _metadata_row_for_file(path: Path, *, source_url: str = "", source_kind: str, user_id: str, title: str = "") -> Dict[str, Any]:
    stat = path.stat()
    digest = _sha256_file(path)
    document_id = "doc_" + hashlib.sha1(digest.encode("ascii")).hexdigest()[:20]
    extension = path.suffix.lower()
    return {
        "document_id": document_id,
        "document_group_id": document_id,
        "source_path": str(path),
        "canonical_path": str(path),
        "source_uri": source_url,
        "sha256": digest,
        "size_bytes": stat.st_size,
        "file_name": path.name,
        "file_stem": path.stem,
        "title": title or path.stem,
        "extension": extension,
        "mime_type": mimetypes.guess_type(str(path))[0] or "",
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "source_root": str(agent_settings.get_knowledge_base_canonical_root()),
        "relative_source_path": str(path.relative_to(agent_settings.get_knowledge_base_canonical_root())),
        "domain": "未分类_待确认",
        "topic": "",
        "subtopic": "",
        "tags": [source_kind, "用户归档"],
        "is_duplicate": False,
        "duplicate_of": "",
        "version_rank": 1,
        "version_label": "latest",
        "tenant_id": "default",
        "owner_user_id": user_id or "agent-system",
        "access_level": "public",
        "active": True,
        "deleted": False,
        "archived": False,
    }


def download_url_to_archive(url: str, *, user_id: str, timeout_seconds: float = 60.0) -> Path:
    target_dir = _archive_dir(user_id=user_id, source_kind="url")
    target_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = str(response.headers.get("content-type") or "")
        filename = safe_filename(_guess_url_filename(url, content_type=content_type, disposition=str(response.headers.get("content-disposition") or "")))
        if not filename:
            filename = "archived-file"
        target = target_dir / filename
        content = response.content
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"文件过大，超过 {MAX_DOWNLOAD_BYTES} bytes")
        target.write_bytes(content)
    if is_skipped_media_file(target):
        target.unlink(missing_ok=True)
        raise RuntimeError("音频、视频、图片类文件当前不入库，已跳过。")
    return target


def save_markdown_to_archive(markdown: str, *, user_id: str, title: str = "") -> Path:
    target_dir = _archive_dir(user_id=user_id, source_kind="conversation")
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(title or datetime.now(timezone.utc).strftime("conversation-%Y%m%d-%H%M%S")) or "conversation"
    target = target_dir / f"{Path(stem).stem}.md"
    target.write_text(str(markdown or "").strip() + "\n", encoding="utf-8")
    return target


def summarize_conversation_to_markdown(content: str, *, title: str = "", user_id: str = "") -> str:
    messages = [
        {
            "role": "system",
            "content": "你是知识库归档助手。将给定对话整理为结构化 Markdown，保留关键事实、决策、操作步骤和待办，不要编造。",
        },
        {
            "role": "user",
            "content": f"标题：{title or '对话归档'}\n\n对话内容：\n{content}",
        },
    ]
    return run_agent_planner_completion(messages, user_id=user_id, error_label="knowledge archive conversation summary")


def archive_file_path(
    path: Path,
    *,
    user_id: str,
    source_url: str = "",
    source_kind: str = "url",
    title: str = "",
    classify: bool = True,
    knowledge_base_id: str = "default",
    refresh: bool = True,
    client: Optional[KnowledgeBaseEsClient] = None,
) -> Dict[str, Any]:
    if is_skipped_media_file(path):
        raise RuntimeError("音频、视频、图片类文件当前不入库，已跳过。")
    row = _metadata_row_for_file(path, source_url=source_url, source_kind=source_kind, user_id=user_id, title=title)
    if classify:
        result = classify_source_row(row, user_id=user_id)
        existing_tags = list(row.get("tags") or [])
        row.update(result)
        row["tags"] = sorted(set(existing_tags + list(result.get("tags") or [])))
    if is_media_manifest_row(row):
        raise RuntimeError("音频、视频、图片类文件当前不入库，已跳过。")

    manifest_path = agent_settings.get_knowledge_base_manifest_path()
    upsert_manifest_rows(manifest_path, [row], key_field="document_id")
    es_client = client or KnowledgeBaseEsClient(
        url=agent_settings.get_knowledge_base_es_url(),
        username=agent_settings.get_knowledge_base_es_username(),
        password=agent_settings.get_knowledge_base_es_password(),
        timeout=agent_settings.get_knowledge_base_request_timeout_seconds(),
    )
    ensure_indices(
        es_client,
        document_index=agent_settings.get_knowledge_base_document_index(),
        chunk_index=agent_settings.get_knowledge_base_chunk_index(),
        dims=agent_settings.get_knowledge_base_embedding_dims(),
    )
    ingest_summary = ingest_rows(
        [row],
        client=es_client,
        document_index=agent_settings.get_knowledge_base_document_index(),
        chunk_index=agent_settings.get_knowledge_base_chunk_index(),
        knowledge_base_id=knowledge_base_id,
    )
    if refresh:
        es_client.request("POST", f"/{agent_settings.get_knowledge_base_document_index()},{agent_settings.get_knowledge_base_chunk_index()}/_refresh")
    return {"document": row, "ingest": ingest_summary, "manifest_path": str(manifest_path)}


def archive_url(url: str, *, user_id: str, title: str = "", classify: bool = True, knowledge_base_id: str = "default") -> Dict[str, Any]:
    path = download_url_to_archive(url, user_id=user_id)
    return archive_file_path(path, user_id=user_id, source_url=url, source_kind="url", title=title, classify=classify, knowledge_base_id=knowledge_base_id)


def archive_conversation(
    content: str,
    *,
    user_id: str,
    title: str = "",
    summarize: bool = True,
    classify: bool = True,
    knowledge_base_id: str = "default",
) -> Dict[str, Any]:
    markdown = summarize_conversation_to_markdown(content, title=title, user_id=user_id) if summarize else content
    path = save_markdown_to_archive(markdown, user_id=user_id, title=title)
    return archive_file_path(path, user_id=user_id, source_kind="conversation", title=title, classify=classify, knowledge_base_id=knowledge_base_id)
