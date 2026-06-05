from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .models import ParsedDocument


TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm"}
METADATA_ONLY_EXTENSIONS = {".zip", ".key"}
MARKITDOWN_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm"}


def metadata_text_from_manifest(row: Dict[str, Any]) -> str:
    parts = [
        f"文件名：{row.get('file_name') or Path(str(row.get('canonical_path') or row.get('source_path') or '')).name}",
        f"路径：{row.get('canonical_path') or row.get('source_path') or ''}",
        f"分类：{row.get('domain') or ''}/{row.get('topic') or ''}/{row.get('subtopic') or ''}",
        f"标签：{', '.join(row.get('tags') or []) if isinstance(row.get('tags'), list) else row.get('tags') or ''}",
    ]
    return "\n".join(item for item in parts if item and not item.endswith("："))


def parse_document(row: Dict[str, Any]) -> ParsedDocument:
    path = Path(str(row.get("canonical_path") or row.get("source_path") or ""))
    document_id = str(row.get("document_id") or "").strip()
    title = str(row.get("title") or row.get("file_stem") or path.stem or document_id).strip()
    metadata_text = metadata_text_from_manifest(row)
    extension = str(row.get("extension") or path.suffix).lower()
    if extension in METADATA_ONLY_EXTENSIONS:
        return ParsedDocument(document_id=document_id, title=title, text="", metadata_text=metadata_text, parse_status="metadata_only")
    if extension in TEXT_EXTENSIONS and path.exists():
        try:
            return ParsedDocument(document_id=document_id, title=title, text=path.read_text(encoding="utf-8", errors="ignore"), metadata_text=metadata_text)
        except Exception as exc:
            return ParsedDocument(document_id=document_id, title=title, text="", metadata_text=metadata_text, parse_status="failed", parse_error=str(exc))
    if extension in MARKITDOWN_EXTENSIONS and path.exists():
        try:
            from markitdown import MarkItDown  # type: ignore[import-not-found]

            result = MarkItDown().convert(str(path))
            text = str(getattr(result, "text_content", "") or "").strip()
            if text:
                return ParsedDocument(document_id=document_id, title=title, text=text, metadata_text=metadata_text)
            return ParsedDocument(document_id=document_id, title=title, text="", metadata_text=metadata_text, parse_status="metadata_only")
        except Exception as exc:
            return ParsedDocument(document_id=document_id, title=title, text="", metadata_text=metadata_text, parse_status="failed", parse_error=str(exc))
    return ParsedDocument(document_id=document_id, title=title, text="", metadata_text=metadata_text, parse_status="metadata_only")
