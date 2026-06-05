from __future__ import annotations

import hashlib
from typing import List

from .models import ChunkRecord, ParsedDocument


def _chunk_id(document_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"chk_{digest}"


def split_text(text: str, *, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    value = str(text or "").strip()
    if not value:
        return []
    size = max(200, chunk_size)
    step = max(1, size - max(0, min(overlap, size // 2)))
    chunks: List[str] = []
    start = 0
    while start < len(value):
        chunks.append(value[start : start + size].strip())
        start += step
    return [item for item in chunks if item]


def chunk_document(document: ParsedDocument, *, chunk_size: int = 900, overlap: int = 120) -> List[ChunkRecord]:
    text_chunks = split_text(document.text, chunk_size=chunk_size, overlap=overlap)
    records: List[ChunkRecord] = []
    if not text_chunks:
        metadata = document.metadata_text or document.title
        return [
            ChunkRecord(
                chunk_id=_chunk_id(document.document_id, 0, metadata),
                document_id=document.document_id,
                chunk_index=0,
                chunk_type="metadata",
                title=document.title,
                text="",
                metadata_text=metadata,
            )
        ]
    for index, text in enumerate(text_chunks):
        records.append(
            ChunkRecord(
                chunk_id=_chunk_id(document.document_id, index, text),
                document_id=document.document_id,
                chunk_index=index,
                chunk_type="body",
                title=document.title,
                text=text,
                metadata_text=document.metadata_text,
            )
        )
    return records
