from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalHit:
    chunk_id: str
    document_id: str
    score: float
    source: Dict[str, Any]
    rank: int = 0
    source_name: str = ""

    @property
    def title(self) -> str:
        return str(self.source.get("title") or self.source.get("file_name") or "").strip()

    @property
    def text(self) -> str:
        return str(self.source.get("text") or self.source.get("metadata_text") or "").strip()

    @property
    def file_name(self) -> str:
        return str(self.source.get("file_name") or self.title or self.document_id).strip()


@dataclass
class QueryContext:
    user_id: str = "agent-system"
    tenant_id: str = "default"
    group_ids: List[str] = field(default_factory=list)


@dataclass
class KnowledgeBaseConfig:
    enabled: bool
    es_url: str
    es_username: str
    es_password: str
    document_index: str
    chunk_index: str
    request_timeout_seconds: float
    bm25_top_k: int
    vector_top_k: int
    vector_num_candidates: int
    rrf_k: int
    merged_top_k: int
    rerank_enabled: bool
    rerank_backend: str
    rerank_top_n: int
    final_top_k: int
    answer_max_context_chars: int
    include_sources: bool
    include_debug: bool


@dataclass
class ParsedDocument:
    document_id: str
    title: str
    text: str
    metadata_text: str
    sections: List[Dict[str, Any]] = field(default_factory=list)
    parse_status: str = "completed"
    parse_error: str = ""
    derived_markdown_path: str = ""


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_type: str
    title: str
    text: str
    metadata_text: str
    section_path: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    language: str = "zh"

