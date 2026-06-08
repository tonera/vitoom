from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from backend.services.agent.embeddings import get_embedding_service

from .es_client import KnowledgeBaseEsClient
from .models import KnowledgeBaseConfig, QueryContext, RetrievalHit
from .rrf import reciprocal_rank_fusion


def _coerce_hits(response: Dict[str, Any], *, source_name: str) -> List[RetrievalHit]:
    raw_hits = ((response.get("hits") or {}).get("hits") or []) if isinstance(response, dict) else []
    hits: List[RetrievalHit] = []
    for rank, item in enumerate(raw_hits, start=1):
        if not isinstance(item, dict):
            continue
        source = item.get("_source") if isinstance(item.get("_source"), dict) else {}
        chunk_id = str(source.get("chunk_id") or item.get("_id") or "").strip()
        document_id = str(source.get("document_id") or "").strip()
        try:
            score = float(item.get("_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if chunk_id:
            hits.append(RetrievalHit(chunk_id=chunk_id, document_id=document_id, score=score, source=source, rank=rank, source_name=source_name))
    return hits


def build_permission_filter(context: QueryContext) -> List[Dict[str, Any]]:
    user_id = str(context.user_id or "agent-system").strip()
    tenant_id = str(context.tenant_id or "default").strip()
    filters: List[Dict[str, Any]] = [
        {"term": {"tenant_id": tenant_id}},
        {"term": {"active": True}},
        {"term": {"deleted": False}},
    ]
    access_should: List[Dict[str, Any]] = [
        {"term": {"access_level": "public"}},
        {"term": {"owner_user_id": user_id}},
        {"term": {"allowed_user_ids": user_id}},
    ]
    for group_id in context.group_ids:
        if group_id:
            access_should.append({"term": {"allowed_group_ids": group_id}})
    filters.append({"bool": {"should": access_should, "minimum_should_match": 1}})
    return filters


def build_metadata_filters(filters: Dict[str, Any], *, knowledge_base_id: str = "") -> List[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    kb_id = str(knowledge_base_id or filters.get("knowledge_base_id") or "").strip()
    if kb_id:
        clauses.append({"term": {"knowledge_base_id": kb_id}})
    for field in ("domain", "topic", "extension", "document_id"):
        value = filters.get(field)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple, set)):
            clauses.append({"terms": {field: [str(item) for item in value if str(item).strip()]}})
        else:
            clauses.append({"term": {field: str(value).strip()}})
    if not bool(filters.get("include_archived", False)):
        clauses.append({"bool": {"should": [{"term": {"archived": False}}, {"bool": {"must_not": [{"exists": {"field": "archived"}}]}}], "minimum_should_match": 1}})
    return clauses


def build_bm25_body(query: str, *, top_k: int, context: QueryContext, filters: Dict[str, Any], knowledge_base_id: str = "") -> Dict[str, Any]:
    filter_clauses = build_permission_filter(context) + build_metadata_filters(filters, knowledge_base_id=knowledge_base_id)
    return {
        "size": max(1, top_k),
        "_source": True,
        "query": {
            "bool": {
                "filter": filter_clauses,
                "must": [
                    {
                        "multi_match": {
                            "query": str(query or ""),
                            "fields": [
                                "title^3",
                                "file_name^4",
                                "file_stem^3",
                                "section_path^2",
                                "metadata_text^2",
                                "text",
                            ],
                            "type": "best_fields",
                        }
                    }
                ],
            }
        },
    }


def build_vector_body(
    vector: List[float],
    *,
    top_k: int,
    num_candidates: int,
    context: QueryContext,
    filters: Dict[str, Any],
    knowledge_base_id: str = "",
) -> Dict[str, Any]:
    return {
        "size": max(1, top_k),
        "_source": True,
        "knn": {
            "field": "embedding",
            "query_vector": vector,
            "k": max(1, top_k),
            "num_candidates": max(num_candidates, top_k),
            "filter": build_permission_filter(context) + build_metadata_filters(filters, knowledge_base_id=knowledge_base_id),
        },
    }


class KnowledgeBaseRetriever:
    def __init__(self, config: KnowledgeBaseConfig, *, client: Optional[KnowledgeBaseEsClient] = None) -> None:
        self.config = config
        self.client = client or KnowledgeBaseEsClient(
            url=config.es_url,
            username=config.es_username,
            password=config.es_password,
            timeout=config.request_timeout_seconds,
        )

    def retrieve(self, query: str, *, context: QueryContext, filters: Optional[Dict[str, Any]] = None, knowledge_base_id: str = "") -> Dict[str, Any]:
        effective_filters = dict(filters or {})
        bm25_hits = self.bm25_search(query, context=context, filters=effective_filters, knowledge_base_id=knowledge_base_id)
        vector_hits: List[RetrievalHit] = []
        vector_error = ""
        embedding_ready = False
        if self.config.vector_top_k > 0:
            embedding_service = get_embedding_service()
            embedding_ready = embedding_service.is_ready()
            vector = embedding_service.embed_query(query) if embedding_ready else None
            if vector:
                try:
                    vector_hits = self.vector_search(vector, context=context, filters=effective_filters, knowledge_base_id=knowledge_base_id)
                except Exception as exc:
                    vector_error = f"{type(exc).__name__}: {exc}"
        fused = reciprocal_rank_fusion(
            [bm25_hits, vector_hits],
            k=self.config.rrf_k,
            top_k=self.config.merged_top_k,
        )
        return {
            "bm25_hits": bm25_hits,
            "vector_hits": vector_hits,
            "hits": fused,
            "debug": {
                "bm25_count": len(bm25_hits),
                "vector_count": len(vector_hits),
                "embedding_ready": embedding_ready,
                "vector_error": vector_error,
            },
        }

    def bm25_search(self, query: str, *, context: QueryContext, filters: Dict[str, Any], knowledge_base_id: str = "") -> List[RetrievalHit]:
        body = build_bm25_body(query, top_k=self.config.bm25_top_k, context=context, filters=filters, knowledge_base_id=knowledge_base_id)
        return _coerce_hits(self.client.search(self.config.chunk_index, body), source_name="bm25")

    def vector_search(self, vector: List[float], *, context: QueryContext, filters: Dict[str, Any], knowledge_base_id: str = "") -> List[RetrievalHit]:
        body = build_vector_body(
            vector,
            top_k=self.config.vector_top_k,
            num_candidates=self.config.vector_num_candidates,
            context=context,
            filters=filters,
            knowledge_base_id=knowledge_base_id,
        )
        return _coerce_hits(self.client.search(self.config.chunk_index, body), source_name="vector")


def dedupe_adjacent_chunks(hits: Iterable[RetrievalHit]) -> List[RetrievalHit]:
    seen: set[str] = set()
    result: List[RetrievalHit] = []
    for hit in hits:
        key = hit.chunk_id or f"{hit.document_id}:{hit.source.get('chunk_index')}"
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)
    return result
