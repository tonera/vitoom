from __future__ import annotations

from typing import Any, Dict


def kb_document_mapping() -> Dict[str, Any]:
    keyword_text = {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}}
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "dynamic": "false",
            "properties": {
                "document_id": {"type": "keyword"},
                "document_group_id": {"type": "keyword"},
                "knowledge_base_id": {"type": "keyword"},
                "title": keyword_text,
                "file_name": keyword_text,
                "file_stem": keyword_text,
                "extension": {"type": "keyword"},
                "mime_type": {"type": "keyword"},
                "source_path": {"type": "keyword", "ignore_above": 2048},
                "canonical_path": {"type": "keyword", "ignore_above": 2048},
                "source_uri": {"type": "keyword", "ignore_above": 2048},
                "domain": {"type": "keyword"},
                "topic": {"type": "keyword"},
                "subtopic": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "summary": {"type": "text"},
                "sha256": {"type": "keyword"},
                "size_bytes": {"type": "long"},
                "created_at": {"type": "date"},
                "modified_at": {"type": "date"},
                "ingested_at": {"type": "date"},
                "owner_user_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "access_level": {"type": "keyword"},
                "allowed_user_ids": {"type": "keyword"},
                "allowed_group_ids": {"type": "keyword"},
                "archived": {"type": "boolean"},
                "active": {"type": "boolean"},
                "deleted": {"type": "boolean"},
                "duplicate_of": {"type": "keyword"},
                "ingest_version": {"type": "keyword"},
            },
        },
    }


def kb_chunk_mapping(*, dims: int = 384) -> Dict[str, Any]:
    keyword_text = {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}}
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "dynamic": "false",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "document_group_id": {"type": "keyword"},
                "knowledge_base_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_type": {"type": "keyword"},
                "title": keyword_text,
                "file_name": keyword_text,
                "file_stem": {"type": "text"},
                "section_path": keyword_text,
                "text": {"type": "text"},
                "metadata_text": {"type": "text"},
                "domain": {"type": "keyword"},
                "topic": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "page_start": {"type": "integer"},
                "page_end": {"type": "integer"},
                "source_uri": {"type": "keyword", "ignore_above": 2048},
                "canonical_path": {"type": "keyword", "ignore_above": 2048},
                "owner_user_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "access_level": {"type": "keyword"},
                "allowed_user_ids": {"type": "keyword"},
                "allowed_group_ids": {"type": "keyword"},
                "archived": {"type": "boolean"},
                "active": {"type": "boolean"},
                "deleted": {"type": "boolean"},
                "embedding_model": {"type": "keyword"},
                "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
                "ingested_at": {"type": "date"},
                "source_modified_at": {"type": "date"},
                "ingest_version": {"type": "keyword"},
            },
        },
    }


def index_mappings(*, dims: int = 384) -> Dict[str, Dict[str, Any]]:
    return {"kb_document_v1": kb_document_mapping(), "kb_chunk_v1": kb_chunk_mapping(dims=dims)}
