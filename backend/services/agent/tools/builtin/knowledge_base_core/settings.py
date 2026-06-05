from __future__ import annotations

from backend.services.agent import settings as agent_settings

from .models import KnowledgeBaseConfig


def load_knowledge_base_config() -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(
        enabled=agent_settings.is_knowledge_base_enabled(),
        es_url=agent_settings.get_knowledge_base_es_url(),
        es_username=agent_settings.get_knowledge_base_es_username(),
        es_password=agent_settings.get_knowledge_base_es_password(),
        document_index=agent_settings.get_knowledge_base_document_index(),
        chunk_index=agent_settings.get_knowledge_base_chunk_index(),
        request_timeout_seconds=agent_settings.get_knowledge_base_request_timeout_seconds(),
        bm25_top_k=agent_settings.get_knowledge_base_retrieval_bm25_top_k(),
        vector_top_k=agent_settings.get_knowledge_base_retrieval_vector_top_k(),
        vector_num_candidates=agent_settings.get_knowledge_base_retrieval_vector_num_candidates(),
        rrf_k=agent_settings.get_knowledge_base_retrieval_rrf_k(),
        merged_top_k=agent_settings.get_knowledge_base_retrieval_merged_top_k(),
        rerank_enabled=agent_settings.is_knowledge_base_rerank_enabled(),
        rerank_backend=agent_settings.get_knowledge_base_rerank_backend(),
        rerank_top_n=agent_settings.get_knowledge_base_rerank_top_n(),
        final_top_k=agent_settings.get_knowledge_base_rerank_final_top_k(),
        answer_max_context_chars=agent_settings.get_knowledge_base_answer_max_context_chars(),
        include_sources=agent_settings.is_knowledge_base_answer_sources_enabled(),
        include_debug=agent_settings.is_knowledge_base_answer_debug_enabled(),
    )
