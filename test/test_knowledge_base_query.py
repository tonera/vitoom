from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tool_catalog import ToolCatalog  # noqa: E402
from backend.services.agent.tools.builtin import knowledge_base_query as kb_query  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.answer import evidence_only_answer  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.models import QueryContext, RetrievalHit  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.reranker import rule_rerank  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.retriever import build_bm25_body, build_vector_body  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.rrf import reciprocal_rank_fusion  # noqa: E402
from backend.services.agent.tools.registry import get_tool_plugin_registry  # noqa: E402


def _hit(chunk_id: str, score: float, *, file_name: str = "文档.md", text: str = "正文") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        score=score,
        source={"chunk_id": chunk_id, "document_id": f"doc_{chunk_id}", "file_name": file_name, "title": file_name, "text": text},
    )


def test_knowledge_base_query_registered_in_plugin_registry_and_catalog():
    registrations = get_tool_plugin_registry().all_registrations()
    assert kb_query.KNOWLEDGE_BASE_QUERY_TOOL_NAME in registrations

    entry = ToolCatalog().get(kb_query.KNOWLEDGE_BASE_QUERY_TOOL_NAME)
    assert entry is not None
    assert entry.enabled is True
    assert "知识库" in entry.description


def test_rrf_merges_duplicate_chunk_and_preserves_stable_order():
    fused = reciprocal_rank_fusion(
        [
            [_hit("a", 2.0), _hit("b", 1.0)],
            [_hit("b", 3.0), _hit("c", 1.0)],
        ],
        k=60,
        top_k=3,
    )

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].source_name == "rrf"


def test_rule_rerank_boosts_exact_file_name_match():
    hits = [
        _hit("a", 1.0, file_name="普通制度.md", text="Vitoom 架构"),
        _hit("b", 0.4, file_name="Vitoom架构设计.md", text="系统分层设计"),
    ]

    reranked = rule_rerank("Vitoom架构设计", hits, top_k=2)

    assert reranked[0].chunk_id == "b"


def test_rule_rerank_extracts_latin_keyword_from_chinese_question():
    hits = [
        _hit("noise", 1.0, file_name="普通文档.md", text="无关内容"),
        _hit("tailscale", 0.1, file_name="Tailscale使用指南.md", text="Tailscale 安装和使用方法"),
    ]

    reranked = rule_rerank("Tailscale如何使用", hits, top_k=2)

    assert reranked[0].chunk_id == "tailscale"


def test_permission_filters_are_applied_to_bm25_and_vector_bodies():
    context = QueryContext(user_id="u1", tenant_id="t1", group_ids=["g1"])

    bm25_body = build_bm25_body("报销流程", top_k=5, context=context, filters={"domain": "财务"}, knowledge_base_id="kb1")
    vector_body = build_vector_body([0.1, 0.2], top_k=5, num_candidates=20, context=context, filters={"domain": "财务"}, knowledge_base_id="kb1")

    bm25_filter = bm25_body["query"]["bool"]["filter"]
    vector_filter = vector_body["knn"]["filter"]
    assert {"term": {"tenant_id": "t1"}} in bm25_filter
    assert {"term": {"knowledge_base_id": "kb1"}} in bm25_filter
    assert {"term": {"domain": "财务"}} in vector_filter
    assert any(clause.get("bool", {}).get("should") for clause in vector_filter)


def test_no_result_answer_has_no_fabricated_sources():
    output = evidence_only_answer("不存在的问题", [])

    assert "知识库未找到足够依据" in output
    assert "### 依据" in output
