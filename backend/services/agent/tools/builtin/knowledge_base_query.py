"""Local knowledge-base query tool."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from backend.services.agent.tools.builtin.knowledge_base_core.answer import append_debug, format_sources, generate_answer
from backend.services.agent.tools.builtin.knowledge_base_core.models import QueryContext
from backend.services.agent.tools.builtin.knowledge_base_core.reranker import rule_rerank
from backend.services.agent.tools.builtin.knowledge_base_core.retriever import KnowledgeBaseRetriever, dedupe_adjacent_chunks
from backend.services.agent.tools.builtin.knowledge_base_core.settings import load_knowledge_base_config
from backend.services.agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_QUERY_TOOL_NAME = "knowledge_base_query"

KNOWLEDGE_BASE_QUERY_DESCRIPTION = (
    "查询本地知识库文档，适合根据已入库的 PDF、Markdown、Word、PPT、Keynote、压缩包文件名、制度、"
    "方案、项目文档、会议纪要和历史工作资料回答问题。支持文件名、路径、正文和语义检索，并返回来源引用。"
    "用户提到“知识库里”“根据已有资料”“历史文档”“查一下文档”时优先使用。"
    "不要用于联网搜索最新新闻；不要用于把文件转换为 PDF/Markdown。"
)

KNOWLEDGE_BASE_QUERY_DOCSTRING = "Query local knowledge-base chunks with BM25/vector retrieval, RRF, rerank and cited Markdown answers."


def _coerce_tool_args(raw_input: Any = None, **kwargs: Any) -> Dict[str, Any]:
    if kwargs:
        return dict(kwargs)
    if raw_input is None:
        return {}
    if isinstance(raw_input, dict):
        return dict(raw_input)
    if isinstance(raw_input, str):
        text = raw_input.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"query": text}
    return {"query": str(raw_input)}


def run_knowledge_base_query(
    query: str,
    *,
    top_k: int = 8,
    include_sources: bool = True,
    include_debug: bool = False,
    knowledge_base_id: str = "",
    filters: Optional[Dict[str, Any]] = None,
    user_id: str = "agent-system",
    tenant_id: str = "default",
) -> str:
    text = str(query or "").strip()
    if not text:
        return "请提供要查询的知识库问题。"
    config = load_knowledge_base_config()
    if not config.enabled:
        return "知识库查询当前未启用。"
    context = QueryContext(user_id=str(user_id or "agent-system").strip() or "agent-system", tenant_id=str(tenant_id or "default").strip() or "default")
    try:
        retrieval = KnowledgeBaseRetriever(config).retrieve(text, context=context, filters=filters or {}, knowledge_base_id=knowledge_base_id)
        hits = dedupe_adjacent_chunks(retrieval["hits"])
        final_top_k = max(1, int(top_k or config.final_top_k or 8))
        if config.rerank_enabled and config.rerank_backend == "rule":
            hits = rule_rerank(text, hits[: config.rerank_top_n], top_k=final_top_k)
        else:
            hits = hits[:final_top_k]
        markdown = generate_answer(text, hits, user_id=context.user_id, max_context_chars=config.answer_max_context_chars)
        sources = format_sources(hits) if include_sources else []
        debug_enabled = include_debug or config.include_debug
        if debug_enabled:
            markdown = append_debug(markdown, sources=sources, debug=retrieval.get("debug") or {})
        elif include_sources and sources and "### 依据" not in markdown:
            markdown = f"{markdown.rstrip()}\n\n### 依据\n" + "\n".join(f"- 《{item['file_name']}》" for item in sources)
        return markdown
    except Exception as exc:
        logger.exception(
            "knowledge_base_query failed: es_url=%s chunk_index=%s query=%r",
            config.es_url,
            config.chunk_index,
            text,
        )
        diagnostic = (
            f"知识库查询执行失败：{type(exc).__name__}: {exc}。"
            f"请检查 ES 地址 `{config.es_url}`、chunk 索引 `{config.chunk_index}`，"
            "并确认后端进程已重启加载最新配置。"
        )
        if include_debug:
            return diagnostic
        return diagnostic


@register_tool(
    name=KNOWLEDGE_BASE_QUERY_TOOL_NAME,
    description=KNOWLEDGE_BASE_QUERY_DESCRIPTION,
    tags=["knowledge", "kb", "rag", "document", "知识库", "文档检索", "资料查询", "历史文档"],
    provider="local",
    enabled=True,
)
def build_knowledge_base_query_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()
    bound_tenant_id = str(ctx.get("tenant_id") or "default").strip() or "default"

    try:
        from crewai.tools import BaseTool  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError("crewai is required to register native agent tools") from e

    try:
        from pydantic import BaseModel, Field  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError("pydantic is required to build knowledge_base_query tool") from e

    class KnowledgeBaseQueryArgs(BaseModel):
        query: str = Field(default="", description="User's natural-language knowledge base question.")
        top_k: int = Field(default=8, ge=1, le=20, description="Number of source snippets used for the final answer.")
        include_sources: bool = Field(default=True, description="Whether to return source citations.")
        include_debug: bool = Field(default=False, description="Whether to append retrieval debug JSON.")
        knowledge_base_id: str = Field(default="", description="Optional knowledge base ID filter.")
        filters: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata filters, e.g. domain/topic/extension/document_id.")

    class KnowledgeBaseQueryTool(BaseTool):
        name: str = KNOWLEDGE_BASE_QUERY_TOOL_NAME
        description: str = KNOWLEDGE_BASE_QUERY_DESCRIPTION
        args_schema: type = KnowledgeBaseQueryArgs

        def _run(
            self,
            query: str = "",
            top_k: int = 8,
            include_sources: bool = True,
            include_debug: bool = False,
            knowledge_base_id: str = "",
            filters: Optional[Dict[str, Any]] = None,
            **_ignored: Any,
        ) -> str:
            payload = _coerce_tool_args(
                query=query,
                top_k=top_k,
                include_sources=include_sources,
                include_debug=include_debug,
                knowledge_base_id=knowledge_base_id,
                filters=filters or {},
            )
            return run_knowledge_base_query(
                str(payload.get("query") or ""),
                top_k=int(payload.get("top_k") or 8),
                include_sources=bool(payload.get("include_sources", True)),
                include_debug=bool(payload.get("include_debug", False)),
                knowledge_base_id=str(payload.get("knowledge_base_id") or ""),
                filters=payload.get("filters") if isinstance(payload.get("filters"), dict) else {},
                user_id=bound_user_id or "agent-system",
                tenant_id=bound_tenant_id,
            )

    KnowledgeBaseQueryTool.__doc__ = KNOWLEDGE_BASE_QUERY_DOCSTRING
    return KnowledgeBaseQueryTool()
