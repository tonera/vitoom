from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from backend.services.agent.tools.builtin.business_query_core.planner_base import run_agent_planner_completion

from .models import RetrievalHit


def _source_label(hit: RetrievalHit) -> str:
    parts = [f"《{hit.file_name}》"]
    section = str(hit.source.get("section_path") or "").strip()
    if section:
        parts.append(section)
    page = hit.source.get("page_start") or hit.source.get("page_end")
    if page:
        parts.append(f"页码 {page}")
    return " / ".join(parts)


def _snippet(text: str, *, limit: int = 220) -> str:
    normalized = " ".join(str(text or "").split())
    return normalized[:limit] + ("..." if len(normalized) > limit else "")


def build_context(hits: Iterable[RetrievalHit], *, max_chars: int) -> str:
    blocks: List[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        text = hit.text or str(hit.source.get("metadata_text") or "")
        block = (
            f"[{index}] 来源：{_source_label(hit)}\n"
            f"文件路径：{hit.source.get('canonical_path') or hit.source.get('source_uri') or ''}\n"
            f"内容：{text.strip()}"
        ).strip()
        if not block:
            continue
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def format_sources(hits: Iterable[RetrievalHit]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for hit in hits:
        sources.append(
            {
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "file_name": hit.file_name,
                "section_path": hit.source.get("section_path") or "",
                "page_start": hit.source.get("page_start"),
                "page_end": hit.source.get("page_end"),
                "canonical_path": hit.source.get("canonical_path") or "",
                "source_uri": hit.source.get("source_uri") or "",
                "score": hit.score,
            }
        )
    return sources


def evidence_only_answer(query: str, hits: List[RetrievalHit]) -> str:
    if not hits:
        return "### 答案\n\n知识库未找到足够依据。\n\n### 依据\n\n- 无\n\n### 未覆盖信息\n\n- 未检索到可引用的知识库片段。"
    display_hits = _lexically_matching_hits(query, hits)
    if not display_hits:
        display_hits = hits[:3]
    lines = ["### 答案", "", "已在知识库中找到以下相关资料。", "", "### 依据"]
    for hit in display_hits:
        lines.append(f"- {_source_label(hit)}：{_snippet(hit.text or hit.source.get('metadata_text') or '')}")
    lines.extend(["", "### 未覆盖信息", "", f"- 如需正文级结论，需要知识库中存在可解析正文；压缩包第一版只索引文件名和路径。问题：{query}"])
    return "\n".join(lines)


def _is_existence_query(query: str) -> bool:
    return bool(re.search(r"(有吗|有没有|是否有|找一下|查一下|资料|文件|文档)", str(query or "")))


def _query_keywords(query: str) -> List[str]:
    text = str(query or "").strip()
    text = re.sub(r"(有没有|是否有|有吗|找一下|查一下|资料|文件|文档|知识库|里面|里|关于|的)", " ", text)
    keywords = [item.strip().lower() for item in re.split(r"[\s,，。！？；;:：/\\()\[\]{}<>《》\"']+", text) if len(item.strip()) >= 2]
    return keywords


def _lexically_matching_hits(query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
    keywords = _query_keywords(query)
    if not keywords:
        return hits
    matched: List[RetrievalHit] = []
    for hit in hits:
        haystack = " ".join(
            str(value or "")
            for value in (
                hit.file_name,
                hit.title,
                hit.source.get("metadata_text"),
                hit.text,
            )
        ).lower()
        if any(keyword in haystack for keyword in keywords):
            matched.append(hit)
    return matched


def generate_answer(query: str, hits: List[RetrievalHit], *, user_id: str, max_context_chars: int) -> str:
    if not hits:
        return evidence_only_answer(query, hits)
    if _is_existence_query(query):
        return evidence_only_answer(query, hits)
    context = build_context(hits, max_chars=max_context_chars)
    messages = [
        {
            "role": "system",
            "content": (
                "你是本地知识库问答助手。只能基于给定检索片段回答；证据不足时必须说“知识库未找到足够依据”。"
                "不得编造不存在的文档、页码或结论。输出固定包含：### 答案、### 依据、### 未覆盖信息。"
            ),
        },
        {
            "role": "user",
            "content": f"用户问题：{query}\n\n检索片段：\n{context}",
        },
    ]
    try:
        return run_agent_planner_completion(messages, user_id=user_id, error_label="knowledge base answer")
    except Exception:
        return evidence_only_answer(query, hits)


def append_debug(markdown: str, *, sources: List[Dict[str, Any]], debug: Dict[str, Any]) -> str:
    payload = json.dumps({"sources": sources, "debug": debug}, ensure_ascii=False, indent=2)
    return f"{markdown.rstrip()}\n\n### Debug\n\n```json\n{payload}\n```"
