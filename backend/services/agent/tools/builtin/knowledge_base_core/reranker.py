from __future__ import annotations

import re
from typing import Iterable, List

from .models import RetrievalHit


def _query_terms(query: str) -> List[str]:
    text = str(query or "").lower()
    terms: List[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9._-]*", text):
        if len(token) >= 2:
            terms.append(token)
    cleaned = re.sub(r"(如何|怎么|怎样|使用|操作|教程|指南|查一下|知识库|资料|文档|文件|关于|的)", " ", text)
    for item in re.split(r"[\s,，。！？；;:：/\\()\[\]{}<>《》\"']+", cleaned):
        item = item.strip()
        if len(item) >= 2:
            terms.append(item)
    if not terms and text:
        terms = [text]
    deduped: List[str] = []
    seen = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def rule_rerank(query: str, hits: Iterable[RetrievalHit], *, top_k: int = 8) -> List[RetrievalHit]:
    terms = _query_terms(query)
    scored: List[tuple[float, int, RetrievalHit]] = []
    for index, hit in enumerate(hits):
        file_name = hit.file_name.lower()
        title = hit.title.lower()
        text = hit.text.lower()
        metadata = str(hit.source.get("metadata_text") or "").lower()
        boost = 0.0
        for term in terms:
            if term in file_name:
                boost += 8.0
            if term in title:
                boost += 5.0
            if term in metadata:
                boost += 3.0
            if term in text:
                boost += 2.0
        if hit.source.get("chunk_type") == "metadata":
            boost += 0.15
        if hit.source.get("archived"):
            boost -= 0.3
        scored.append((hit.score + boost, index, hit))
    ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
    result: List[RetrievalHit] = []
    for rank, (score, _index, hit) in enumerate(ordered[: max(1, top_k)], start=1):
        result.append(
            RetrievalHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                score=score,
                source=hit.source,
                rank=rank,
                source_name="rule_rerank",
            )
        )
    return result
