from __future__ import annotations

from typing import Iterable, List

from .models import RetrievalHit


def reciprocal_rank_fusion(result_sets: Iterable[Iterable[RetrievalHit]], *, k: int = 60, top_k: int = 30) -> List[RetrievalHit]:
    scores: dict[str, float] = {}
    best_hits: dict[str, RetrievalHit] = {}
    for result_set in result_sets:
        for rank, hit in enumerate(result_set, start=1):
            key = hit.chunk_id or f"{hit.document_id}:{rank}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (max(1, k) + rank)
            previous = best_hits.get(key)
            if previous is None or hit.score > previous.score:
                best_hits[key] = hit
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused: List[RetrievalHit] = []
    for rank, (key, score) in enumerate(ordered[: max(1, top_k)], start=1):
        hit = best_hits[key]
        fused.append(
            RetrievalHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                score=score,
                source=hit.source,
                rank=rank,
                source_name="rrf",
            )
        )
    return fused
