"""Result-list fusion strategies.

When the engine retrieves candidates from multiple sources (dense embeddings,
BM25, focus-term scan, structured queries) we need to combine the lists into a
single ranked result.

Two strategies are provided:

- `reciprocal_rank_fusion` (RRF): rank-based, robust to score-scale differences,
  default. Each item's fused score is the sum of `1/(k + rank_i)` over the
  lists it appears in. This is the recommended default for hybrid retrieval.

- `weighted_score_fusion`: classic weighted-sum on the raw similarity scores,
  with optional per-source min/max normalization. Kept for back-compat with
  the previous hybrid retriever.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from .types import MemoryHit


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Iterable[MemoryHit]],
    *,
    k: float = 60.0,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
    annotate: bool = True,
) -> list[MemoryHit]:
    """RRF over multiple ranked candidate lists.

    `weights` lets you up- or down-weight an entire list (e.g. give dense
    retrieval 1.0 and BM25 0.7). When None, all lists are weighted 1.0.
    """
    weight_vector = list(weights) if weights is not None else [1.0] * len(ranked_lists)
    if len(weight_vector) != len(ranked_lists):
        raise ValueError("weights must align with ranked_lists")

    fused: dict[str, MemoryHit] = {}
    fused_score: dict[str, float] = {}
    sources: dict[str, list[str]] = {}

    for list_idx, hits in enumerate(ranked_lists):
        weight = weight_vector[list_idx]
        for rank, hit in enumerate(hits):
            mid = hit.record.memory_id
            increment = weight / (k + rank + 1)
            fused_score[mid] = fused_score.get(mid, 0.0) + increment
            kept = fused.get(mid)
            if kept is None or hit.score > kept.score:
                fused[mid] = hit
            sources.setdefault(mid, []).append(
                hit.embedding_model or f"src{list_idx}"
            )

    output = []
    max_score = max(fused_score.values()) if fused_score else 1.0
    max_score = max_score if max_score > 1e-9 else 1.0
    for mid, hit in fused.items():
        # Re-scale to [0, 1] so downstream similarity gates still make sense.
        hit.score = fused_score[mid] / max_score
        if annotate:
            unique_sources = list(dict.fromkeys(sources[mid]))
            tag = "rrf+" + "+".join(unique_sources)
            if hit.reasons is None:
                hit.reasons = []
            hit.reasons.append(tag)
        output.append(hit)

    output.sort(key=lambda h: h.score, reverse=True)
    if top_k:
        output = output[:top_k]
    return output


def _minmax(values):
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return lo, lo + 1.0
    return lo, hi


def weighted_score_fusion(
    ranked_lists: Sequence[Iterable[MemoryHit]],
    *,
    weights: Sequence[float],
    top_k: int | None = None,
    normalize: bool = True,
    annotate: bool = True,
) -> list[MemoryHit]:
    """Weighted-sum fusion over the raw scores in each list.

    With `normalize=True` (default), each list's scores are min-max normalized to
    [0,1] before combining, which reduces the impact of different score scales.
    """
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must align with ranked_lists")

    normalized_lists: list[list[tuple[MemoryHit, float]]] = []
    for hits in ranked_lists:
        hits_list = list(hits)
        scores = [hit.score for hit in hits_list]
        if normalize and scores:
            lo, hi = _minmax(scores)
            normalized_lists.append([(hit, (hit.score - lo) / (hi - lo)) for hit in hits_list])
        else:
            normalized_lists.append([(hit, hit.score) for hit in hits_list])

    fused: dict[str, MemoryHit] = {}
    fused_score: dict[str, float] = {}
    sources: dict[str, list[str]] = {}

    for list_idx, items in enumerate(normalized_lists):
        weight = weights[list_idx]
        for hit, normalized_score in items:
            mid = hit.record.memory_id
            fused_score[mid] = fused_score.get(mid, 0.0) + weight * normalized_score
            kept = fused.get(mid)
            if kept is None or hit.score > kept.score:
                fused[mid] = hit
            sources.setdefault(mid, []).append(hit.embedding_model or f"src{list_idx}")

    output = []
    max_score = max(fused_score.values()) if fused_score else 1.0
    max_score = max_score if max_score > 1e-9 else 1.0
    for mid, hit in fused.items():
        hit.score = fused_score[mid] / max_score
        if annotate:
            unique_sources = list(dict.fromkeys(sources[mid]))
            tag = "wsf+" + "+".join(unique_sources)
            if hit.reasons is None:
                hit.reasons = []
            hit.reasons.append(tag)
        output.append(hit)

    output.sort(key=lambda h: h.score, reverse=True)
    if top_k:
        output = output[:top_k]
    return output
