"""Optional cross-encoder reranker.

A cross-encoder reads (query, candidate) jointly and produces a single
relevance score. It's much more accurate than dual-encoder cosine but ~10x
slower, so we use it as a top-N rerank stage after gating.

This module is import-safe even when sentence-transformers is missing — the
reranker becomes a no-op (returns inputs untouched, with `available=False`).

Example::

    reranker = CrossEncoderReranker.try_load("cross-encoder/ms-marco-MiniLM-L-6-v2")
    if reranker.available:
        hits = reranker.rerank(query, hits, top_k=20)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .types import MemoryHit


@dataclass
class _NullReranker:
    available: bool = False

    def rerank(self, query: str, hits: Sequence[MemoryHit], *, top_k: int | None = None) -> list[MemoryHit]:
        result = list(hits)
        if top_k is not None:
            result = result[:top_k]
        return result


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder. Lazy-loaded."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self.available = True
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._cls = CrossEncoder
        except ImportError:
            self._cls = None
            self.available = False

    def _ensure(self):
        if self._cls is None:
            return None
        if self._model is None:
            self._model = self._cls(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        hits: Sequence[MemoryHit],
        *,
        top_k: int | None = None,
        blend: float = 0.7,
    ) -> list[MemoryHit]:
        """Rerank `hits` by cross-encoder score and (optionally) keep some of
        the original ranking signal.

        `blend=1.0` -> pure cross-encoder.
        `blend=0.0` -> identical to input order.
        """
        model = self._ensure()
        if model is None or not hits:
            return list(hits)[:top_k] if top_k else list(hits)

        pairs = [(query, hit.record.text) for hit in hits]
        scores = model.predict(pairs).tolist()

        new_hits = []
        max_input = max((hit.score for hit in hits), default=1.0) or 1.0
        max_ce = max(scores) if scores else 1.0
        max_ce = max_ce if abs(max_ce) > 1e-9 else 1.0
        for hit, score in zip(hits, scores):
            normalized_input = hit.score / max_input
            normalized_ce = score / max_ce
            hit.score = blend * normalized_ce + (1.0 - blend) * normalized_input
            if hit.reasons is None:
                hit.reasons = []
            hit.reasons.append(f"ce={score:.2f}")
            new_hits.append(hit)
        new_hits.sort(key=lambda h: h.score, reverse=True)
        if top_k is not None:
            new_hits = new_hits[:top_k]
        return new_hits

    @classmethod
    def try_load(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        instance = cls(model_name)
        if not instance.available:
            return _NullReranker()
        return instance


def null_reranker():
    return _NullReranker()
