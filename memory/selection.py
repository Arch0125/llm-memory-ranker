"""Diversity-aware memory selection.

Maximal Marginal Relevance (MMR) picks the next-best item by trading off
relevance to the query against redundancy with what's already been selected:

    mmr(d_i) = lambda * sim(d_i, q) - (1 - lambda) * max_{d_j in selected} sim(d_i, d_j)

Higher `lambda` -> closer to vanilla top-k.
Lower `lambda`  -> more diverse selections.

We use cosine similarity over either the items' stored embeddings (when
present) or hashed Jaccard token overlap as a fallback.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from .types import MemoryHit
from .utils import tokenize


def _embedding(hit: MemoryHit):
    metadata = hit.record.metadata or {}
    if "embedding" in metadata:
        return np.asarray(metadata["embedding"], dtype=np.float32)
    if "vector_json" in metadata:
        try:
            return np.asarray(json.loads(metadata["vector_json"]), dtype=np.float32)
        except ValueError:
            return None
    return None


def _cosine(left, right):
    if left is None or right is None:
        return None
    if left.shape != right.shape:
        return None
    norm = (np.linalg.norm(left) * np.linalg.norm(right))
    if norm < 1e-12:
        return 0.0
    return float(np.dot(left, right) / norm)


def _jaccard(left_text: str, right_text: str) -> float:
    left = set(tokenize(left_text, drop_stopwords=True))
    right = set(tokenize(right_text, drop_stopwords=True))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def mmr_select(
    hits: Sequence[MemoryHit],
    *,
    k: int,
    lambda_: float = 0.7,
    min_lambda: float = 0.4,
) -> list[MemoryHit]:
    """Return up to `k` hits picked via MMR.

    `lambda_` is gradually decayed each pick (down to `min_lambda`) so that as
    the selection grows we lean more on diversity.
    """
    if k <= 0 or not hits:
        return []
    remaining = list(hits)
    selected: list[MemoryHit] = []
    selected_embeddings: list[np.ndarray | None] = []
    selected_texts: list[str] = []

    embeddings = [_embedding(hit) for hit in remaining]
    while remaining and len(selected) < k:
        best_index = 0
        best_score = float("-inf")
        for i, hit in enumerate(remaining):
            relevance = hit.score
            redundancy = 0.0
            if selected:
                similarities = []
                for sel_emb, sel_text in zip(selected_embeddings, selected_texts):
                    sim = None
                    if embeddings[i] is not None and sel_emb is not None:
                        sim = _cosine(embeddings[i], sel_emb)
                    if sim is None:
                        sim = _jaccard(hit.record.text, sel_text)
                    similarities.append(sim)
                redundancy = max(similarities) if similarities else 0.0
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_index = i
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        selected_embeddings.append(embeddings.pop(best_index))
        selected_texts.append(chosen.record.text)
        lambda_ = max(min_lambda, lambda_ * 0.95)
    return selected
