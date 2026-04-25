"""Pluggable vector indexes.

The default `SQLiteMemoryStore` reads every memory's embedding back from the
database on each query. That's fine at LongMemEval scale (haystacks of a few
hundred to a few thousand items) but does O(n) work per query.

This module provides drop-in indexes that the pipeline can use to bypass the
full table scan:

- `InMemoryCosineIndex`: numpy matmul against a stacked float32 matrix.
- `FaissFlatIndex`: same semantics via faiss when the dependency is available
  (auto-detected; falls back to InMemoryCosineIndex if `faiss` is missing).

Both implement a tiny `add`, `remove`, and `search` API and return
`MemoryHit`s populated with `score`, `record`, and `age_days`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import MemoryHit
from .utils import parse_timestamp, utc_now


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return matrix / norms


class InMemoryCosineIndex:
    """Stacked-matrix cosine search. Thread-unsafe; use one per pipeline."""

    def __init__(self, dim: int | None = None):
        self.dim = dim
        self._matrix: np.ndarray | None = None
        self._records: list = []
        self._memory_id_to_index: dict[str, int] = {}

    @property
    def size(self) -> int:
        return 0 if self._matrix is None else self._matrix.shape[0]

    def add(self, record, embedding: Sequence[float]) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        if self.dim is None:
            self.dim = vector.shape[0]
        elif vector.shape[0] != self.dim:
            raise ValueError(f"embedding dim mismatch: {vector.shape[0]} vs {self.dim}")
        normalized = vector / max(np.linalg.norm(vector), 1e-12)
        if self._matrix is None:
            self._matrix = normalized.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, normalized])
        self._memory_id_to_index[record.memory_id] = len(self._records)
        self._records.append(record)

    def remove(self, memory_id: str) -> None:
        idx = self._memory_id_to_index.pop(memory_id, None)
        if idx is None or self._matrix is None:
            return
        self._matrix = np.delete(self._matrix, idx, axis=0)
        self._records.pop(idx)
        self._memory_id_to_index = {
            rec.memory_id: i for i, rec in enumerate(self._records)
        }

    def search(self, vector: Sequence[float], top_k: int = 20, *, type_allowlist=None) -> list[MemoryHit]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        query = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm < 1e-12:
            return []
        query = query / norm
        scores = self._matrix @ query

        candidate_indices = np.argsort(-scores)
        results: list[MemoryHit] = []
        now = utc_now()
        for idx in candidate_indices:
            record = self._records[idx]
            if type_allowlist and record.memory_type not in type_allowlist:
                continue
            if record.status != "active":
                continue
            age_days = max(0, (now - parse_timestamp(record.last_accessed_at)).days)
            results.append(
                MemoryHit(
                    record=record,
                    score=float(scores[idx]),
                    embedding_model="memory-index",
                    age_days=age_days,
                )
            )
            if len(results) >= top_k:
                break
        return results


class FaissFlatIndex(InMemoryCosineIndex):
    """Faiss-backed flat cosine index. Falls back to numpy when faiss missing."""

    def __init__(self, dim: int | None = None):
        super().__init__(dim=dim)
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self._index = None
        except ImportError:
            self._faiss = None
            self._index = None

    def _ensure_index(self):
        if self._faiss is None or self.dim is None:
            return
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(self.dim)
            if self._matrix is not None:
                self._index.add(self._matrix)

    def add(self, record, embedding):
        super().add(record, embedding)
        if self._faiss is None:
            return
        self._ensure_index()
        if self._index is not None:
            normalized = (
                np.asarray(embedding, dtype=np.float32)
                / max(np.linalg.norm(np.asarray(embedding, dtype=np.float32)), 1e-12)
            )
            self._index.add(normalized.reshape(1, -1))

    def search(self, vector, top_k=20, *, type_allowlist=None):
        if self._faiss is None or self._index is None:
            return super().search(vector, top_k=top_k, type_allowlist=type_allowlist)
        query = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm < 1e-12:
            return []
        query = (query / norm).reshape(1, -1)
        scores, indices = self._index.search(query, min(top_k * 4, self.size or top_k))
        results: list[MemoryHit] = []
        now = utc_now()
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            record = self._records[idx]
            if type_allowlist and record.memory_type not in type_allowlist:
                continue
            if record.status != "active":
                continue
            age_days = max(0, (now - parse_timestamp(record.last_accessed_at)).days)
            results.append(
                MemoryHit(
                    record=record,
                    score=float(score),
                    embedding_model="faiss-flat",
                    age_days=age_days,
                )
            )
            if len(results) >= top_k:
                break
        return results


def build_index(kind: str = "auto", **kwargs):
    """Factory: 'auto' picks faiss when available, otherwise in-memory."""
    if kind == "memory":
        return InMemoryCosineIndex(**kwargs)
    if kind == "faiss":
        return FaissFlatIndex(**kwargs)
    if kind == "auto":
        index = FaissFlatIndex(**kwargs)
        if index._faiss is None:
            return InMemoryCosineIndex(**kwargs)
        return index
    raise ValueError(f"Unknown index kind: {kind!r}")
