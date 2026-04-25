"""BM25 keyword scorer using `rank_bm25`.

This replaces the hand-rolled IDF-weighted overlap in `SQLiteMemoryStore`. We
keep the fallback path inside the store for environments without `rank_bm25`,
but when available we get the real Okapi BM25 scoring with proper saturation
and document-length normalization.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

try:
    from rank_bm25 import BM25Okapi
    HAVE_RANK_BM25 = True
except ImportError:  # pragma: no cover - exercised only when dep missing
    BM25Okapi = None
    HAVE_RANK_BM25 = False

from .types import MemoryHit, MemoryRecord
from .utils import parse_timestamp, tokenize, utc_now


def _document_text(row) -> str:
    metadata = json.loads(row["metadata_json"] or "{}") if hasattr(row, "keys") else {}
    text = row["text"] if hasattr(row, "keys") else getattr(row, "text", "")
    return " ".join(
        v
        for v in (
            text,
            metadata.get("fact_text", ""),
            metadata.get("summary", ""),
            " ".join(metadata.get("entities", []) or []),
            " ".join(metadata.get("event_aliases", []) or []),
            " ".join(metadata.get("aggregate_labels", []) or []),
        )
        if v
    )


def _row_to_record(store, row) -> MemoryRecord:
    return store._record_from_row(row)


def bm25_search(
    store,
    query_text: str,
    user_id: str,
    *,
    top_k: int = 20,
    type_allowlist: Sequence[str] | None = None,
    status: str = "active",
) -> list[MemoryHit]:
    """Run BM25 over the memories belonging to `user_id`."""
    query_tokens = tokenize(query_text, drop_stopwords=True)
    if not query_tokens:
        return []

    clauses = ["m.user_id = ?", "m.status = ?"]
    params = [user_id, status]
    if type_allowlist:
        placeholders = ", ".join("?" for _ in type_allowlist)
        clauses.append(f"m.type IN ({placeholders})")
        params.extend(type_allowlist)
    rows = store.conn.execute(
        f"SELECT m.* FROM memory_item m WHERE {' AND '.join(clauses)}",
        params,
    ).fetchall()
    if not rows:
        return []

    corpus = [tokenize(_document_text(row), drop_stopwords=True) for row in rows]
    if HAVE_RANK_BM25:
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
        max_score = max(scores) if len(scores) else 1.0
        max_score = max_score if max_score > 1e-9 else 1.0
        normalized = [float(score) / max_score for score in scores]
    else:
        # Lightweight fallback: IDF-weighted token overlap.
        import math

        doc_count = len(corpus)
        token_doc_freq: dict[str, int] = {}
        for doc in corpus:
            for token in set(doc):
                token_doc_freq[token] = token_doc_freq.get(token, 0) + 1
        idf = {
            token: math.log((doc_count + 1) / (df + 1)) + 1.0
            for token, df in token_doc_freq.items()
        }
        query_tokens_set = set(query_tokens)
        raw_scores = []
        for doc in corpus:
            doc_set = set(doc)
            shared = query_tokens_set & doc_set
            if not shared:
                raw_scores.append(0.0)
                continue
            score = sum(idf.get(token, 1.0) for token in shared)
            raw_scores.append(score)
        max_score = max(raw_scores) if raw_scores else 1.0
        max_score = max_score if max_score > 1e-9 else 1.0
        normalized = [s / max_score for s in raw_scores]

    now = utc_now()
    results: list[MemoryHit] = []
    for row, score in zip(rows, normalized):
        if score <= 0:
            continue
        record = _row_to_record(store, row)
        age_days = max(0, (now - parse_timestamp(record.last_accessed_at)).days)
        results.append(
            MemoryHit(
                record=record,
                score=float(score),
                embedding_model="bm25",
                age_days=age_days,
            )
        )

    results.sort(key=lambda hit: hit.score, reverse=True)
    return results[:top_k]
