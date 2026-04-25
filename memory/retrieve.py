from dataclasses import dataclass, field


DEFAULT_MAX_AGE_BY_TYPE = {
    "constraint": 180,
    "ephemeral": 14,
    "identity": 365,
    "preference": 180,
    "project": 120,
    "safety": 365,
    "skill": 180,
}


@dataclass
class RetrievalConfig:
    top_k: int = 12
    similarity_threshold: float = 0.18
    max_age_days: int | None = None
    stable_importance_threshold: float = 0.8
    type_allowlist: list[str] = field(default_factory=list)


def parse_type_allowlist(raw_value):
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        return [value for value in raw_value if value]
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def infer_capture_type(text):
    lowered = text.lower()
    if any(token in lowered for token in ("prefer", "favorite", "like to", "usually")):
        return "preference"
    if any(token in lowered for token in ("working on", "project", "repo", "codebase", "feature")):
        return "project"
    if any(token in lowered for token in ("must", "never", "cannot", "can't", "constraint")):
        return "constraint"
    if any(token in lowered for token in ("skill", "experienced", "know how")):
        return "skill"
    if any(token in lowered for token in ("today", "this week", "temporary", "for now")):
        return "ephemeral"
    return "ephemeral"


def retrieve_candidates(query_text, store, embedder, user_id, top_k=20, type_allowlist=None):
    query_embedding = embedder.embed(query_text)
    return store.search(
        vector=query_embedding,
        user_id=user_id,
        top_k=top_k,
        type_allowlist=type_allowlist,
        model_name=embedder.model_name,
    )


def retrieve_hybrid_candidates(
    query_text,
    store,
    embedder,
    user_id,
    top_k=20,
    type_allowlist=None,
    keyword_weight=0.35,
    fusion="weighted",
    use_bm25=False,
):
    """Retrieve candidates using embeddings and keyword/BM25, then fuse.

    `fusion` selects the merge strategy:
        - "rrf"      -> Reciprocal Rank Fusion (recommended, scale-free).
        - "weighted" -> classic weighted-sum (kept for back-compat).
    """
    from .bm25 import bm25_search
    from .fusion import reciprocal_rank_fusion, weighted_score_fusion

    embedding_hits = retrieve_candidates(
        query_text, store, embedder, user_id, top_k=top_k, type_allowlist=type_allowlist,
    )
    if use_bm25:
        keyword_hits = bm25_search(
            store, query_text, user_id, top_k=top_k, type_allowlist=type_allowlist,
        )
    else:
        keyword_hits = store.keyword_search(
            query_text, user_id, top_k=top_k, type_allowlist=type_allowlist,
        )

    if fusion == "rrf":
        return reciprocal_rank_fusion(
            [embedding_hits, keyword_hits],
            weights=[1.0, max(keyword_weight * 2.0, 0.5)],
            top_k=top_k,
        )
    return weighted_score_fusion(
        [embedding_hits, keyword_hits],
        weights=[1.0 - keyword_weight, keyword_weight],
        top_k=top_k,
    )


def retrieve_for_query(
    query_text,
    store,
    embedder,
    user_id,
    *,
    top_k=20,
    type_allowlist=None,
    expansions=None,
    fusion="rrf",
    use_bm25=True,
    keyword_weight=0.5,
):
    """Top-level retrieval helper that combines query expansion + hybrid fusion.

    `expansions` is a list of additional query strings to retrieve in parallel
    (typically produced by `memory.expansion.expand_query`). All ranked lists
    are merged via `fusion` ("rrf" or "weighted"). The original query is
    always retrieved first.
    """
    from .bm25 import bm25_search
    from .fusion import reciprocal_rank_fusion, weighted_score_fusion

    queries = [query_text]
    if expansions:
        for q in expansions:
            if q and q not in queries:
                queries.append(q)

    ranked_lists = []
    weights = []
    for index, q in enumerate(queries):
        # Per-query weight decays so the original query has the largest signal.
        per_query_weight = 1.0 if index == 0 else max(0.6, 1.0 - 0.15 * index)
        ranked_lists.append(
            retrieve_candidates(q, store, embedder, user_id, top_k=top_k, type_allowlist=type_allowlist)
        )
        weights.append(per_query_weight)
        if use_bm25:
            ranked_lists.append(
                bm25_search(store, q, user_id, top_k=top_k, type_allowlist=type_allowlist)
            )
            weights.append(per_query_weight * keyword_weight * 2.0)
        else:
            ranked_lists.append(
                store.keyword_search(q, user_id, top_k=top_k, type_allowlist=type_allowlist)
            )
            weights.append(per_query_weight * keyword_weight)

    if fusion == "rrf":
        return reciprocal_rank_fusion(ranked_lists, weights=weights, top_k=top_k)
    return weighted_score_fusion(ranked_lists, weights=weights, top_k=top_k)


def gate_hits(
    hits,
    sim_threshold,
    max_age_days=None,
    stable_importance_threshold=0.8,
):
    kept = []
    for hit in hits:
        if hit.score < sim_threshold:
            continue
        age_limit = max_age_days
        if age_limit is None:
            age_limit = DEFAULT_MAX_AGE_BY_TYPE.get(hit.record.memory_type, 90)
        if hit.age_days > age_limit and hit.record.importance < stable_importance_threshold:
            hit.reasons.append(f"stale>{age_limit}d")
            continue
        if (
            hit.record.memory_type == "ephemeral"
            and hit.age_days > 7
            and hit.record.importance < stable_importance_threshold
        ):
            hit.reasons.append("ephemeral-aged-out")
            continue
        kept.append(hit)
    return kept
