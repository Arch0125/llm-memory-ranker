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
):
    """Retrieve candidates using both embedding similarity and keyword overlap.

    Merges results from embedding search and keyword search, combining scores
    for memories found by both methods.  This catches memories with matching
    keywords that hash embeddings might rank low.
    """
    embedding_hits = retrieve_candidates(
        query_text, store, embedder, user_id, top_k=top_k, type_allowlist=type_allowlist,
    )
    keyword_hits = store.keyword_search(
        query_text, user_id, top_k=top_k, type_allowlist=type_allowlist,
    )

    # Merge: index by memory_id, combine scores
    merged = {}
    for hit in embedding_hits:
        merged[hit.record.memory_id] = hit

    embedding_weight = 1.0 - keyword_weight
    for kw_hit in keyword_hits:
        mid = kw_hit.record.memory_id
        if mid in merged:
            # Blend: keep the embedding hit but boost its score with keyword score
            existing = merged[mid]
            existing.score = (
                embedding_weight * existing.score
                + keyword_weight * kw_hit.score
            )
            if not existing.reasons:
                existing.reasons = []
            existing.reasons.append(f"kw-boost={kw_hit.score:.2f}")
        else:
            # New hit from keyword search only - scale its score
            kw_hit.score = keyword_weight * kw_hit.score
            kw_hit.reasons = [f"keyword-only={kw_hit.score:.2f}"]
            merged[mid] = kw_hit

    all_hits = list(merged.values())
    all_hits.sort(key=lambda h: h.score, reverse=True)
    return all_hits[:top_k]


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
