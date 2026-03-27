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
