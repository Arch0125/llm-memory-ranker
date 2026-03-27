from dataclasses import dataclass

from .types import CriticDecision
from .utils import clamp, tokenize


PERSONALIZATION_CUES = {
    "advice",
    "favorite",
    "gift",
    "like",
    "plan",
    "prefer",
    "preference",
    "recommend",
    "should",
    "suggest",
}

CONTINUATION_CUES = {
    "again",
    "continue",
    "continuing",
    "resume",
    "revisit",
    "repo",
    "project",
    "workspace",
}

CONSTRAINT_CUES = {
    "constraint",
    "must",
    "never",
    "rule",
    "safe",
    "safety",
}


def _overlap_features(query_text, memory_text):
    query_tokens = set(tokenize(query_text))
    memory_tokens = set(tokenize(memory_text))
    shared = query_tokens & memory_tokens
    if not query_tokens or not memory_tokens:
        return 0.0, 0.0, False, False, []
    overlap = len(shared) / max(1, min(len(query_tokens), len(memory_tokens)))
    long_shared = sorted(token for token in shared if len(token) >= 5)
    lexical_anchor = bool(long_shared)
    return overlap, len(shared) / max(1, len(query_tokens)), lexical_anchor, bool(shared), long_shared[:4]


@dataclass
class HeuristicCritic:
    use_threshold: float = 0.72
    maybe_threshold: float = 0.48

    def predict(self, query_text, hit):
        overlap, query_coverage, lexical_anchor, has_shared, shared_tokens = _overlap_features(
            query_text,
            hit.record.text,
        )
        query_tokens = set(tokenize(query_text))
        personalization = bool(query_tokens & PERSONALIZATION_CUES)
        continuation = bool(query_tokens & CONTINUATION_CUES)
        constraint_request = bool(query_tokens & CONSTRAINT_CUES)
        recency = 1.0 if hit.age_days <= 14 else max(0.0, 1.0 - (hit.age_days / 180.0))
        type_name = hit.record.memory_type
        importance = hit.record.importance

        if type_name == "project":
            base = max(overlap * 1.05, 0.7 if continuation and lexical_anchor else 0.0)
        elif type_name == "preference":
            base = overlap * 0.9
            if personalization and lexical_anchor:
                base = max(base, 0.72)
        elif type_name in {"constraint", "safety"}:
            base = max(overlap, 0.75 if constraint_request and lexical_anchor else 0.0)
        elif type_name == "identity":
            base = max(overlap * 0.9, 0.55 if continuation and lexical_anchor else 0.0)
        elif type_name == "skill":
            base = max(overlap, 0.6 if continuation and lexical_anchor else 0.0)
        else:
            base = overlap * 0.95

        if type_name == "ephemeral" and hit.age_days > 7:
            base *= 0.75
        if type_name == "preference" and personalization and not has_shared:
            base *= 0.4

        confidence = clamp(
            (0.52 * hit.score)
            + (0.30 * base)
            + (0.10 * importance)
            + (0.08 * recency),
            0.0,
            1.0,
        )

        reasons = []
        if shared_tokens:
            reasons.append("shared=" + ",".join(shared_tokens))
        if continuation:
            reasons.append("continuation-cue")
        if personalization:
            reasons.append("personalization-cue")
        if constraint_request:
            reasons.append("constraint-cue")
        if hit.age_days > 30:
            reasons.append(f"aged={hit.age_days}d")
        if query_coverage > 0.5:
            reasons.append("covers-query")

        if confidence >= self.use_threshold and (has_shared or type_name in {"constraint", "safety"}):
            return CriticDecision(label="use", confidence=confidence, reasons=reasons)
        if confidence >= self.maybe_threshold and has_shared:
            return CriticDecision(label="maybe", confidence=confidence, reasons=reasons)
        return CriticDecision(label="ignore", confidence=confidence, reasons=reasons or ["low-applicability"])


def rerank_with_critic(query_text, hits, critic_model):
    ranked = []
    label_rank = {"use": 2, "maybe": 1, "ignore": 0}
    for hit in hits:
        decision = critic_model.predict(query_text, hit)
        hit.critic_label = decision.label
        hit.critic_confidence = decision.confidence
        hit.reasons.extend(decision.reasons)
        ranked.append(hit)
    ranked.sort(
        key=lambda hit: (
            label_rank.get(hit.critic_label, 0),
            hit.critic_confidence,
            hit.score,
            hit.record.importance,
        ),
        reverse=True,
    )
    return ranked
