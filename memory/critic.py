from dataclasses import dataclass

from .types import CriticDecision
from .utils import clamp, extract_entities, normalize_entity, tokenize


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

TEMPORAL_CUES = {
    "after",
    "before",
    "days",
    "first",
    "last",
    "months",
    "recent",
    "since",
    "time",
    "timeline",
    "when",
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
        query_entities = set(extract_entities(query_text))
        memory_entities = {
            normalize_entity(value) for value in hit.record.metadata.get("entities", []) if value
        }
        query_text_normalized = normalize_entity(query_text)
        memory_aliases = {
            normalize_entity(value) for value in hit.record.metadata.get("event_aliases", []) if value
        }
        personalization = bool(query_tokens & PERSONALIZATION_CUES)
        continuation = bool(query_tokens & CONTINUATION_CUES)
        constraint_request = bool(query_tokens & CONSTRAINT_CUES)
        temporal_request = bool(query_tokens & TEMPORAL_CUES)
        recency = 1.0 if hit.age_days <= 14 else max(0.0, 1.0 - (hit.age_days / 180.0))
        type_name = hit.record.memory_type
        importance = hit.record.importance
        entity_overlap = len(query_entities & memory_entities) / max(1, len(query_entities)) if query_entities else 0.0
        has_event_date = bool(hit.record.metadata.get("event_date"))
        date_confidence = float(hit.record.metadata.get("date_confidence", 0.0) or 0.0)
        alias_overlap = 0.0
        if memory_aliases:
            alias_overlap = max(
                1.0
                for alias in memory_aliases
                if alias and (alias in query_text_normalized or all(part in query_text_normalized for part in alias.split() if len(part) >= 3))
            ) if any(
                alias and (alias in query_text_normalized or all(part in query_text_normalized for part in alias.split() if len(part) >= 3))
                for alias in memory_aliases
            ) else 0.0
        granularity = hit.record.metadata.get("granularity", "")

        if type_name == "project":
            base = max(overlap * 1.05, 0.7 if continuation and lexical_anchor else 0.0)
        elif type_name == "preference":
            base = overlap * 0.9
            if personalization and lexical_anchor:
                base = max(base, 0.72)
        elif type_name == "timeline":
            base = overlap * 0.9
            if temporal_request:
                base = max(base, 0.62 if has_event_date else 0.52)
        elif type_name == "episode":
            base = overlap * 0.95
            if temporal_request:
                base = max(base, 0.58 if has_event_date else 0.48)
        elif type_name == "event":
            base = overlap
            if temporal_request and has_event_date:
                base = max(base, 0.72 if entity_overlap > 0.0 else 0.56)
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
        if temporal_request:
            base += 0.22 * entity_overlap
            base += 0.18 * alias_overlap
            if granularity in {"timeline-global", "timeline"}:
                base += 0.08
            if has_event_date:
                base += 0.06
                base += 0.10 * date_confidence

        confidence = clamp(
            (0.44 * hit.score)
            + (0.28 * base)
            + (0.10 * importance)
            + (0.08 * recency),
            0.0,
            1.0,
        )
        if temporal_request:
            confidence = clamp(confidence + (0.10 * entity_overlap) + (0.08 * alias_overlap) + (0.06 if has_event_date else 0.0) + (0.05 * date_confidence))

        reasons = []
        if shared_tokens:
            reasons.append("shared=" + ",".join(shared_tokens))
        if entity_overlap > 0.0:
            reasons.append(f"entity-overlap={entity_overlap:.2f}")
        if alias_overlap > 0.0:
            reasons.append(f"alias-overlap={alias_overlap:.2f}")
        if continuation:
            reasons.append("continuation-cue")
        if personalization:
            reasons.append("personalization-cue")
        if constraint_request:
            reasons.append("constraint-cue")
        if temporal_request:
            reasons.append("temporal-cue")
        if has_event_date:
            reasons.append("dated")
        if date_confidence > 0.0:
            reasons.append(f"date-conf={date_confidence:.2f}")
        if hit.age_days > 30:
            reasons.append(f"aged={hit.age_days}d")
        if query_coverage > 0.5:
            reasons.append("covers-query")

        if confidence >= self.use_threshold and (
            has_shared or entity_overlap > 0.0 or type_name in {"constraint", "safety", "timeline"}
        ):
            return CriticDecision(label="use", confidence=confidence, reasons=reasons)
        if confidence >= self.maybe_threshold and (has_shared or entity_overlap > 0.0 or temporal_request):
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
