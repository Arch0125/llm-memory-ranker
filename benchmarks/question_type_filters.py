from __future__ import annotations

import re


QUESTION_TYPE_ALIASES = {
    "temporal": "temporal-reasoning",
    "temporal-reasoning": "temporal-reasoning",
    "temporal_reasoning": "temporal-reasoning",
    "multi": "multi-session",
    "multisession": "multi-session",
    "multi-session": "multi-session",
    "multi_session": "multi-session",
    "assistant": "single-session-assistant",
    "single-session-assistant": "single-session-assistant",
    "single_session_assistant": "single-session-assistant",
    "user": "single-session-user",
    "single-session-user": "single-session-user",
    "single_session_user": "single-session-user",
    "preference": "single-session-preference",
    "single-session-preference": "single-session-preference",
    "single_session_preference": "single-session-preference",
    "knowledge": "knowledge-update",
    "knowledge-update": "knowledge-update",
    "knowledge_update": "knowledge-update",
}


def normalize_question_types(value: str) -> str:
    if not value:
        return ""
    canonical = []
    seen = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        lookup = re.sub(r"\s+", "-", item.strip().lower())
        lookup = lookup.replace("_", "-")
        resolved = QUESTION_TYPE_ALIASES.get(lookup, item)
        resolved = resolved.strip()
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        canonical.append(resolved)
    return ",".join(canonical)


def question_type_matches(question_type: str, normalized_filter: str) -> bool:
    """Return True if ``question_type`` is selected by ``normalized_filter``.

    ``normalized_filter`` is the comma-separated string returned by
    :func:`normalize_question_types`. An empty filter matches everything.
    """
    if not normalized_filter:
        return True
    allowed = {item.strip() for item in normalized_filter.split(",") if item.strip()}
    if not allowed:
        return True
    return (question_type or "").strip() in allowed


def question_type_slug(value: str) -> str:
    normalized = normalize_question_types(value)
    if not normalized:
        return ""
    parts = []
    for item in normalized.split(","):
        slug = re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_")
        if slug:
            parts.append(slug)
    return "__".join(parts)
