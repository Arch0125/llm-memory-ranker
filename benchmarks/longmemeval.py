from __future__ import annotations

import json
import re
import string
from calendar import monthrange
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from memory.utils import (
    extract_entities,
    normalize_date,
    preview,
)
from prompt.budget import estimate_token_count


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ABSTENTION_MARKERS = (
    "do not know",
    "don't know",
    "cannot determine",
    "can't determine",
    "insufficient evidence",
    "not enough information",
    "not mentioned",
    "unknown",
    "unsure",
)
_QUESTION_OR_RE = re.compile(
    r"(?:^|,|\bbetween\b|\bfirst\b|\blast\b|\bhappened\b).*?\b(?:the\s+)?([^,?]+?)\s+or\s+(?:the\s+)?([^?]+?)\??$",
    re.IGNORECASE,
)
_QUESTION_AND_RE = re.compile(
    r"\bbetween\s+(?:the\s+)?([^,?]+?)\s+and\s+(?:the\s+)?([^?]+?)\??$",
    re.IGNORECASE,
)
_MONTH_DAY_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)
_DATE_LINE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MONTH_DIFF_RE = re.compile(r"\b(\d+)\s*(month|months)\b", re.IGNORECASE)
_DAY_DIFF_RE = re.compile(r"\b(\d+)\s*(day|days)\b", re.IGNORECASE)
_RELATIVE_SPAN_RE = re.compile(
    r"\b(?:about|around|approximately|roughly)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(day|days|week|weeks|month|months|year|years)\s+ago\b",
    re.IGNORECASE,
)
_LAST_SPAN_RE = re.compile(r"\blast\s+(week|month|year)\b", re.IGNORECASE)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass
class QuestionPlan:
    question_id: str
    question_type: str
    question: str
    query_text: str
    question_date: str
    normalized_question_date: str
    reasoning_kind: str
    is_temporal: bool
    unit_hint: str
    targets: list[str]
    normalized_targets: list[str]
    query_entities: list[str]
    question_month: str


def load_longmemeval_instances(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("LongMemEval data must be a JSON list of instances")
    return data


def _collapse(text):
    return " ".join((text or "").split())


def _trim(text, limit=200):
    collapsed = _collapse(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _clean_answer_variant(text):
    value = _collapse(text)
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"\bis also acceptable\b\.?$", "", value, flags=re.IGNORECASE)
    value = value.strip(" .")
    return value


def acceptable_answers(answer):
    parts = re.split(r"\.\s+", answer or "")
    variants = []
    seen = set()
    for part in parts:
        cleaned = _clean_answer_variant(part)
        if not cleaned:
            continue
        normalized = normalize_answer(cleaned)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append(cleaned)
    if not variants and (answer or "").strip():
        variants.append((answer or "").strip())
    return variants


def format_session_text(session_id, session_date, session):
    header = f"Session {session_id}"
    if session_date:
        header += f" on {session_date}"
    lines = [header]
    for turn in session:
        content = _collapse(turn.get("content") or "")
        if not content:
            continue
        role = (turn.get("role") or "user").strip().capitalize()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_activity_type(text):
    lowered = (text or "").lower()
    patterns = (
        ("attend", "attendance"),
        ("bought", "purchase"),
        ("purchased", "purchase"),
        ("booked", "booking"),
        ("service", "service"),
        ("serviced", "service"),
        ("repaired", "repair"),
        ("repair", "repair"),
        ("cleaned", "cleaning"),
        ("workshop", "workshop"),
        ("webinar", "webinar"),
        ("meeting", "meeting"),
        ("festival", "festival"),
        ("mass", "service"),
        ("event", "event"),
    )
    for needle, label in patterns:
        if needle in lowered:
            return label
    return "event"


def _compact_turn_text(turn, session_date):
    role = (turn.get("role") or "user").strip().capitalize()
    content = _trim(turn.get("content") or "", limit=180)
    date_value = normalize_date(session_date)
    if date_value:
        return f"{date_value} | {role}: {content}"
    return f"{role}: {content}"


def _compact_turn_text_with_date(turn, event_date, fallback_session_date):
    role = (turn.get("role") or "user").strip().capitalize()
    content = _trim(turn.get("content") or "", limit=180)
    date_value = event_date or normalize_date(fallback_session_date)
    if date_value:
        return f"{date_value} | {role}: {content}"
    return f"{role}: {content}"


def _parse_iso_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _subtract_months(value, months):
    year = value.year
    month = value.month - months
    while month <= 0:
        year -= 1
        month += 12
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


def _subtract_years(value, years):
    year = value.year - years
    day = min(value.day, monthrange(year, value.month)[1])
    return date(year, value.month, day)


def _parse_numeric_token(value):
    token = (value or "").strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _derived_event_dates(text, session_date):
    base_date = _parse_iso_date(normalize_date(session_date))
    if not text:
        return []

    values = []
    seen = set()

    def add_date(date_value):
        if not date_value:
            return
        iso = date_value.isoformat()
        if iso in seen:
            return
        seen.add(iso)
        values.append(iso)

    explicit_iso = normalize_date(text)
    if explicit_iso:
        parsed = _parse_iso_date(explicit_iso)
        add_date(parsed)

    if base_date is not None:
        for match in _MONTH_DAY_RE.finditer(text):
            month_token = match.group(1).lower()
            month_value = _MONTHS.get(month_token)
            day_value = int(match.group(2))
            year_value = int(match.group(3)) if match.group(3) else base_date.year
            try:
                derived = date(year_value, month_value, day_value)
            except ValueError:
                continue
            if derived > base_date + timedelta(days=1) and match.group(3) is None:
                derived = date(year_value - 1, month_value, day_value)
            add_date(derived)

        for match in _RELATIVE_SPAN_RE.finditer(text):
            amount = _parse_numeric_token(match.group(1))
            unit = match.group(2).lower()
            if not amount:
                continue
            if unit.startswith("day"):
                derived = base_date - timedelta(days=amount)
            elif unit.startswith("week"):
                derived = base_date - timedelta(days=7 * amount)
            elif unit.startswith("month"):
                derived = _subtract_months(base_date, amount)
            else:
                derived = _subtract_years(base_date, amount)
            add_date(derived)

        for match in _LAST_SPAN_RE.finditer(text):
            unit = match.group(1).lower()
            if unit == "week":
                derived = base_date - timedelta(days=7)
            elif unit == "month":
                derived = _subtract_months(base_date, 1)
            else:
                derived = _subtract_years(base_date, 1)
            add_date(derived)

        lowered = text.lower()
        if "yesterday" in lowered:
            add_date(base_date - timedelta(days=1))
        if "today" in lowered:
            add_date(base_date)

    return values


def _primary_event_date(text, session_date):
    derived = _derived_event_dates(text, session_date)
    if derived:
        return derived[0], derived
    normalized = normalize_date(session_date)
    if normalized:
        return normalized, [normalized]
    return "", []


def _normalize_target_text(text):
    return normalize_answer(text)


def _phrase_matches_text(normalized_target, normalized_text):
    if not normalized_target or not normalized_text:
        return False
    if normalized_target in normalized_text:
        return True
    parts = [part for part in normalized_target.split() if len(part) >= 3]
    return bool(parts) and all(part in normalized_text for part in parts)


def _session_entities(session):
    entities = []
    seen = set()
    for turn in session:
        for entity in extract_entities(turn.get("content") or ""):
            if entity in seen:
                continue
            seen.add(entity)
            entities.append(entity)
    return entities


def _session_key_snippets(session, include_assistant_turns=True, limit=3):
    preferred = []
    fallback = []
    for turn in session:
        role = (turn.get("role") or "user").strip().lower()
        if role == "assistant" and not include_assistant_turns:
            continue
        content = _collapse(turn.get("content") or "")
        if not content:
            continue
        if turn.get("has_answer") or role == "user":
            preferred.append(turn)
        else:
            fallback.append(turn)
    ordered = preferred + fallback
    return ordered[:limit]


def _build_episode_memory(session_id, session_date, session, include_assistant_turns=True):
    snippets = _session_key_snippets(
        session,
        include_assistant_turns=include_assistant_turns,
        limit=3,
    )
    summary_lines = [_compact_turn_text(turn, session_date) for turn in snippets]
    entities = _session_entities(session)
    summary_text = "\n".join(summary_lines) if summary_lines else format_session_text(session_id, session_date, session)
    date_value = normalize_date(session_date)
    return {
        "text": f"Episode summary for {session_id}:\n{summary_text}",
        "memory_type": "episode",
        "importance": 0.95 if any(turn.get("has_answer") for turn in session) else 0.7,
        "metadata": {
            "session_id": session_id,
            "session_date": session_date,
            "event_date": date_value,
            "event_dates": [date_value] if date_value else [],
            "entities": entities,
            "activity_type": _extract_activity_type(summary_text),
            "summary": _trim(summary_text, limit=220),
            "granularity": "episode",
        },
    }


def _build_timeline_memory(session_id, session_date, session, include_assistant_turns=True):
    snippets = _session_key_snippets(
        session,
        include_assistant_turns=include_assistant_turns,
        limit=2,
    )
    date_value = normalize_date(session_date)
    facts = []
    for turn in snippets:
        content = _trim(turn.get("content") or "", limit=120)
        facts.append(f"{date_value}: {content}" if date_value else content)
    timeline_text = "\n".join(facts) if facts else format_session_text(session_id, session_date, session)
    entities = _session_entities(session)
    return {
        "text": f"Timeline evidence for {session_id}:\n{timeline_text}",
        "memory_type": "timeline",
        "importance": 0.9 if any(turn.get("has_answer") for turn in session) else 0.68,
        "metadata": {
            "session_id": session_id,
            "session_date": session_date,
            "event_date": date_value,
            "event_dates": [date_value] if date_value else [],
            "entities": entities,
            "activity_type": _extract_activity_type(timeline_text),
            "summary": _trim(timeline_text, limit=220),
            "granularity": "timeline",
        },
    }


def _build_fact_memories(session_id, session_date, session, include_assistant_turns=True):
    for turn_index, turn in enumerate(session):
        role = (turn.get("role") or "user").strip().lower()
        content = _collapse(turn.get("content") or "")
        if not content:
            continue
        if role == "assistant" and not include_assistant_turns:
            continue
        entities = extract_entities(content)
        date_value, event_dates = _primary_event_date(content, session_date)
        fact_text = _compact_turn_text_with_date(turn, date_value, session_date)
        yield {
            "text": fact_text,
            "memory_type": "event",
            "importance": 0.98 if turn.get("has_answer") else (0.78 if role == "user" else 0.55),
            "metadata": {
                "session_id": session_id,
                "session_date": session_date,
                "event_date": date_value,
                "event_dates": event_dates,
                "entities": entities,
                "activity_type": _extract_activity_type(content),
                "fact_text": _trim(content, limit=180),
                "turn_index": turn_index,
                "role": role,
                "has_answer": bool(turn.get("has_answer")),
                "granularity": "fact",
            },
        }


def _build_global_timeline_memory(instance, include_assistant_turns=True):
    session_ids = instance.get("haystack_session_ids", [])
    session_dates = instance.get("haystack_dates", [])
    sessions = instance.get("haystack_sessions", [])
    rows = []
    entities = []
    seen_entities = set()
    for session_id, session_date, session in zip(session_ids, session_dates, sessions):
        snippets = _session_key_snippets(
            session,
            include_assistant_turns=include_assistant_turns,
            limit=1,
        )
        if not snippets:
            continue
        summary = _trim(snippets[0].get("content") or "", limit=120)
        date_value = normalize_date(session_date)
        rows.append((date_value or "unknown", session_id, summary))
        for entity in _session_entities(session):
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            entities.append(entity)
    if not rows:
        return None
    rows.sort(key=lambda item: item[0])
    lines = [f"{date_value} | {session_id} | {summary}" for date_value, session_id, summary in rows]
    return {
        "text": "Global timeline evidence:\n" + "\n".join(lines),
        "memory_type": "timeline",
        "importance": 0.92,
        "metadata": {
            "session_ids": list(session_ids),
            "session_date": rows[0][0],
            "event_date": rows[0][0],
            "event_dates": [value for value, _, _ in rows if value and value != "unknown"],
            "entities": entities,
            "activity_type": "timeline",
            "summary": _trim(" ; ".join(lines), limit=240),
            "granularity": "timeline-global",
        },
    }


def iter_history_memories(instance, granularity="hybrid", include_assistant_turns=True):
    session_ids = instance.get("haystack_session_ids", [])
    session_dates = instance.get("haystack_dates", [])
    sessions = instance.get("haystack_sessions", [])

    if granularity not in {"turn", "session", "hybrid"}:
        raise ValueError("granularity must be 'turn', 'session', or 'hybrid'")

    if granularity == "hybrid":
        global_timeline = _build_global_timeline_memory(
            instance,
            include_assistant_turns=include_assistant_turns,
        )
        if global_timeline is not None:
            global_timeline["metadata"].update(
                {
                    "question_id": instance.get("question_id"),
                }
            )
            yield global_timeline

    for session_id, session_date, session in zip(session_ids, session_dates, sessions):
        if granularity in {"session", "hybrid"}:
            episode = _build_episode_memory(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            )
            episode["metadata"].update({"question_id": instance.get("question_id")})
            yield episode
        if granularity == "hybrid":
            timeline = _build_timeline_memory(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            )
            timeline["metadata"].update({"question_id": instance.get("question_id")})
            yield timeline
        if granularity in {"turn", "hybrid"}:
            for fact_memory in _build_fact_memories(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            ):
                fact_memory["metadata"].update({"question_id": instance.get("question_id")})
                yield fact_memory


def build_query_text(instance, include_question_date=True):
    question = (instance.get("question") or "").strip()
    question_date = (instance.get("question_date") or "").strip()
    if include_question_date and question_date:
        normalized = normalize_date(question_date)
        if normalized:
            return f"Question date: {normalized}\nQuestion: {question}"
        return f"Question date: {question_date}\nQuestion: {question}"
    return question


def analyze_question(instance, include_question_date=True):
    question = (instance.get("question") or "").strip()
    lowered = question.lower()
    question_date = (instance.get("question_date") or "").strip()
    normalized_question_date = normalize_date(question_date)
    quoted_targets = [match.strip() for match in re.findall(r"['\"]([^'\"]+)['\"]", question)]

    targets = []
    if len(quoted_targets) >= 2:
        targets = quoted_targets[:2]
    else:
        match = _QUESTION_OR_RE.search(question)
        if match:
            targets = [match.group(1).strip(" '\""), match.group(2).strip(" '\"")]
        else:
            match = _QUESTION_AND_RE.search(question)
            if match:
                targets = [match.group(1).strip(" '\""), match.group(2).strip(" '\"")]

    reasoning_kind = "factual"
    unit_hint = ""
    if "how many days" in lowered:
        reasoning_kind = "difference"
        unit_hint = "days"
    elif "how many months" in lowered:
        reasoning_kind = "difference"
        unit_hint = "months"
    elif any(token in lowered for token in ("first", "last", "before", "after")):
        reasoning_kind = "ordering"
    elif "what was the date" in lowered or "which date" in lowered:
        reasoning_kind = "date"

    return QuestionPlan(
        question_id=instance.get("question_id", ""),
        question_type=instance.get("question_type", ""),
        question=question,
        query_text=build_query_text(instance, include_question_date=include_question_date),
        question_date=question_date,
        normalized_question_date=normalized_question_date,
        reasoning_kind=reasoning_kind,
        is_temporal="temporal" in instance.get("question_type", "") or reasoning_kind in {"ordering", "difference", "date"},
        unit_hint=unit_hint,
        targets=targets,
        normalized_targets=[_normalize_target_text(target) for target in targets],
        query_entities=extract_entities(question),
        question_month=(normalized_question_date or "")[5:7] if normalized_question_date else "",
    )


def normalize_answer(text):
    normalized = (text or "").lower().replace("\u2019", "'")
    normalized = normalized.translate(_PUNCT_TABLE)
    normalized = _ARTICLES_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def exact_match_score(prediction, answer):
    variants = acceptable_answers(answer)
    if not variants:
        return 0.0
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(variant) for variant in variants))


def contains_match_score(prediction, answer):
    normalized_prediction = normalize_answer(prediction)
    variants = acceptable_answers(answer)
    if not normalized_prediction or not variants:
        return 0.0
    return float(
        any(
            normalize_answer(variant) in normalized_prediction
            or normalized_prediction in normalize_answer(variant)
            for variant in variants
        )
    )


def token_f1_score(prediction, answer):
    variants = acceptable_answers(answer)
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens or not variants:
        return 0.0

    best = 0.0
    for variant in variants:
        answer_tokens = normalize_answer(variant).split()
        if not answer_tokens:
            continue
        overlap = Counter(prediction_tokens) & Counter(answer_tokens)
        shared = sum(overlap.values())
        if shared == 0:
            continue
        precision = shared / len(prediction_tokens)
        recall = shared / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def abstention_score(prediction, is_abstention_question):
    if not is_abstention_question:
        return None
    normalized_prediction = normalize_answer(prediction)
    return float(any(marker in normalized_prediction for marker in _ABSTENTION_MARKERS))


def selected_session_ids(selected_hits):
    seen = set()
    ordered = []
    for hit in selected_hits:
        session_ids = []
        if hit.record.metadata.get("session_id"):
            session_ids.append(hit.record.metadata["session_id"])
        session_ids.extend(hit.record.metadata.get("session_ids", []))
        for session_id in session_ids:
            if not session_id or session_id in seen:
                continue
            seen.add(session_id)
            ordered.append(session_id)
    return ordered


def selected_session_recall(selected_hits, answer_session_ids):
    gold = set(answer_session_ids or [])
    if not gold:
        return None
    selected = set(selected_session_ids(selected_hits))
    return len(selected & gold) / len(gold)


def question_policy(plan):
    policy = {
        "top_k": 24,
        "max_items": 8,
        "token_budget": 768,
        "critic_threshold": 0.52,
        "maybe_threshold": 0.42,
        "similarity_threshold": 0.12,
    }
    if plan.is_temporal:
        policy.update(
            {
                "top_k": 40,
                "max_items": 12,
                "token_budget": 1200,
                "critic_threshold": 0.44,
                "maybe_threshold": 0.34,
                "similarity_threshold": 0.08,
            }
        )
    return policy


def _target_matches(plan, hit):
    if not plan.normalized_targets:
        return []
    text = normalize_answer(hit.record.text)
    metadata = hit.record.metadata
    entities = {
        _normalize_target_text(value)
        for value in metadata.get("entities", [])
        if value
    }
    matches = []
    for raw, normalized in zip(plan.targets, plan.normalized_targets):
        if normalized and (
            normalized in entities
            or _phrase_matches_text(normalized, text)
            or any(_phrase_matches_text(normalized, entity) for entity in entities)
        ):
            matches.append(raw)
    return matches


def _target_coverage(plan, hit):
    matches = {_normalize_target_text(value) for value in _target_matches(plan, hit)}
    metadata = hit.record.metadata
    source_text = " ".join(
        value
        for value in (
            hit.record.text,
            metadata.get("summary", ""),
            metadata.get("fact_text", ""),
        )
        if value
    )
    normalized_text = normalize_answer(source_text)
    for normalized in plan.normalized_targets:
        if not normalized:
            continue
        if _phrase_matches_text(normalized, normalized_text):
            matches.add(normalized)
    return matches


def _hit_event_dates(hit):
    metadata = hit.record.metadata
    dates = []
    for value in metadata.get("event_dates", []):
        if value and value not in dates:
            dates.append(value)
    primary = metadata.get("event_date")
    if primary and primary not in dates:
        dates.append(primary)
    return dates


def _granularity(hit):
    return hit.record.metadata.get("granularity", "")


def _granularity_priority(hit):
    granularity = _granularity(hit)
    if granularity == "fact":
        return 4
    if granularity == "timeline":
        return 3
    if granularity == "timeline-global":
        return 2
    if granularity == "episode":
        return 1
    return 0


def _candidate_selection_score(plan, hit):
    score = temporal_bundle_score(plan, hit)
    score += 0.35 * _granularity_priority(hit)
    score += 0.45 * len(_target_matches(plan, hit))
    if _hit_event_dates(hit):
        score += 0.15
    if hit.record.memory_type == "ephemeral" and not _target_matches(plan, hit):
        score -= 0.8
    return score


def temporal_bundle_score(plan, hit):
    metadata = hit.record.metadata
    score = (
        (2.2 * hit.score)
        + (1.8 * hit.critic_confidence)
        + (0.6 * hit.record.importance)
    )
    granularity = metadata.get("granularity", "")
    if _hit_event_dates(hit):
        score += 0.45
    if granularity == "timeline-global":
        score += 0.7
    elif granularity == "timeline":
        score += 0.35
    elif granularity == "episode":
        score -= 0.15
    elif granularity == "fact":
        score += 0.75
    if metadata.get("has_answer"):
        score += 0.2
    score += 0.35 * len(_target_matches(plan, hit))
    score += 0.1 * len(set(plan.query_entities) & set(metadata.get("entities", [])))
    if hit.record.memory_type == "ephemeral" and not _target_matches(plan, hit):
        score -= 0.6
    return score


def render_evidence_line(hit, index=None):
    metadata = hit.record.metadata
    date_value = metadata.get("event_date") or normalize_date(metadata.get("session_date", ""))
    anchor_date = normalize_date(metadata.get("session_date", ""))
    granularity = metadata.get("granularity") or hit.record.memory_type
    source = metadata.get("session_id") or ",".join(metadata.get("session_ids", [])[:2]) or "unknown"
    entities = ", ".join(metadata.get("entities", [])[:3]) or "n/a"
    evidence = metadata.get("fact_text") or metadata.get("summary") or hit.record.text
    prefix = f"{index}. " if index is not None else ""
    anchor_fragment = ""
    if anchor_date and anchor_date != date_value:
        anchor_fragment = f" ; session_date={anchor_date}"
    return (
        f"{prefix}date={date_value or 'unknown'} ; "
        f"kind={granularity} ; "
        f"source={source} ; "
        f"entities={entities} ; "
        f"evidence={preview(evidence, limit=180)}"
        f"{anchor_fragment}"
    )


def _hit_token_cost(hit, encode=None):
    return estimate_token_count(render_evidence_line(hit), encode=encode)


def select_bundled_hits(plan, hits, max_items, max_tokens, encode=None):
    ranked = sorted(hits, key=lambda hit: _candidate_selection_score(plan, hit), reverse=True)
    selected = []
    selected_ids = set()
    used_tokens = 0
    covered_targets = set()
    covered_sessions = set()
    global_timeline_selected = False

    def add_hit(hit):
        nonlocal used_tokens, global_timeline_selected
        if hit.record.memory_id in selected_ids:
            return False
        token_cost = _hit_token_cost(hit, encode=encode)
        hit.token_cost = token_cost
        if len(selected) >= max_items or used_tokens + token_cost > max_tokens:
            return False
        selected.append(hit)
        selected_ids.add(hit.record.memory_id)
        used_tokens += token_cost
        covered_sessions.update(selected_session_ids([hit]))
        covered_targets.update(_target_coverage(plan, hit))
        if hit.record.metadata.get("granularity") == "timeline-global":
            global_timeline_selected = True
        return True

    def current_answerability():
        return assess_answerability(plan, selected)

    def useful_increment(hit):
        hit_targets = _target_coverage(plan, hit)
        hit_sessions = set(selected_session_ids([hit]))
        hit_dates = set(_hit_event_dates(hit))
        current_dates = {
            value
            for item in selected
            for value in _hit_event_dates(item)
        }
        if hit.record.memory_type == "ephemeral" and not hit_targets and not hit_dates:
            return False
        if hit_targets - covered_targets:
            return True
        if hit_sessions - covered_sessions:
            return True
        if hit_dates - current_dates:
            return True
        return _granularity(hit) == "fact" and bool(hit_targets)

    if plan.is_temporal:
        global_timeline = next(
            (hit for hit in ranked if hit.record.metadata.get("granularity") == "timeline-global"),
            None,
        )
        if global_timeline is not None and len(plan.normalized_targets) >= 2:
            add_hit(global_timeline)

        for target in plan.normalized_targets:
            candidates = [
                hit
                for hit in ranked
                if target in _target_coverage(plan, hit)
            ]
            candidates.sort(
                key=lambda hit: (
                    _granularity_priority(hit),
                    1 if _hit_event_dates(hit) else 0,
                    _candidate_selection_score(plan, hit),
                ),
                reverse=True,
            )
            target_hit = next(iter(candidates), None)
            if target_hit is not None:
                add_hit(target_hit)

        for hit in ranked:
            if len(selected) >= max_items:
                break
            if hit.record.memory_id in selected_ids:
                continue
            if not useful_increment(hit):
                continue
            if _granularity(hit) == "episode" and current_answerability()["sufficient"]:
                continue
            add_hit(hit)
            if current_answerability()["sufficient"] and len(selected) >= max(2, min(4, len(plan.normalized_targets) + 1)):
                break

        if not current_answerability()["sufficient"]:
            for hit in ranked:
                if len(selected) >= max_items:
                    break
                if hit.record.memory_id in selected_ids:
                    continue
                if _granularity(hit) == "episode" and current_answerability()["sufficient"]:
                    continue
                if _granularity(hit) in {"fact", "timeline"} or (
                    global_timeline_selected and _granularity(hit) == "episode"
                ):
                    add_hit(hit)
                if current_answerability()["sufficient"] and len(selected) >= max(2, min(4, len(plan.normalized_targets) + 1)):
                    break
    else:
        for hit in ranked:
            if not add_hit(hit):
                continue
            if len(selected) >= max_items:
                break

    return selected, used_tokens


def assess_answerability(plan, selected_hits):
    distinct_dates = {
        value
        for hit in selected_hits
        for value in _hit_event_dates(hit)
        if value
    }
    distinct_sessions = set(selected_session_ids(selected_hits))
    covered_targets = set()
    for hit in selected_hits:
        covered_targets.update(_target_coverage(plan, hit))

    reasons = []
    sufficient = True
    if plan.reasoning_kind == "difference":
        if len(distinct_dates) < 2:
            sufficient = False
            reasons.append("need-two-dates")
        if plan.normalized_targets:
            required_coverage = min(2, len(plan.normalized_targets))
            if len(covered_targets) < required_coverage:
                sufficient = False
                reasons.append("missing-target-coverage")
    elif plan.reasoning_kind == "ordering":
        if len(distinct_dates) < 2 and len(distinct_sessions) < 2:
            sufficient = False
            reasons.append("need-two-events")
        if plan.normalized_targets:
            required_coverage = min(2, len(plan.normalized_targets))
            if len(covered_targets) < required_coverage:
                has_global_timeline = any(
                    hit.record.metadata.get("granularity") == "timeline-global"
                    for hit in selected_hits
                )
                if not (has_global_timeline and len(distinct_sessions) >= 2):
                    sufficient = False
                    reasons.append("missing-target-coverage")
    elif plan.reasoning_kind == "date":
        if not distinct_dates:
            sufficient = False
            reasons.append("missing-date")

    return {
        "sufficient": sufficient,
        "reasons": reasons or ["enough-evidence"],
        "distinct_dates": sorted(value for value in distinct_dates if value),
        "distinct_sessions": sorted(distinct_sessions),
        "covered_targets": sorted(covered_targets),
    }


def build_evidence_table(plan, selected_hits):
    if not selected_hits:
        return "Evidence:\n- none"
    lines = ["Evidence table:"]
    for index, hit in enumerate(selected_hits, start=1):
        lines.append(render_evidence_line(hit, index=index))
    return "\n".join(lines)


def build_benchmark_instructions(plan, selected_hits, answerability, base_system_prompt):
    parts = [base_system_prompt]
    parts.append("Use only the evidence table when it is present. Do not invent dates or events.")
    if plan.reasoning_kind == "ordering":
        parts.append("For ordering questions, return only the event/item that happened first or last, with no explanation.")
        if plan.targets:
            parts.append("Valid answer options: " + " | ".join(plan.targets))
    elif plan.reasoning_kind == "difference":
        unit = plan.unit_hint or "days"
        parts.append(f"For duration questions, return only the final duration like '7 {unit}', with no explanation.")
    elif plan.reasoning_kind == "date":
        parts.append("For date questions, return only the final date or short date phrase, with no explanation.")
    else:
        parts.append("Return only the final answer.")
    if answerability["sufficient"]:
        parts.append("The evidence table is sufficient. Answer directly and do not reply with 'Insufficient evidence'.")
    else:
        parts.append("If the evidence table is insufficient, reply exactly: Insufficient evidence.")
    parts.append(build_evidence_table(plan, selected_hits))
    return "\n\n".join(parts)


def postprocess_prediction(plan, text):
    value = _collapse(text)
    if not value:
        return value
    lowered_value = value.lower()
    has_abstention_marker = any(
        marker in lowered_value
        for marker in (
            "insufficient evidence",
            "cannot determine",
            "not possible to determine",
            "do not know",
            "i do not know",
            "there is no information",
        )
    )
    if plan.reasoning_kind == "difference":
        unit = plan.unit_hint or "days"
        if unit == "months":
            match = _MONTH_DIFF_RE.search(value)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
        else:
            match = _DAY_DIFF_RE.search(value)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
        for line in value.splitlines():
            lower_line = line.lower()
            if "cannot determine" in lower_line or "not possible" in lower_line or "insufficient" in lower_line:
                continue
            match = (_MONTH_DIFF_RE if unit == "months" else _DAY_DIFF_RE).search(line)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
    if plan.reasoning_kind == "date":
        month_day = _MONTH_DAY_RE.search(value)
        if month_day:
            month = month_day.group(1).capitalize()
            day = month_day.group(2)
            return f"{month} {day}"
        date_value = normalize_date(value)
        if date_value:
            return date_value
    if "what was the date" in plan.question.lower() or "which date" in plan.question.lower():
        month_day = _MONTH_DAY_RE.search(value)
        if month_day:
            month = month_day.group(1).capitalize()
            day = month_day.group(2)
            return f"{month} {day}"
    if plan.normalized_targets:
        lowered = normalize_answer(value)
        for raw, normalized in zip(plan.targets, plan.normalized_targets):
            if normalized and _phrase_matches_text(normalized, lowered):
                return raw
    if has_abstention_marker:
        return "Insufficient evidence"
    return value.splitlines()[0].strip(" \"'")


def summarize_records(records):
    summary = {
        "examples": len(records),
        "exact_match": 0.0,
        "contains_match": 0.0,
        "token_f1": 0.0,
        "abstention_accuracy": None,
        "avg_selected_memory_count": 0.0,
        "avg_selected_session_recall": None,
        "avg_answerable": None,
        "by_question_type": {},
    }
    if not records:
        return summary

    summary["exact_match"] = sum(row["exact_match"] for row in records) / len(records)
    summary["contains_match"] = sum(row["contains_match"] for row in records) / len(records)
    summary["token_f1"] = sum(row["token_f1"] for row in records) / len(records)
    summary["avg_selected_memory_count"] = sum(
        row["selected_memory_count"] for row in records
    ) / len(records)
    summary["avg_answerable"] = sum(1.0 if row["answerable"] else 0.0 for row in records) / len(records)

    recall_values = [
        row["selected_session_recall"]
        for row in records
        if row["selected_session_recall"] is not None
    ]
    if recall_values:
        summary["avg_selected_session_recall"] = sum(recall_values) / len(recall_values)

    abstention_values = [
        row["abstention_accuracy"]
        for row in records
        if row["abstention_accuracy"] is not None
    ]
    if abstention_values:
        summary["abstention_accuracy"] = sum(abstention_values) / len(abstention_values)

    grouped = {}
    for row in records:
        grouped.setdefault(row["question_type"], []).append(row)

    for question_type, items in sorted(grouped.items()):
        grouped_recall = [
            item["selected_session_recall"]
            for item in items
            if item["selected_session_recall"] is not None
        ]
        grouped_abstention = [
            item["abstention_accuracy"]
            for item in items
            if item["abstention_accuracy"] is not None
        ]
        summary["by_question_type"][question_type] = {
            "examples": len(items),
            "exact_match": sum(item["exact_match"] for item in items) / len(items),
            "contains_match": sum(item["contains_match"] for item in items) / len(items),
            "token_f1": sum(item["token_f1"] for item in items) / len(items),
            "avg_selected_memory_count": sum(
                item["selected_memory_count"] for item in items
            )
            / len(items),
            "avg_selected_session_recall": (
                sum(grouped_recall) / len(grouped_recall) if grouped_recall else None
            ),
            "avg_answerable": sum(1.0 if item["answerable"] else 0.0 for item in items) / len(items),
            "abstention_accuracy": (
                sum(grouped_abstention) / len(grouped_abstention)
                if grouped_abstention
                else None
            ),
        }

    return summary
