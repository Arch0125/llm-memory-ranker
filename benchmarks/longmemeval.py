from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass

from memory.retrieve import infer_capture_type
from memory.utils import (
    extract_entities,
    normalize_date,
    normalize_entity,
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
            "entities": entities,
            "activity_type": _extract_activity_type(timeline_text),
            "summary": _trim(timeline_text, limit=220),
            "granularity": "timeline",
        },
    }


def _build_fact_memories(session_id, session_date, session, include_assistant_turns=True):
    date_value = normalize_date(session_date)
    for turn_index, turn in enumerate(session):
        role = (turn.get("role") or "user").strip().lower()
        content = _collapse(turn.get("content") or "")
        if not content:
            continue
        if role == "assistant" and not include_assistant_turns:
            continue
        entities = extract_entities(content)
        fact_text = _compact_turn_text(turn, session_date)
        yield {
            "text": fact_text,
            "memory_type": "event",
            "importance": 0.98 if turn.get("has_answer") else (0.78 if role == "user" else 0.55),
            "metadata": {
                "session_id": session_id,
                "session_date": session_date,
                "event_date": date_value,
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
                if fact_memory["memory_type"] == "event" and not fact_memory["metadata"]["entities"]:
                    fact_memory["memory_type"] = infer_capture_type(fact_memory["text"])
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
        normalized_targets=[normalize_entity(target) for target in targets],
        query_entities=extract_entities(question),
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
    entities = {normalize_entity(value) for value in hit.record.metadata.get("entities", [])}
    matches = []
    for raw, normalized in zip(plan.targets, plan.normalized_targets):
        if normalized and (normalized in entities or normalized in text):
            matches.append(raw)
    return matches


def temporal_bundle_score(plan, hit):
    metadata = hit.record.metadata
    score = (
        (2.2 * hit.score)
        + (1.8 * hit.critic_confidence)
        + (0.6 * hit.record.importance)
    )
    granularity = metadata.get("granularity", "")
    if metadata.get("event_date"):
        score += 0.45
    if granularity == "timeline-global":
        score += 0.7
    elif granularity == "timeline":
        score += 0.5
    elif granularity == "episode":
        score += 0.25
    elif granularity == "fact":
        score += 0.35
    if metadata.get("has_answer"):
        score += 0.2
    score += 0.35 * len(_target_matches(plan, hit))
    score += 0.1 * len(set(plan.query_entities) & set(metadata.get("entities", [])))
    return score


def render_evidence_line(hit, index=None):
    metadata = hit.record.metadata
    date_value = metadata.get("event_date") or normalize_date(metadata.get("session_date", ""))
    granularity = metadata.get("granularity") or hit.record.memory_type
    source = metadata.get("session_id") or ",".join(metadata.get("session_ids", [])[:2]) or "unknown"
    entities = ", ".join(metadata.get("entities", [])[:3]) or "n/a"
    evidence = metadata.get("fact_text") or metadata.get("summary") or hit.record.text
    prefix = f"{index}. " if index is not None else ""
    return (
        f"{prefix}date={date_value or 'unknown'} ; "
        f"kind={granularity} ; "
        f"source={source} ; "
        f"entities={entities} ; "
        f"evidence={preview(evidence, limit=180)}"
    )


def _hit_token_cost(hit, encode=None):
    return estimate_token_count(render_evidence_line(hit), encode=encode)


def select_bundled_hits(plan, hits, max_items, max_tokens, encode=None):
    ranked = sorted(hits, key=lambda hit: temporal_bundle_score(plan, hit), reverse=True)
    selected = []
    selected_ids = set()
    used_tokens = 0
    covered_targets = set()
    covered_sessions = set()

    def add_hit(hit):
        nonlocal used_tokens
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
        covered_targets.update(normalize_entity(value) for value in _target_matches(plan, hit))
        return True

    if plan.is_temporal:
        global_timeline = next(
            (hit for hit in ranked if hit.record.metadata.get("granularity") == "timeline-global"),
            None,
        )
        if global_timeline is not None:
            add_hit(global_timeline)

        for target in plan.normalized_targets:
            target_hit = next(
                (
                    hit
                    for hit in ranked
                    if target in {normalize_entity(value) for value in _target_matches(plan, hit)}
                ),
                None,
            )
            if target_hit is not None:
                add_hit(target_hit)

        for hit in ranked:
            if len(selected) >= max_items:
                break
            if hit.record.metadata.get("session_id") in covered_sessions and hit.record.metadata.get("granularity") == "fact":
                continue
            add_hit(hit)

        for hit in ranked:
            if len(selected) >= max_items:
                break
            if hit.record.metadata.get("granularity") != "episode":
                continue
            if hit.record.metadata.get("session_id") not in covered_sessions:
                continue
            add_hit(hit)
    else:
        for hit in ranked:
            if not add_hit(hit):
                continue
            if len(selected) >= max_items:
                break

    return selected, used_tokens


def assess_answerability(plan, selected_hits):
    distinct_dates = {
        hit.record.metadata.get("event_date")
        for hit in selected_hits
        if hit.record.metadata.get("event_date")
    }
    distinct_sessions = set(selected_session_ids(selected_hits))
    covered_targets = set()
    for hit in selected_hits:
        covered_targets.update(normalize_entity(value) for value in _target_matches(plan, hit))

    reasons = []
    sufficient = True
    if plan.reasoning_kind in {"ordering", "difference"}:
        if len(distinct_dates) < 2 and len(distinct_sessions) < 2:
            sufficient = False
            reasons.append("need-two-events")
        if plan.normalized_targets and len(covered_targets) < min(2, len(plan.normalized_targets)):
            sufficient = False
            reasons.append("missing-target-coverage")

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
        parts.append("For ordering questions, return only the event/item that happened first or last.")
    elif plan.reasoning_kind == "difference":
        unit = plan.unit_hint or "days"
        parts.append(f"For duration questions, return only the final duration like '7 {unit}'.")
    elif plan.reasoning_kind == "date":
        parts.append("For date questions, return only the final date or short date phrase.")
    else:
        parts.append("Return only the final answer.")
    if not answerability["sufficient"]:
        parts.append("If the evidence table is insufficient, reply exactly: Insufficient evidence.")
    parts.append(build_evidence_table(plan, selected_hits))
    return "\n\n".join(parts)


def postprocess_prediction(plan, text):
    value = _collapse(text)
    if not value:
        return value
    if "insufficient evidence" in value.lower():
        return "Insufficient evidence"
    if plan.reasoning_kind == "difference":
        match = re.search(r"\b(\d+)\s*(day|days|month|months)\b", value, flags=re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2).lower()}"
        match = re.search(r"\b(\d+)\b", value)
        if match and plan.unit_hint:
            return f"{match.group(1)} {plan.unit_hint}"
    if plan.reasoning_kind == "date":
        date_value = normalize_date(value)
        if date_value:
            return date_value
    if plan.normalized_targets:
        lowered = normalize_answer(value)
        for raw, normalized in zip(plan.targets, plan.normalized_targets):
            if normalized and normalized in lowered:
                return raw
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
