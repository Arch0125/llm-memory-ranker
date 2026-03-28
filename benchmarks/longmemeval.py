from __future__ import annotations

import json
import re
import string
from collections import Counter

from memory.retrieve import infer_capture_type


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ABSTENTION_MARKERS = (
    "do not know",
    "don't know",
    "cannot determine",
    "can't determine",
    "not enough information",
    "not mentioned",
    "unknown",
    "unsure",
)


def load_longmemeval_instances(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("LongMemEval data must be a JSON list of instances")
    return data


def format_session_text(session_id, session_date, session):
    header = f"Session {session_id}"
    if session_date:
        header += f" on {session_date}"
    lines = [header]
    for turn in session:
        content = " ".join((turn.get("content") or "").split())
        if not content:
            continue
        role = (turn.get("role") or "user").strip().capitalize()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def iter_history_memories(instance, granularity="turn", include_assistant_turns=True):
    session_ids = instance.get("haystack_session_ids", [])
    session_dates = instance.get("haystack_dates", [])
    sessions = instance.get("haystack_sessions", [])

    for session_id, session_date, session in zip(session_ids, session_dates, sessions):
        if granularity == "session":
            text = format_session_text(session_id, session_date, session)
            yield {
                "text": text,
                "memory_type": "ephemeral",
                "importance": 0.95 if any(turn.get("has_answer") for turn in session) else 0.65,
                "metadata": {
                    "session_id": session_id,
                    "session_date": session_date,
                    "question_id": instance.get("question_id"),
                    "granularity": "session",
                },
            }
            continue

        for turn_index, turn in enumerate(session):
            role = (turn.get("role") or "user").strip().lower()
            content = " ".join((turn.get("content") or "").split())
            if not content:
                continue
            if role == "assistant" and not include_assistant_turns:
                continue

            text = format_session_text(
                session_id,
                session_date,
                [{"role": role, "content": content}],
            )
            yield {
                "text": text,
                "memory_type": infer_capture_type(content),
                "importance": 0.95 if turn.get("has_answer") else (0.75 if role == "user" else 0.6),
                "metadata": {
                    "session_id": session_id,
                    "session_date": session_date,
                    "turn_index": turn_index,
                    "role": role,
                    "has_answer": bool(turn.get("has_answer")),
                    "question_id": instance.get("question_id"),
                    "granularity": "turn",
                },
            }


def build_query_text(instance, include_question_date=True):
    question = (instance.get("question") or "").strip()
    question_date = (instance.get("question_date") or "").strip()
    if include_question_date and question_date:
        return f"Question date: {question_date}\nQuestion: {question}"
    return question


def normalize_answer(text):
    normalized = (text or "").lower().replace("\u2019", "'")
    normalized = normalized.translate(_PUNCT_TABLE)
    normalized = _ARTICLES_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def exact_match_score(prediction, answer):
    return float(normalize_answer(prediction) == normalize_answer(answer))


def contains_match_score(prediction, answer):
    normalized_prediction = normalize_answer(prediction)
    normalized_answer = normalize_answer(answer)
    if not normalized_prediction or not normalized_answer:
        return 0.0
    return float(
        normalized_answer in normalized_prediction
        or normalized_prediction in normalized_answer
    )


def token_f1_score(prediction, answer):
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return 0.0

    overlap = Counter(prediction_tokens) & Counter(answer_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0

    precision = shared / len(prediction_tokens)
    recall = shared / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def abstention_score(prediction, is_abstention_question):
    if not is_abstention_question:
        return None
    normalized_prediction = normalize_answer(prediction)
    return float(any(marker in normalized_prediction for marker in _ABSTENTION_MARKERS))


def selected_session_ids(selected_hits):
    seen = set()
    ordered = []
    for hit in selected_hits:
        session_id = hit.record.metadata.get("session_id")
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


def summarize_records(records):
    summary = {
        "examples": len(records),
        "exact_match": 0.0,
        "contains_match": 0.0,
        "token_f1": 0.0,
        "abstention_accuracy": None,
        "avg_selected_memory_count": 0.0,
        "avg_selected_session_recall": None,
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
            "abstention_accuracy": (
                sum(grouped_abstention) / len(grouped_abstention)
                if grouped_abstention
                else None
            ),
        }

    return summary
