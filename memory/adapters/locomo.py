"""LoCoMo adapter (Long Conversational Memory benchmark).

LoCoMo (https://arxiv.org/abs/2402.17753) tests very-long-horizon dialogue
memory. Each instance is a multi-session conversation between two speakers
plus a list of evaluation questions. The published format puts conversational
sessions under `conversation` keyed by `session_<n>` and dates as
`session_<n>_date_time` strings.

This adapter:
  * loads a JSON list of (conversation, qa) pairs;
  * ingests each session via the generic `memory.granularity` builders;
  * answers questions using the generic critic + retrieval pipeline;
  * scores predictions with token-F1 + contains-match (LoCoMo provides free-form gold answers).

The scoring helpers are intentionally simple — LoCoMo's official eval involves
GPT-4 judges, which we don't replicate; users who want that can wrap this
adapter and hook in their own judge.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence

from ..extractors import extract_iso_dates, normalize_iso_date
from ..granularity import (
    build_episode_memory,
    build_fact_memories,
    build_global_timeline_memory,
    build_timeline_memory,
)
from ..postprocess import extract_final_answer_marker
from ..query import QueryPlan
from ..utils import extract_entities, normalize_entity, tokenize
from .base import BenchmarkAdapter, BenchmarkInstance, register_adapter


_DATE_TIME_RE = re.compile(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", re.IGNORECASE)
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_locomo_date(text):
    """LoCoMo session dates look like '8 may 2023 13:25'. Return ISO yyyy-mm-dd."""
    if not text:
        return ""
    m = _DATE_TIME_RE.search(text)
    if not m:
        return normalize_iso_date(text)
    day = int(m.group(1))
    month = _MONTHS.get(m.group(2).lower(), 0)
    year = int(m.group(3))
    if not month:
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalize_conversation(raw):
    """Coerce LoCoMo `conversation` blob into the granularity module's schema."""
    sessions = []
    if not isinstance(raw, dict):
        return sessions
    for key, value in raw.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        session_id = key
        session_date = _parse_locomo_date(raw.get(f"{key}_date_time", ""))
        turns = []
        if isinstance(value, list):
            for turn in value:
                if isinstance(turn, dict):
                    turns.append({
                        "role": "user" if turn.get("speaker") in {"user", None} else "assistant",
                        "content": turn.get("text") or turn.get("dia_id") or "",
                        "speaker": turn.get("speaker", ""),
                    })
        sessions.append({
            "session_id": session_id,
            "session_date": session_date,
            "turns": turns,
        })
    sessions.sort(key=lambda s: s["session_date"] or "")
    return sessions


@register_adapter("locomo")
class LoCoMoAdapter:
    """LoCoMo benchmark adapter."""

    name = "locomo"

    def __init__(self, *, granularity: str = "hybrid", include_assistant_turns: bool = True):
        self.granularity = granularity
        self.include_assistant_turns = include_assistant_turns

    # ---- loading -----------------------------------------------------------

    def load(self, path: str) -> list[BenchmarkInstance]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("LoCoMo data must be a JSON list of conversation/qa records")

        instances: list[BenchmarkInstance] = []
        for record in data:
            conversation = _normalize_conversation(record.get("conversation") or {})
            for qa in record.get("qa") or []:
                question = (qa.get("question") or "").strip()
                if not question:
                    continue
                instances.append(
                    BenchmarkInstance(
                        instance_id=qa.get("question_id") or qa.get("dia_id") or f"locomo-{len(instances)}",
                        question=question,
                        answer=qa.get("answer", ""),
                        raw={
                            "question": question,
                            "answer": qa.get("answer", ""),
                            "category": qa.get("category", ""),
                            "evidence": qa.get("evidence", []),
                            "sessions": conversation,
                        },
                        question_type=str(qa.get("category") or "").lower(),
                        extras={
                            "evidence": qa.get("evidence", []),
                            "category": qa.get("category", ""),
                        },
                    )
                )
        return instances

    def filter(
        self,
        instances: Sequence[BenchmarkInstance],
        *,
        question_types: str = "",
        start_index: int = 0,
        max_examples: int = 0,
    ) -> list[BenchmarkInstance]:
        result = list(instances)[start_index:]
        if question_types:
            wanted = {q.strip().lower() for q in question_types.split(",") if q.strip()}
            result = [inst for inst in result if inst.question_type in wanted]
        if max_examples > 0:
            result = result[:max_examples]
        return result

    # ---- ingest ------------------------------------------------------------

    def ingest(self, instance: BenchmarkInstance) -> Iterable[dict]:
        sessions = list(instance.raw.get("sessions") or [])
        if self.granularity in {"hybrid", "session", "episode"}:
            for session in sessions:
                yield build_episode_memory(session, include_assistant_turns=self.include_assistant_turns)
        if self.granularity in {"hybrid", "timeline"}:
            for session in sessions:
                yield build_timeline_memory(session, include_assistant_turns=self.include_assistant_turns)
        if self.granularity in {"hybrid", "turn", "fact"}:
            for session in sessions:
                yield from build_fact_memories(session, include_assistant_turns=self.include_assistant_turns)
        global_view = build_global_timeline_memory(sessions, include_assistant_turns=False)
        if global_view is not None:
            yield global_view

    # ---- analyze -----------------------------------------------------------

    def analyze(self, instance: BenchmarkInstance, *, include_anchor_date: bool = True) -> QueryPlan:
        question = instance.question
        lowered = question.lower()
        category = (instance.question_type or "").lower()

        reasoning_kind = "factual"
        unit_hint = ""
        ordering_direction = "first"
        is_temporal = False
        is_multi_source = False

        if "how many" in lowered:
            reasoning_kind = "difference"
            for unit in ("days", "weeks", "months", "years"):
                if unit in lowered:
                    unit_hint = unit
                    is_temporal = True
                    break
        elif "when" in lowered or "what date" in lowered or "which date" in lowered:
            reasoning_kind = "date"
            is_temporal = True
            ordering_direction = "last" if " last" in lowered else "first"
        elif "first" in lowered or "last" in lowered or "earliest" in lowered or "latest" in lowered:
            reasoning_kind = "ordering"
            is_temporal = True
            ordering_direction = "last" if any(t in lowered for t in (" last", "latest")) else "first"
        elif category in {"5", "5_temporal"}:
            is_temporal = True
            reasoning_kind = "ordering"
        elif category in {"3", "3_multi", "3_multi_hop"}:
            is_multi_source = True

        anchor_dates = extract_iso_dates(question)
        anchor_date = anchor_dates[0] if anchor_dates else ""
        entities = extract_entities(question)
        targets = entities[:2]

        return QueryPlan(
            query_id=instance.instance_id,
            question=question,
            query_text=(f"{anchor_date} | {question}" if anchor_date and include_anchor_date else question),
            question_type=category,
            reasoning_kind=reasoning_kind,
            is_multi_source=is_multi_source,
            is_temporal=is_temporal,
            unit_hint=unit_hint,
            targets=targets,
            normalized_targets=[normalize_entity(t) for t in targets],
            entities=entities,
            anchor_date=anchor_date,
            ordering_direction=ordering_direction,
            focus_terms=entities[:4],
            metadata={"category": category, "evidence": instance.extras.get("evidence", [])},
        )

    # ---- postprocess + score ----------------------------------------------

    def postprocess(self, plan: QueryPlan, raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""
        # Prefer the canonical "Final answer:" marker when the model emits it.
        marker = extract_final_answer_marker(text)
        if marker:
            return marker
        first_line = text.split("\n", 1)[0].strip()
        if first_line.lower().startswith("answer:"):
            first_line = first_line[len("answer:"):].strip()
        return first_line

    def score(self, plan: QueryPlan, prediction: str, instance: BenchmarkInstance) -> dict:
        gold = str(instance.answer or "").strip()
        pred = (prediction or "").strip()
        gold_norm = gold.lower()
        pred_norm = pred.lower()

        contains = 1.0 if gold_norm and gold_norm in pred_norm else 0.0
        exact = 1.0 if gold_norm and pred_norm == gold_norm else 0.0

        gold_tokens = set(tokenize(gold))
        pred_tokens = set(tokenize(pred))
        if not gold_tokens or not pred_tokens:
            f1 = 0.0
        else:
            shared = gold_tokens & pred_tokens
            if not shared:
                f1 = 0.0
            else:
                precision = len(shared) / len(pred_tokens)
                recall = len(shared) / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall)

        return {
            "exact_match": exact,
            "contains_match": contains,
            "token_f1": f1,
            "abstention": None,
        }
