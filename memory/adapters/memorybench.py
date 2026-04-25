"""MemoryBench / generic JSONL adapter.

This is the "bring your own dataset" adapter. It expects a JSONL file where
each line is a record describing a single haystack + question:

    {
        "id": "bench-1",
        "question": "Where did Alice go on her birthday?",
        "answer": "Tokyo",
        "question_type": "single_session",   // free-form
        "anchor_date": "2024-09-12",         // optional
        "sessions": [
            {
                "session_id": "s1",
                "session_date": "2024-09-12",
                "turns": [
                    {"role": "user",      "content": "Headed to Tokyo for my birthday!"},
                    {"role": "assistant", "content": "Have a great trip."}
                ]
            },
            ...
        ],
        "metadata": {...}                    // optional, surfaced into the QueryPlan
    }

The adapter applies the generic granularity builders, runs a lightweight
question analysis (date / ordering / factual), and scores with token-F1 and
contains-match. Subclass and override anything that doesn't fit your dataset.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from ..extractors import extract_iso_dates, normalize_iso_date
from ..granularity import (
    build_episode_memory,
    build_fact_memories,
    build_global_timeline_memory,
    build_timeline_memory,
    iter_sessions,
)
from ..query import QueryPlan
from ..utils import extract_entities, normalize_entity, tokenize
from .base import BenchmarkAdapter, BenchmarkInstance, register_adapter


def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@register_adapter("memorybench")
class MemoryBenchAdapter:
    """Generic JSONL-based memory benchmark adapter."""

    name = "memorybench"

    def __init__(self, *, granularity: str = "hybrid", include_assistant_turns: bool = True):
        self.granularity = granularity
        self.include_assistant_turns = include_assistant_turns

    # ---- loading -----------------------------------------------------------

    def load(self, path: str) -> list[BenchmarkInstance]:
        if path.endswith(".jsonl"):
            rows = _read_jsonl(path)
        else:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            rows = data if isinstance(data, list) else data.get("records", [])

        instances: list[BenchmarkInstance] = []
        for row in rows:
            question = (row.get("question") or "").strip()
            if not question:
                continue
            instances.append(
                BenchmarkInstance(
                    instance_id=str(row.get("id") or row.get("question_id") or f"mb-{len(instances)}"),
                    question=question,
                    answer=row.get("answer", ""),
                    raw=row,
                    question_type=str(row.get("question_type") or row.get("category") or ""),
                    extras={
                        "anchor_date": row.get("anchor_date") or row.get("question_date") or "",
                        "metadata": row.get("metadata") or {},
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
            result = [inst for inst in result if inst.question_type.lower() in wanted]
        if max_examples > 0:
            result = result[:max_examples]
        return result

    # ---- ingest ------------------------------------------------------------

    def ingest(self, instance: BenchmarkInstance) -> Iterable[dict]:
        sessions = list(iter_sessions(instance.raw))
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

        reasoning_kind = "factual"
        unit_hint = ""
        is_temporal = False
        ordering_direction = "first"
        is_multi_source = False

        if any(token in lowered for token in ("how many days", "how many weeks", "how many months", "how many years")):
            reasoning_kind = "difference"
            for unit in ("days", "weeks", "months", "years"):
                if unit in lowered:
                    unit_hint = unit
                    is_temporal = True
                    break
        elif "what date" in lowered or "which date" in lowered or "when" in lowered:
            reasoning_kind = "date"
            is_temporal = True
            ordering_direction = "last" if " last" in lowered else "first"
        elif any(t in lowered for t in ("first", "earliest", "last", "latest", "before", "after")):
            reasoning_kind = "ordering"
            is_temporal = True
            ordering_direction = "last" if any(t in lowered for t in (" last", "latest")) else "first"
        elif any(t in lowered for t in ("how many times", "how often", "across", "over the past", "in total")):
            is_multi_source = True
            reasoning_kind = "multi-session"

        anchor_date = normalize_iso_date(instance.extras.get("anchor_date", ""))
        if not anchor_date:
            dates = extract_iso_dates(question)
            anchor_date = dates[0] if dates else ""

        entities = extract_entities(question)
        targets = entities[:2]
        return QueryPlan(
            query_id=instance.instance_id,
            question=question,
            query_text=(f"{anchor_date} | {question}" if anchor_date and include_anchor_date else question),
            question_type=instance.question_type,
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
            metadata={**instance.extras.get("metadata", {})},
        )

    # ---- postprocess + score ----------------------------------------------

    def postprocess(self, plan: QueryPlan, raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""
        first_line = text.split("\n", 1)[0].strip()
        if first_line.lower().startswith("answer:"):
            first_line = first_line[len("answer:"):].strip()
        return first_line

    def score(self, plan: QueryPlan, prediction: str, instance: BenchmarkInstance) -> dict:
        gold = str(instance.answer or "").strip().lower()
        pred = (prediction or "").strip().lower()

        exact = 1.0 if gold and gold == pred else 0.0
        contains = 1.0 if gold and gold in pred else 0.0

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

        return {"exact_match": exact, "contains_match": contains, "token_f1": f1, "abstention": None}
