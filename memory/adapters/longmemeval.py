"""LongMemEval benchmark adapter.

This is a thin wrapper around the existing `benchmarks.longmemeval` harness so
the rest of the codebase can interact with LongMemEval through the generic
`BenchmarkAdapter` interface. Behaviour is intended to be identical to calling
the harness directly.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, is_dataclass

from ..query import QueryPlan
from .base import BenchmarkAdapter, BenchmarkInstance, register_adapter


def _question_plan_to_query_plan(qp) -> QueryPlan:
    """Convert the harness's `QuestionPlan` dataclass to a generic `QueryPlan`."""
    payload = asdict(qp) if is_dataclass(qp) else dict(qp)

    metadata = {
        "question_id": payload.get("question_id", ""),
        "normalized_question_date": payload.get("normalized_question_date", ""),
        "question_month": payload.get("question_month", ""),
        "multi_session_kind": payload.get("multi_session_kind", ""),
        "multi_session_subject": payload.get("multi_session_subject", ""),
        "multi_session_actions": payload.get("multi_session_actions", []),
        "multi_session_focus_terms": payload.get("multi_session_focus_terms", []),
        "range_start": payload.get("range_start", ""),
        "range_end": payload.get("range_end", ""),
        "requires_distinct": payload.get("requires_distinct", False),
        "requires_current_state": payload.get("requires_current_state", False),
        "raw_question_plan": qp,
    }

    return QueryPlan(
        query_id=payload.get("question_id", ""),
        question=payload.get("question", ""),
        query_text=payload.get("query_text", ""),
        question_type=payload.get("question_type", ""),
        reasoning_kind=payload.get("reasoning_kind", "factual"),
        is_multi_source=bool(payload.get("is_multi_session")),
        is_temporal=bool(payload.get("is_temporal")),
        unit_hint=payload.get("unit_hint", ""),
        targets=list(payload.get("targets") or []),
        normalized_targets=list(payload.get("normalized_targets") or []),
        entities=list(payload.get("query_entities") or []),
        anchor_date=payload.get("normalized_question_date", "") or payload.get("question_date", ""),
        ordering_direction=payload.get("ordering_direction", "first"),
        filter_month=payload.get("filter_month", ""),
        focus_terms=list(payload.get("multi_session_focus_terms") or []),
        metadata=metadata,
    )


@register_adapter("longmemeval")
class LongMemEvalAdapter:
    """Adapter wrapping the LongMemEval harness."""

    name = "longmemeval"

    def __init__(self, *, granularity: str = "hybrid", include_assistant_turns: bool = True):
        self.granularity = granularity
        self.include_assistant_turns = include_assistant_turns

    # ---- loading -----------------------------------------------------------

    def load(self, path: str) -> list[BenchmarkInstance]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("LongMemEval data must be a JSON list of instances")
        return [self._wrap(raw) for raw in data]

    def filter(
        self,
        instances: Sequence[BenchmarkInstance],
        *,
        question_types: str = "",
        start_index: int = 0,
        max_examples: int = 0,
    ) -> list[BenchmarkInstance]:
        from benchmarks.question_type_filters import normalize_question_types, question_type_matches

        result = list(instances)[start_index:]
        normalized = normalize_question_types(question_types) if question_types else []
        if normalized:
            result = [inst for inst in result if question_type_matches(inst.question_type, normalized)]
        if max_examples > 0:
            result = result[:max_examples]
        return result

    # ---- ingest ------------------------------------------------------------

    def ingest(self, instance: BenchmarkInstance) -> Iterable[dict]:
        from benchmarks.longmemeval import iter_history_memories

        yield from iter_history_memories(
            instance.raw,
            granularity=self.granularity,
            include_assistant_turns=self.include_assistant_turns,
        )

    # ---- analyze -----------------------------------------------------------

    def analyze(self, instance: BenchmarkInstance, *, include_anchor_date: bool = True) -> QueryPlan:
        from benchmarks.longmemeval import analyze_question

        question_plan = analyze_question(instance.raw, include_question_date=include_anchor_date)
        return _question_plan_to_query_plan(question_plan)

    # ---- postprocess + score ----------------------------------------------

    def postprocess(self, plan: QueryPlan, raw_text: str) -> str:
        from benchmarks.longmemeval import postprocess_prediction

        question_plan = plan.metadata.get("raw_question_plan")
        return postprocess_prediction(question_plan, raw_text)

    def score(self, plan: QueryPlan, prediction: str, instance: BenchmarkInstance) -> dict:
        from benchmarks.longmemeval import (
            abstention_score,
            contains_match_score,
            exact_match_score,
            token_f1_score,
        )

        gold = instance.answer
        question_type = instance.question_type or plan.question_type
        is_abstention = "abstention" in (question_type or "").lower()
        return {
            "exact_match": exact_match_score(prediction, gold),
            "contains_match": contains_match_score(prediction, gold),
            "token_f1": token_f1_score(prediction, gold),
            "abstention": (
                abstention_score(prediction, is_abstention) if is_abstention else None
            ),
        }

    # ---- helpers -----------------------------------------------------------

    def _wrap(self, raw: dict) -> BenchmarkInstance:
        return BenchmarkInstance(
            instance_id=str(raw.get("question_id") or raw.get("id") or ""),
            question=str(raw.get("question") or ""),
            answer=raw.get("answer"),
            raw=raw,
            question_type=str(raw.get("question_type") or ""),
            extras={
                "question_date": raw.get("question_date", ""),
                "answer_session_ids": list(raw.get("answer_session_ids") or []),
                "haystack_session_ids": list(raw.get("haystack_session_ids") or []),
            },
        )
