"""Generic query analysis types shared across benchmarks.

A `QueryPlan` is the benchmark-agnostic description of what a question is asking
for: its reasoning kind, the targets it mentions, the entities it references,
its temporal anchor, and any benchmark-specific extras placed in `metadata`.

Benchmark adapters subclass `QueryAnalyzer` to populate this in a way that
matches their dataset (LongMemEval has its own subclass at
`memory.adapters.longmemeval.LongMemEvalAnalyzer`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


REASONING_KINDS = (
    "factual",
    "ordering",
    "difference",
    "date",
    "multi-session",
    "preference",
    "constraint",
    "knowledge-update",
)


@dataclass
class QueryPlan:
    """A benchmark-agnostic plan for a single query.

    Attributes:
        query_id: stable per-query identifier (used for caching and traces).
        question: the user's question text as-is.
        query_text: the text actually fed to the retriever (may include date prefix etc.).
        question_type: free-form benchmark label, e.g. "single-session-user".
        reasoning_kind: one of REASONING_KINDS, used by selectors and the solver.
        is_multi_source: True when the answer must combine evidence from many sessions/docs.
        is_temporal: True when the answer requires date/time reasoning.
        unit_hint: e.g. "days", "weeks", "months", "$", "items".
        targets: surface forms of the entities being compared/asked about.
        entities: entities mentioned in the question (free-form).
        anchor_date: ISO date the question is anchored to (e.g. the question_date).
        ordering_direction: "first" or "last".
        filter_month: "01"..."12" if a month filter is implied, else "".
        focus_terms: tokens we want to score memories against (subject + actions).
        metadata: benchmark-specific extras (range filters, distinct flags, etc.).
    """

    query_id: str = ""
    question: str = ""
    query_text: str = ""
    question_type: str = ""
    reasoning_kind: str = "factual"
    is_multi_source: bool = False
    is_temporal: bool = False
    unit_hint: str = ""
    targets: list[str] = field(default_factory=list)
    normalized_targets: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    anchor_date: str = ""
    ordering_direction: str = "first"
    filter_month: str = ""
    focus_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_session(self) -> bool:
        # Backwards-compatible alias for benchmarks that grew up calling it
        # multi-session.
        return self.is_multi_source


class QueryAnalyzer(Protocol):
    """Per-benchmark analyzer that produces a QueryPlan from a raw instance."""

    def analyze(self, instance: dict, *, include_anchor_date: bool = True) -> QueryPlan: ...
