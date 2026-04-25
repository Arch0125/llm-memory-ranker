"""Recency-aware reranking helper.

A small, benchmark-agnostic post-fusion step. For plans that ask "what is the
*current* state of X?" (e.g. knowledge-update questions in LongMemEval, or
``reasoning_kind in {"knowledge-update", "constraint"}`` more broadly) the
ranker is biased to prefer memories with the most recent ``event_date`` or
``session_date`` metadata.

The bias is multiplicative on top of the existing fused score, so it does not
change the relative order within memories of the same date and never *removes*
candidates - it only re-weights them.

Usage::

    from memory.recency import apply_recency_bias

    apply_recency_bias(hits, strength=0.25, plan=plan)

The default ``trigger_kinds`` set is small and conservative; pass an explicit
set to extend it to other benchmarks/use-cases.
"""
from __future__ import annotations

from typing import Iterable


_DEFAULT_TRIGGER_KINDS: frozenset[str] = frozenset(
    {
        "knowledge-update",
        "knowledge_update",
    }
)


def _hit_event_date(hit) -> str:
    """Best-effort 'when did this memory happen?' date string for a hit."""
    metadata = getattr(hit.record, "metadata", None) or {}
    for key in ("event_date", "session_date"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _plan_triggers_recency(plan, trigger_kinds: Iterable[str]) -> bool:
    if plan is None:
        return False
    triggers = set(trigger_kinds or ())
    if not triggers:
        return False
    fields = (
        getattr(plan, "reasoning_kind", "") or "",
        getattr(plan, "question_type", "") or "",
    )
    return any(field in triggers for field in fields if field)


def apply_recency_bias(
    hits,
    *,
    strength: float,
    plan,
    trigger_kinds: Iterable[str] = _DEFAULT_TRIGGER_KINDS,
):
    """Multiplicatively rescale hit scores by recency rank.

    Parameters
    ----------
    hits:
        Mutable list of `MemoryHit`. Modified in-place; also returned for
        chaining convenience.
    strength:
        Maximum bonus, in proportion of the original score, that the most
        recent memory receives. ``0`` disables the bias entirely. Typical
        values: ``0.15`` - ``0.5``.
    plan:
        The active `QueryPlan` (or ``None``). Used to decide whether the
        plan's reasoning kind triggers recency promotion.
    trigger_kinds:
        Iterable of strings. The bias is applied iff
        ``plan.reasoning_kind`` *or* ``plan.question_type`` is in this set.

    Returns
    -------
    The same ``hits`` list, with ``score`` adjusted and a ``recency+<rank>``
    string appended to ``reasons`` for each adjusted hit. Memories without a
    parseable date are left untouched (rank treated as 0).
    """
    if strength <= 0:
        return hits
    if not _plan_triggers_recency(plan, trigger_kinds):
        return hits
    if not hits:
        return hits

    dated = [(_hit_event_date(hit), hit) for hit in hits]
    distinct_dates = sorted({d for d, _ in dated if d})
    if len(distinct_dates) < 2:
        return hits

    # Linear ranking in [0, 1] over the distinct dates we observed in this
    # candidate set: oldest -> 0.0, newest -> 1.0.
    rank_by_date = {d: idx / (len(distinct_dates) - 1) for idx, d in enumerate(distinct_dates)}

    for date, hit in dated:
        if not date:
            continue
        rank = rank_by_date.get(date, 0.0)
        if rank <= 0.0:
            continue
        bonus = 1.0 + strength * rank
        hit.score = hit.score * bonus
        hit.reasons.append(f"recency+{rank:.2f}")

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits
