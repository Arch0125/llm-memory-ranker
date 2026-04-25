"""Deterministic temporal/aggregate solver.

The solver tries to compute an answer directly from the structured metadata
attached to selected `MemoryHit`s, without invoking the LLM. When it succeeds
with high confidence the caller can choose to:

- Surface the answer as a hint in the prompt ("Solver suggestion: …"), or
- Short-circuit the LLM call entirely and return the answer.

This module is benchmark-agnostic: it consumes a `QueryPlan` and a list of
hits whose metadata follows the convention emitted by `memory.granularity`
(plus optional `event_date`, `event_aliases`, `aggregate_*` enrichments).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .extractors import (
    difference_between_dates,
    first_month_day,
    format_month_day,
    parse_iso_date,
)
from .utils import normalize_entity


@dataclass
class SolverResult:
    resolved: bool = False
    answer: str = ""
    confidence: float = 0.0
    rationale: str = ""
    mode: str = "insufficient"
    supporting_memory_ids: list[str] = field(default_factory=list)
    supporting_dates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hit_metadata(hit):
    return hit.record.metadata or {}


def _hit_event_date(hit):
    md = _hit_metadata(hit)
    return md.get("event_date") or md.get("session_date") or ""


def _hit_aliases(hit):
    md = _hit_metadata(hit)
    aliases = list(md.get("event_aliases", []) or [])
    aliases.extend(md.get("entities", []) or [])
    return [normalize_entity(a) for a in aliases if a]


def _matches_target(hit, target):
    if not target:
        return False
    needle = normalize_entity(target)
    if not needle:
        return False
    text_blob = " ".join(
        str(value)
        for value in (
            hit.record.text,
            _hit_metadata(hit).get("fact_text", ""),
            _hit_metadata(hit).get("summary", ""),
        )
        if value
    ).lower()
    if needle in text_blob:
        return True
    return any(needle == alias for alias in _hit_aliases(hit))


def _select_target_hit(hits, target):
    """Pick the highest-confidence hit that mentions `target` and has an event_date."""
    candidates = [hit for hit in hits if _matches_target(hit, target) and _hit_event_date(hit)]
    if not candidates:
        return None
    candidates.sort(
        key=lambda hit: (
            float(_hit_metadata(hit).get("date_confidence", 0.0) or 0.0),
            getattr(hit, "critic_confidence", 0.0),
            getattr(hit, "score", 0.0),
        ),
        reverse=True,
    )
    return candidates[0]


def _hit_to_event(hit):
    md = _hit_metadata(hit)
    return {
        "memory_id": hit.record.memory_id,
        "event_date": md.get("event_date") or md.get("session_date") or "",
        "session_id": md.get("session_id"),
        "date_confidence": float(md.get("date_confidence", 0.0) or 0.0),
        "label": md.get("summary") or md.get("fact_text") or hit.record.text,
        "aliases": md.get("event_aliases", []) or [],
    }


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------

def solve_ordering(plan, hits) -> SolverResult:
    """For ordering questions ("which came first, X or Y?")."""
    if len(plan.targets) < 2:
        return SolverResult(rationale="ordering-needs-2-targets")

    matched = []
    for target in plan.targets:
        hit = _select_target_hit(hits, target)
        if hit is None:
            continue
        date_value = parse_iso_date(_hit_event_date(hit))
        if date_value is None:
            continue
        matched.append((target, hit, date_value))

    if plan.metadata.get("requires_full_order") and len(matched) >= 2:
        matched.sort(key=lambda item: item[2], reverse=(plan.ordering_direction == "last"))
        return SolverResult(
            resolved=True,
            answer=", ".join(item[0] for item in matched),
            confidence=min(float(_hit_metadata(item[1]).get("date_confidence", 0.0) or 0.0) for item in matched),
            rationale=", ".join(item[2].isoformat() for item in matched),
            mode="full-ordering",
            supporting_memory_ids=[item[1].record.memory_id for item in matched],
            supporting_dates=[item[2].isoformat() for item in matched],
        )

    if len(matched) < 2:
        return SolverResult(rationale="missing-target-events")

    left, right = matched[0], matched[1]
    if left[2] == right[2]:
        return SolverResult(
            confidence=min(_hit_metadata(left[1]).get("date_confidence", 0.0), _hit_metadata(right[1]).get("date_confidence", 0.0)),
            rationale="same-date",
            mode="ambiguous",
            supporting_memory_ids=[left[1].record.memory_id, right[1].record.memory_id],
            supporting_dates=[left[2].isoformat(), right[2].isoformat()],
        )
    answer = left[0] if left[2] < right[2] else right[0]
    if plan.ordering_direction == "last":
        answer = left[0] if left[2] > right[2] else right[0]
    return SolverResult(
        resolved=True,
        answer=answer,
        confidence=min(
            float(_hit_metadata(left[1]).get("date_confidence", 0.0) or 0.0),
            float(_hit_metadata(right[1]).get("date_confidence", 0.0) or 0.0),
        ),
        rationale=f"{left[2].isoformat()} vs {right[2].isoformat()}",
        mode="pair-ordering",
        supporting_memory_ids=[left[1].record.memory_id, right[1].record.memory_id],
        supporting_dates=[left[2].isoformat(), right[2].isoformat()],
    )


def solve_difference(plan, hits) -> SolverResult:
    """For 'how many days/weeks/months between X and Y?' questions."""
    left = right = None
    if len(plan.targets) >= 2:
        left = _select_target_hit(hits, plan.targets[0])
        right = _select_target_hit(hits, plan.targets[1])
    elif len(plan.targets) == 1 and plan.anchor_date:
        left = _select_target_hit(hits, plan.targets[0])
        right_date = parse_iso_date(plan.anchor_date)
        if left is not None and right_date is not None:
            unit = plan.unit_hint or "days"
            value = difference_between_dates(parse_iso_date(_hit_event_date(left)), right_date, unit=unit)
            return SolverResult(
                resolved=True,
                answer=f"{value} {unit}",
                confidence=float(_hit_metadata(left).get("date_confidence", 0.0) or 0.0),
                rationale=f"{_hit_event_date(left)} vs {plan.anchor_date}",
                mode="deterministic-difference",
                supporting_memory_ids=[left.record.memory_id],
                supporting_dates=[_hit_event_date(left), plan.anchor_date],
            )

    if not left or not right:
        return SolverResult(rationale="missing-difference-pair")

    left_date = parse_iso_date(_hit_event_date(left))
    right_date = parse_iso_date(_hit_event_date(right))
    if not left_date or not right_date:
        return SolverResult(rationale="missing-difference-dates")
    unit = plan.unit_hint or "days"
    value = difference_between_dates(left_date, right_date, unit=unit)
    return SolverResult(
        resolved=True,
        answer=f"{value} {unit}",
        confidence=min(
            float(_hit_metadata(left).get("date_confidence", 0.0) or 0.0),
            float(_hit_metadata(right).get("date_confidence", 0.0) or 0.0),
        ),
        rationale=f"{left_date.isoformat()} vs {right_date.isoformat()}",
        mode="deterministic-difference",
        supporting_memory_ids=[left.record.memory_id, right.record.memory_id],
        supporting_dates=[left_date.isoformat(), right_date.isoformat()],
    )


def solve_date(plan, hits) -> SolverResult:
    """For 'what was the date when…' questions."""
    candidates = []
    for hit in hits:
        date_str = _hit_event_date(hit)
        if not date_str:
            continue
        if plan.targets and not any(_matches_target(hit, target) for target in plan.targets):
            continue
        if plan.filter_month and date_str[5:7] != plan.filter_month:
            continue
        candidates.append(hit)
    if not candidates:
        return SolverResult(rationale="missing-date-target")

    candidates.sort(
        key=lambda hit: (
            float(_hit_metadata(hit).get("date_confidence", 0.0) or 0.0),
            getattr(hit, "critic_confidence", 0.0),
            getattr(hit, "score", 0.0),
        ),
        reverse=True,
    )
    if "first" in plan.question.lower():
        candidates.sort(key=lambda hit: _hit_event_date(hit) or "9999-99-99")
    elif "last" in plan.question.lower():
        candidates.sort(key=lambda hit: _hit_event_date(hit) or "")
        candidates.reverse()

    best = candidates[0]
    date_value = _hit_event_date(best)
    parsed = parse_iso_date(date_value)
    label = _hit_metadata(best).get("summary") or _hit_metadata(best).get("fact_text") or best.record.text
    answer = date_value
    month_day = first_month_day(label)
    if month_day:
        answer = f"{month_day[0]} {month_day[1]}"
    elif parsed is not None:
        answer = format_month_day(parsed)
    return SolverResult(
        resolved=True,
        answer=answer,
        confidence=float(_hit_metadata(best).get("date_confidence", 0.0) or 0.0),
        rationale=date_value,
        mode="deterministic-date",
        supporting_memory_ids=[best.record.memory_id],
        supporting_dates=[date_value],
    )


def solve_aggregate(plan, hits) -> SolverResult:
    """Use a pre-computed aggregate memory if one is in the candidate set."""
    aggregates = [
        hit for hit in hits
        if hit.record.memory_type == "aggregate"
        or _hit_metadata(hit).get("granularity") == "aggregate"
    ]
    if not aggregates:
        return SolverResult(rationale="no-aggregate")

    aggregates.sort(
        key=lambda hit: float(_hit_metadata(hit).get("aggregate_confidence", 0.0) or 0.0),
        reverse=True,
    )
    best = aggregates[0]
    md = _hit_metadata(best)
    if not md.get("aggregate_answer") or not md.get("aggregate_answer_public", True):
        return SolverResult(rationale="aggregate-private")
    return SolverResult(
        resolved=True,
        answer=str(md["aggregate_answer"]),
        confidence=float(md.get("aggregate_confidence", 0.0) or 0.0),
        rationale=md.get("aggregate_kind", "aggregate"),
        mode=md.get("aggregate_mode", "multi-source-aggregate"),
        supporting_memory_ids=[entry.get("entry_id", "") for entry in md.get("aggregate_entries", [])[:8]],
        supporting_dates=[entry.get("event_date", "") for entry in md.get("aggregate_entries", [])[:8]],
    )


def solve(plan, hits) -> SolverResult:
    """Dispatch to the appropriate per-kind solver for `plan`."""
    if plan.is_multi_source:
        return solve_aggregate(plan, hits)
    if plan.reasoning_kind == "ordering":
        return solve_ordering(plan, hits)
    if plan.reasoning_kind == "difference":
        return solve_difference(plan, hits)
    if plan.reasoning_kind == "date":
        return solve_date(plan, hits)
    return SolverResult(rationale="unsupported-kind", mode="unsupported")
