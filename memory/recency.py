"""Recency- and time-anchor-aware reranking helpers.

Two small, benchmark-agnostic post-fusion steps:

1. ``apply_recency_bias``: for plans that ask "what is the *current* state of
   X?" (e.g. knowledge-update), promote the most recent memories.
2. ``apply_time_anchor_bias``: for plans whose question contains an explicit
   relative time anchor (e.g. "10 days ago", "last Tuesday", "in the past two
   months"), promote memories whose ``event_date``/``session_date`` matches
   one of the derived target dates (or falls inside a derived target window).

Both biases are multiplicative on top of the existing fused score, so they
never *remove* candidates - they only re-weight them.

Usage::

    from memory.recency import apply_recency_bias, apply_time_anchor_bias

    apply_recency_bias(hits, strength=0.25, plan=plan)
    apply_time_anchor_bias(hits, strength=0.5, plan=plan)

The biases are gated so they only fire when the plan signals it.
"""
from __future__ import annotations

from datetime import date, datetime
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


# ---------------------------------------------------------------------------
# Time-anchor bias
# ---------------------------------------------------------------------------


def _parse_iso(value: str):
    if not value:
        return None
    s = value.strip()
    # Accept "YYYY-MM-DD" first, then "YYYY/MM/DD" with optional trailing time.
    s = s.replace("/", "-")
    # Take only the date prefix (before any space, tab, or 'T').
    for sep in (" ", "T", "\t"):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _hit_iso_date(hit):
    metadata = getattr(hit.record, "metadata", None) or {}
    # Prefer event_date (already ISO-normalized in our adapters), fall back to
    # session_date which may be the raw "YYYY/MM/DD (Wed) HH:MM" timestamp.
    for key in ("event_date", "session_date"):
        value = metadata.get(key)
        parsed = _parse_iso(value) if isinstance(value, str) else None
        if parsed is not None:
            return parsed
    return None


def _coerce_target_dates(values):
    out = []
    seen = set()
    for v in values or ():
        d = _parse_iso(v) if isinstance(v, str) else None
        if d is None:
            continue
        iso = d.isoformat()
        if iso in seen:
            continue
        seen.add(iso)
        out.append(d)
    return out


def apply_time_anchor_bias(
    hits,
    *,
    strength: float,
    plan,
    window_days: int = 3,
    taper_days: int = 14,
):
    """Boost memories whose date matches a query's relative time anchor.

    Looks at ``plan.metadata`` for either or both of:

    - ``retrieval_target_dates``: list of ISO date strings derived from
      phrases like "10 days ago", "last Tuesday", "two months ago" applied
      to ``plan.anchor_date``. Each target is treated as a point.
    - ``retrieval_target_window``: ``(start_iso, end_iso)`` pair derived from
      phrases like "in the past two months" or "in March". Hits whose date
      lies inside the window receive the full bonus.

    Scoring (per hit, bonus is the maximum across all targets/windows):

    - hit date inside window               -> bonus = strength
    - |hit_date - target| <= window_days   -> bonus = strength
    - window_days < delta <= taper_days    -> linear taper to 0
    - delta > taper_days                   -> no bonus

    Returns the same ``hits`` list, sorted by adjusted score.
    """
    if strength <= 0 or not hits or plan is None:
        return hits

    metadata = getattr(plan, "metadata", None) or {}
    target_dates = _coerce_target_dates(metadata.get("retrieval_target_dates") or ())
    window_pair = metadata.get("retrieval_target_window") or ()
    win_start = _parse_iso(window_pair[0]) if len(window_pair) >= 1 else None
    win_end = _parse_iso(window_pair[1]) if len(window_pair) >= 2 else None
    if not target_dates and not (win_start and win_end):
        return hits

    if window_days < 0:
        window_days = 0
    if taper_days < window_days:
        taper_days = window_days

    for hit in hits:
        hit_date = _hit_iso_date(hit)
        if hit_date is None:
            continue
        # Window match (full bonus).
        scale = 0.0
        if win_start and win_end and win_start <= hit_date <= win_end:
            scale = 1.0
        if scale < 1.0 and target_dates:
            min_delta = min(abs((hit_date - t).days) for t in target_dates)
            if min_delta <= window_days:
                scale = 1.0
            elif min_delta <= taper_days:
                span = max(taper_days - window_days, 1)
                linear = 1.0 - (min_delta - window_days) / span
                scale = max(scale, linear)
        if scale <= 0.0:
            continue
        bonus = 1.0 + strength * scale
        hit.score = hit.score * bonus
        hit.reasons.append(f"anchor+{scale:.2f}")

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits
