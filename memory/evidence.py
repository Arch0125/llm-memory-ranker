"""Render selected MemoryHits into a textual evidence view.

The previous LongMemEval harness inlined two views:

- A "structured event view" of (date, label, source) rows for temporal queries.
- An "evidence table" of detailed memory rows.

Both are parameter-free in their core logic, so they live here and adapters
just call them.
"""
from __future__ import annotations

from .extractors import normalize_iso_date
from .utils import preview


def render_structured_event_line(event, index=None):
    prefix = f"{index}. " if index is not None else ""
    return (
        f"{prefix}date={event.get('event_date') or 'unknown'} ; "
        f"date_source={event.get('date_source', 'unknown')} ; "
        f"date_confidence={float(event.get('date_confidence', 0.0) or 0.0):.2f} ; "
        f"kind={event.get('granularity', 'unknown')} ; "
        f"label={preview(event.get('label', ''), limit=90)} ; "
        f"aliases={', '.join(event.get('aliases', [])[:3]) or 'n/a'} ; "
        f"source={event.get('session_id') or 'unknown'}"
    )


def render_evidence_line(hit, index=None):
    """Render one memory hit as a compact evidence line.

    Aggregate-kind memories get a different shape because they pre-encode the
    answer and a confidence; everything else gets a date/kind/source/entities/
    evidence shape.
    """
    record = hit.record
    metadata = record.metadata or {}
    if metadata.get("granularity") == "aggregate" or record.memory_type == "aggregate":
        aggregate_kind = metadata.get("aggregate_kind", "aggregate")
        public = metadata.get("aggregate_answer_public", True)
        answer = metadata.get("aggregate_answer") if public else "withheld"
        confidence = float(metadata.get("aggregate_confidence", 0.0) or 0.0)
        prefix = f"{index}. " if index is not None else ""
        return (
            f"{prefix}kind=aggregate ; "
            f"aggregate_kind={aggregate_kind} ; "
            f"answer={preview(answer, limit=72)} ; "
            f"confidence={confidence:.2f} ; "
            f"evidence={preview(metadata.get('summary') or record.text, limit=180)}"
        )

    date_value = metadata.get("event_date") or normalize_iso_date(metadata.get("session_date", ""))
    anchor_date = normalize_iso_date(metadata.get("session_date", ""))
    granularity = metadata.get("granularity") or record.memory_type
    source = (
        metadata.get("session_id")
        or ",".join(metadata.get("session_ids", [])[:2])
        or "unknown"
    )
    entities = ", ".join(metadata.get("entities", [])[:3]) or "n/a"
    evidence = metadata.get("fact_text") or metadata.get("summary") or record.text
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


def build_evidence_table(selected_hits, *, structured_events=None, header="Evidence table:"):
    """Build a multi-section text view: optional structured events + evidence rows."""
    lines = []
    if structured_events:
        lines.append("Structured event view:")
        for index, event in enumerate(structured_events, start=1):
            lines.append(render_structured_event_line(event, index=index))
    if selected_hits:
        if lines:
            lines.append("")
        lines.append(header)
        for index, hit in enumerate(selected_hits, start=1):
            lines.append(render_evidence_line(hit, index=index))
    if not lines:
        return "Evidence:\n- none"
    return "\n".join(lines)


def build_temporal_event_view(plan, structured_events):
    """Return a one-shot textual view of the structured event list, or empty if not temporal."""
    if not getattr(plan, "is_temporal", False) or not structured_events:
        return ""
    lines = ["Structured event/date view:"]
    for index, event in enumerate(structured_events, start=1):
        lines.append(render_structured_event_line(event, index=index))
    return "\n".join(lines)
