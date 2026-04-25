"""Generic prompt fragments for memory-aware QA.

Adapters can compose these snippets into their reader prompts so that
postprocessing helpers in :mod:`memory.postprocess` have a predictable hook to
extract literal short answers.

Keep these strings short and benchmark-neutral — anything LongMemEval- or
LoCoMo-specific belongs in the adapter.
"""
from __future__ import annotations

from .query import QueryPlan


_FINAL_ANSWER_INSTRUCTION = (
    "After any reasoning, end your reply with one final line in this exact format:\n"
    "Final answer: <terse>\n"
    "Replace <terse> with the shortest literal value or noun phrase that "
    "answers the question — copy it verbatim from the evidence when possible. "
    "Do NOT restate the question, do NOT wrap the value in a sentence, and do "
    "NOT add explanations on that line. Examples: "
    "'Final answer: Business Administration', 'Final answer: $800', "
    "'Final answer: 45 minutes each way', 'Final answer: 2024-03-14', "
    "'Final answer: Insufficient evidence'."
)


def final_answer_instruction() -> str:
    """Return the canonical 'end with Final answer: <terse>' instruction.

    Pair with :func:`memory.postprocess.extract_final_answer_marker` in your
    adapter's ``postprocess`` step.
    """
    return _FINAL_ANSWER_INSTRUCTION


def terse_answer_hint(plan: QueryPlan | None = None) -> str:
    """Return a short reasoning-kind-aware hint about answer terseness.

    Useful when an adapter wants a one-line hint *in addition* to the
    explicit Final answer marker (e.g. to nudge the model towards a number
    for counting questions). Returns an empty string when no specific hint
    applies.
    """
    if plan is None:
        return ""
    kind = (plan.reasoning_kind or "").lower()
    if kind == "ordering":
        return "Reply with the single event/item that satisfies the ordering, with no explanation."
    if kind == "difference":
        unit = plan.unit_hint or "days"
        return f"Reply with the duration like '7 {unit}', with no explanation."
    if kind == "date":
        return "Reply with the literal date or short date phrase, with no explanation."
    if kind in {"count", "aggregate"}:
        return "Reply with only the final number, with no explanation."
    return ""


__all__ = ["final_answer_instruction", "terse_answer_hint"]
