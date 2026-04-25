"""Generic prompt fragments for memory-aware QA.

Adapters can compose these snippets into their reader prompts so that
postprocessing helpers in :mod:`memory.postprocess` have a predictable hook to
extract literal short answers.

Keep these strings short and benchmark-neutral — anything LongMemEval- or
LoCoMo-specific belongs in the adapter.
"""
from __future__ import annotations

import re

from .query import QueryPlan


# Question opens with one of these auxiliary verbs and is therefore
# a candidate yes/no question. We further exclude polite-request patterns
# like "Can you suggest …" / "Would you recommend …" which are really
# preference / open-ended asks.
_YES_NO_OPENER_RE = re.compile(
    r"^\s*(do|does|did|is|are|was|were|am|have|has|had|will|would|can|could|should|shall|may|might|must)\b",
    re.IGNORECASE,
)
_OPEN_REQUEST_RE = re.compile(
    r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:suggest|recommend|help|tell|show|give|share|list|describe|explain|provide|find|name)\b",
    re.IGNORECASE,
)


def is_yes_no_question(question: str) -> bool:
    """Heuristic: does ``question`` look like a literal yes/no question?

    Returns ``True`` for first-person yes/no asks like "Do I have …?",
    "Was it raining?", "Have I tried this restaurant?".

    Returns ``False`` for polite-request openers like "Can you suggest …?"
    or "Would you recommend …?" which expect a recommendation, not a
    yes/no token.
    """
    if not question:
        return False
    text = question.strip()
    if not text:
        return False
    if not _YES_NO_OPENER_RE.match(text):
        return False
    if _OPEN_REQUEST_RE.match(text):
        return False
    return True


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


_YES_NO_INSTRUCTION = (
    "This is a yes/no question. After any reasoning, end your reply with "
    "exactly one of these two final lines:\n"
    "Final answer: Yes\n"
    "Final answer: No\n"
    "Do NOT add details on the Final-answer line and do NOT replace 'Yes'/'No' "
    "with a frequency, quantity, or descriptive phrase. If the evidence "
    "clearly does not support either, write 'Final answer: Insufficient evidence'."
)


_PREFERENCE_INSTRUCTION = (
    "This question asks for a recommendation tailored to the user's preferences. "
    "Read the supporting evidence carefully and synthesise the user's stated "
    "tastes, goals, constraints, and dislikes. Then write 1-3 sentences that "
    "describe what the user WOULD prefer for the question at hand and what "
    "they would NOT prefer, grounding each claim in the evidence. "
    "Use the phrasing 'The user would prefer ...' and 'They would not prefer "
    "...'. Do NOT abstain when relevant context is present even if it does "
    "not literally state the answer — infer the preference from the closest "
    "signals (recent behaviour, explicit likes/dislikes, ongoing projects, "
    "etc.). Do NOT end with a 'Final answer:' line for this question type."
)


def final_answer_instruction() -> str:
    """Return the canonical 'end with Final answer: <terse>' instruction.

    Pair with :func:`memory.postprocess.extract_final_answer_marker` in your
    adapter's ``postprocess`` step.
    """
    return _FINAL_ANSWER_INSTRUCTION


def yes_no_answer_instruction() -> str:
    """Return the canonical 'Final answer: Yes/No' instruction.

    Use only for plans whose underlying question is genuinely yes/no
    (see :func:`is_yes_no_question`). Pair with
    :func:`memory.postprocess.extract_final_answer_marker`.
    """
    return _YES_NO_INSTRUCTION


def preference_answer_instruction() -> str:
    """Return the multi-sentence preference-profile instruction.

    Use for plans whose ``reasoning_kind == "preference"``. The expected
    output is a short paragraph describing what the user would and would
    not prefer, not a terse literal answer.
    """
    return _PREFERENCE_INSTRUCTION


def answer_instruction_for_plan(plan: QueryPlan | None = None) -> str:
    """Return the appropriate instruction fragment for a plan.

    Routing:
    - ``reasoning_kind == "preference"`` -> :func:`preference_answer_instruction`
    - factual yes/no question (per :func:`is_yes_no_question`) and the
      plan has no other structured shape (not multi-session, not ordering,
      not difference, not date) -> :func:`yes_no_answer_instruction`
    - everything else -> :func:`final_answer_instruction`

    This keeps benchmark adapters honest about which output shape they want
    without each adapter re-implementing the routing logic.
    """
    if plan is None:
        return final_answer_instruction()
    kind = (plan.reasoning_kind or "").lower()
    if kind == "preference":
        return preference_answer_instruction()
    structured = {"multi-session", "ordering", "difference", "date", "count", "aggregate"}
    if kind not in structured and is_yes_no_question(plan.question):
        return yes_no_answer_instruction()
    return final_answer_instruction()


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


__all__ = [
    "answer_instruction_for_plan",
    "final_answer_instruction",
    "is_yes_no_question",
    "preference_answer_instruction",
    "terse_answer_hint",
    "yes_no_answer_instruction",
]
