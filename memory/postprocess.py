"""Generic postprocessing primitives for memory-aware QA.

These helpers are benchmark-agnostic and operate on raw model output. The
canonical pattern is to ask the model to end its reply with::

    Final answer: <terse>

then call :func:`extract_final_answer_marker` to pull the literal short answer
out for grading.

Adapters may compose these primitives with their own benchmark-specific
extraction passes (see e.g. ``benchmarks/longmemeval.py``).
"""
from __future__ import annotations

import re


# Match a "Final answer:" / "Answer:" / "Final:" line. Anchored to a line
# boundary so chain-of-thought mentions like "the final answer might be ..."
# in mid-sentence don't accidentally match.
_FINAL_ANSWER_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:final\s*answer|final|answer)\s*[:\-]\s*([^\n]+)"
)


def extract_final_answer_marker(raw_text: str) -> str:
    """Return the value after the *last* ``Final answer:`` line, if any.

    The matcher accepts ``Final answer:``, ``Answer:`` or ``Final:`` (case
    insensitive) on its own line. Returns an empty string when no marker is
    found.

    Surrounding ASCII or smart quotes are stripped, and a trailing period is
    removed unless it appears in a decimal number (e.g. ``3.14``).
    """
    if not raw_text:
        return ""
    matches = list(_FINAL_ANSWER_RE.finditer(raw_text))
    if not matches:
        return ""
    candidate = matches[-1].group(1).strip()
    candidate = candidate.strip(" \t\r\n.,;")
    candidate = _strip_wrapping_quotes(candidate)
    if candidate.endswith(".") and not re.search(r"\d\.\d", candidate):
        candidate = candidate.rstrip(".").strip()
    return candidate


def _strip_wrapping_quotes(value: str) -> str:
    quote_pairs = (
        ("\"", "\""),
        ("'", "'"),
        ("\u201c", "\u201d"),  # smart double quotes
        ("\u2018", "\u2019"),  # smart single quotes
    )
    for opener, closer in quote_pairs:
        if value.startswith(opener) and value.endswith(closer) and len(value) >= 2:
            return value[len(opener) : -len(closer)].strip()
    return value


__all__ = ["extract_final_answer_marker"]
