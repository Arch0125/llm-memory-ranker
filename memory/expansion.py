"""Generic query expansion.

Two cheap signals reliably help retrieval recall:

1. **Entity-only sub-queries** — queries built from the entities mentioned in
   the question. A "fact" memory referencing the same entity but with very
   different surface form will often outrank under entity-only retrieval.

2. **Reformulations** — drop interrogative scaffolding ("how many", "what was
   the date when", "do you remember") and keep the noun-phrase content.

We don't call out to an LLM. Each expansion is a heuristic transformation of
the original query; the engine then runs retrieval against each expansion and
fuses the results via Reciprocal Rank Fusion (see `memory.fusion`).
"""
from __future__ import annotations

import re

from .extractors import normalize_iso_date
from .utils import extract_entities, tokenize


_INTERROGATIVE_PREFIXES = (
    r"how many (?:days|weeks|months|years|times|times have|times did)\s+",
    r"how much\s+",
    r"how often\s+",
    r"what (?:was|is|were|are) (?:the (?:date|time|month|year) when)?\s*",
    r"when (?:did|was|is|will)\s+",
    r"do you remember\s+",
    r"can you (?:tell me|recall|remember)\s+",
)
_INTERROGATIVE_RE = re.compile(
    "|".join(f"(?:^|\\s){pattern}" for pattern in _INTERROGATIVE_PREFIXES),
    re.IGNORECASE,
)
_TRAILING_QUESTION_RE = re.compile(r"[?]\s*$")


def _strip_interrogative(text: str) -> str:
    text = _INTERROGATIVE_RE.sub(" ", text or "").strip()
    text = _TRAILING_QUESTION_RE.sub("", text)
    return " ".join(text.split())


def _entity_query(entities: list[str], anchor_date: str) -> str:
    cleaned = [e for e in entities if e]
    if not cleaned:
        return ""
    head = f"{anchor_date} | " if anchor_date else ""
    return f"{head}{' '.join(cleaned)}".strip()


def expand_query(
    query_text: str,
    *,
    entities: list[str] | None = None,
    anchor_date: str = "",
    targets: list[str] | None = None,
    focus_terms: list[str] | None = None,
    max_variants: int = 4,
) -> list[str]:
    """Return a list of query strings, original first, deduplicated.

    Variants:
      0. The original query.
      1. Reformulation (interrogative stripped).
      2. Entity-only query.
      3. Targets-only query.
      4. Focus-terms query (already a noun-phrase list).
    """
    variants: list[str] = []

    def _push(candidate: str) -> None:
        candidate = " ".join((candidate or "").split())
        if not candidate:
            return
        for existing in variants:
            if candidate == existing:
                return
        variants.append(candidate)

    _push(query_text)

    reformulation = _strip_interrogative(query_text)
    if reformulation and reformulation.lower() != (query_text or "").strip().lower():
        prefix = f"{normalize_iso_date(anchor_date)} | " if anchor_date else ""
        _push(f"{prefix}{reformulation}".strip())

    entity_query = _entity_query(entities or [], normalize_iso_date(anchor_date))
    if entity_query:
        _push(entity_query)

    if targets:
        prefix = f"{normalize_iso_date(anchor_date)} | " if anchor_date else ""
        _push(f"{prefix}{' '.join(targets)}".strip())

    if focus_terms:
        _push(" ".join(focus_terms))

    if not (entities or targets or focus_terms):
        head = extract_entities(query_text)
        if head:
            _push(" ".join(head))

    return variants[:max_variants]


def is_useful_expansion(original: str, candidate: str) -> bool:
    if not candidate:
        return False
    if candidate.lower() == (original or "").lower():
        return False
    return len(set(tokenize(candidate, drop_stopwords=True))) >= 1
