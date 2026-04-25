"""Generic memory granularity builders.

The memory layer ingests raw conversational data at multiple granularities:

- `turn`     : one memory per relevant user/assistant turn (the highest fidelity).
- `session`  : a per-session episode summary (good for "what did we discuss?").
- `timeline` : a date-anchored short summary (good for ordering questions).
- `global`   : a single haystack-wide list of (date, label) pairs, useful for
               recall across very long histories.

Adapters describe their data using a small uniform schema:

    Session = {
        "session_id": str,
        "session_date": str,            # ISO yyyy-mm-dd, optional
        "turns": [{"role": str, "content": str, "has_answer": bool?}, ...],
    }

This module emits dictionaries shaped like:

    {
        "text": str,                # rendered text fed to the embedder
        "memory_type": str,         # "episode" / "timeline" / "fact" / "timeline-global"
        "importance": float,
        "metadata": dict,           # session_id, session_date, granularity, etc.
    }

Adapters are free to enrich the metadata further (event_date, event_aliases,
entities, currency_values, etc.) before persisting.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .extractors import normalize_iso_date
from .utils import extract_entities, preview


def _collapse(text):
    return " ".join((text or "").split())


def _trim(text, limit=200):
    text = _collapse(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _role(turn):
    return (turn.get("role") or "user").strip().lower()


def _content(turn):
    return _collapse(turn.get("content") or "")


def session_key_snippets(session, include_assistant_turns=True, limit=3):
    """Pick the top-N most informative turns from a session for an episode summary."""
    snippets = []
    for turn in session.get("turns", []):
        if _role(turn) == "assistant" and not include_assistant_turns:
            continue
        text = _trim(_content(turn), limit=180)
        if text:
            snippets.append(text)
        if len(snippets) >= limit:
            break
    return snippets


def session_entities(session, max_entities=8):
    blob = " ".join(_content(turn) for turn in session.get("turns", []))
    return extract_entities(blob)[:max_entities]


def build_episode_memory(session, *, include_assistant_turns=True, importance=0.55):
    """Emit a coarse per-session episode memory (summary + entities + key snippets)."""
    session_id = session.get("session_id", "")
    session_date = normalize_iso_date(session.get("session_date", ""))
    snippets = session_key_snippets(session, include_assistant_turns=include_assistant_turns)
    summary_lines = [f"Session {session_id}"]
    if session_date:
        summary_lines[0] += f" on {session_date}"
    if snippets:
        summary_lines.extend(f"- {snippet}" for snippet in snippets)
    text = "\n".join(summary_lines)
    has_answer = any(turn.get("has_answer") for turn in session.get("turns", []))
    return {
        "text": text,
        "memory_type": "episode",
        "importance": max(importance, 0.7) if has_answer else importance,
        "metadata": {
            "session_id": session_id,
            "session_date": session_date,
            "granularity": "episode",
            "summary": text,
            "entities": session_entities(session),
            "has_answer": has_answer,
        },
    }


def build_timeline_memory(session, *, include_assistant_turns=True, importance=0.55):
    """Emit a date-anchored summary memory for a session."""
    session_id = session.get("session_id", "")
    session_date = normalize_iso_date(session.get("session_date", ""))
    user_snippets = []
    for turn in session.get("turns", []):
        if _role(turn) == "assistant" and not include_assistant_turns:
            continue
        snippet = _trim(_content(turn), limit=140)
        if snippet:
            user_snippets.append(snippet)
        if len(user_snippets) >= 2:
            break
    summary = "; ".join(user_snippets)
    head = f"{session_date} | " if session_date else ""
    text = f"{head}{summary}".strip()
    return {
        "text": text or f"Session {session_id}",
        "memory_type": "timeline",
        "importance": importance,
        "metadata": {
            "session_id": session_id,
            "session_date": session_date,
            "event_date": session_date,
            "granularity": "timeline",
            "summary": summary,
            "entities": session_entities(session),
        },
    }


def build_fact_memories(session, *, include_assistant_turns=True, importance=0.55):
    """Emit one fact-grain memory per relevant turn.

    Each fact memory's metadata starts with shared fields the adapter can extend
    (event_date, event_aliases, currency_values, ...).
    """
    session_id = session.get("session_id", "")
    session_date = normalize_iso_date(session.get("session_date", ""))
    out = []
    for index, turn in enumerate(session.get("turns", [])):
        role = _role(turn)
        if role == "assistant" and not include_assistant_turns:
            continue
        content = _content(turn)
        if not content:
            continue
        head = f"{session_date} | " if session_date else ""
        text = f"{head}{role.capitalize()}: {_trim(content, limit=240)}"
        has_answer = bool(turn.get("has_answer"))
        out.append(
            {
                "text": text,
                "memory_type": "fact",
                "importance": max(importance, 0.7) if has_answer else importance,
                "metadata": {
                    "session_id": session_id,
                    "session_date": session_date,
                    "event_date": session_date,
                    "granularity": "fact",
                    "role": role,
                    "turn_index": index,
                    "fact_text": content,
                    "summary": _trim(content, limit=140),
                    "entities": extract_entities(content),
                    "has_answer": has_answer,
                },
            }
        )
    return out


def build_global_timeline_memory(sessions, *, include_assistant_turns=False, max_entries=24, importance=0.7):
    """Emit a single 'timeline-global' memory listing dated session beats.

    `sessions` is an iterable of `Session` dicts. Returns None if the haystack
    is empty or has no useful dates.
    """
    entries: list[tuple[str, str, str]] = []
    for session in sessions:
        session_id = session.get("session_id", "")
        session_date = normalize_iso_date(session.get("session_date", ""))
        if not session_date:
            continue
        for turn in session.get("turns", []):
            if _role(turn) == "assistant" and not include_assistant_turns:
                continue
            snippet = _trim(_content(turn), limit=120)
            if not snippet:
                continue
            entries.append((session_date, session_id, snippet))
            break
    if not entries:
        return None
    entries.sort(key=lambda item: item[0])
    if len(entries) > max_entries:
        entries = entries[:max_entries]
    text_lines = ["Global timeline:"]
    for session_date, session_id, snippet in entries:
        text_lines.append(f"{session_date} | {session_id}: {preview(snippet, limit=120)}")
    text = "\n".join(text_lines)
    return {
        "text": text,
        "memory_type": "timeline-global",
        "importance": importance,
        "metadata": {
            "granularity": "timeline-global",
            "session_ids": [item[1] for item in entries],
            "event_dates": [item[0] for item in entries],
            "summary": text,
        },
    }


def normalize_session(raw: Mapping[str, Any]) -> dict:
    """Coerce a benchmark's session record into the schema this module expects."""
    return {
        "session_id": str(raw.get("session_id") or raw.get("id") or ""),
        "session_date": raw.get("session_date") or raw.get("date") or "",
        "turns": list(raw.get("turns") or []),
    }


def iter_sessions(instance: Mapping[str, Any], *, session_key="sessions") -> Iterable[dict]:
    """Yield normalized sessions from an instance dict."""
    for raw in instance.get(session_key) or []:
        yield normalize_session(raw)
