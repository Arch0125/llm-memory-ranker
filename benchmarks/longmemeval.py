from __future__ import annotations

import json
import re
import string
from calendar import monthrange
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from memory.utils import (
    extract_entities,
    normalize_date,
    preview,
)
from prompt.budget import estimate_token_count


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_ORDINAL_SUFFIX_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ABSTENTION_MARKERS = (
    "do not know",
    "don't know",
    "cannot determine",
    "can't determine",
    "insufficient evidence",
    "not enough information",
    "not mentioned",
    "unknown",
    "unsure",
)
_QUESTION_OR_RE = re.compile(
    r"(?:^|,|\bbetween\b|\bfirst\b|\blast\b|\bhappened\b).*?\b(?:the\s+)?([^,?]+?)\s+or\s+(?:the\s+)?([^?]+?)\??$",
    re.IGNORECASE,
)
_QUESTION_AND_RE = re.compile(
    r"\bbetween\s+(?:the\s+)?([^,?]+?)\s+and\s+(?:the\s+)?([^?]+?)\??$",
    re.IGNORECASE,
)
_DIFFERENCE_BEFORE_AFTER_RE = re.compile(
    r"\bhow many\s+(days|weeks|months)\s+(before|after)\s+(.+?)\s+did\s+i\s+(.+?)\??$",
    re.IGNORECASE,
)
_DURATION_AFTER_RE = re.compile(
    r"\bhow many\s+(days|weeks|months)\s+did it take(?:\s+for\s+me)?\s+to\s+(.+?)\s+after\s+(.+?)\??$",
    re.IGNORECASE,
)
_AGO_RE = re.compile(
    r"\bhow many\s+(days|weeks|months)\s+ago\s+did\s+i\s+(.+?)\??$",
    re.IGNORECASE,
)
_DATE_QUESTION_RE = re.compile(
    r"\b(?:what was the date|which date)\s+(?:on which|when)\s+i\s+(.+?)\??$",
    re.IGNORECASE,
)
_IN_MONTH_RE = re.compile(
    r"\bin\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_LEADING_I_RE = re.compile(r"^(?:i\s+|my\s+)", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_LEADING_PREPOSITION_RE = re.compile(r"^(?:to|for|with)\s+", re.IGNORECASE)
_TRAILING_MONTH_CLAUSE_RE = re.compile(
    r"\s+in\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)$",
    re.IGNORECASE,
)
_ORDINAL_PREFIX_RE = re.compile(r"^(?:first|last|annual)\s+", re.IGNORECASE)
_GERUND_ACTION_RE = re.compile(
    r"^(attending|buying|purchasing|booking|ordering|getting|setting up|upgrading|repairing|fixing|trimming|watching|visiting|starting|finding|working with|participating in|going to)\s+",
    re.IGNORECASE,
)
_PAST_ACTION_RE = re.compile(
    r"^(attended|bought|got|purchased|booked|ordered|set up|upgraded|repaired|fixed|trimmed|watched|visited|started|found|worked with|participated in|went to)\s+",
    re.IGNORECASE,
)
_BASE_ACTION_RE = re.compile(
    r"^(attend|buy|get|purchase|book|order|set up|upgrade|repair|fix|trim|watch|visit|start|find|work with|participate in|go to)\s+",
    re.IGNORECASE,
)
_OF_PREFIX_RE = re.compile(r"^(purchase|malfunction|repair|fix|setup|upgrade|trimming|attendance|visit|order|booking)\s+of\s+", re.IGNORECASE)
_EVENT_CAPTURE_PATTERNS = (
    ("attendance", re.compile(r"\battend(?:ed|ing)?\s+(?:the\s+)?(.+?)(?:\s+on\b|\s+in\b|\s+before\b|\s+after\b|\s+about\b|\s+ago\b|[.,;!?]|$)", re.IGNORECASE)),
    ("purchase", re.compile(r"\b(?:bought|got|purchased|buying|getting|ordering|ordered|booked|booking|set up|upgraded)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+on\b|\s+in\b|\s+before\b|\s+after\b|\s+about\b|\s+ago\b|[.,;!?]|$)", re.IGNORECASE)),
    ("repair", re.compile(r"\b(?:repaired|repairing|fixed|fixing|trimmed|trimming|started|starting|visited|visiting|joined|joining|completed|completing|participated in|participating in|watched|watching)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+on\b|\s+in\b|\s+before\b|\s+after\b|\s+about\b|\s+ago\b|[.,;!?]|$)", re.IGNORECASE)),
    ("status", re.compile(r"\b(?:malfunction of|breakdown of)\s+(?:the\s+)?(.+?)(?:\s+on\b|\s+in\b|[.,;!?]|$)", re.IGNORECASE)),
)
_MONTH_DAY_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)
_DATE_LINE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MONTH_DIFF_RE = re.compile(r"\b(\d+)\s*(month|months)\b", re.IGNORECASE)
_DAY_DIFF_RE = re.compile(r"\b(\d+)\s*(day|days)\b", re.IGNORECASE)
_RELATIVE_SPAN_RE = re.compile(
    r"\b(?:about|around|approximately|roughly)?\s*(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(day|days|week|weeks|month|months|year|years)\s+ago\b",
    re.IGNORECASE,
)
_LAST_SPAN_RE = re.compile(r"\blast\s+(week|month|year)\b", re.IGNORECASE)
_RELATIVE_TO_ANCHOR_RE = re.compile(
    r"\b(?:about|around|approximately|roughly)?\s*(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(day|days|week|weeks|month|months|year|years)\s+(before|after)\s+(black friday|cyber monday)\b",
    re.IGNORECASE,
)
_LOWERCASE_LIST_RE = re.compile(
    r"(?:-\s*|:\s*|including\s+)"
    r"([a-z][a-z0-9'/-]*(?:\s+[a-z][a-z0-9'/-]*)?"
    r"(?:\s*,\s*[a-z][a-z0-9'/-]*(?:\s+[a-z][a-z0-9'/-]*)?)+"
    r"(?:\s*,?\s*and\s+[a-z][a-z0-9'/-]*(?:\s+[a-z][a-z0-9'/-]*)?)?)"
    r"\s+(?:are|is|were|was|have|has|did)\b",
    re.IGNORECASE,
)
_TRAILING_RELATIVE_FRAGMENT_RE = re.compile(
    r"\s+(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:day|days|week|weeks|month|months|year|years)\b.*$",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


@dataclass
class QuestionPlan:
    question_id: str
    question_type: str
    question: str
    query_text: str
    question_date: str
    normalized_question_date: str
    reasoning_kind: str
    is_temporal: bool
    unit_hint: str
    targets: list[str]
    normalized_targets: list[str]
    query_entities: list[str]
    question_month: str
    ordering_direction: str
    filter_month: str


@dataclass
class TemporalSolution:
    resolved: bool
    answer: str
    confidence: float
    rationale: str
    mode: str
    supporting_memory_ids: list[str]
    supporting_dates: list[str]


def load_longmemeval_instances(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("LongMemEval data must be a JSON list of instances")
    return data


def _collapse(text):
    return " ".join((text or "").split())


def _trim(text, limit=200):
    collapsed = _collapse(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _clean_answer_variant(text):
    value = _collapse(text)
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"\bis also acceptable\b\.?$", "", value, flags=re.IGNORECASE)
    value = value.strip(" .")
    return value


def acceptable_answers(answer):
    parts = re.split(r"\.\s+", answer or "")
    variants = []
    seen = set()
    for part in parts:
        cleaned = _clean_answer_variant(part)
        if not cleaned:
            continue
        normalized = normalize_answer(cleaned)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append(cleaned)
    if not variants and (answer or "").strip():
        variants.append((answer or "").strip())
    return variants


def format_session_text(session_id, session_date, session):
    header = f"Session {session_id}"
    if session_date:
        header += f" on {session_date}"
    lines = [header]
    for turn in session:
        content = _collapse(turn.get("content") or "")
        if not content:
            continue
        role = (turn.get("role") or "user").strip().capitalize()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_activity_type(text):
    lowered = (text or "").lower()
    patterns = (
        ("attend", "attendance"),
        ("bought", "purchase"),
        ("got", "purchase"),
        ("purchased", "purchase"),
        ("booked", "booking"),
        ("service", "service"),
        ("serviced", "service"),
        ("repaired", "repair"),
        ("repair", "repair"),
        ("cleaned", "cleaning"),
        ("workshop", "workshop"),
        ("webinar", "webinar"),
        ("meeting", "meeting"),
        ("festival", "festival"),
        ("mass", "service"),
        ("event", "event"),
    )
    for needle, label in patterns:
        if needle in lowered:
            return label
    return "event"


def _compact_turn_text(turn, session_date):
    role = (turn.get("role") or "user").strip().capitalize()
    content = _trim(turn.get("content") or "", limit=180)
    date_value = normalize_date(session_date)
    if date_value:
        return f"{date_value} | {role}: {content}"
    return f"{role}: {content}"


def _compact_turn_text_with_date(turn, event_date, fallback_session_date):
    role = (turn.get("role") or "user").strip().capitalize()
    content = _trim(turn.get("content") or "", limit=180)
    date_value = event_date or normalize_date(fallback_session_date)
    if date_value:
        return f"{date_value} | {role}: {content}"
    return f"{role}: {content}"


def _parse_iso_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _subtract_months(value, months):
    year = value.year
    month = value.month - months
    while month <= 0:
        year -= 1
        month += 12
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


def _subtract_years(value, years):
    year = value.year - years
    day = min(value.day, monthrange(year, value.month)[1])
    return date(year, value.month, day)


def _thanksgiving_date(year):
    november_first = date(year, 11, 1)
    days_until_thursday = (3 - november_first.weekday()) % 7
    return november_first + timedelta(days=days_until_thursday + 21)


def _named_anchor_date(name, base_date):
    if not name or base_date is None:
        return None
    anchor_name = name.lower()
    if anchor_name not in {"black friday", "cyber monday"}:
        return None
    thanksgiving = _thanksgiving_date(base_date.year)
    if anchor_name == "black friday":
        anchor = thanksgiving + timedelta(days=1)
    else:
        anchor = thanksgiving + timedelta(days=4)
    if anchor > base_date + timedelta(days=7):
        thanksgiving = _thanksgiving_date(base_date.year - 1)
        anchor = thanksgiving + timedelta(days=1 if anchor_name == "black friday" else 4)
    return anchor


def _normalize_word_form(token):
    value = (token or "").strip().lower()
    if len(value) <= 3:
        return value
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("es") and len(value) > 4:
        return value[:-2]
    if value.endswith("s") and len(value) > 4:
        return value[:-1]
    return value


def _normalized_token_forms(text):
    return {
        _normalize_word_form(token)
        for token in normalize_answer(text).split()
        if len(token) >= 3
    }


def _parse_numeric_token(value):
    token = (value or "").strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _derive_event_date_candidates(text, session_date):
    base_date = _parse_iso_date(normalize_date(session_date))
    if not text:
        return []

    values = []
    seen = set()

    def add_date(date_value, source, confidence):
        if not date_value:
            return
        iso = date_value.isoformat()
        if iso in seen:
            return
        seen.add(iso)
        values.append(
            {
                "date": iso,
                "source": source,
                "confidence": confidence,
            }
        )

    explicit_iso = normalize_date(text)
    if explicit_iso:
        parsed = _parse_iso_date(explicit_iso)
        add_date(parsed, "explicit-date", 1.0)

    if base_date is not None:
        for match in _MONTH_DAY_RE.finditer(text):
            month_token = match.group(1).lower()
            month_value = _MONTHS.get(month_token)
            day_value = int(match.group(2))
            year_value = int(match.group(3)) if match.group(3) else base_date.year
            try:
                derived = date(year_value, month_value, day_value)
            except ValueError:
                continue
            if derived > base_date + timedelta(days=1) and match.group(3) is None:
                derived = date(year_value - 1, month_value, day_value)
            add_date(derived, "explicit-month-day", 0.95 if match.group(3) else 0.9)

        for match in _DAY_MONTH_RE.finditer(text):
            day_value = int(match.group(1))
            month_token = match.group(2).lower()
            month_value = _MONTHS.get(month_token)
            year_value = int(match.group(3)) if match.group(3) else base_date.year
            try:
                derived = date(year_value, month_value, day_value)
            except ValueError:
                continue
            if derived > base_date + timedelta(days=1) and match.group(3) is None:
                derived = date(year_value - 1, month_value, day_value)
            add_date(derived, "explicit-day-month", 0.95 if match.group(3) else 0.9)

        for match in _RELATIVE_SPAN_RE.finditer(text):
            amount = _parse_numeric_token(match.group(1))
            unit = match.group(2).lower()
            if not amount:
                continue
            if unit.startswith("day"):
                derived = base_date - timedelta(days=amount)
            elif unit.startswith("week"):
                derived = base_date - timedelta(days=7 * amount)
            elif unit.startswith("month"):
                derived = _subtract_months(base_date, amount)
            else:
                derived = _subtract_years(base_date, amount)
            add_date(derived, f"relative-{unit}", 0.84 if amount <= 12 else 0.76)

        for match in _LAST_SPAN_RE.finditer(text):
            unit = match.group(1).lower()
            if unit == "week":
                derived = base_date - timedelta(days=7)
            elif unit == "month":
                derived = _subtract_months(base_date, 1)
            else:
                derived = _subtract_years(base_date, 1)
            add_date(derived, f"relative-last-{unit}", 0.78)

        lowered = text.lower()
        for match in _RELATIVE_TO_ANCHOR_RE.finditer(text):
            amount = _parse_numeric_token(match.group(1))
            unit = match.group(2).lower()
            direction = match.group(3).lower()
            anchor_name = match.group(4).lower()
            anchor = _named_anchor_date(anchor_name, base_date)
            if not amount or anchor is None:
                continue
            if unit.startswith("day"):
                delta = timedelta(days=amount)
                derived = anchor - delta if direction == "before" else anchor + delta
                add_date(derived, f"relative-{direction}-{anchor_name.replace(' ', '-')}", 0.9)
            elif unit.startswith("week"):
                delta = timedelta(days=7 * amount)
                derived = anchor - delta if direction == "before" else anchor + delta
                add_date(derived, f"relative-{direction}-{anchor_name.replace(' ', '-')}", 0.9)
            elif unit.startswith("month"):
                if direction == "before":
                    derived = _subtract_months(anchor, amount)
                else:
                    year = anchor.year
                    month = anchor.month + amount
                    while month > 12:
                        year += 1
                        month -= 12
                    derived = date(year, month, min(anchor.day, monthrange(year, month)[1]))
                add_date(derived, f"relative-{direction}-{anchor_name.replace(' ', '-')}", 0.88)
            else:
                if direction == "before":
                    derived = _subtract_years(anchor, amount)
                else:
                    derived = date(anchor.year + amount, anchor.month, min(anchor.day, monthrange(anchor.year + amount, anchor.month)[1]))
                add_date(derived, f"relative-{direction}-{anchor_name.replace(' ', '-')}", 0.88)

        for anchor_name in ("black friday", "cyber monday"):
            if anchor_name in lowered:
                anchor = _named_anchor_date(anchor_name, base_date)
                add_date(anchor, f"named-anchor-{anchor_name.replace(' ', '-')}", 0.88)

        if "yesterday" in lowered:
            add_date(base_date - timedelta(days=1), "relative-yesterday", 0.82)
        if "today" in lowered:
            add_date(base_date, "relative-today", 0.82)

    return values


def _primary_event_date(text, session_date):
    derived = _derive_event_date_candidates(text, session_date)
    if derived:
        return derived[0]["date"], derived
    normalized = normalize_date(session_date)
    if normalized:
        return normalized, [{"date": normalized, "source": "session-date", "confidence": 0.36}]
    return "", []


def _normalize_target_text(text):
    return normalize_answer(text)


def _phrase_matches_text(normalized_target, normalized_text):
    if not normalized_target or not normalized_text:
        return False
    if normalized_target in normalized_text:
        return True
    target_tokens = [_normalize_word_form(part) for part in normalized_target.split() if len(part) >= 3]
    text_tokens = _normalized_token_forms(normalized_text)
    return bool(target_tokens) and all(token in text_tokens for token in target_tokens)


def _event_dates_from_candidates(candidates):
    return [item["date"] for item in candidates if item.get("date")]


def _cleanup_event_surface(text):
    value = _collapse(text).strip(" ,.")
    value = _LEADING_PREPOSITION_RE.sub("", value)
    value = _TRAILING_RELATIVE_FRAGMENT_RE.sub("", value)
    value = _LEADING_PREPOSITION_RE.sub("", value)
    return value.strip(" ,.")


def _dedupe_preserve(values):
    ordered = []
    seen = set()
    for value in values:
        cleaned = _collapse(value).strip(" ,.")
        normalized = normalize_answer(cleaned)
        if not cleaned or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(cleaned)
    return ordered


def _focus_fact_detail(content, event_meta):
    text = _collapse(content)
    if not text:
        return ""

    candidates = []
    for item in event_meta.get("event_items", []):
        candidates.append(item.get("label", ""))
        candidates.extend(item.get("aliases", [])[:3])
    candidates = _dedupe_preserve(candidates)
    lowered = text.lower()

    best_index = None
    best_candidate = ""
    for candidate in candidates:
        index = lowered.find(candidate.lower())
        if index == -1:
            continue
        if best_index is None or index < best_index:
            best_index = index
            best_candidate = candidate

    if best_index is None:
        return _trim(text, limit=160)

    start = 0
    for marker in (". ", "? ", "! ", "; "):
        position = text.rfind(marker, 0, best_index)
        if position != -1:
            start = max(start, position + len(marker))

    if start == 0:
        for marker in (", by the way, ", ". by the way, ", ", also, ", ". also, "):
            position = lowered.rfind(marker, 0, best_index)
            if position != -1:
                start = max(start, position + 2)

    end = len(text)
    search_from = best_index + len(best_candidate)
    for marker in (". ", "? ", "! ", "; "):
        position = text.find(marker, search_from)
        if position != -1:
            end = min(end, position + 1)

    snippet = text[start:end]
    if len(snippet) > 180:
        local_start = max(0, best_index - start - 36)
        local_end = min(len(snippet), local_start + 180)
        snippet = snippet[local_start:local_end]
    return _trim(snippet, limit=180)


def _clean_question_clause(text):
    value = _collapse(text).strip(" ?.")
    value = _LEADING_I_RE.sub("", value)
    return value.strip()


def _extract_question_targets(question):
    targets = []
    lowered = question.lower()
    quoted_targets = [match.strip() for match in re.findall(r"['\"]([^'\"]+)['\"]", question)]
    if len(quoted_targets) >= 2:
        targets = quoted_targets[:2]
    else:
        match = _QUESTION_OR_RE.search(question)
        if match:
            targets = [match.group(1).strip(" '\""), match.group(2).strip(" '\"")]
        else:
            match = _QUESTION_AND_RE.search(question)
            if match:
                targets = [match.group(1).strip(" '\""), match.group(2).strip(" '\"")]
            else:
                match = _DIFFERENCE_BEFORE_AFTER_RE.search(question)
                if match:
                    targets = [
                        _clean_question_clause(match.group(4)),
                        _clean_question_clause(match.group(3)),
                    ]
                else:
                    match = _DURATION_AFTER_RE.search(question)
                    if match:
                        targets = [
                            _clean_question_clause(match.group(1)),
                            _clean_question_clause(match.group(2)),
                        ]
                    else:
                        match = _AGO_RE.search(question)
                        if match:
                            targets = [_clean_question_clause(match.group(2))]
                        else:
                            match = _DATE_QUESTION_RE.search(question)
                            if match:
                                targets = [_clean_question_clause(match.group(1))]
    if not targets and quoted_targets:
        targets = quoted_targets[:1]
    if not targets and "before" in lowered and "did i" in lowered:
        fragments = re.split(r"\bbefore\b|\bafter\b", question, maxsplit=1, flags=re.IGNORECASE)
        if len(fragments) == 2:
            right = _clean_question_clause(fragments[1].replace("?", ""))
            left = re.split(r"\bdid i\b", fragments[0], flags=re.IGNORECASE)
            if len(left) > 1:
                inferred = _clean_question_clause(left[-1])
                if inferred and right:
                    targets = [inferred, right]
    deduped = []
    seen = set()
    for target in targets:
        normalized = normalize_answer(target)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(target)
    return deduped


def _target_aliases(text):
    normalized = _normalize_target_text(text)
    if not normalized:
        return []
    values = {normalized}
    simplified = _LEADING_ARTICLE_RE.sub("", normalized)
    values.add(simplified)
    values.add(_OF_PREFIX_RE.sub("", simplified))
    values.add(_GERUND_ACTION_RE.sub("", simplified))
    values.add(_PAST_ACTION_RE.sub("", simplified))
    values.add(_BASE_ACTION_RE.sub("", simplified))
    values.add(_ORDINAL_PREFIX_RE.sub("", simplified))
    values.add(_TRAILING_MONTH_CLAUSE_RE.sub("", simplified))
    if " of " in simplified:
        values.add(simplified.split(" of ", 1)[1])
    if " with " in simplified:
        values.add(simplified.split(" with ", 1)[1])
    if " event" in simplified:
        values.add(simplified.replace(" event", " party"))
    if " party" in simplified:
        values.add(simplified.replace(" party", " event"))
    aliases = []
    for value in values:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            continue
        variants = {
            cleaned,
            _LEADING_ARTICLE_RE.sub("", cleaned),
            _ORDINAL_PREFIX_RE.sub("", cleaned),
            _TRAILING_MONTH_CLAUSE_RE.sub("", cleaned),
        }
        expanded = set()
        for variant in variants:
            candidate = _LEADING_ARTICLE_RE.sub("", _ORDINAL_PREFIX_RE.sub("", _TRAILING_MONTH_CLAUSE_RE.sub("", variant))).strip()
            if candidate:
                expanded.add(candidate)
                if " event" in candidate:
                    expanded.add(candidate.replace(" event", " party"))
                if " party" in candidate:
                    expanded.add(candidate.replace(" party", " event"))
        for variant in expanded | variants:
            variant = " ".join(variant.split()).strip()
            if variant and variant not in aliases:
                aliases.append(variant)
    return aliases


def _extract_event_mentions(text):
    mentions = []
    seen = set()
    for activity_type, pattern in _EVENT_CAPTURE_PATTERNS:
        for match in pattern.finditer(text or ""):
            phrase = _cleanup_event_surface(match.group(1))
            normalized = normalize_answer(phrase)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            mentions.append(
                {
                    "activity_type": activity_type,
                    "surface": phrase,
                    "aliases": _target_aliases(phrase),
                }
            )
    return mentions


def _extract_lowercase_list_mentions(text):
    mentions = []
    seen = set()
    for match in _LOWERCASE_LIST_RE.finditer(text or ""):
        raw = match.group(1)
        parts = re.split(r",|\band\b", raw)
        for part in parts:
            cleaned = _cleanup_event_surface(part)
            normalized = normalize_answer(cleaned)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            mentions.append(
                {
                    "activity_type": _extract_activity_type(text),
                    "surface": cleaned,
                    "aliases": _target_aliases(cleaned),
                }
            )
    return mentions


def _build_event_items(text, session_date, session_id=None, turn_index=None, role=None, has_answer=False):
    primary_date, date_candidates = _primary_event_date(text, session_date)
    mentions = _extract_event_mentions(text)
    mentions.extend(_extract_lowercase_list_mentions(text))
    if not mentions:
        mentions = [
            {
                "activity_type": _extract_activity_type(text),
                "surface": _trim(text, limit=72),
                "aliases": _target_aliases(_trim(text, limit=72)),
            }
        ]
    primary_candidate = date_candidates[0] if date_candidates else {"date": primary_date, "source": "missing", "confidence": 0.0}
    items = []
    for mention in mentions:
        items.append(
            {
                "label": mention["surface"],
                "normalized_label": normalize_answer(mention["surface"]),
                "aliases": mention["aliases"],
                "activity_type": mention["activity_type"],
                "event_date": primary_candidate.get("date", ""),
                "date_source": primary_candidate.get("source", "missing"),
                "date_confidence": primary_candidate.get("confidence", 0.0),
                "date_candidates": date_candidates,
                "session_id": session_id,
                "turn_index": turn_index,
                "role": role,
                "has_answer": bool(has_answer),
            }
        )
    return {
        "event_items": items,
        "event_dates": _event_dates_from_candidates(date_candidates),
        "event_date": primary_candidate.get("date", ""),
        "date_source": primary_candidate.get("source", "missing"),
        "date_confidence": primary_candidate.get("confidence", 0.0),
        "date_candidates": date_candidates,
        "event_aliases": [alias for item in items for alias in item["aliases"]],
    }


def _build_fact_memory_text(content, event_meta, session_date):
    date_value = event_meta.get("event_date") or normalize_date(session_date)
    labels = []
    for item in event_meta.get("event_items", []):
        labels.append(item.get("label", ""))
        labels.extend(item.get("aliases", [])[:2])
    label_text = "; ".join(_dedupe_preserve(labels)[:4])
    detail_text = _focus_fact_detail(content, event_meta)
    parts = []
    if date_value:
        parts.append(date_value)
    if label_text:
        parts.append(f"events: {label_text}")
    if detail_text:
        parts.append(f"details: {detail_text}")
    return " | ".join(parts)


def _session_entities(session):
    entities = []
    seen = set()
    for turn in session:
        for entity in extract_entities(turn.get("content") or ""):
            if entity in seen:
                continue
            seen.add(entity)
            entities.append(entity)
    return entities


def _session_key_snippets(session, include_assistant_turns=True, limit=3):
    preferred = []
    fallback = []
    for turn in session:
        role = (turn.get("role") or "user").strip().lower()
        if role == "assistant" and not include_assistant_turns:
            continue
        content = _collapse(turn.get("content") or "")
        if not content:
            continue
        if turn.get("has_answer") or role == "user":
            preferred.append(turn)
        else:
            fallback.append(turn)
    ordered = preferred + fallback
    return ordered[:limit]


def _build_episode_memory(session_id, session_date, session, include_assistant_turns=True):
    snippets = _session_key_snippets(
        session,
        include_assistant_turns=include_assistant_turns,
        limit=3,
    )
    summary_lines = [_compact_turn_text(turn, session_date) for turn in snippets]
    entities = _session_entities(session)
    summary_text = "\n".join(summary_lines) if summary_lines else format_session_text(session_id, session_date, session)
    event_meta = _build_event_items(summary_text, session_date, session_id=session_id, role="summary")
    date_value = event_meta["event_date"] or normalize_date(session_date)
    return {
        "text": f"Episode summary for {session_id}:\n{summary_text}",
        "memory_type": "episode",
        "importance": 0.95 if any(turn.get("has_answer") for turn in session) else 0.7,
        "metadata": {
            "session_id": session_id,
            "session_date": session_date,
            "event_date": date_value,
            "event_dates": event_meta["event_dates"] or ([date_value] if date_value else []),
            "entities": entities,
            "activity_type": _extract_activity_type(summary_text),
            "summary": _trim(summary_text, limit=220),
            "granularity": "episode",
            "event_items": event_meta["event_items"],
            "event_aliases": event_meta["event_aliases"],
            "date_source": event_meta["date_source"] if event_meta["event_items"] else "session-date",
            "date_confidence": event_meta["date_confidence"] if event_meta["event_items"] else 0.36,
            "date_candidates": event_meta["date_candidates"],
        },
    }


def _build_timeline_memory(session_id, session_date, session, include_assistant_turns=True):
    snippets = _session_key_snippets(
        session,
        include_assistant_turns=include_assistant_turns,
        limit=2,
    )
    date_value = normalize_date(session_date)
    facts = []
    for turn in snippets:
        content = _trim(turn.get("content") or "", limit=120)
        facts.append(f"{date_value}: {content}" if date_value else content)
    timeline_text = "\n".join(facts) if facts else format_session_text(session_id, session_date, session)
    entities = _session_entities(session)
    event_meta = _build_event_items(timeline_text, session_date, session_id=session_id, role="timeline")
    return {
        "text": f"Timeline evidence for {session_id}:\n{timeline_text}",
        "memory_type": "timeline",
        "importance": 0.9 if any(turn.get("has_answer") for turn in session) else 0.68,
        "metadata": {
            "session_id": session_id,
            "session_date": session_date,
            "event_date": event_meta["event_date"] or date_value,
            "event_dates": event_meta["event_dates"] or ([date_value] if date_value else []),
            "entities": entities,
            "activity_type": _extract_activity_type(timeline_text),
            "summary": _trim(timeline_text, limit=220),
            "granularity": "timeline",
            "event_items": event_meta["event_items"],
            "event_aliases": event_meta["event_aliases"],
            "date_source": event_meta["date_source"] if event_meta["event_items"] else "session-date",
            "date_confidence": event_meta["date_confidence"] if event_meta["event_items"] else 0.36,
            "date_candidates": event_meta["date_candidates"],
        },
    }


def _build_fact_memories(session_id, session_date, session, include_assistant_turns=True):
    for turn_index, turn in enumerate(session):
        role = (turn.get("role") or "user").strip().lower()
        content = _collapse(turn.get("content") or "")
        if not content:
            continue
        if role == "assistant" and not include_assistant_turns:
            continue
        entities = extract_entities(content)
        event_meta = _build_event_items(
            content,
            session_date,
            session_id=session_id,
            turn_index=turn_index,
            role=role,
            has_answer=turn.get("has_answer"),
        )
        date_value = event_meta["event_date"]
        fact_text = _build_fact_memory_text(content, event_meta, session_date)
        yield {
            "text": fact_text,
            "memory_type": "event",
            "importance": 0.98 if turn.get("has_answer") else (0.78 if role == "user" else 0.55),
            "metadata": {
                "session_id": session_id,
                "session_date": session_date,
                "event_date": date_value,
                "event_dates": event_meta["event_dates"],
                "entities": entities,
                "activity_type": _extract_activity_type(content),
                "fact_text": _collapse(content),
                "turn_index": turn_index,
                "role": role,
                "has_answer": bool(turn.get("has_answer")),
                "granularity": "fact",
                "event_items": event_meta["event_items"],
                "event_aliases": event_meta["event_aliases"],
                "date_source": event_meta["date_source"],
                "date_confidence": event_meta["date_confidence"],
                "date_candidates": event_meta["date_candidates"],
            },
        }


def _build_global_timeline_memory(instance, include_assistant_turns=True):
    session_ids = instance.get("haystack_session_ids", [])
    session_dates = instance.get("haystack_dates", [])
    sessions = instance.get("haystack_sessions", [])
    rows = []
    entities = []
    seen_entities = set()
    for session_id, session_date, session in zip(session_ids, session_dates, sessions):
        snippets = _session_key_snippets(
            session,
            include_assistant_turns=include_assistant_turns,
            limit=1,
        )
        if not snippets:
            continue
        summary = _trim(snippets[0].get("content") or "", limit=120)
        date_value = normalize_date(session_date)
        rows.append((date_value or "unknown", session_id, summary))
        for entity in _session_entities(session):
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            entities.append(entity)
    if not rows:
        return None
    rows.sort(key=lambda item: item[0])
    lines = [f"{date_value} | {session_id} | {summary}" for date_value, session_id, summary in rows]
    event_items = []
    for date_value, session_id, summary in rows:
        event_items.append(
            {
                "label": summary,
                "normalized_label": normalize_answer(summary),
                "aliases": _target_aliases(summary),
                "activity_type": "timeline",
                "event_date": date_value if date_value != "unknown" else "",
                "date_source": "session-date",
                "date_confidence": 0.36,
                "date_candidates": (
                    [{"date": date_value, "source": "session-date", "confidence": 0.36}]
                    if date_value != "unknown"
                    else []
                ),
                "session_id": session_id,
                "turn_index": None,
                "role": "timeline",
                "has_answer": False,
            }
        )
    return {
        "text": "Global timeline evidence:\n" + "\n".join(lines),
        "memory_type": "timeline",
        "importance": 0.92,
        "metadata": {
            "session_ids": list(session_ids),
            "session_date": rows[0][0],
            "event_date": rows[0][0],
            "event_dates": [value for value, _, _ in rows if value and value != "unknown"],
            "entities": entities,
            "activity_type": "timeline",
            "summary": _trim(" ; ".join(lines), limit=240),
            "granularity": "timeline-global",
            "event_items": event_items,
            "event_aliases": [alias for item in event_items for alias in item["aliases"]],
            "date_source": "session-date",
            "date_confidence": 0.36,
            "date_candidates": (
                [{"date": value, "source": "session-date", "confidence": 0.36} for value, _, _ in rows if value and value != "unknown"]
            ),
        },
    }


def iter_history_memories(instance, granularity="hybrid", include_assistant_turns=True):
    session_ids = instance.get("haystack_session_ids", [])
    session_dates = instance.get("haystack_dates", [])
    sessions = instance.get("haystack_sessions", [])

    if granularity not in {"turn", "session", "hybrid"}:
        raise ValueError("granularity must be 'turn', 'session', or 'hybrid'")

    if granularity == "hybrid":
        global_timeline = _build_global_timeline_memory(
            instance,
            include_assistant_turns=include_assistant_turns,
        )
        if global_timeline is not None:
            global_timeline["metadata"].update(
                {
                    "question_id": instance.get("question_id"),
                }
            )
            yield global_timeline

    for session_id, session_date, session in zip(session_ids, session_dates, sessions):
        if granularity in {"session", "hybrid"}:
            episode = _build_episode_memory(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            )
            episode["metadata"].update({"question_id": instance.get("question_id")})
            yield episode
        if granularity == "hybrid":
            timeline = _build_timeline_memory(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            )
            timeline["metadata"].update({"question_id": instance.get("question_id")})
            yield timeline
        if granularity in {"turn", "hybrid"}:
            for fact_memory in _build_fact_memories(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            ):
                fact_memory["metadata"].update({"question_id": instance.get("question_id")})
                yield fact_memory


def build_query_text(instance, include_question_date=True):
    question = (instance.get("question") or "").strip()
    question_date = (instance.get("question_date") or "").strip()
    if include_question_date and question_date:
        normalized = normalize_date(question_date)
        if normalized:
            return f"Question date: {normalized}\nQuestion: {question}"
        return f"Question date: {question_date}\nQuestion: {question}"
    return question


def analyze_question(instance, include_question_date=True):
    question = (instance.get("question") or "").strip()
    lowered = question.lower()
    question_date = (instance.get("question_date") or "").strip()
    normalized_question_date = normalize_date(question_date)
    targets = _extract_question_targets(question)

    reasoning_kind = "factual"
    unit_hint = ""
    ordering_direction = "first"
    if "how many days" in lowered:
        reasoning_kind = "difference"
        unit_hint = "days"
    elif "how many weeks" in lowered:
        reasoning_kind = "difference"
        unit_hint = "weeks"
    elif "how many months" in lowered:
        reasoning_kind = "difference"
        unit_hint = "months"
    elif "what was the date" in lowered or "which date" in lowered:
        reasoning_kind = "date"
        ordering_direction = "last" if " last" in lowered else "first"
    elif any(token in lowered for token in ("first", "last", "before", "after")):
        reasoning_kind = "ordering"
        ordering_direction = "last" if " last" in lowered else "first"

    filter_month = ""
    month_match = _IN_MONTH_RE.search(question)
    if month_match:
        filter_month = f"{_MONTHS[month_match.group(1).lower()]:02d}"

    return QuestionPlan(
        question_id=instance.get("question_id", ""),
        question_type=instance.get("question_type", ""),
        question=question,
        query_text=build_query_text(instance, include_question_date=include_question_date),
        question_date=question_date,
        normalized_question_date=normalized_question_date,
        reasoning_kind=reasoning_kind,
        is_temporal="temporal" in instance.get("question_type", "") or reasoning_kind in {"ordering", "difference", "date"},
        unit_hint=unit_hint,
        targets=targets,
        normalized_targets=[_normalize_target_text(target) for target in targets],
        query_entities=extract_entities(question),
        question_month=(normalized_question_date or "")[5:7] if normalized_question_date else "",
        ordering_direction=ordering_direction,
        filter_month=filter_month,
    )


def normalize_answer(text):
    normalized = (text or "").lower().replace("\u2019", "'")
    normalized = _ORDINAL_SUFFIX_RE.sub(r"\1", normalized)
    normalized = normalized.translate(_PUNCT_TABLE)
    normalized = _ARTICLES_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def exact_match_score(prediction, answer):
    variants = acceptable_answers(answer)
    if not variants:
        return 0.0
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(variant) for variant in variants))


def contains_match_score(prediction, answer):
    normalized_prediction = normalize_answer(prediction)
    variants = acceptable_answers(answer)
    if not normalized_prediction or not variants:
        return 0.0
    return float(
        any(
            normalize_answer(variant) in normalized_prediction
            or normalized_prediction in normalize_answer(variant)
            for variant in variants
        )
    )


def token_f1_score(prediction, answer):
    variants = acceptable_answers(answer)
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens or not variants:
        return 0.0

    best = 0.0
    for variant in variants:
        answer_tokens = normalize_answer(variant).split()
        if not answer_tokens:
            continue
        overlap = Counter(prediction_tokens) & Counter(answer_tokens)
        shared = sum(overlap.values())
        if shared == 0:
            continue
        precision = shared / len(prediction_tokens)
        recall = shared / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def abstention_score(prediction, is_abstention_question):
    if not is_abstention_question:
        return None
    normalized_prediction = normalize_answer(prediction)
    return float(any(marker in normalized_prediction for marker in _ABSTENTION_MARKERS))


def selected_session_ids(selected_hits):
    seen = set()
    ordered = []
    for hit in selected_hits:
        session_ids = []
        if hit.record.metadata.get("session_id"):
            session_ids.append(hit.record.metadata["session_id"])
        session_ids.extend(hit.record.metadata.get("session_ids", []))
        for session_id in session_ids:
            if not session_id or session_id in seen:
                continue
            seen.add(session_id)
            ordered.append(session_id)
    return ordered


def selected_session_recall(selected_hits, answer_session_ids):
    gold = set(answer_session_ids or [])
    if not gold:
        return None
    selected = set(selected_session_ids(selected_hits))
    return len(selected & gold) / len(gold)


def question_policy(plan):
    policy = {
        "top_k": 24,
        "max_items": 8,
        "token_budget": 768,
        "critic_threshold": 0.52,
        "maybe_threshold": 0.42,
        "similarity_threshold": 0.12,
    }
    if plan.is_temporal:
        policy.update(
            {
                "top_k": 40,
                "max_items": 12,
                "token_budget": 1200,
                "critic_threshold": 0.44,
                "maybe_threshold": 0.34,
                "similarity_threshold": 0.08,
            }
        )
    return policy


def _target_matches(plan, hit):
    if not plan.normalized_targets:
        return []
    text = normalize_answer(hit.record.text)
    metadata = hit.record.metadata
    entities = {
        _normalize_target_text(value)
        for value in metadata.get("entities", [])
        if value
    }
    aliases = {
        _normalize_target_text(value)
        for value in metadata.get("event_aliases", [])
        if value
    }
    matches = []
    for raw, normalized in zip(plan.targets, plan.normalized_targets):
        target_aliases = _target_aliases(raw)
        if normalized and (
            normalized in entities
            or normalized in aliases
            or _phrase_matches_text(normalized, text)
            or any(_phrase_matches_text(normalized, entity) for entity in entities)
            or any(
                _phrase_matches_text(alias, text) or alias in aliases
                for alias in target_aliases
            )
        ):
            matches.append(raw)
    return matches


def _target_coverage(plan, hit):
    matches = {_normalize_target_text(value) for value in _target_matches(plan, hit)}
    metadata = hit.record.metadata
    source_text = " ".join(
        value
        for value in (
            hit.record.text,
            metadata.get("summary", ""),
            metadata.get("fact_text", ""),
        )
        if value
    )
    normalized_text = normalize_answer(source_text)
    alias_text = " ".join(metadata.get("event_aliases", []))
    normalized_alias_text = normalize_answer(alias_text)
    for normalized in plan.normalized_targets:
        if not normalized:
            continue
        if _phrase_matches_text(normalized, normalized_text) or _phrase_matches_text(normalized, normalized_alias_text):
            matches.add(normalized)
    return matches


def _hit_event_dates(hit):
    metadata = hit.record.metadata
    dates = []
    for value in metadata.get("event_dates", []):
        if value and value not in dates:
            dates.append(value)
    primary = metadata.get("event_date")
    if primary and primary not in dates:
        dates.append(primary)
    return dates


def _hit_date_confidence(hit):
    return float(hit.record.metadata.get("date_confidence", 0.0) or 0.0)


def _granularity(hit):
    return hit.record.metadata.get("granularity", "")


def _granularity_priority(hit):
    granularity = _granularity(hit)
    if granularity == "fact":
        return 4
    if granularity == "timeline":
        return 3
    if granularity == "timeline-global":
        return 2
    if granularity == "episode":
        return 1
    return 0


def _candidate_selection_score(plan, hit):
    score = temporal_bundle_score(plan, hit)
    score += 0.35 * _granularity_priority(hit)
    score += 0.45 * len(_target_matches(plan, hit))
    if _hit_event_dates(hit):
        score += 0.15
    score += 0.22 * _hit_date_confidence(hit)
    if hit.record.memory_type == "ephemeral" and not _target_matches(plan, hit):
        score -= 0.8
    return score


def temporal_bundle_score(plan, hit):
    metadata = hit.record.metadata
    score = (
        (2.2 * hit.score)
        + (1.8 * hit.critic_confidence)
        + (0.6 * hit.record.importance)
    )
    granularity = metadata.get("granularity", "")
    if _hit_event_dates(hit):
        score += 0.45
        score += 0.18 * _hit_date_confidence(hit)
    if granularity == "timeline-global":
        score += 0.7
    elif granularity == "timeline":
        score += 0.35
    elif granularity == "episode":
        score -= 0.15
    elif granularity == "fact":
        score += 0.75
    if metadata.get("has_answer"):
        score += 0.2
    score += 0.35 * len(_target_matches(plan, hit))
    score += 0.1 * len(set(plan.query_entities) & set(metadata.get("entities", [])))
    if hit.record.memory_type == "ephemeral" and not _target_matches(plan, hit):
        score -= 0.6
    return score


def render_evidence_line(hit, index=None):
    metadata = hit.record.metadata
    date_value = metadata.get("event_date") or normalize_date(metadata.get("session_date", ""))
    anchor_date = normalize_date(metadata.get("session_date", ""))
    granularity = metadata.get("granularity") or hit.record.memory_type
    source = metadata.get("session_id") or ",".join(metadata.get("session_ids", [])[:2]) or "unknown"
    entities = ", ".join(metadata.get("entities", [])[:3]) or "n/a"
    evidence = metadata.get("fact_text") or metadata.get("summary") or hit.record.text
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


def _hit_token_cost(hit, encode=None):
    return estimate_token_count(render_evidence_line(hit), encode=encode)


def select_bundled_hits(plan, hits, max_items, max_tokens, encode=None):
    ranked = sorted(hits, key=lambda hit: _candidate_selection_score(plan, hit), reverse=True)
    selected = []
    selected_ids = set()
    used_tokens = 0
    covered_targets = set()
    covered_sessions = set()
    global_timeline_selected = False

    def add_hit(hit):
        nonlocal used_tokens, global_timeline_selected
        if hit.record.memory_id in selected_ids:
            return False
        token_cost = _hit_token_cost(hit, encode=encode)
        hit.token_cost = token_cost
        if len(selected) >= max_items or used_tokens + token_cost > max_tokens:
            return False
        selected.append(hit)
        selected_ids.add(hit.record.memory_id)
        used_tokens += token_cost
        covered_sessions.update(selected_session_ids([hit]))
        covered_targets.update(_target_coverage(plan, hit))
        if hit.record.metadata.get("granularity") == "timeline-global":
            global_timeline_selected = True
        return True

    def current_answerability():
        return assess_answerability(plan, selected)

    def useful_increment(hit):
        hit_targets = _target_coverage(plan, hit)
        hit_sessions = set(selected_session_ids([hit]))
        hit_dates = set(_hit_event_dates(hit))
        current_dates = {
            value
            for item in selected
            for value in _hit_event_dates(item)
        }
        if hit.record.memory_type == "ephemeral" and not hit_targets and not hit_dates:
            return False
        if hit_targets - covered_targets:
            return True
        if hit_sessions - covered_sessions:
            return True
        if hit_dates - current_dates:
            return True
        return _granularity(hit) == "fact" and bool(hit_targets)

    if plan.is_temporal:
        global_timeline = next(
            (hit for hit in ranked if hit.record.metadata.get("granularity") == "timeline-global"),
            None,
        )
        for target in plan.normalized_targets:
            candidates = [
                hit
                for hit in ranked
                if target in _target_coverage(plan, hit)
            ]
            candidates.sort(
                key=lambda hit: (
                    _granularity_priority(hit),
                    float(hit.record.metadata.get("date_confidence", 0.0) or 0.0),
                    1 if hit.record.metadata.get("has_answer") else 0,
                    1 if _hit_event_dates(hit) else 0,
                    _candidate_selection_score(plan, hit),
                ),
                reverse=True,
            )
            target_hit = next(iter(candidates), None)
            if target_hit is not None:
                add_hit(target_hit)

        if global_timeline is not None and (
            len(plan.normalized_targets) >= 2 and len(covered_targets) < min(2, len(plan.normalized_targets))
        ):
            add_hit(global_timeline)

        for hit in ranked:
            if len(selected) >= max_items:
                break
            if hit.record.memory_id in selected_ids:
                continue
            if not useful_increment(hit):
                continue
            if _granularity(hit) == "episode" and current_answerability()["sufficient"]:
                continue
            add_hit(hit)
            if current_answerability()["sufficient"] and len(selected) >= max(2, min(4, len(plan.normalized_targets) + 1)):
                break

        if not current_answerability()["sufficient"]:
            for hit in ranked:
                if len(selected) >= max_items:
                    break
                if hit.record.memory_id in selected_ids:
                    continue
                if _granularity(hit) == "episode" and current_answerability()["sufficient"]:
                    continue
                if _granularity(hit) in {"fact", "timeline"} or (
                    global_timeline_selected and _granularity(hit) == "episode"
                ):
                    add_hit(hit)
                if current_answerability()["sufficient"] and len(selected) >= max(2, min(4, len(plan.normalized_targets) + 1)):
                    break
    else:
        for hit in ranked:
            if not add_hit(hit):
                continue
            if len(selected) >= max_items:
                break

    return selected, used_tokens


def assess_answerability(plan, selected_hits):
    distinct_dates = {
        value
        for hit in selected_hits
        for value in _hit_event_dates(hit)
        if value
    }
    distinct_sessions = set(selected_session_ids(selected_hits))
    covered_targets = set()
    for hit in selected_hits:
        covered_targets.update(_target_coverage(plan, hit))

    reasons = []
    sufficient = True
    if plan.reasoning_kind == "difference":
        if len(distinct_dates) < 2:
            sufficient = False
            reasons.append("need-two-dates")
        if plan.normalized_targets:
            required_coverage = min(2, len(plan.normalized_targets))
            if len(covered_targets) < required_coverage:
                sufficient = False
                reasons.append("missing-target-coverage")
    elif plan.reasoning_kind == "ordering":
        if len(distinct_dates) < 2 and len(distinct_sessions) < 2:
            sufficient = False
            reasons.append("need-two-events")
        if plan.normalized_targets:
            required_coverage = min(2, len(plan.normalized_targets))
            if len(covered_targets) < required_coverage:
                has_global_timeline = any(
                    hit.record.metadata.get("granularity") == "timeline-global"
                    for hit in selected_hits
                )
                if not (has_global_timeline and len(distinct_sessions) >= 2):
                    sufficient = False
                    reasons.append("missing-target-coverage")
    elif plan.reasoning_kind == "date":
        if not distinct_dates:
            sufficient = False
            reasons.append("missing-date")

    return {
        "sufficient": sufficient,
        "reasons": reasons or ["enough-evidence"],
        "distinct_dates": sorted(value for value in distinct_dates if value),
        "distinct_sessions": sorted(distinct_sessions),
        "covered_targets": sorted(covered_targets),
    }


def _flatten_structured_events(plan, selected_hits):
    events = []
    for hit in selected_hits:
        metadata = hit.record.metadata
        event_items = metadata.get("event_items") or []
        if not event_items:
            event_items = [
                {
                    "label": metadata.get("fact_text") or metadata.get("summary") or hit.record.text,
                    "normalized_label": normalize_answer(metadata.get("fact_text") or metadata.get("summary") or hit.record.text),
                    "aliases": metadata.get("event_aliases", []),
                    "activity_type": metadata.get("activity_type", hit.record.memory_type),
                    "event_date": metadata.get("event_date", ""),
                    "date_source": metadata.get("date_source", "unknown"),
                    "date_confidence": metadata.get("date_confidence", 0.0),
                    "date_candidates": metadata.get("date_candidates", []),
                    "session_id": metadata.get("session_id"),
                    "turn_index": metadata.get("turn_index"),
                    "role": metadata.get("role"),
                    "has_answer": metadata.get("has_answer", False),
                }
            ]
        for item in event_items:
            aliases = []
            for alias in item.get("aliases", []):
                for normalized_alias in _target_aliases(alias):
                    if normalized_alias and normalized_alias not in aliases:
                        aliases.append(normalized_alias)
            label = item.get("label") or metadata.get("fact_text") or metadata.get("summary") or hit.record.text
            normalized_label = normalize_answer(label)
            normalized_text = normalize_answer(hit.record.text)
            target_matches = []
            for raw_target, normalized_target in zip(plan.targets, plan.normalized_targets):
                target_aliases = _target_aliases(raw_target)
                if any(
                    _phrase_matches_text(candidate, normalized_text)
                    or _phrase_matches_text(candidate, normalized_label)
                    or candidate in aliases
                    for candidate in [normalized_target] + target_aliases
                    if candidate
                ):
                    target_matches.append(raw_target)
            date_value = item.get("event_date") or metadata.get("event_date", "")
            event = {
                "memory_id": hit.record.memory_id,
                "memory_type": hit.record.memory_type,
                "granularity": metadata.get("granularity", hit.record.memory_type),
                "label": label,
                "normalized_label": normalized_label,
                "aliases": aliases or _target_aliases(label),
                "activity_type": item.get("activity_type") or metadata.get("activity_type", hit.record.memory_type),
                "event_date": date_value,
                "date_source": item.get("date_source") or metadata.get("date_source", "unknown"),
                "date_confidence": float(item.get("date_confidence", metadata.get("date_confidence", 0.0)) or 0.0),
                "session_id": item.get("session_id") or metadata.get("session_id"),
                "turn_index": item.get("turn_index"),
                "role": item.get("role") or metadata.get("role"),
                "hit_score": hit.score,
                "critic_confidence": hit.critic_confidence,
                "importance": hit.record.importance,
                "target_matches": target_matches,
                "preview": metadata.get("fact_text") or metadata.get("summary") or preview(hit.record.text, limit=160),
            }
            event["event_score"] = (
                (2.0 * event["hit_score"])
                + (1.4 * event["critic_confidence"])
                + (0.5 * event["importance"])
                + (0.5 * event["date_confidence"])
                + (0.45 * len(target_matches))
                + (0.12 * _granularity_priority(hit))
            )
            events.append(event)
    deduped = []
    seen = set()
    for event in sorted(events, key=lambda item: item["event_score"], reverse=True):
        key = (
            event["memory_id"],
            event["normalized_label"],
            event["event_date"],
            tuple(event["aliases"][:4]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _event_matches_target(event, target):
    if not target:
        return False
    target_aliases = _target_aliases(target)
    text = normalize_answer(" ".join([event.get("label", ""), event.get("preview", "")]))
    for candidate in target_aliases:
        if not candidate:
            continue
        if candidate in event.get("aliases", []):
            return True
        if _phrase_matches_text(candidate, text):
            return True
        target_tokens = {token for token in candidate.split() if len(token) >= 3}
        event_tokens = {token for token in text.split() if len(token) >= 3}
        alias_tokens = {
            token
            for alias in event.get("aliases", [])
            for token in alias.split()
            if len(token) >= 3
        }
        universe = event_tokens | alias_tokens
        if target_tokens and len(target_tokens & universe) / len(target_tokens) >= 0.6:
            return True
    return False


def _select_target_event(plan, events, target):
    matching = [event for event in events if _event_matches_target(event, target)]
    matching.sort(
        key=lambda event: (
            event["date_confidence"],
            1 if event["event_date"] else 0,
            event["event_score"],
        ),
        reverse=True,
    )
    return matching[0] if matching else None


def _date_from_event(event):
    return _parse_iso_date(event.get("event_date", "")) if event else None


def _difference_between_dates(left, right, unit):
    if unit == "weeks":
        return abs((left - right).days) // 7
    if unit == "months":
        return abs((left.year - right.year) * 12 + (left.month - right.month))
    return abs((left - right).days)


def _format_month_day(date_value):
    return f"{_MONTH_NAMES[date_value.month]} {date_value.day}"


def build_structured_event_view(plan, selected_hits, limit=8):
    events = _flatten_structured_events(plan, selected_hits)
    if plan.filter_month:
        filtered = [
            event for event in events
            if event.get("event_date", "")[5:7] == plan.filter_month
        ]
        if filtered:
            events = filtered
    return events[:limit]


def solve_temporal_question(plan, selected_hits):
    events = build_structured_event_view(plan, selected_hits, limit=24)
    if not events:
        return TemporalSolution(
            resolved=False,
            answer="",
            confidence=0.0,
            rationale="no-structured-events",
            mode="insufficient",
            supporting_memory_ids=[],
            supporting_dates=[],
        )

    if plan.reasoning_kind == "ordering" and len(plan.targets) >= 2:
        first = _select_target_event(plan, events, plan.targets[0])
        second = _select_target_event(plan, events, plan.targets[1])
        if not first or not second:
            return TemporalSolution(False, "", 0.0, "missing-target-events", "insufficient", [], [])
        left_date = _date_from_event(first)
        right_date = _date_from_event(second)
        if not left_date or not right_date:
            return TemporalSolution(False, "", min(first["date_confidence"], second["date_confidence"]), "missing-dates", "insufficient", [first["memory_id"], second["memory_id"]], [first.get("event_date", ""), second.get("event_date", "")])
        if left_date == right_date:
            return TemporalSolution(False, "", min(first["date_confidence"], second["date_confidence"]), "same-date", "ambiguous", [first["memory_id"], second["memory_id"]], [first["event_date"], second["event_date"]])
        answer = plan.targets[0] if left_date < right_date else plan.targets[1]
        if plan.ordering_direction == "last":
            answer = plan.targets[0] if left_date > right_date else plan.targets[1]
        return TemporalSolution(
            resolved=True,
            answer=answer,
            confidence=min(first["date_confidence"], second["date_confidence"]),
            rationale=f"{first['event_date']} vs {second['event_date']}",
            mode="pair-ordering",
            supporting_memory_ids=[first["memory_id"], second["memory_id"]],
            supporting_dates=[first["event_date"], second["event_date"]],
        )

    if plan.reasoning_kind == "difference":
        left = None
        right = None
        if len(plan.targets) >= 2:
            left = _select_target_event(plan, events, plan.targets[0])
            right = _select_target_event(plan, events, plan.targets[1])
        elif len(plan.targets) == 1:
            left = _select_target_event(plan, events, plan.targets[0])
            if plan.normalized_question_date:
                right = {
                    "memory_id": "question-date",
                    "event_date": plan.normalized_question_date,
                    "date_confidence": 1.0,
                }
        if not left or not right:
            return TemporalSolution(False, "", 0.0, "missing-difference-pair", "insufficient", [], [])
        left_date = _date_from_event(left)
        right_date = _date_from_event(right)
        if not left_date or not right_date:
            return TemporalSolution(False, "", min(left.get("date_confidence", 0.0), right.get("date_confidence", 0.0)), "missing-difference-dates", "insufficient", [left.get("memory_id", ""), right.get("memory_id", "")], [left.get("event_date", ""), right.get("event_date", "")])
        unit = plan.unit_hint or "days"
        value = _difference_between_dates(left_date, right_date, unit)
        return TemporalSolution(
            resolved=True,
            answer=f"{value} {unit}",
            confidence=min(left.get("date_confidence", 0.0), right.get("date_confidence", 0.0)),
            rationale=f"{left.get('event_date', '')} vs {right.get('event_date', '')}",
            mode="deterministic-difference",
            supporting_memory_ids=[left.get("memory_id", ""), right.get("memory_id", "")],
            supporting_dates=[left.get("event_date", ""), right.get("event_date", "")],
        )

    if plan.reasoning_kind == "date":
        candidates = events
        if plan.targets:
            candidates = [event for event in candidates if _event_matches_target(event, plan.targets[0])]
        if plan.filter_month:
            month_candidates = [event for event in candidates if event.get("event_date", "")[5:7] == plan.filter_month]
            if month_candidates:
                candidates = month_candidates
        if not candidates:
            return TemporalSolution(False, "", 0.0, "missing-date-target", "insufficient", [], [])
        candidates.sort(
            key=lambda event: (
                event["date_confidence"],
                1 if event["event_date"] else 0,
                event["event_score"],
            ),
            reverse=True,
        )
        if "first" in plan.question.lower():
            candidates.sort(key=lambda event: event.get("event_date", "9999-99-99"))
        elif "last" in plan.question.lower():
            candidates.sort(key=lambda event: event.get("event_date", ""))
            candidates.reverse()
        best = candidates[0]
        if not best.get("event_date"):
            return TemporalSolution(False, "", best.get("date_confidence", 0.0), "missing-date", "insufficient", [best["memory_id"]], [])
        parsed_date = _parse_iso_date(best["event_date"])
        month_day = _MONTH_DAY_RE.search(best["preview"])
        answer = best["event_date"]
        if month_day:
            answer = f"{month_day.group(1).capitalize()} {month_day.group(2)}"
        elif parsed_date is not None:
            answer = _format_month_day(parsed_date)
        return TemporalSolution(
            resolved=True,
            answer=answer,
            confidence=best.get("date_confidence", 0.0),
            rationale=best.get("event_date", ""),
            mode="deterministic-date",
            supporting_memory_ids=[best["memory_id"]],
            supporting_dates=[best.get("event_date", "")],
        )

    return TemporalSolution(False, "", 0.0, "unsupported-kind", "unsupported", [], [])


def render_structured_event_line(event, index=None):
    prefix = f"{index}. " if index is not None else ""
    return (
        f"{prefix}date={event.get('event_date') or 'unknown'} ; "
        f"date_source={event.get('date_source', 'unknown')} ; "
        f"date_confidence={event.get('date_confidence', 0.0):.2f} ; "
        f"kind={event.get('granularity', 'unknown')} ; "
        f"label={preview(event.get('label', ''), limit=90)} ; "
        f"aliases={', '.join(event.get('aliases', [])[:3]) or 'n/a'} ; "
        f"source={event.get('session_id') or 'unknown'}"
    )


def build_evidence_table(plan, selected_hits, structured_events=None):
    lines = []
    if structured_events:
        lines.append("Structured event view:")
        for index, event in enumerate(structured_events, start=1):
            lines.append(render_structured_event_line(event, index=index))
    if selected_hits:
        if lines:
            lines.append("")
        lines.append("Evidence table:")
        for index, hit in enumerate(selected_hits, start=1):
            lines.append(render_evidence_line(hit, index=index))
    if not lines:
        return "Evidence:\n- none"
    return "\n".join(lines)


def build_benchmark_instructions(plan, selected_hits, answerability, base_system_prompt, structured_events=None, solver_result=None):
    parts = [base_system_prompt]
    parts.append("Use only the evidence table when it is present. Do not invent dates or events.")
    if plan.reasoning_kind == "ordering":
        parts.append("For ordering questions, return only the event/item that happened first or last, with no explanation.")
        if plan.targets:
            parts.append("Valid answer options: " + " | ".join(plan.targets))
    elif plan.reasoning_kind == "difference":
        unit = plan.unit_hint or "days"
        parts.append(f"For duration questions, return only the final duration like '7 {unit}', with no explanation.")
    elif plan.reasoning_kind == "date":
        parts.append("For date questions, return only the final date or short date phrase, with no explanation.")
    else:
        parts.append("Return only the final answer.")
    if solver_result and solver_result.resolved:
        parts.append(
            "Deterministic memory solver suggestion: "
            f"{solver_result.answer} "
            f"(mode={solver_result.mode}, confidence={solver_result.confidence:.2f}, rationale={solver_result.rationale})."
        )
    if answerability["sufficient"]:
        parts.append("The evidence table is sufficient. Answer directly and do not reply with 'Insufficient evidence'.")
    else:
        parts.append("If the evidence table is insufficient, reply exactly: Insufficient evidence.")
    parts.append(build_evidence_table(plan, selected_hits, structured_events=structured_events))
    return "\n\n".join(parts)


def postprocess_prediction(plan, text):
    value = _collapse(text)
    if not value:
        return value
    lowered_value = value.lower()
    has_abstention_marker = any(
        marker in lowered_value
        for marker in (
            "insufficient evidence",
            "cannot determine",
            "not possible to determine",
            "do not know",
            "i do not know",
            "there is no information",
        )
    )
    if plan.reasoning_kind == "difference":
        unit = plan.unit_hint or "days"
        if unit == "months":
            match = _MONTH_DIFF_RE.search(value)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
        else:
            match = _DAY_DIFF_RE.search(value)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
        for line in value.splitlines():
            lower_line = line.lower()
            if "cannot determine" in lower_line or "not possible" in lower_line or "insufficient" in lower_line:
                continue
            match = (_MONTH_DIFF_RE if unit == "months" else _DAY_DIFF_RE).search(line)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
    if plan.reasoning_kind == "date":
        month_day = _MONTH_DAY_RE.search(value)
        if month_day:
            month = month_day.group(1).capitalize()
            day = month_day.group(2)
            return f"{month} {day}"
        day_month = _DAY_MONTH_RE.search(value)
        if day_month:
            month = day_month.group(2).capitalize()
            day = day_month.group(1)
            return f"{month} {day}"
        date_value = normalize_date(value)
        if date_value:
            return date_value
    if "what was the date" in plan.question.lower() or "which date" in plan.question.lower():
        month_day = _MONTH_DAY_RE.search(value)
        if month_day:
            month = month_day.group(1).capitalize()
            day = month_day.group(2)
            return f"{month} {day}"
        day_month = _DAY_MONTH_RE.search(value)
        if day_month:
            month = day_month.group(2).capitalize()
            day = day_month.group(1)
            return f"{month} {day}"
    if has_abstention_marker and plan.reasoning_kind in {"difference", "date"}:
        return "Insufficient evidence"
    if plan.normalized_targets:
        lowered = normalize_answer(value)
        for raw, normalized in zip(plan.targets, plan.normalized_targets):
            if normalized and _phrase_matches_text(normalized, lowered):
                return raw
    if has_abstention_marker:
        return "Insufficient evidence"
    return value.splitlines()[0].strip(" \"'")


def summarize_records(records):
    summary = {
        "examples": len(records),
        "exact_match": 0.0,
        "contains_match": 0.0,
        "token_f1": 0.0,
        "abstention_accuracy": None,
        "avg_selected_memory_count": 0.0,
        "avg_structured_event_count": 0.0,
        "avg_selected_session_recall": None,
        "avg_answerable": None,
        "avg_solver_resolved": None,
        "avg_solver_confidence": None,
        "by_question_type": {},
    }
    if not records:
        return summary

    summary["exact_match"] = sum(row["exact_match"] for row in records) / len(records)
    summary["contains_match"] = sum(row["contains_match"] for row in records) / len(records)
    summary["token_f1"] = sum(row["token_f1"] for row in records) / len(records)
    summary["avg_selected_memory_count"] = sum(
        row["selected_memory_count"] for row in records
    ) / len(records)
    summary["avg_structured_event_count"] = sum(
        row.get("structured_event_count", 0) for row in records
    ) / len(records)
    summary["avg_answerable"] = sum(1.0 if row["answerable"] else 0.0 for row in records) / len(records)
    summary["avg_solver_resolved"] = sum(1.0 if row.get("solver_resolved") else 0.0 for row in records) / len(records)
    summary["avg_solver_confidence"] = sum(row.get("solver_confidence", 0.0) for row in records) / len(records)

    recall_values = [
        row["selected_session_recall"]
        for row in records
        if row["selected_session_recall"] is not None
    ]
    if recall_values:
        summary["avg_selected_session_recall"] = sum(recall_values) / len(recall_values)

    abstention_values = [
        row["abstention_accuracy"]
        for row in records
        if row["abstention_accuracy"] is not None
    ]
    if abstention_values:
        summary["abstention_accuracy"] = sum(abstention_values) / len(abstention_values)

    grouped = {}
    for row in records:
        grouped.setdefault(row["question_type"], []).append(row)

    for question_type, items in sorted(grouped.items()):
        grouped_recall = [
            item["selected_session_recall"]
            for item in items
            if item["selected_session_recall"] is not None
        ]
        grouped_abstention = [
            item["abstention_accuracy"]
            for item in items
            if item["abstention_accuracy"] is not None
        ]
        summary["by_question_type"][question_type] = {
            "examples": len(items),
            "exact_match": sum(item["exact_match"] for item in items) / len(items),
            "contains_match": sum(item["contains_match"] for item in items) / len(items),
            "token_f1": sum(item["token_f1"] for item in items) / len(items),
            "avg_selected_memory_count": sum(
                item["selected_memory_count"] for item in items
            )
            / len(items),
            "avg_structured_event_count": sum(
                item.get("structured_event_count", 0) for item in items
            )
            / len(items),
            "avg_selected_session_recall": (
                sum(grouped_recall) / len(grouped_recall) if grouped_recall else None
            ),
            "avg_answerable": sum(1.0 if item["answerable"] else 0.0 for item in items) / len(items),
            "avg_solver_resolved": sum(1.0 if item.get("solver_resolved") else 0.0 for item in items) / len(items),
            "avg_solver_confidence": sum(item.get("solver_confidence", 0.0) for item in items) / len(items),
            "abstention_accuracy": (
                sum(grouped_abstention) / len(grouped_abstention)
                if grouped_abstention
                else None
            ),
        }

    return summary
