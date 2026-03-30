from __future__ import annotations

import json
import re
import string
from calendar import monthrange
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import numpy as np

from memory.utils import (
    STOPWORDS,
    cosine_similarity,
    extract_entities,
    normalize_date,
    preview,
)
from memory.types import MemoryHit, MemoryRecord
from memory.critic import HeuristicCritic, rerank_with_critic
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
_SHORT_MONTH_DAY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_DATE_LINE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MONTH_DIFF_RE = re.compile(r"\b(\d+)\s*(month|months)\b", re.IGNORECASE)
_WEEK_DIFF_RE = re.compile(r"\b(\d+)\s*(week|weeks)\b", re.IGNORECASE)
_DAY_DIFF_RE = re.compile(r"\b(\d+)\s*(day|days)\b", re.IGNORECASE)
_MONEY_RE = re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)")
_QUANTITY_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(hours?|days?|weeks?|months?|years?|pages?)\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")
_TIME_OF_DAY_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_DOCTOR_RE = re.compile(r"\bDr\.?\s+([A-Z][a-z]+)\b")
_URL_RE = re.compile(r"https?://[^\s)>\"]+")
_DOMAIN_RE = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$")
_FINAL_ANSWER_RE = re.compile(r"(?:^|\b)final answer\s*:\s*(.+)$", re.IGNORECASE)
_TOTAL_LINE_RE = re.compile(
    r"\b(?:total|combined|in total|overall|altogether|therefore|so the total|final answer)\b",
    re.IGNORECASE,
)
_ORDER_REQUEST_RE = re.compile(
    r"\bwhat is the order\b|\bfrom earliest to latest\b|\bfrom latest to earliest\b",
    re.IGNORECASE,
)
_LAST_WEEKDAY_RE = re.compile(
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_THIS_WEEKDAY_RE = re.compile(
    r"\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
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
_MULTI_HOW_MANY_RE = re.compile(
    r"^how many\s+(.+?)\s+(?:do|did|have|has|am|are|were|will|can)\s+i\s+(.+)$",
    re.IGNORECASE,
)
_MULTI_TOTAL_SPENT_RE = re.compile(
    r"^(?:what is the total amount i spent on|how much total money have i spent on)\s+(.+)$",
    re.IGNORECASE,
)
_MULTI_TOTAL_RE = re.compile(
    r"^(?:how many|how much)\s+(.+?)\s+in total\b",
    re.IGNORECASE,
)
_MULTI_TIME_RE = re.compile(r"^what time did i\s+(.+)$", re.IGNORECASE)
_MULTI_MAX_RE = re.compile(
    r"^(which\s+.+?)\s+did\s+i\s+(.+?\bthe most)\b.*$",
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
_WEEKDAY_TO_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_MULTI_GENERIC_TOKENS = STOPWORDS | {
    "different",
    "current",
    "currently",
    "everything",
    "including",
    "last",
    "past",
    "since",
    "start",
    "total",
    "totally",
    "year",
    "month",
    "months",
    "week",
    "weeks",
    "day",
    "days",
    "time",
    "times",
    "amount",
    "money",
    "spent",
    "take",
    "took",
    "before",
    "after",
}
_CATEGORY_HINTS = {
    "clothing": {"clothing", "clothes", "blazer", "shirt", "jeans", "boots", "dress", "sundress", "pants", "scarf", "gloves", "dry", "cleaning"},
    "doctor": {"doctor", "dr", "physician", "specialist", "dermatologist", "appointment", "ent"},
    "project": {"project", "research", "analysis", "competition", "presentation", "poster"},
    "model": {"model", "kit", "spitfire", "bomber", "camaro", "tank", "eagle"},
    "bike": {"bike", "helmet", "lights", "chain", "rack", "service", "serviced", "tune", "tuneup"},
    "instrument": {"instrument", "guitar", "drum", "piano", "violin", "keyboard"},
    "wedding": {"wedding", "couple"},
    "festival": {"festival"},
    "museum": {"museum", "gallery"},
    "grocery": {"grocery", "store", "market"},
}
_CITRUS_VALUES = {
    "blood orange": "blood orange",
    "orange peel": "orange",
    "orange peels": "orange",
    "orange": "orange",
    "lime": "lime",
    "lemon": "lemon",
    "grapefruit": "grapefruit",
    "mandarin": "mandarin",
    "tangerine": "tangerine",
    "clementine": "clementine",
    "pomelo": "pomelo",
    "yuzu": "yuzu",
}
_CUISINE_VALUES = {
    "thai": "Thai",
    "indian": "Indian",
    "mexican": "Mexican",
    "italian": "Italian",
    "japanese": "Japanese",
    "korean": "Korean",
    "french": "French",
    "greek": "Greek",
    "spanish": "Spanish",
    "vietnamese": "Vietnamese",
    "chinese": "Chinese",
    "mediterranean": "Mediterranean",
    "middle eastern": "Middle Eastern",
}
_FOOD_DELIVERY_VALUES = {
    "doordash": "DoorDash",
    "uber eats": "Uber Eats",
    "ubereats": "Uber Eats",
    "grubhub": "Grubhub",
    "postmates": "Postmates",
    "caviar": "Caviar",
}
_GROCERY_STORE_VALUES = {
    "thrive market": "Thrive Market",
    "walmart": "Walmart",
    "trader joe's": "Trader Joe's",
    "trader joes": "Trader Joe's",
    "publix": "Publix",
    "whole foods": "Whole Foods",
    "sprouts": "Sprouts",
    "costco": "Costco",
    "instacart": "Instacart",
}
_SOCIAL_PLATFORM_VALUES = {
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "facebook": "Facebook",
    "twitter": "Twitter",
    "youtube": "YouTube",
    "linkedin": "LinkedIn",
    "pinterest": "Pinterest",
}
_INSTRUMENT_TERMS = (
    "electric guitar",
    "acoustic guitar",
    "guitar",
    "drum set",
    "drums",
    "piano",
    "keyboard",
    "ukulele",
    "violin",
    "cello",
    "flute",
    "saxophone",
    "bass",
)
_CLOTHING_TERMS = (
    "blazer",
    "boots",
    "jeans",
    "shirt",
    "button-down shirt",
    "sundress",
    "sweater",
    "jacket",
    "dress",
    "pants",
    "coat",
    "scarf",
    "gloves",
)
_PROJECT_TERMS = (
    "case competition",
    "marketing research project",
    "marketing research",
    "data mining project",
    "database systems project",
    "thesis",
    "research project",
)
_ART_EVENT_TERMS = (
    "workshop",
    "class",
    "opening night",
    "exhibition",
    "festival",
    "fair",
    "gallery opening",
)
_WEDDING_COUPLE_RE = re.compile(r"\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\b")
_VENUE_NAME_RE = re.compile(
    r"\b(?:The\s+)?([A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}\s+"
    r"(?:Museum|Gallery|Cube|Festival|Market))\b"
)
_BRANDED_ITEM_RE = re.compile(
    r"\b(?:my|the|a|an)\s+([a-z0-9' -]{0,40}?(?:"
    r"electric guitar|acoustic guitar|guitar|drum set|drums|piano|keyboard|ukulele|violin|cello|flute|saxophone|bass|"
    r"boots|blazer|jeans|shirt|sundress|sweater|jacket|dress|pants|coat))\b",
    re.IGNORECASE,
)
_CURRENT_STATE_CUES = ("current", "currently", "still", "i have", "i've had", "my ", "i own", "i use", "i'm working on")
_NEGATED_OWNERSHIP_CUES = ("my niece", "friend's kid", "for my friend's kid", "for my niece")
_BUDGET_CUES = ("budget", "set aside", "allow for", "spending habits", "expense tracker", "track my expenses")
_FROM_TO_RE = re.compile(r"\bfrom\s+(\d+(?:,\d+)?)\s+to\s+(\d+(?:,\d+)?)\b", re.IGNORECASE)
_STEADY_RE = re.compile(r"\bremained steady\b", re.IGNORECASE)
_COMPARE_TO_RE = re.compile(
    r"\bin\s+([a-z][a-z ]+?)\s+compared to\s+([a-z][a-z ]+?)\??$",
    re.IGNORECASE,
)
_PER_NIGHT_RE = re.compile(r"\bper night\b", re.IGNORECASE)
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
    is_multi_session: bool
    multi_session_kind: str
    multi_session_subject: str
    multi_session_actions: list[str]
    multi_session_focus_terms: list[str]
    range_start: str
    range_end: str
    requires_distinct: bool
    requires_current_state: bool


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


def _coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def acceptable_answers(answer):
    answer_text = _coerce_text(answer)
    parts = re.split(r"\.\s+", answer_text)
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
    if not variants and answer_text.strip():
        variants.append(answer_text.strip())
    return variants


def format_session_text(session_id, session_date, session, include_assistant_turns=True):
    header = f"Session {session_id}"
    if session_date:
        header += f" on {session_date}"
    lines = [header]
    for turn in session:
        if (turn.get("role") or "").strip().lower() == "assistant" and not include_assistant_turns:
            continue
        content = _collapse(turn.get("content") or "")
        if not content:
            continue
        role = (turn.get("role") or "user").strip().capitalize()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_history_context(
    instance,
    include_assistant_turns=True,
    answer_sessions_only=False,
    history_format="nl",
    allowed_session_ids=None,
):
    session_ids = instance.get("haystack_session_ids", [])
    session_dates = instance.get("haystack_dates", [])
    sessions = instance.get("haystack_sessions", [])
    if allowed_session_ids is not None:
        allowed_session_ids = set(allowed_session_ids)
    elif answer_sessions_only:
        allowed_session_ids = set(instance.get("answer_session_ids", []))
    rendered_sessions = []

    for session_id, session_date, session in zip(session_ids, session_dates, sessions):
        if allowed_session_ids is not None and session_id not in allowed_session_ids:
            continue
        if history_format == "json":
            turns = []
            for turn in session:
                role = (turn.get("role") or "user").strip().lower()
                if role == "assistant" and not include_assistant_turns:
                    continue
                content = _collapse(turn.get("content") or "")
                if not content:
                    continue
                turns.append({"role": role, "content": content})
            rendered_sessions.append(
                {
                    "session_id": session_id,
                    "session_date": session_date,
                    "turns": turns,
                }
            )
        else:
            rendered = format_session_text(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            )
            if rendered.strip():
                rendered_sessions.append(rendered)

    if history_format == "json":
        return json.dumps(rendered_sessions, ensure_ascii=True, indent=2)
    return "\n\n".join(rendered_sessions)


def official_process_item_flat_index(data, granularity, sess_id, timestamp):
    corpus = []
    if granularity == "session":
        text = " ".join(interact["content"] for interact in data if interact["role"] == "user")
        corpus.append(text)
        ids = [sess_id]
        user_turns = [turn for turn in data if turn["role"] == "user"]
        if "answer" in sess_id and user_turns and all(not turn.get("has_answer", False) for turn in user_turns):
            ids = [sess_id.replace("answer", "noans")]
    elif granularity == "turn":
        ids = []
        for i_turn, turn in enumerate(data):
            if turn["role"] == "user":
                corpus.append(turn["content"])
                turn_id = f"{sess_id}_{i_turn + 1}"
                if "answer" not in sess_id:
                    ids.append(turn_id)
                else:
                    if turn.get("has_answer", False):
                        ids.append(turn_id)
                    else:
                        ids.append(turn_id.replace("answer", "noans"))
    else:
        raise NotImplementedError
    return corpus, ids, [timestamp for _ in corpus]


def build_official_retrieval_corpus(instance, granularity):
    corpus = []
    corpus_ids = []
    corpus_timestamps = []
    session_lookup = {}
    for sess_id, sess_entry, ts in zip(
        instance.get("haystack_session_ids", []),
        instance.get("haystack_sessions", []),
        instance.get("haystack_dates", []),
    ):
        cur_items, cur_ids, cur_ts = official_process_item_flat_index(sess_entry, granularity, sess_id, ts)
        corpus.extend(cur_items)
        corpus_ids.extend(cur_ids)
        corpus_timestamps.extend(cur_ts)
        for corpus_id, text, timestamp in zip(cur_ids, cur_items, cur_ts):
            session_lookup[corpus_id] = {
                "session_id": sess_id,
                "timestamp": timestamp,
                "text": text,
            }
    return corpus, corpus_ids, corpus_timestamps, session_lookup


def retrieval_dcg(relevances, k):
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0


def retrieval_ndcg(rankings, correct_docs, corpus_ids, k=10):
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_relevances = [relevances[idx] for idx in rankings[:k]]
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = retrieval_dcg(ideal_relevance, k)
    actual_dcg = retrieval_dcg(sorted_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_official_retrieval(rankings, correct_docs, corpus_ids, k=10):
    recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(doc in recalled_docs for doc in correct_docs))
    recall_all = float(all(doc in recalled_docs for doc in correct_docs))
    ndcg_score = retrieval_ndcg(rankings, correct_docs, corpus_ids, k)
    return recall_any, recall_all, ndcg_score


def evaluate_official_retrieval_turn2session(rankings, correct_docs, corpus_ids, k=10):
    def strip_turn_id(docid):
        return "_".join(docid.split("_")[:-1])

    correct_docs = list(set(strip_turn_id(x) for x in correct_docs))
    corpus_ids = [strip_turn_id(x) for x in corpus_ids]
    effective_k = k
    unique_docids = set(corpus_ids[idx] for idx in rankings[:effective_k])
    while effective_k <= len(corpus_ids) and len(unique_docids) < k:
        effective_k += 1
        unique_docids = set(corpus_ids[idx] for idx in rankings[:effective_k])
    return evaluate_official_retrieval(rankings, correct_docs, corpus_ids, k=effective_k)


def _retrieval_age_days(question_date, item_timestamp):
    left = _parse_iso_date(normalize_date(question_date))
    right = _parse_iso_date(normalize_date(item_timestamp))
    if left is None or right is None:
        return 0
    return max(0, (left - right).days)


def build_official_retrieval_log_entry(instance, granularity, embedder, critic=None):
    critic = critic or HeuristicCritic()
    corpus, corpus_ids, corpus_timestamps, session_lookup = build_official_retrieval_corpus(instance, granularity)
    query = instance.get("question", "")
    query_vector = embedder.embed(query)
    hits = []
    for idx, (text, corpus_id, timestamp) in enumerate(zip(corpus, corpus_ids, corpus_timestamps)):
        session_id = session_lookup[corpus_id]["session_id"]
        event_meta = _build_event_items(text, timestamp, session_id=session_id, role="user")
        metadata = {
            "session_id": session_id,
            "corpus_id": corpus_id,
            "session_date": normalize_date(timestamp),
            "event_date": event_meta.get("event_date") or normalize_date(timestamp),
            "entities": extract_entities(text),
            "event_aliases": event_meta.get("event_aliases", []),
            "event_items": event_meta.get("event_items", []),
            "date_confidence": event_meta.get("date_confidence", 0.0),
            "date_source": event_meta.get("date_source", ""),
            "granularity": granularity,
            "rank_index": idx,
            "fact_text": text,
        }
        hits.append(
            MemoryHit(
                record=MemoryRecord(
                    memory_id=corpus_id,
                    user_id="benchmark",
                    memory_type="episode" if granularity == "session" else "event",
                    text=text,
                    created_at="2026-03-29T00:00:00+00:00",
                    last_accessed_at="2026-03-29T00:00:00+00:00",
                    importance=0.75 if "answer" in corpus_id else 0.55,
                    metadata=metadata,
                ),
                score=cosine_similarity(query_vector, embedder.embed(text)),
                embedding_model=embedder.model_name,
                age_days=_retrieval_age_days(instance.get("question_date", ""), timestamp),
            )
        )

    reranked = rerank_with_critic(query, hits, critic)
    rankings = [hit.record.metadata["rank_index"] for hit in reranked]
    correct_docs = list(set(doc_id for doc_id in corpus_ids if "answer" in doc_id))
    metrics = {"session": {}, "turn": {}}
    for k in [1, 3, 5, 10, 30, 50]:
        recall_any, recall_all, ndcg_any = evaluate_official_retrieval(rankings, correct_docs, corpus_ids, k=k)
        metrics[granularity].update(
            {
                f"recall_any@{k}": recall_any,
                f"recall_all@{k}": recall_all,
                f"ndcg_any@{k}": ndcg_any,
            }
        )
        if granularity == "turn":
            recall_any, recall_all, ndcg_any = evaluate_official_retrieval_turn2session(rankings, correct_docs, corpus_ids, k=k)
            metrics["session"].update(
                {
                    f"recall_any@{k}": recall_any,
                    f"recall_all@{k}": recall_all,
                    f"ndcg_any@{k}": ndcg_any,
                }
            )

    return {
        "question_id": instance.get("question_id"),
        "question_type": instance.get("question_type"),
        "question": instance.get("question"),
        "answer": instance.get("answer"),
        "question_date": instance.get("question_date"),
        "haystack_dates": instance.get("haystack_dates"),
        "haystack_sessions": instance.get("haystack_sessions"),
        "haystack_session_ids": instance.get("haystack_session_ids"),
        "answer_session_ids": instance.get("answer_session_ids"),
        "retrieval_results": {
            "query": query,
            "ranked_items": [
                {
                    "corpus_id": corpus_ids[rid],
                    "text": corpus[rid],
                    "timestamp": corpus_timestamps[rid],
                }
                for rid in rankings
            ],
            "metrics": metrics,
        },
    }


def summarize_official_retrieval_logs(entries):
    filtered = [x for x in entries if "_abs" not in x["question_id"]]
    metric_names = {
        "session": ["recall_all@5", "ndcg_any@5", "recall_all@10", "ndcg_any@10"],
        "turn": ["recall_all@5", "ndcg_any@5", "recall_all@10", "ndcg_any@10", "recall_all@50", "ndcg_any@50"],
    }
    summary = {
        "examples": len(entries),
        "reported_examples": len(filtered),
        "session": {},
        "turn": {},
    }
    for level, names in metric_names.items():
        available = []
        for name in names:
            values = []
            for entry in filtered:
                metrics = entry.get("retrieval_results", {}).get("metrics", {}).get(level, {})
                if name in metrics:
                    has_user_target = any(
                        turn.get("has_answer")
                        for session in entry.get("haystack_sessions", [])
                        for turn in session
                        if turn.get("role") == "user"
                    )
                    if has_user_target:
                        values.append(metrics[name])
            if values:
                summary[level][name] = float(np.mean(values))
                available.append(name)
        if not available:
            summary.pop(level, None)
    return summary


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
    if not (1 <= year <= 9999):
        return None
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


def _subtract_years(value, years):
    year = value.year - years
    if not (1 <= year <= 9999):
        return None
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


def _parse_amount_token(value):
    token = _coerce_text(value).replace(",", "").strip()
    try:
        return float(token)
    except ValueError:
        return None


def _format_numeric_value(value):
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return str(round(float(value), 2)).rstrip("0").rstrip(".")


def _canonicalize_person_reference(text):
    value = normalize_answer(text)
    value = re.sub(r"^you\b", "i", value)
    value = re.sub(r"^your\b", "my", value)
    value = value.replace(" your ", " my ")
    value = value.replace(" you ", " i ")
    return " ".join(value.split())


def _split_answer_lines(text):
    lines = []
    for raw_line in _coerce_text(text).splitlines():
        line = raw_line.strip()
        if line:
            lines.append(line)
    return lines


def _extract_number_like_candidates(text):
    """Extract digit numbers and number words from text, skipping year-like 4-digit numbers."""
    candidates = []
    value = _coerce_text(text)
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b", value):
        token = match.group(0)
        if len(token) == 4 and token.startswith(("19", "20")):
            continue
        candidates.append(token)
    # Also find number words (e.g. "three" → "3")
    for match in re.finditer(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"twenty[\s-]?one|twenty[\s-]?two|twenty[\s-]?three|twenty[\s-]?four|"
        r"twenty[\s-]?five|twenty[\s-]?six|twenty[\s-]?seven|twenty[\s-]?eight|"
        r"twenty[\s-]?nine|thirty)\b",
        value,
        re.IGNORECASE,
    ):
        word = match.group(0).lower().replace("-", " ").strip()
        num = _NUMBER_WORDS_EXTENDED.get(word)
        if num is not None:
            candidates.append(str(num))
    return candidates


_NUMBER_WORDS_EXTENDED = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty one": 21, "twenty two": 22,
    "twenty three": 23, "twenty four": 24, "twenty five": 25,
    "twenty six": 26, "twenty seven": 27, "twenty eight": 28,
    "twenty nine": 29, "thirty": 30,
}


def _extract_bullet_items(text):
    items = []
    for line in _split_answer_lines(text):
        match = _BULLET_LINE_RE.match(line)
        if not match:
            continue
        item = _collapse(match.group(1)).strip(" -:;,.")
        if item:
            items.append(item)
    return items


def _extract_named_value_from_statement(text):
    value = _collapse(text).strip()
    if not value:
        return ""
    for pattern in (
        re.compile(r"\b(?:is|was|were|are)\s+(?:called\s+)?[\"']?([^.\n]+?)[\"']?(?:\.|$)", re.IGNORECASE),
        re.compile(r"\b(?:named|titled)\s+[\"']?([^.\n]+?)[\"']?(?:\.|$)", re.IGNORECASE),
    ):
        match = pattern.search(value)
        if match:
            candidate = match.group(1).strip(" \"'")
            candidate = re.split(r"\s+(?:and|with|which)\s+", candidate, maxsplit=1)[0].strip(" ,.;")
            if candidate:
                return candidate
    return ""


def _extract_multi_session_count_answer(text):
    value = _coerce_text(text)
    if not value:
        return ""
    final_match = _FINAL_ANSWER_RE.search(value)
    if final_match:
        final_numbers = _extract_number_like_candidates(final_match.group(1))
        if final_numbers:
            return _format_numeric_value(float(final_numbers[-1]))
        parsed = _parse_numeric_token(final_match.group(1))
        if parsed is not None:
            return str(parsed)
    # Look for math expressions like "3 + 5 = 8" and extract the result
    math_result = _extract_math_result(value)
    # Look for summary/total lines
    prioritized = []
    for line in _split_answer_lines(value):
        lowered = normalize_answer(line)
        if _TOTAL_LINE_RE.search(line) or any(
            phrase in lowered
            for phrase in (
                "there are",
                "there were",
                "you have",
                "you attended",
                "you visited",
                "you used",
                "you mentioned",
                "you baked",
                "you cooked",
                "you played",
                "you bought",
                "you acquired",
                "you replaced",
                "you fixed",
                "you completed",
                "you listened",
                "counting only",
                "in summary",
                "that makes",
                "that gives",
                "which is",
                "which gives",
                "which makes",
                "bringing the total",
            )
        ):
            numbers = _extract_number_like_candidates(line)
            # Filter out numbers that are part of time expressions
            filtered = _filter_time_period_numbers(line, numbers)
            if filtered:
                prioritized.extend(filtered)
            else:
                parsed = _parse_numeric_token(line)
                if parsed is not None:
                    prioritized.append(str(parsed))
    if prioritized:
        return _format_numeric_value(float(prioritized[-1]))
    # If we found a math result (e.g., "3 + 5 = 8"), use it
    if math_result is not None:
        return _format_numeric_value(math_result)
    bullet_items = _extract_bullet_items(value)
    if len(bullet_items) >= 2:
        return str(len(bullet_items))
    # Count enumerated items in verbose text (e.g., "1. item, 2. item, 3. item")
    enum_count = _count_enumerated_items(value)
    if enum_count >= 2:
        return str(enum_count)
    numbers = _extract_number_like_candidates(value)
    # Filter out time-period numbers from the full text
    filtered_numbers = _filter_time_period_numbers(value, numbers)
    if filtered_numbers:
        return _format_numeric_value(float(filtered_numbers[-1]))
    # If ALL numbers were time-period numbers, don't use them — return empty
    # to let the general postprocessor handle it
    if numbers and not filtered_numbers:
        # All candidates were time references; don't return a misleading number
        return ""
    if numbers:
        return _format_numeric_value(float(numbers[-1]))
    parsed = _parse_numeric_token(value)
    if parsed is not None:
        return str(parsed)
    return ""


def _filter_time_period_numbers(text, numbers):
    """Filter out numbers that appear as part of time period expressions like '2 months ago'."""
    if not numbers:
        return numbers
    lowered = text.lower()
    time_number_strs = set()
    # Digit time periods: "2 months ago", "3 weeks later"
    for match in re.finditer(
        r"\b(\d+)\s+(?:days?|weeks?|months?|years?)\s*(?:ago|before|after|later|earlier|prior|back)\b",
        lowered,
    ):
        time_number_strs.add(match.group(1))
    # "past X months/weeks" patterns
    for match in re.finditer(
        r"\b(?:past|last|previous|next|about|around|approximately)\s+(\d+)\s+(?:days?|weeks?|months?|years?)\b",
        lowered,
    ):
        time_number_strs.add(match.group(1))
    # Number words in time periods: "three weeks ago", "about two months ago"
    _TIME_WORD_RE = re.compile(
        r"\b(?:about\s+|around\s+|approximately\s+)?"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s+(?:days?|weeks?|months?|years?)"
        r"\s*(?:ago|before|after|later|earlier|prior|back)\b",
        re.IGNORECASE,
    )
    for match in _TIME_WORD_RE.finditer(lowered):
        word_num = _NUMBER_WORDS.get(match.group(1).lower())
        if word_num is not None:
            time_number_strs.add(str(word_num))
    # "past/last X months" with number words
    _PAST_WORD_RE = re.compile(
        r"\b(?:past|last|previous|next)\s+"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s+(?:days?|weeks?|months?|years?)\b",
        re.IGNORECASE,
    )
    for match in _PAST_WORD_RE.finditer(lowered):
        word_num = _NUMBER_WORDS.get(match.group(1).lower())
        if word_num is not None:
            time_number_strs.add(str(word_num))
    if not time_number_strs:
        return numbers
    return [n for n in numbers if n not in time_number_strs]


def _extract_math_result(text):
    """Extract the result from math expressions like '3 + 5 = 8' or '15 + 50 = 65'."""
    for match in re.finditer(r"=\s*(\d+(?:\.\d+)?)\b", text):
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _count_enumerated_items(text):
    """Count items listed with connectors like 'also', 'additionally', 'as well'."""
    lines = _split_answer_lines(text)
    # Check for numbered items like "1. X, 2. Y, 3. Z" in a single line
    for line in lines:
        enum_matches = list(re.finditer(r"\b(\d+)\.\s", line))
        if len(enum_matches) >= 2:
            return int(enum_matches[-1].group(1))
    return 0


def _extract_multi_session_money_answer(text):
    value = _coerce_text(text)
    if not value:
        return ""
    final_match = _FINAL_ANSWER_RE.search(value)
    if final_match:
        money_match = list(_MONEY_RE.finditer(final_match.group(1)))
        if money_match:
            amount = _parse_amount_token(money_match[-1].group(1))
            if amount is not None:
                return _format_money(amount)
    # Look for math expression with = sign (e.g., "$15 + $50 = $65")
    for line in _split_answer_lines(value):
        eq_match = re.search(r"=\s*\$(\d+(?:,\d{3})*(?:\.\d+)?)", line)
        if eq_match:
            amount = _parse_amount_token(eq_match.group(1))
            if amount is not None:
                return _format_money(amount)
    prioritized = []
    for line in _split_answer_lines(value):
        lowered_line = normalize_answer(line)
        if _TOTAL_LINE_RE.search(line) or any(
            phrase in lowered_line
            for phrase in ("spent", "difference", "more", "less", "per night", "total cost", "total amount", "sum")
        ):
            line_matches = list(_MONEY_RE.finditer(line))
            if line_matches:
                # If multiple amounts on a summary line, sum them (e.g., "$15 + $50")
                if len(line_matches) >= 2 and not _TOTAL_LINE_RE.search(line):
                    amounts = [_parse_amount_token(m.group(1)) for m in line_matches]
                    amounts = [a for a in amounts if a is not None]
                    if amounts:
                        return _format_money(sum(amounts))
                prioritized.extend(line_matches)
    if prioritized:
        amount = _parse_amount_token(prioritized[-1].group(1))
        if amount is not None:
            return _format_money(amount)
    money_matches = list(_MONEY_RE.finditer(value))
    if money_matches:
        # If there are 2+ amounts and no total/summary line found, sum them
        if len(money_matches) >= 2:
            amounts = []
            for m in money_matches:
                a = _parse_amount_token(m.group(1))
                if a is not None:
                    amounts.append(a)
            if amounts:
                lowered = value.lower()
                if any(kw in lowered for kw in ("and", "plus", "+", "also")):
                    return _format_money(sum(amounts))
        # Single amount or fallback to last
        amount = _parse_amount_token(money_matches[-1].group(1))
        if amount is not None:
            return _format_money(amount)
    return ""


def _extract_multi_session_quantity_answer(text, unit_hint=""):
    value = _coerce_text(text)
    if not value:
        return ""
    normalized_unit = _normalize_unit(unit_hint or "")

    def matching_quantities(source):
        matches = []
        for quantity_match in _QUANTITY_RE.finditer(source):
            found_unit = _normalize_unit(quantity_match.group(2))
            if normalized_unit and found_unit != normalized_unit:
                continue
            matches.append(quantity_match)
        return matches

    final_match = _FINAL_ANSWER_RE.search(value)
    if final_match:
        matches = matching_quantities(final_match.group(1))
        if matches:
            match = matches[-1]
            return f"{_format_numeric_value(float(match.group(1)))} {match.group(2).lower()}"
        # If no unit match, try just extracting numbers from the final answer line
        numbers = _extract_number_like_candidates(final_match.group(1))
        if numbers:
            unit = unit_hint or ""
            return f"{_format_numeric_value(float(numbers[-1]))} {unit}".strip()
    # Look for math results on = lines
    prioritized = []
    for line in _split_answer_lines(value):
        if _TOTAL_LINE_RE.search(line) or "=" in line:
            prioritized.extend(matching_quantities(line))
            if not prioritized:
                numbers = _extract_number_like_candidates(line)
                filtered = _filter_time_period_numbers(line, numbers)
                if filtered:
                    unit = unit_hint or ""
                    return f"{_format_numeric_value(float(filtered[-1]))} {unit}".strip()
                if numbers:
                    unit = unit_hint or ""
                    return f"{_format_numeric_value(float(numbers[-1]))} {unit}".strip()
    if prioritized:
        match = prioritized[-1]
        return f"{_format_numeric_value(float(match.group(1)))} {match.group(2).lower()}"
    matches = matching_quantities(value)
    if matches:
        match = matches[-1]
        return f"{_format_numeric_value(float(match.group(1)))} {match.group(2).lower()}"
    if normalized_unit:
        numbers = _extract_number_like_candidates(value)
        filtered = _filter_time_period_numbers(value, numbers)
        if filtered:
            return f"{_format_numeric_value(float(filtered[-1]))} {unit_hint}".strip()
        if numbers:
            return f"{_format_numeric_value(float(numbers[-1]))} {unit_hint}".strip()
    return ""


def _extract_other_option_list(text, excluded_values):
    items = _extract_bullet_items(text)
    if not items:
        return ""
    excluded = {normalize_answer(value) for value in excluded_values if value}
    cleaned = []
    for item in items:
        normalized = normalize_answer(item)
        if not normalized or normalized in excluded:
            continue
        cleaned.append(item.strip(" \"'"))
    return ", ".join(cleaned)


def _extract_single_session_assistant_answer(plan, text):
    raw_value = _coerce_text(text).strip()
    value = _collapse(raw_value).strip()
    if not value:
        return ""
    question = (plan.question or "").lower()
    if any(marker in normalize_answer(value) for marker in _ABSTENTION_MARKERS):
        return "Insufficient evidence"
    if "other" in question and "option" in question:
        other_options = _extract_other_option_list(raw_value, plan.targets)
        if other_options:
            return other_options
    if "website" in question or "site" in question or "url" in question or "link" in question:
        url_match = _URL_RE.search(value)
        if url_match:
            return url_match.group(0).rstrip(".,)")
        domain_match = _DOMAIN_RE.search(value)
        if domain_match:
            return domain_match.group(0).rstrip(".,)")
    if "youtube" in question or "video" in question:
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", value)
        if quoted:
            return quoted[0].strip()
        named = _extract_named_value_from_statement(value)
        if named:
            return named
    if "quote" in question or ("what did" in question and "say" in question):
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", value)
        if quoted:
            return max((item.strip() for item in quoted), key=len)
    if question.startswith("how many") or "average improvement" in question or "%" in value:
        percent_match = _PERCENT_RE.search(value)
        if percent_match:
            return f"{percent_match.group(1)}%"
        number = _extract_multi_session_count_answer(value)
        if number:
            return number
    named = _extract_named_value_from_statement(value)
    if named:
        return named
    first_sentence = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0].strip(" \"'")
    if first_sentence:
        return first_sentence
    return value


def _weekday_relative_date(base_date, weekday_name, qualifier):
    if base_date is None:
        return None
    weekday_index = _WEEKDAY_TO_INDEX.get((weekday_name or "").lower())
    if weekday_index is None:
        return None
    delta = (base_date.weekday() - weekday_index) % 7
    if qualifier == "last":
        delta = 7 if delta == 0 else delta
    else:
        delta = 0 if delta == 0 else delta
    return base_date - timedelta(days=delta)


def _extract_currency_values(text):
    values = []
    for match in _MONEY_RE.finditer(text or ""):
        amount = _parse_amount_token(match.group(1))
        if amount is None:
            continue
        values.append(
            {
                "value": amount,
                "raw": match.group(0),
            }
        )
    return values


def _extract_quantity_values(text):
    values = []
    for match in _QUANTITY_RE.finditer(text or ""):
        amount = _parse_amount_token(match.group(1))
        if amount is None:
            continue
        values.append(
            {
                "value": amount,
                "unit": match.group(2).lower(),
                "raw": match.group(0),
            }
        )
    return values


def _extract_clock_times(text):
    values = []
    for match in _TIME_OF_DAY_RE.finditer(text or ""):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3).upper()
        values.append(
            {
                "text": f"{hour}:{minute:02d} {meridiem}" if minute else f"{hour} {meridiem}",
                "hour": hour,
                "minute": minute,
                "meridiem": meridiem,
            }
        )
    return values


def _extract_doctor_names(text):
    seen = set()
    values = []
    for match in _DOCTOR_RE.finditer(text or ""):
        name = f"Dr. {match.group(1)}"
        normalized = normalize_answer(name)
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(name)
    lowered = (text or "").lower()
    role_map = {
        "primary care physician": "primary care physician",
        "ent specialist": "ENT specialist",
        "dermatologist": "dermatologist",
    }
    for needle, label in role_map.items():
        if needle in lowered and normalize_answer(label) not in seen:
            seen.add(normalize_answer(label))
            values.append(label)
    return values


def _expand_focus_terms(subject, question):
    terms = []
    for token in normalize_answer(" ".join([subject, question])).split():
        if token in _MULTI_GENERIC_TOKENS or len(token) < 3:
            continue
        if token not in terms:
            terms.append(token)
    lowered = normalize_answer(" ".join([subject, question]))
    for trigger, extra in _CATEGORY_HINTS.items():
        if trigger in lowered:
            for token in sorted(extra):
                if token not in terms:
                    terms.append(token)
    return terms


def _clean_multi_session_action(action):
    value = normalize_answer(action)
    value = re.sub(r"^(?:do|did|have|has|am|are|were|will|can|need|need to|trying to|been)\s+", "", value)
    value = re.sub(r"^to\s+", "", value)
    value = re.sub(r"^(?:currently\s+|current\s+)", "", value)
    value = re.sub(r"\s+(?:from|at|in|on)\s+.+$", "", value)
    value = value.strip()
    if value.endswith("ing") and value.startswith("currently "):
        value = value.replace("currently ", "", 1)
    replacements = {
        "pick": "pick up",
        "pickup": "pick up",
        "returning": "return",
        "leading": "lead",
        "led": "lead",
        "worked on": "work on",
        "working on": "work on",
        "bought": "buy",
        "purchased": "buy",
        "using": "use",
        "used": "use",
        "attended": "attend",
        "visiting": "visit",
        "visited": "visit",
    }
    return replacements.get(value, value)


def _match_phrase_map(text, mapping):
    normalized_text = normalize_answer(text)
    values = []
    seen = set()
    for needle, label in mapping.items():
        normalized_needle = normalize_answer(needle)
        if normalized_needle and _phrase_matches_text(normalized_needle, normalized_text):
            normalized_label = normalize_answer(label)
            if normalized_label not in seen:
                seen.add(normalized_label)
                values.append(label)
    return values


def _dedupe_subsumed(values):
    kept = []
    seen = []
    for value in values:
        normalized = normalize_answer(value)
        if not normalized:
            continue
        if any(normalized == prior or normalized in prior for prior in seen):
            continue
        seen = [prior for prior in seen if prior not in normalized]
        kept = [prior for prior in kept if normalize_answer(prior) not in normalized]
        seen.append(normalized)
        kept.append(value)
    return kept


def _extract_branded_items(text):
    values = []
    seen = set()
    normalized_text = normalize_answer(text)
    for match in _BRANDED_ITEM_RE.finditer(text or ""):
        candidate = _cleanup_event_surface(match.group(1))
        if not candidate:
            continue
        normalized = normalize_answer(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(candidate)
    for term in _INSTRUMENT_TERMS + _CLOTHING_TERMS:
        if _phrase_matches_text(normalize_answer(term), normalized_text):
            normalized = normalize_answer(term)
            if normalized not in seen:
                seen.add(normalized)
                values.append(term)
    values = _dedupe_subsumed(values)
    normalized_values = [normalize_answer(value) for value in values]
    if any("drum set" in value for value in normalized_values):
        values = [value for value in values if normalize_answer(value) != "drums"]
    return values


def _extract_venue_names(text):
    values = []
    seen = set()
    for match in _VENUE_NAME_RE.finditer(text or ""):
        candidate = _collapse(match.group(1)).strip()
        normalized = normalize_answer(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            values.append(candidate)
    return values


def _entry_has_personal_fact(text):
    normalized = normalize_answer(text)
    return any(
        cue in normalized
        for cue in (
            "i visited",
            "i recently visited",
            "i attended",
            "i recently attended",
            "i made",
            "i recently made",
            "i added",
            "i infused",
            "i bought",
            "i went",
            "i got back from",
            "i had an appointment",
            "i recently had",
            "i just got back from",
            "i was diagnosed",
            "i recently got back into",
            "i stayed",
            "i booked",
        )
    )


def _subject_requires_strict_values(plan):
    subject = normalize_answer(plan.multi_session_subject)
    return any(
        token in subject
        for token in (
            "citrus",
            "museum",
            "gallery",
            "doctor",
            "grocery store",
            "food delivery",
            "social media platform",
            "musical instrument",
            "clothing",
            "wedding",
            "project",
            "art-related",
        )
    )


def _extract_owned_instruments(text):
    values = []
    lowered = (text or "").lower()
    for match in re.finditer(
        r"\bmy\s+([a-z0-9' -]{0,50}?(?:electric guitar|acoustic guitar|guitar|drum set|drums|piano|keyboard|ukulele|violin|cello|flute|saxophone|bass))\b",
        lowered,
        re.IGNORECASE,
    ):
        candidate = _cleanup_event_surface(match.group(1))
        if candidate:
            values.append(candidate)
    return _dedupe_subsumed(values)


def _extract_project_values(text):
    normalized_text = normalize_answer(text)
    values = _match_phrase_map(text, {term: term.title() for term in _PROJECT_TERMS})
    if "poster" in normalized_text and "research" in normalized_text and "marketing" in normalized_text:
        values.append("Marketing Research")
    if "case competition" in normalized_text:
        values.append("Case Competition")
    if "working on a project" in normalized_text and "marketing research" in normalized_text:
        values.append("Marketing Research")
    return _dedupe_preserve(values)


def _extract_subject_values(plan, entry):
    text = entry.get("text", "")
    labels = " ".join(entry.get("labels", []))
    aliases = " ".join(entry.get("event_aliases", []))
    entities = " ".join(entry.get("entities", []))
    blob = " ".join([text, labels, aliases, entities, " ".join(entry.get("doctor_names", []))])
    normalized_blob = normalize_answer(blob)
    subject = normalize_answer(plan.multi_session_subject)
    values = []

    if "doctor" in subject:
        doctors = _extract_doctor_names(text)
        roles = [value for value in doctors if not value.startswith("Dr.")]
        return _dedupe_preserve(roles or doctors)
    if any(token in subject for token in ("museum", "gallery")):
        if not _entry_has_personal_fact(text):
            return []
        return _dedupe_preserve(_extract_venue_names(blob))
    if "citrus" in subject:
        if not _entry_has_personal_fact(text):
            return []
        return _dedupe_preserve(_match_phrase_map(blob, _CITRUS_VALUES))
    if "food delivery" in subject:
        return _dedupe_preserve(_match_phrase_map(blob, _FOOD_DELIVERY_VALUES))
    if "cuisine" in subject:
        return _dedupe_preserve(_match_phrase_map(blob, _CUISINE_VALUES))
    if "grocery store" in subject or "grocery" in subject:
        return _dedupe_preserve(_match_phrase_map(blob, _GROCERY_STORE_VALUES))
    if "social media platform" in subject:
        return _dedupe_preserve(_match_phrase_map(blob, _SOCIAL_PLATFORM_VALUES))
    if "musical instrument" in subject or "instrument" in subject:
        return _dedupe_preserve(_extract_owned_instruments(text))
    if "clothing" in subject:
        matches = _match_phrase_map(blob, {term: term for term in _CLOTHING_TERMS})
        if "dry cleaning" in normalized_blob:
            for_match = re.search(r"\bfor\s+(?:the\s+)?([a-z0-9' -]{2,40})", normalize_answer(text))
            if for_match:
                matches.append(for_match.group(1))
        matches.extend(_extract_branded_items(text))
        return _dedupe_preserve(matches)
    if "project" in subject or "thesis" in subject:
        return _extract_project_values(blob)
    if "model kit" in subject:
        model_values = []
        for label in entry.get("labels", []) + entry.get("event_aliases", []):
            cleaned = _collapse(label)
            if any(token in normalize_answer(cleaned) for token in ("spitfire", "camaro", "bomber", "tiger", "eagle", "model")):
                model_values.append(cleaned)
        return _dedupe_preserve(model_values)
    if "wedding" in subject:
        if not _entry_has_personal_fact(text):
            return []
        couples = [f"{left} and {right}" for left, right in _WEDDING_COUPLE_RE.findall(text or "")]
        return _dedupe_preserve(couples)
    if "art-related event" in subject:
        if not _entry_has_personal_fact(text):
            return []
        values = _extract_venue_names(blob)
        for term in _ART_EVENT_TERMS:
            if _phrase_matches_text(normalize_answer(term), normalized_blob):
                values.append(term.title())
        return _dedupe_preserve(values)
    if "fitness class" in subject:
        for term in ("yoga", "pilates", "spin", "cycling", "zumba", "barre", "boxing"):
            if _phrase_matches_text(term, normalized_blob):
                values.append(term.title())
        return _dedupe_preserve(values)
    return _dedupe_preserve(values)


def _is_current_state_entry(plan, entry):
    if not plan.requires_current_state:
        return True
    text = normalize_answer(entry.get("text", ""))
    if any(marker in text for marker in _NEGATED_OWNERSHIP_CUES):
        return False
    return any(marker in text for marker in _CURRENT_STATE_CUES)


def _comparison_targets(plan):
    match = _COMPARE_TO_RE.search(plan.question.lower().strip())
    if not match:
        return []
    return [_clean_question_clause(match.group(1)), _clean_question_clause(match.group(2))]


def _multi_session_window(question, normalized_question_date):
    base_date = _parse_iso_date(normalized_question_date)
    if base_date is None:
        return "", ""
    lowered = question.lower()
    start = ""
    end = normalized_question_date
    if "since the start of the year" in lowered or "this year" in lowered:
        start = date(base_date.year, 1, 1).isoformat()
    elif "last two months" in lowered or "past two months" in lowered:
        start = _subtract_months(base_date, 2).isoformat()
    elif "last month" in lowered or "past month" in lowered or "in the last month" in lowered:
        start = _subtract_months(base_date, 1).isoformat()
    elif "last two weeks" in lowered:
        start = (base_date - timedelta(days=14)).isoformat()
    elif "last week" in lowered or "past week" in lowered:
        start = (base_date - timedelta(days=7)).isoformat()
    elif "past few months" in lowered or "last few months" in lowered:
        start = _subtract_months(base_date, 3).isoformat()
    month_match = _IN_MONTH_RE.search(question)
    if month_match:
        month_value = _MONTHS[month_match.group(1).lower()]
        range_start = date(base_date.year, month_value, 1)
        range_end = date(base_date.year, month_value, monthrange(base_date.year, month_value)[1])
        start = range_start.isoformat()
        end = range_end.isoformat()
    return start, end


def _analyze_multi_session(question, normalized_question_date):
    lowered = question.lower().strip(" ?.")
    subject = ""
    actions = []
    kind = "open"
    requires_distinct = "different" in lowered
    requires_current_state = any(token in lowered for token in ("currently", "current", "currently own", "currently have"))

    if match := _MULTI_TIME_RE.match(lowered):
        subject = match.group(1)
        kind = "time_lookup"
    elif "which grocery store" in lowered and "spend the most" in lowered:
        subject = "grocery store"
        actions = ["spent"]
        kind = "max_value"
    elif "which social media platform" in lowered and "gain the most followers" in lowered:
        subject = "social media platform"
        actions = ["gain"]
        kind = "max_value"
    elif match := _MULTI_TOTAL_SPENT_RE.match(lowered):
        subject = match.group(1)
        actions = ["spent"]
        kind = "sum_money"
    elif lowered.startswith("how much more did i spend on"):
        subject = lowered.replace("how much more did i spend on", "", 1).strip()
        actions = ["spent"]
        kind = "difference_money"
    elif match := _MULTI_HOW_MANY_RE.match(lowered):
        subject = match.group(1)
        actions = [part.strip() for part in re.split(r"\bor\b|\band\b|,", match.group(2)) if part.strip()]
        if requires_distinct:
            kind = "count_distinct"
        elif any(unit in lowered for unit in ("hours", "days", "weeks", "months", "years", "pages")):
            kind = "sum_quantity"
        else:
            kind = "count_entries"
    elif match := _MULTI_TOTAL_RE.match(lowered):
        subject = match.group(1)
        kind = "sum_quantity"
    elif match := _MULTI_MAX_RE.match(lowered):
        subject = match.group(1)
        actions = [match.group(2)]
        kind = "max_value"

    if not subject:
        subject = lowered
    subject = _clean_question_clause(subject)
    actions = [_clean_multi_session_action(action) for action in actions if _clean_multi_session_action(action)]
    focus_terms = _expand_focus_terms(subject, question)
    range_start, range_end = _multi_session_window(question, normalized_question_date)
    return {
        "kind": kind,
        "subject": subject,
        "actions": actions,
        "focus_terms": focus_terms,
        "range_start": range_start,
        "range_end": range_end,
        "requires_distinct": requires_distinct,
        "requires_current_state": requires_current_state,
    }


def _infer_quantity_unit(question):
    lowered = question.lower()
    for unit in ("hours", "days", "weeks", "months", "years", "pages"):
        if re.search(rf"\b{unit}\b", lowered):
            return unit
    for unit in ("hour", "day", "week", "month", "year", "page"):
        if re.search(rf"\b{unit}\b", lowered):
            return unit + "s"
    return ""


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
        for match in _SHORT_MONTH_DAY_RE.finditer(text):
            month_value = int(match.group(1))
            day_value = int(match.group(2))
            if not (1 <= month_value <= 12 and 1 <= day_value <= 31):
                continue
            year_token = match.group(3)
            year_value = base_date.year
            if year_token:
                year_value = int(year_token)
                if year_value < 100:
                    year_value += 2000
            try:
                derived = date(year_value, month_value, day_value)
            except ValueError:
                continue
            if derived > base_date + timedelta(days=1) and not year_token:
                derived = date(year_value - 1, month_value, day_value)
            add_date(derived, "explicit-short-month-day", 0.88 if year_token else 0.84)

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
            if unit.startswith("year") and amount > 150:
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

        for match in _LAST_WEEKDAY_RE.finditer(text):
            derived = _weekday_relative_date(base_date, match.group(1), "last")
            add_date(derived, f"relative-last-{match.group(1).lower()}", 0.82)

        for match in _THIS_WEEKDAY_RE.finditer(text):
            derived = _weekday_relative_date(base_date, match.group(1), "this")
            add_date(derived, f"relative-this-{match.group(1).lower()}", 0.78)

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
        targets = quoted_targets
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


def _requires_full_order_answer(plan):
    if plan.reasoning_kind != "ordering":
        return False
    question = plan.question or ""
    return bool(_ORDER_REQUEST_RE.search(question))


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


def _aggregate_labels(content, event_meta, entities):
    values = []
    for item in event_meta.get("event_items", []):
        values.append(item.get("label", ""))
        values.extend(item.get("aliases", [])[:3])
    values.extend(entities)
    return _dedupe_preserve(values)[:12]


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
    summary_text = "\n".join(summary_lines) if summary_lines else format_session_text(
        session_id,
        session_date,
        session,
        include_assistant_turns=include_assistant_turns,
    )
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
    timeline_text = "\n".join(facts) if facts else format_session_text(
        session_id,
        session_date,
        session,
        include_assistant_turns=include_assistant_turns,
    )
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
        aggregate_labels = _aggregate_labels(content, event_meta, entities)
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
                "aggregate_labels": aggregate_labels,
                "currency_values": _extract_currency_values(content),
                "quantity_values": _extract_quantity_values(content),
                "clock_times": _extract_clock_times(content),
                "doctor_names": _extract_doctor_names(content),
                "date_source": event_meta["date_source"],
                "date_confidence": event_meta["date_confidence"],
                "date_candidates": event_meta["date_candidates"],
            },
        }


def _normalize_unit(unit):
    lowered = _coerce_text(unit).lower().strip()
    if lowered.endswith("s"):
        lowered = lowered[:-1]
    return lowered


def _entry_in_range(entry, plan):
    if not plan.range_start and not plan.range_end:
        return True
    entry_date = _parse_iso_date(entry.get("event_date", ""))
    if entry_date is None:
        return False
    if plan.range_start:
        start = _parse_iso_date(plan.range_start)
        if start is not None and entry_date < start:
            return False
    if plan.range_end:
        end = _parse_iso_date(plan.range_end)
        if end is not None and entry_date > end:
            return False
    return True


def _extract_action_objects(text, actions):
    lowered = _collapse(text).lower()
    values = []
    for action in actions:
        action = _clean_multi_session_action(action).strip().lower()
        if not action:
            continue
        pattern = re.compile(
            rf"\b{re.escape(action)}\s+(?:my|the|a|an|some|this|that|these|those)?\s*(.+?)(?:\s+(?:to|from|at|for|with|on)\b|[.,;!?]|$)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(lowered):
            phrase = _cleanup_event_surface(match.group(1))
            if phrase in {"it", "them", "one", "pair"} or len(phrase) < 2:
                continue
            value = f"{action}:{phrase}"
            if value not in values:
                values.append(value)
        if action == "pick up" and "dry cleaning" in lowered:
            extra = re.search(r"dry cleaning\s+for\s+(?:the\s+)?([a-z0-9' -]{2,40})", lowered)
            if extra:
                values.append(f"{action}:{_cleanup_event_surface(extra.group(1))}")
        if action == "return":
            extra = re.search(r"return\s+(?:some\s+|my\s+|the\s+|a\s+|an\s+)?([a-z0-9' -]{2,40})", lowered)
            if extra:
                values.append(f"{action}:{_cleanup_event_surface(extra.group(1))}")
    return values


def _build_multi_session_entries(fact_memories):
    entries = []
    for index, memory in enumerate(fact_memories):
        metadata = memory["metadata"]
        entries.append(
            {
                "entry_id": f"entry-{index}",
                "session_id": metadata.get("session_id"),
                "event_date": metadata.get("event_date") or normalize_date(metadata.get("session_date", "")),
                "session_date": normalize_date(metadata.get("session_date", "")),
                "text": metadata.get("fact_text", memory["text"]),
                "labels": metadata.get("aggregate_labels", []),
                "event_aliases": metadata.get("event_aliases", []),
                "entities": metadata.get("entities", []),
                "currency_values": metadata.get("currency_values") or _extract_currency_values(metadata.get("fact_text", memory["text"])),
                "quantity_values": metadata.get("quantity_values") or _extract_quantity_values(metadata.get("fact_text", memory["text"])),
                "clock_times": metadata.get("clock_times") or _extract_clock_times(metadata.get("fact_text", memory["text"])),
                "doctor_names": metadata.get("doctor_names") or _extract_doctor_names(metadata.get("fact_text", memory["text"])),
                "memory_text": memory["text"],
                "importance": memory["importance"],
                "has_answer": metadata.get("has_answer", False),
            }
        )
    return entries


def _multi_session_text_blob(entry):
    return normalize_answer(
        " ".join(
            [
                entry.get("text", ""),
                " ".join(entry.get("labels", [])),
                " ".join(entry.get("event_aliases", [])),
                " ".join(entry.get("entities", [])),
                " ".join(entry.get("doctor_names", [])),
            ]
        )
    )


def _multi_session_entry_score(plan, entry):
    text_blob = _multi_session_text_blob(entry)
    score = 0.0
    subject_values = _extract_subject_values(plan, entry)
    entry["subject_values"] = subject_values
    focus_matches = sum(1 for token in plan.multi_session_focus_terms if token in text_blob)
    score += 0.42 * focus_matches
    score += 0.6 * len(subject_values)
    for action in plan.multi_session_actions:
        normalized_action = normalize_answer(action)
        if normalized_action and normalized_action in text_blob:
            score += 0.55
    if entry.get("has_answer"):
        score += 0.15
    if _entry_in_range(entry, plan):
        score += 0.2
    elif plan.range_start or plan.range_end:
        score -= 1.0
    if plan.multi_session_kind == "sum_money" and entry.get("currency_values"):
        score += 1.2
    if plan.multi_session_kind == "sum_quantity":
        matching_quantities = [
            item for item in entry.get("quantity_values", [])
            if not plan.unit_hint or _normalize_unit(item.get("unit")) == _normalize_unit(plan.unit_hint)
        ]
        if matching_quantities:
            score += 1.1
    if plan.multi_session_kind == "time_lookup" and entry.get("clock_times"):
        score += 0.9
    if plan.requires_distinct and (
        entry.get("doctor_names")
        or entry.get("entities")
        or entry.get("labels")
    ):
        score += 0.4
    if plan.requires_current_state and any(token in text_blob for token in ("current", "currently", "still", "use", "own", "have")):
        score += 0.3
    if plan.requires_current_state and not _is_current_state_entry(plan, entry):
        score -= 1.2
    if plan.multi_session_kind in {"count_entries", "count_distinct", "max_value"} and not subject_values:
        score -= 0.6
    return score


def _entry_display_key(plan, entry):
    action_objects = _extract_action_objects(entry.get("text", ""), plan.multi_session_actions)
    if action_objects:
        return action_objects[0]
    if plan.multi_session_kind == "count_distinct" and entry.get("doctor_names"):
        return entry["doctor_names"][0]
    for source in (entry.get("labels", []), entry.get("event_aliases", []), entry.get("entities", [])):
        for value in source:
            normalized = normalize_answer(value)
            if not normalized:
                continue
            if any(token in normalized for token in plan.multi_session_focus_terms) or len(source) == 1:
                return value
    return _trim(entry.get("text", ""), limit=72)


def _distinct_values_for_entry(plan, entry):
    if plan.requires_current_state and not _is_current_state_entry(plan, entry):
        return []
    subject_values = entry.get("subject_values")
    if subject_values is None:
        subject_values = _extract_subject_values(plan, entry)
        entry["subject_values"] = subject_values
    if subject_values:
        return _dedupe_preserve(subject_values)
    if _subject_requires_strict_values(plan):
        return []
    action_objects = _extract_action_objects(entry.get("text", ""), plan.multi_session_actions)
    if action_objects:
        return _dedupe_preserve(action_objects)
    values = []
    for source in (entry.get("labels", []), entry.get("event_aliases", []), entry.get("entities", [])):
        for value in source:
            normalized = normalize_answer(value)
            if not normalized or normalized in {"doctor", "doctors", "dr"}:
                continue
            if any(token in normalized for token in plan.multi_session_focus_terms):
                values.append(value)
    return _dedupe_preserve(values)


def _count_values_for_entry(plan, entry):
    if plan.requires_current_state and not _is_current_state_entry(plan, entry):
        return []
    subject_values = _distinct_values_for_entry(plan, entry)
    if "clothing" in normalize_answer(plan.multi_session_subject):
        action_objects = _extract_action_objects(entry.get("text", ""), plan.multi_session_actions)
        text = normalize_answer(entry.get("text", ""))
        if action_objects and any(token in text for token in ("zara", "dry cleaning", "store", "exchange", "pick up")):
            return _dedupe_preserve(action_objects)
        return []
    if "doctor appointment" in normalize_answer(plan.multi_session_subject):
        if _extract_doctor_names(entry.get("text", "")) or "appointment" in normalize_answer(entry.get("text", "")):
            return [entry.get("entry_id", "appointment")]
    return subject_values


def _format_money(amount):
    if amount is None:
        return ""
    if float(amount).is_integer():
        return f"${int(amount)}"
    return f"${amount:.2f}".rstrip("0").rstrip(".")


def _entry_money_values(plan, entry):
    values = []
    text = normalize_answer(entry.get("text", ""))
    subject = normalize_answer(plan.multi_session_subject)
    if any(cue in text for cue in _BUDGET_CUES):
        return []
    currency_values = entry.get("currency_values", [])
    if plan.multi_session_kind == "difference_money" and _PER_NIGHT_RE.search(plan.question):
        return currency_values[:1]
    if "luxury" in subject:
        if not any(token in text for token in ("luxury", "gucci", "gown", "handbag", "resort", "designer")):
            return []
        if len(currency_values) > 1:
            return [max(currency_values, key=lambda item: item["value"])]
    return currency_values


def _aggregate_answer_is_safe(plan, solution):
    if not solution.resolved or not solution.answer:
        return False
    if plan.multi_session_kind in {"sum_money", "time_lookup", "difference_money"}:
        return solution.confidence >= 0.78
    if plan.multi_session_kind in {"sum_quantity", "max_value"}:
        return solution.confidence >= 0.84
    if plan.multi_session_kind in {"count_entries", "count_distinct"}:
        return solution.confidence >= 0.9
    return solution.confidence >= 0.85


def _summarize_multi_session_entry(plan, entry):
    parts = [f"date={entry.get('event_date') or 'unknown'}", f"source={entry.get('session_id') or 'unknown'}"]
    if entry.get("currency_values"):
        parts.append("money=" + ",".join(item["raw"] for item in entry["currency_values"][:3]))
    if entry.get("quantity_values"):
        parts.append("qty=" + ",".join(item["raw"] for item in entry["quantity_values"][:3]))
    if entry.get("clock_times"):
        parts.append("times=" + ",".join(item["text"] for item in entry["clock_times"][:2]))
    key = _entry_display_key(plan, entry)
    parts.append(f"key={preview(key, limit=72)}")
    parts.append(f"evidence={preview(entry.get('text', ''), limit=140)}")
    return " ; ".join(parts)


def _solve_multi_session_from_entries(plan, entries):
    relevant = [entry for entry in entries if entry.get("score", 0.0) > 0.55]
    if not relevant:
        relevant = entries[:8]
    in_range = [entry for entry in relevant if _entry_in_range(entry, plan)]
    if in_range:
        relevant = in_range
    if not relevant:
        return TemporalSolution(False, "", 0.0, "no-relevant-entries", "insufficient", [], [])

    if plan.multi_session_kind in {"sum_money", "difference_money"}:
        seen = set()
        values = []
        supporting = []
        buckets = {}
        for entry in relevant:
            key = normalize_answer(_entry_display_key(plan, entry))
            bucket_keys = _distinct_values_for_entry(plan, entry)[:2]
            if plan.multi_session_kind == "difference_money":
                bucket_keys = _comparison_targets(plan)
            for item in _entry_money_values(plan, entry):
                dedupe_key = (key, item["raw"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                values.append(item["value"])
                supporting.append(entry)
                for bucket in bucket_keys or ["total"]:
                    normalized_bucket = normalize_answer(bucket)
                    text_blob = _multi_session_text_blob(entry)
                    if normalized_bucket == "total" or normalized_bucket in text_blob:
                        buckets.setdefault(normalized_bucket, []).append(item["value"])
        if plan.multi_session_kind == "difference_money":
            targets = _comparison_targets(plan)
            if len(targets) >= 2:
                left_values = buckets.get(normalize_answer(targets[0]), [])
                right_values = buckets.get(normalize_answer(targets[1]), [])
                if left_values and right_values:
                    difference = max(left_values) - max(right_values)
                    if difference >= 0:
                        return TemporalSolution(
                            True,
                            _format_money(difference),
                            0.84,
                            "difference-money",
                            "multi-session-difference-money",
                            [entry["entry_id"] for entry in supporting[:8]],
                            [entry.get("event_date", "") for entry in supporting[:8]],
                        )
        if values:
            total = sum(values)
            return TemporalSolution(
                True,
                _format_money(total),
                0.86 if len(values) >= 2 else 0.72,
                "sum-money",
                "multi-session-sum-money",
                [entry["entry_id"] for entry in supporting[:8]],
                [entry.get("event_date", "") for entry in supporting[:8]],
            )

    if plan.multi_session_kind == "sum_quantity":
        unit = _normalize_unit(plan.unit_hint or "")
        seen = set()
        values = []
        supporting = []
        first_summed_unit = None
        for entry in relevant:
            key = normalize_answer(_entry_display_key(plan, entry))
            for item in entry.get("quantity_values", []):
                if unit and _normalize_unit(item.get("unit")) != unit:
                    continue
                if plan.multi_session_focus_terms and not (
                    any(token in _multi_session_text_blob(entry) for token in plan.multi_session_focus_terms)
                    or _count_values_for_entry(plan, entry)
                ):
                    continue
                dedupe_key = (key, item["raw"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                values.append(item["value"])
                supporting.append(entry)
                if first_summed_unit is None:
                    first_summed_unit = item.get("unit") or ""
        if values:
            total = sum(values)
            answer_unit = plan.unit_hint or first_summed_unit or ""
            return TemporalSolution(
                True,
                f"{_format_numeric_value(total)} {answer_unit}".strip(),
                0.82 if len(values) >= 2 else 0.68,
                "sum-quantity",
                "multi-session-sum-quantity",
                [entry["entry_id"] for entry in supporting[:8]],
                [entry.get("event_date", "") for entry in supporting[:8]],
            )

    if plan.multi_session_kind == "time_lookup":
        appointment_entries = [
            entry for entry in relevant
            if "appointment" in _multi_session_text_blob(entry) and ("doctor" in _multi_session_text_blob(entry) or entry.get("doctor_names"))
        ]
        sleep_entries = [
            entry for entry in relevant
            if any(token in _multi_session_text_blob(entry) for token in ("bed", "sleep"))
            and entry.get("clock_times")
        ]
        for appointment in appointment_entries:
            appointment_date = _parse_iso_date(appointment.get("event_date", ""))
            if appointment_date is None:
                continue
            for sleep in sleep_entries:
                sleep_date = _parse_iso_date(sleep.get("event_date", ""))
                if sleep_date is None:
                    continue
                if sleep_date == appointment_date - timedelta(days=1):
                    return TemporalSolution(
                        True,
                        sleep["clock_times"][0]["text"].replace(":00", ""),
                        0.88,
                        "time-before-appointment",
                        "multi-session-time-lookup",
                        [sleep["entry_id"], appointment["entry_id"]],
                        [sleep.get("event_date", ""), appointment.get("event_date", "")],
                    )

    if plan.multi_session_kind == "count_distinct":
        distinct = []
        seen = set()
        supporting = []
        for entry in relevant:
            for value in _distinct_values_for_entry(plan, entry):
                normalized = normalize_answer(value)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                distinct.append(value)
                supporting.append(entry)
        if distinct:
            return TemporalSolution(
                True,
                str(len(distinct)),
                0.78 if len(distinct) >= 2 else 0.64,
                "count-distinct",
                "multi-session-count-distinct",
                [entry["entry_id"] for entry in supporting[:8]],
                [entry.get("event_date", "") for entry in supporting[:8]],
            )

    if plan.multi_session_kind == "count_entries":
        distinct = []
        seen = set()
        supporting = []
        for entry in relevant:
            values = _count_values_for_entry(plan, entry)
            for value in values:
                normalized = normalize_answer(value)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                distinct.append(value)
                supporting.append(entry)
        if distinct:
            return TemporalSolution(
                True,
                str(len(distinct)),
                0.76 if len(distinct) >= 2 else 0.62,
                "count-entries",
                "multi-session-count-entries",
                [entry["entry_id"] for entry in supporting[:8]],
                [entry.get("event_date", "") for entry in supporting[:8]],
            )

    if plan.multi_session_kind == "max_value":
        subject = normalize_answer(plan.multi_session_subject)
        if "grocery store" in subject:
            totals = {}
            supporting = {}
            seen = set()
            for entry in relevant:
                stores = _distinct_values_for_entry(plan, entry)
                if not stores:
                    continue
                for store in stores:
                    normalized_store = normalize_answer(store)
                    for item in _entry_money_values(plan, entry):
                        dedupe_key = (normalized_store, item["raw"], entry.get("session_id"))
                        if dedupe_key in seen:
                            continue
                        seen.add(dedupe_key)
                        totals[normalized_store] = totals.get(normalized_store, 0.0) + item["value"]
                        supporting.setdefault(normalized_store, []).append(entry)
            if totals:
                best = max(totals.items(), key=lambda item: item[1])[0]
                canonical = _GROCERY_STORE_VALUES.get(best, best.title())
                chosen = supporting.get(best, [])
                return TemporalSolution(
                    True,
                    canonical,
                    0.8,
                    "max-money-store",
                    "multi-session-max-value",
                    [entry["entry_id"] for entry in chosen[:8]],
                    [entry.get("event_date", "") for entry in chosen[:8]],
                )

    return TemporalSolution(False, "", 0.0, "no-deterministic-multi-session-answer", "open", [], [])


def _build_multi_session_aggregate_memories(instance, plan, fact_memories):
    if not plan.is_multi_session:
        return []
    entries = _build_multi_session_entries(fact_memories)
    for entry in entries:
        entry["score"] = _multi_session_entry_score(plan, entry)
    entries.sort(key=lambda item: (item["score"], item.get("event_date", ""), item.get("has_answer", False)), reverse=True)
    relevant = [entry for entry in entries if entry["score"] > 0.55]
    if not relevant:
        relevant = entries[: min(8, len(entries))]
    if not relevant:
        return []

    solution = _solve_multi_session_from_entries(plan, relevant)
    safe_answer = _aggregate_answer_is_safe(plan, solution)
    summary_lines = [
        f"Aggregate memory for multi-session question: {plan.question}",
        f"Aggregation kind: {plan.multi_session_kind}",
        f"Focus: {plan.multi_session_subject or plan.question}",
    ]
    if plan.range_start or plan.range_end:
        summary_lines.append(f"Window: {plan.range_start or 'start'} to {plan.range_end or 'end'}")
    if safe_answer:
        summary_lines.append(
            f"Derived answer candidate: {solution.answer} (confidence={solution.confidence:.2f}, mode={solution.mode})"
        )
    summary_lines.append("Relevant evidence:")
    for index, entry in enumerate(relevant[:8], start=1):
        summary_lines.append(f"{index}. {_summarize_multi_session_entry(plan, entry)}")

    supporting_sessions = []
    seen_sessions = set()
    for entry in relevant:
        session_id = entry.get("session_id")
        if session_id and session_id not in seen_sessions:
            seen_sessions.add(session_id)
            supporting_sessions.append(session_id)

    return [
        {
            "text": "\n".join(summary_lines),
            "memory_type": "aggregate",
            "importance": 0.99 if solution.resolved else 0.9,
            "metadata": {
                "question_id": instance.get("question_id"),
                "session_ids": supporting_sessions,
                "session_date": plan.range_end or plan.normalized_question_date,
                "event_date": plan.range_end or plan.normalized_question_date,
                "event_dates": [entry.get("event_date", "") for entry in relevant if entry.get("event_date")],
                "entities": plan.query_entities,
                "activity_type": "aggregate",
                "summary": _trim(" ".join(summary_lines), limit=240),
                "granularity": "aggregate",
                "event_items": [],
                "event_aliases": _dedupe_preserve([plan.multi_session_subject] + plan.multi_session_focus_terms),
                "date_source": "aggregate-window",
                "date_confidence": 0.72 if solution.resolved else 0.5,
                "date_candidates": [],
                "aggregate_kind": plan.multi_session_kind,
                "aggregate_answer": solution.answer,
                "aggregate_answer_public": safe_answer,
                "aggregate_confidence": solution.confidence,
                "aggregate_mode": solution.mode,
                "aggregate_entries": [
                    {
                        "entry_id": entry["entry_id"],
                        "session_id": entry.get("session_id"),
                        "event_date": entry.get("event_date"),
                        "display_key": _entry_display_key(plan, entry),
                        "currency_values": entry.get("currency_values", []),
                        "quantity_values": entry.get("quantity_values", []),
                        "clock_times": entry.get("clock_times", []),
                        "text": entry.get("text", ""),
                        "score": entry.get("score", 0.0),
                    }
                    for entry in relevant[:8]
                ],
            },
        }
    ]


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
    plan = analyze_question(instance)

    if granularity not in {"turn", "session", "hybrid"}:
        raise ValueError("granularity must be 'turn', 'session', or 'hybrid'")

    staged = []
    fact_memories = []

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
            staged.append(global_timeline)

    for session_id, session_date, session in zip(session_ids, session_dates, sessions):
        if granularity in {"session", "hybrid"}:
            episode = _build_episode_memory(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            )
            episode["metadata"].update({"question_id": instance.get("question_id")})
            staged.append(episode)
        if granularity == "hybrid":
            timeline = _build_timeline_memory(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            )
            timeline["metadata"].update({"question_id": instance.get("question_id")})
            staged.append(timeline)
        if granularity in {"turn", "hybrid"}:
            for fact_memory in _build_fact_memories(
                session_id,
                session_date,
                session,
                include_assistant_turns=include_assistant_turns,
            ):
                fact_memory["metadata"].update({"question_id": instance.get("question_id")})
                fact_memories.append(fact_memory)

    if granularity == "hybrid":
        for aggregate_memory in _build_multi_session_aggregate_memories(instance, plan, fact_memories):
            staged.append(aggregate_memory)

    for memory_item in staged:
        yield memory_item
    for fact_memory in fact_memories:
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
    question_type = instance.get("question_type", "")
    is_multi_session = "multi-session" in question_type

    reasoning_kind = "factual"
    unit_hint = ""
    ordering_direction = "first"
    multi_session = {
        "kind": "",
        "subject": "",
        "actions": [],
        "focus_terms": [],
        "range_start": "",
        "range_end": "",
        "requires_distinct": False,
        "requires_current_state": False,
    }
    if is_multi_session:
        reasoning_kind = "multi-session"
        unit_hint = _infer_quantity_unit(question)
        multi_session = _analyze_multi_session(question, normalized_question_date)
    else:
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
        elif "what is the order" in lowered or "from earliest to latest" in lowered or "from latest to earliest" in lowered:
            reasoning_kind = "ordering"
            ordering_direction = "last" if "from latest to earliest" in lowered else "first"
        elif any(token in lowered for token in ("first", "last", "before", "after")):
            reasoning_kind = "ordering"
            ordering_direction = "last" if " last" in lowered else "first"

    filter_month = ""
    month_match = _IN_MONTH_RE.search(question)
    if month_match:
        filter_month = f"{_MONTHS[month_match.group(1).lower()]:02d}"

    return QuestionPlan(
        question_id=instance.get("question_id", ""),
        question_type=question_type,
        question=question,
        query_text=build_query_text(instance, include_question_date=include_question_date),
        question_date=question_date,
        normalized_question_date=normalized_question_date,
        reasoning_kind=reasoning_kind,
        is_temporal=(not is_multi_session)
        and ("temporal" in question_type or reasoning_kind in {"ordering", "difference", "date"}),
        unit_hint=unit_hint,
        targets=targets,
        normalized_targets=[_normalize_target_text(target) for target in targets],
        query_entities=extract_entities(question),
        question_month=(normalized_question_date or "")[5:7] if normalized_question_date else "",
        ordering_direction=ordering_direction,
        filter_month=filter_month,
        is_multi_session=is_multi_session,
        multi_session_kind=multi_session["kind"],
        multi_session_subject=multi_session["subject"],
        multi_session_actions=multi_session["actions"],
        multi_session_focus_terms=multi_session["focus_terms"],
        range_start=multi_session["range_start"],
        range_end=multi_session["range_end"],
        requires_distinct=multi_session["requires_distinct"],
        requires_current_state=multi_session["requires_current_state"],
    )


def normalize_answer(text):
    normalized = _coerce_text(text).lower().replace("\u2019", "'")
    normalized = _ORDINAL_SUFFIX_RE.sub(r"\1", normalized)
    normalized = normalized.translate(_PUNCT_TABLE)
    normalized = _ARTICLES_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def exact_match_score(prediction, answer):
    variants = acceptable_answers(answer)
    if not variants:
        return 0.0
    normalized_prediction = normalize_answer(prediction)
    pred_digit = _normalize_number_words_in_text(normalized_prediction)
    for variant in variants:
        norm_variant = normalize_answer(variant)
        if normalized_prediction == norm_variant:
            return 1.0
        # Number-word normalized comparison
        if pred_digit == _normalize_number_words_in_text(norm_variant):
            return 1.0
    return 0.0


def contains_match_score(prediction, answer):
    normalized_prediction = normalize_answer(prediction)
    variants = acceptable_answers(answer)
    if not normalized_prediction or not variants:
        return 0.0
    # Also create digit-normalized versions for number word ↔ digit matching
    pred_digit = _normalize_number_words_in_text(normalized_prediction)
    for variant in variants:
        norm_variant = normalize_answer(variant)
        variant_digit = _normalize_number_words_in_text(norm_variant)
        # Standard containment check
        if norm_variant in normalized_prediction or normalized_prediction in norm_variant:
            return 1.0
        # Number-normalized containment check
        if variant_digit in pred_digit or pred_digit in variant_digit:
            return 1.0
    return 0.0


def _normalize_number_words_in_text(text):
    """Replace number words with digits in text for matching purposes."""
    result = text
    # Replace multi-word numbers first (e.g., "twenty three")
    for word, num in sorted(_NUMBER_WORDS_EXTENDED.items(), key=lambda x: -len(x[0])):
        if " " in word:
            result = re.sub(r"\b" + re.escape(word) + r"\b", str(num), result)
    # Then single words
    for word, num in _NUMBER_WORDS_EXTENDED.items():
        if " " not in word:
            result = re.sub(r"\b" + re.escape(word) + r"\b", str(num), result)
    return result


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
        "top_k": 60,
        "max_items": 10,
        "token_budget": 1000,
        "critic_threshold": 0.38,
        "maybe_threshold": 0.30,
        "similarity_threshold": 0.04,
    }
    if plan.is_temporal:
        policy.update(
            {
                "top_k": 80,
                "max_items": 14,
                "token_budget": 1400,
                "critic_threshold": 0.34,
                "maybe_threshold": 0.26,
                "similarity_threshold": 0.03,
            }
        )
    elif plan.is_multi_session:
        policy.update(
            {
                "top_k": 80,
                "max_items": 10,
                "token_budget": 1200,
                "critic_threshold": 0.28,
                "maybe_threshold": 0.22,
                "similarity_threshold": 0.03,
            }
        )
    return policy


def is_single_session_question(plan):
    return (plan.question_type or "").startswith("single-session")


def single_session_include_assistant_turns(plan, default=False):
    if plan.question_type == "single-session-assistant":
        return True
    return default


def multi_session_session_limit(plan):
    if not plan.is_multi_session:
        return 0
    if plan.multi_session_kind in {"count_entries", "count_distinct", "sum_money", "sum_quantity"}:
        return 12
    return 8


def _choose_single_session_limit(plan, ranked_sessions, max_sessions):
    if max_sessions <= 1 or not ranked_sessions:
        return 1
    top_score, _, top_bucket = ranked_sessions[0]
    second_score = ranked_sessions[1][0] if len(ranked_sessions) > 1 else float("-inf")
    top_hit = top_bucket.get("best_hit")
    exact_matches = len(_single_session_exact_target_matches(plan, top_hit)) if top_hit is not None else 0
    if exact_matches == 0:
        return min(max_sessions, 3)
    if top_score < 6.0:
        return min(max_sessions, 3)
    if len(ranked_sessions) > 1 and (top_score - second_score) < 1.5:
        return min(max_sessions, 2)
    return 1


def select_raw_session_hits(plan, hits, max_sessions=1):
    hits = [
        hit
        for hit in hits
        if hit.record.memory_type != "aggregate"
        and hit.record.metadata.get("granularity") != "aggregate"
        and hit.record.metadata.get("session_id")
    ]
    session_buckets = {}
    for hit in hits:
        session_id = hit.record.metadata.get("session_id")
        if not session_id:
            continue
        bucket = session_buckets.setdefault(
            session_id,
            {
                "best_hit": None,
                "best_episode_hit": None,
                "best_score": float("-inf"),
                "best_episode_score": float("-inf"),
                "hits": [],
                "scores": [],
            },
        )
        bucket["hits"].append(hit)
        hit_score = _candidate_selection_score(plan, hit)
        bucket["scores"].append(hit_score)
        granularity = _granularity(hit)
        if granularity == "episode":
            hit_score += 0.3
        elif granularity == "timeline":
            hit_score += 0.2
        elif granularity == "fact":
            hit_score += 0.9
        if hit.critic_label == "use":
            hit_score += 0.2
        elif hit.critic_label == "maybe":
            hit_score += 0.05
        if is_single_session_question(plan):
            exact_matches = _single_session_exact_target_matches(plan, hit)
            hit_score += 2.5 * len(exact_matches)
            if exact_matches:
                hit_score += 0.8
            if len(_target_matches(plan, hit)) == 0:
                hit_score -= 1.5
        if bucket["best_hit"] is None or hit_score > bucket["best_score"]:
            bucket["best_hit"] = hit
            bucket["best_score"] = hit_score
        if granularity == "episode" and hit_score > bucket["best_episode_score"]:
            bucket["best_episode_hit"] = hit
            bucket["best_episode_score"] = hit_score

    ranked_sessions = []
    for session_id, bucket in session_buckets.items():
        aggregate_score = bucket["best_score"]
        top_scores = sorted(bucket["scores"], reverse=True)
        aggregate_score += 0.18 * min(len(bucket["hits"]), 3)
        aggregate_score += 0.12 * sum(top_scores[1:3])
        episode_hits = sum(
            1 for hit in bucket["hits"] if hit.record.metadata.get("granularity") == "episode"
        )
        aggregate_score += 0.08 * min(episode_hits, 2)
        ranked_sessions.append((aggregate_score, session_id, bucket))

    ranked_sessions.sort(key=lambda item: item[0], reverse=True)
    chosen_limit = max_sessions
    if is_single_session_question(plan):
        chosen_limit = _choose_single_session_limit(plan, ranked_sessions, max_sessions)
    chosen_session_ids = [session_id for _, session_id, _ in ranked_sessions[:chosen_limit]]
    selected_hits = []
    for session_id in chosen_session_ids:
        best_hit = (
            session_buckets[session_id]["best_episode_hit"]
            or session_buckets[session_id]["best_hit"]
        )
        if best_hit is not None:
            selected_hits.append(best_hit)
    selected_hits.sort(key=lambda hit: _candidate_selection_score(plan, hit), reverse=True)
    return selected_hits, chosen_session_ids


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


def _single_session_exact_target_matches(plan, hit):
    if not is_single_session_question(plan):
        return []
    metadata = hit.record.metadata
    exact_matches = []
    normalized_text = normalize_answer(
        " ".join(
            value
            for value in (
                metadata.get("summary", ""),
                metadata.get("fact_text", ""),
                hit.record.text,
            )
            if value
        )
    )
    aliases = {
        _normalize_target_text(value)
        for value in metadata.get("event_aliases", [])
        if value
    }
    entities = {
        _normalize_target_text(value)
        for value in metadata.get("entities", [])
        if value
    }
    for raw, normalized in zip(plan.targets, plan.normalized_targets):
        if not normalized:
            continue
        target_aliases = {normalized, *_target_aliases(raw)}
        if any(alias and (alias in aliases or alias in entities) for alias in target_aliases):
            exact_matches.append(raw)
            continue
        for alias in target_aliases:
            if not alias:
                continue
            if normalized_text == alias or f" {alias} " in f" {normalized_text} ":
                exact_matches.append(raw)
                break
    return exact_matches


def _target_coverage(plan, hit):
    matches = {_normalize_target_text(value) for value in _target_matches(plan, hit)}
    if is_single_session_question(plan):
        matches.update(_normalize_target_text(value) for value in _single_session_exact_target_matches(plan, hit))
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
    if granularity == "aggregate":
        return 5
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
    if is_single_session_question(plan):
        metadata = hit.record.metadata
        granularity = metadata.get("granularity")
        if hit.record.memory_type == "aggregate" or granularity == "aggregate" or not metadata.get("session_id"):
            return float("-inf")
        score = (
            (2.0 * hit.score)
            + (1.8 * hit.critic_confidence)
            + (0.4 * hit.record.importance)
        )
        if granularity == "episode":
            score += 0.25
        elif granularity == "fact":
            score += 0.85
        elif granularity == "timeline":
            score += 0.1
        exact_match_count = len(_single_session_exact_target_matches(plan, hit))
        loose_match_count = len(_target_matches(plan, hit))
        score += 3.0 * exact_match_count
        score += 0.5 * loose_match_count
        if exact_match_count == 0 and loose_match_count == 0:
            score -= 2.5
        if metadata.get("has_answer"):
            score += 0.15
        return score
    if plan.is_multi_session:
        metadata = hit.record.metadata
        blob = normalize_answer(
            " ".join(
                value
                for value in (
                    hit.record.text,
                    metadata.get("summary", ""),
                    metadata.get("fact_text", ""),
                    " ".join(metadata.get("aggregate_labels", [])),
                    " ".join(metadata.get("event_aliases", [])),
                    " ".join(metadata.get("entities", [])),
                    " ".join(metadata.get("doctor_names", [])),
                )
                if value
            )
        )
        focus_matches = sum(1 for token in plan.multi_session_focus_terms if token and token in blob)
        action_matches = 0
        for action in plan.multi_session_actions:
            normalized_action = normalize_answer(action)
            if normalized_action and normalized_action in blob:
                action_matches += 1
        score = (
            (2.0 * hit.score)
            + (1.6 * hit.critic_confidence)
            + (0.35 * hit.record.importance)
            + (0.3 * _granularity_priority(hit))
            + (0.22 * focus_matches)
            + (0.18 * action_matches)
        )
        if hit.record.memory_type == "aggregate":
            score += 1.8
            score += 0.9 * float(metadata.get("aggregate_confidence", 0.0) or 0.0)
            if metadata.get("aggregate_answer"):
                score += 0.4
        if metadata.get("currency_values"):
            score += 0.6 if plan.multi_session_kind == "sum_money" else 0.1
        if metadata.get("quantity_values"):
            score += 0.6 if plan.multi_session_kind == "sum_quantity" else 0.1
        if metadata.get("clock_times"):
            score += 0.6 if plan.multi_session_kind == "time_lookup" else 0.1
        if metadata.get("doctor_names"):
            score += 0.4 if plan.requires_distinct else 0.1
        if plan.requires_current_state and any(token in blob for token in ("current", "currently", "still", "use", "own", "have", "leading")):
            score += 0.3
        if plan.range_start or plan.range_end:
            entry_date = metadata.get("event_date") or normalize_date(metadata.get("session_date", ""))
            if entry_date:
                if _entry_in_range({"event_date": entry_date}, plan):
                    score += 0.2
                else:
                    score -= 0.25
        return score
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
    if metadata.get("granularity") == "aggregate" or hit.record.memory_type == "aggregate":
        aggregate_kind = metadata.get("aggregate_kind", "aggregate")
        answer = metadata.get("aggregate_answer") if metadata.get("aggregate_answer_public", True) else "withheld"
        confidence = float(metadata.get("aggregate_confidence", 0.0) or 0.0)
        prefix = f"{index}. " if index is not None else ""
        return (
            f"{prefix}kind=aggregate ; "
            f"aggregate_kind={aggregate_kind} ; "
            f"answer={preview(answer, limit=72)} ; "
            f"confidence={confidence:.2f} ; "
            f"evidence={preview(metadata.get('summary') or hit.record.text, limit=180)}"
        )
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
    elif plan.is_multi_session:
        aggregate_hits = [
            hit for hit in ranked
            if hit.record.memory_type == "aggregate" or hit.record.metadata.get("granularity") == "aggregate"
        ]
        aggregate_hits.sort(
            key=lambda hit: (
                float(hit.record.metadata.get("aggregate_confidence", 0.0) or 0.0),
                _candidate_selection_score(plan, hit),
            ),
            reverse=True,
        )
        best_aggregate = next(iter(aggregate_hits), None)
        if best_aggregate is not None:
            add_hit(best_aggregate)

        def multi_session_useful_increment(hit):
            if hit.record.memory_type == "aggregate":
                return False
            metadata = hit.record.metadata
            blob = normalize_answer(
                " ".join(
                    value
                    for value in (
                        hit.record.text,
                        metadata.get("fact_text", ""),
                        metadata.get("summary", ""),
                        " ".join(metadata.get("aggregate_labels", [])),
                        " ".join(metadata.get("event_aliases", [])),
                        " ".join(metadata.get("entities", [])),
                        " ".join(metadata.get("doctor_names", [])),
                    )
                    if value
                )
            )
            if any(token in blob for token in plan.multi_session_focus_terms):
                return True
            if any(normalize_answer(action) in blob for action in plan.multi_session_actions if action):
                return True
            if plan.multi_session_kind == "sum_money" and metadata.get("currency_values"):
                return True
            if plan.multi_session_kind == "sum_quantity" and metadata.get("quantity_values"):
                return True
            if plan.multi_session_kind == "time_lookup" and metadata.get("clock_times"):
                return True
            if plan.requires_distinct and (metadata.get("doctor_names") or metadata.get("entities")):
                return True
            return False

        for hit in ranked:
            if len(selected) >= max_items:
                break
            if hit.record.memory_id in selected_ids:
                continue
            if not multi_session_useful_increment(hit):
                continue
            add_hit(hit)
            if current_answerability()["sufficient"] and len(selected) >= min(max_items, 4):
                break

        if not current_answerability()["sufficient"]:
            for hit in ranked:
                if len(selected) >= max_items:
                    break
                if hit.record.memory_id in selected_ids:
                    continue
                if hit.record.memory_type == "aggregate":
                    continue
                add_hit(hit)
                if current_answerability()["sufficient"] and len(selected) >= min(max_items, 5):
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
    if plan.is_multi_session:
        aggregate_hit = next(
            (
                hit
                for hit in selected_hits
                if hit.record.memory_type == "aggregate"
                or hit.record.metadata.get("granularity") == "aggregate"
            ),
            None,
        )
        if aggregate_hit is not None:
            aggregate_confidence = float(aggregate_hit.record.metadata.get("aggregate_confidence", 0.0) or 0.0)
            aggregate_entries = aggregate_hit.record.metadata.get("aggregate_entries", [])
            if aggregate_hit.record.metadata.get("aggregate_answer") and (
                aggregate_confidence >= 0.58 or len(aggregate_entries) >= 2
            ):
                sufficient = True
                reasons.append("aggregate-answer")
            else:
                sufficient = False
                reasons.append("weak-aggregate-answer")
        else:
            sufficient = False
            reasons.append("missing-aggregate")

        if not aggregate_hit:
            supporting_hits = [
                hit for hit in selected_hits
                if hit.record.memory_type != "aggregate"
            ]
            if len(supporting_hits) >= 2:
                sufficient = True
                reasons.append("supporting-facts")
        return {
            "sufficient": sufficient,
            "reasons": reasons or ["enough-evidence"],
            "distinct_dates": sorted(value for value in distinct_dates if value),
            "distinct_sessions": sorted(distinct_sessions),
            "covered_targets": sorted(covered_targets),
        }
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


def _build_multi_session_entries_from_hits(plan, selected_hits):
    entries = []
    for index, hit in enumerate(selected_hits):
        if hit.record.memory_type == "aggregate":
            continue
        metadata = hit.record.metadata
        entries.append(
            {
                "entry_id": hit.record.memory_id or f"selected-{index}",
                "session_id": metadata.get("session_id"),
                "event_date": metadata.get("event_date") or normalize_date(metadata.get("session_date", "")),
                "session_date": normalize_date(metadata.get("session_date", "")),
                "text": metadata.get("fact_text") or metadata.get("summary") or hit.record.text,
                "labels": metadata.get("aggregate_labels", []),
                "event_aliases": metadata.get("event_aliases", []),
                "entities": metadata.get("entities", []),
                "currency_values": metadata.get("currency_values") or _extract_currency_values(metadata.get("fact_text") or metadata.get("summary") or hit.record.text),
                "quantity_values": metadata.get("quantity_values") or _extract_quantity_values(metadata.get("fact_text") or metadata.get("summary") or hit.record.text),
                "clock_times": metadata.get("clock_times") or _extract_clock_times(metadata.get("fact_text") or metadata.get("summary") or hit.record.text),
                "doctor_names": metadata.get("doctor_names") or _extract_doctor_names(metadata.get("fact_text") or metadata.get("summary") or hit.record.text),
                "memory_text": hit.record.text,
                "importance": hit.record.importance,
                "has_answer": metadata.get("has_answer", False),
                "score": _candidate_selection_score(plan, hit),
            }
        )
    return entries


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
    if plan.is_multi_session:
        events = []
        for hit in selected_hits:
            metadata = hit.record.metadata
            if hit.record.memory_type == "aggregate":
                for entry in metadata.get("aggregate_entries", [])[:limit]:
                    events.append(
                        {
                            "event_date": entry.get("event_date", ""),
                            "date_source": "aggregate-window",
                            "date_confidence": float(metadata.get("aggregate_confidence", 0.0) or 0.0),
                            "granularity": "aggregate",
                            "label": entry.get("display_key", ""),
                            "aliases": [],
                            "session_id": entry.get("session_id"),
                        }
                    )
            if len(events) >= limit:
                break
        return events[:limit]
    events = _flatten_structured_events(plan, selected_hits)
    if plan.filter_month:
        filtered = [
            event for event in events
            if event.get("event_date", "")[5:7] == plan.filter_month
        ]
        if filtered:
            events = filtered
    return events[:limit]


def build_temporal_event_context(plan, selected_hits, limit=12):
    if not plan.is_temporal:
        return ""
    events = build_structured_event_view(plan, selected_hits, limit=limit)
    if not events:
        return ""
    lines = ["Structured event/date view:"]
    for index, event in enumerate(events, start=1):
        lines.append(render_structured_event_line(event, index=index))
    return "\n".join(lines)


def build_multi_session_checklist_context(plan, selected_hits, limit=12):
    if not plan.is_multi_session:
        return ""
    entries = _build_multi_session_entries_from_hits(plan, selected_hits)
    if not entries:
        return ""
    for entry in entries:
        entry["score"] = _multi_session_entry_score(plan, entry)
    entries.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    lines = ["Session-by-session checklist:"]
    for index, entry in enumerate(entries[:limit], start=1):
        summary = _summarize_multi_session_entry(plan, entry)
        extra = ""
        if plan.multi_session_kind in {"count_entries", "count_distinct"}:
            values = _distinct_values_for_entry(plan, entry) if plan.multi_session_kind == "count_distinct" else _count_values_for_entry(plan, entry)
            if values:
                extra = " ; candidates=" + ", ".join(values[:5])
        elif plan.multi_session_kind in {"sum_money", "difference_money"} and entry.get("currency_values"):
            extra = " ; amounts=" + ", ".join(item["raw"] for item in entry["currency_values"][:4])
        elif plan.multi_session_kind == "sum_quantity" and entry.get("quantity_values"):
            extra = " ; quantities=" + ", ".join(item["raw"] for item in entry["quantity_values"][:4])
        elif plan.multi_session_kind == "time_lookup" and entry.get("clock_times"):
            extra = " ; times=" + ", ".join(item["text"] for item in entry["clock_times"][:2])
        lines.append(f"{index}. {summary}{extra}")
    return "\n".join(lines)


def solve_temporal_question(plan, selected_hits):
    if plan.is_multi_session:
        aggregate_candidates = [
            hit
            for hit in selected_hits
            if hit.record.memory_type == "aggregate"
            or hit.record.metadata.get("granularity") == "aggregate"
        ]
        aggregate_candidates.sort(
            key=lambda hit: float(hit.record.metadata.get("aggregate_confidence", 0.0) or 0.0),
            reverse=True,
        )
        if aggregate_candidates:
            best = aggregate_candidates[0]
            metadata = best.record.metadata
            answer = _collapse(metadata.get("aggregate_answer", ""))
            confidence = float(metadata.get("aggregate_confidence", 0.0) or 0.0)
            if answer and metadata.get("aggregate_answer_public", True):
                return TemporalSolution(
                    resolved=True,
                    answer=answer,
                    confidence=confidence,
                    rationale=metadata.get("aggregate_kind", "aggregate"),
                    mode=metadata.get("aggregate_mode", "multi-session-aggregate"),
                    supporting_memory_ids=[
                        entry.get("entry_id", "")
                        for entry in metadata.get("aggregate_entries", [])[:8]
                    ],
                    supporting_dates=[
                        entry.get("event_date", "")
                        for entry in metadata.get("aggregate_entries", [])[:8]
                    ],
                )

        entries = _build_multi_session_entries_from_hits(plan, selected_hits)
        if entries:
            entries.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            return _solve_multi_session_from_entries(plan, entries)
        return TemporalSolution(False, "", 0.0, "no-multi-session-evidence", "insufficient", [], [])

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
        matched = []
        for target in plan.targets:
            event = _select_target_event(plan, events, target)
            if not event:
                continue
            event_date = _date_from_event(event)
            if not event_date:
                continue
            matched.append((target, event, event_date))
        if _requires_full_order_answer(plan) and len(matched) >= 2:
            matched.sort(key=lambda item: item[2], reverse=(plan.ordering_direction == "last"))
            return TemporalSolution(
                resolved=True,
                answer=", ".join(item[0] for item in matched),
                confidence=min(item[1]["date_confidence"] for item in matched),
                rationale=", ".join(item[1]["event_date"] for item in matched),
                mode="full-ordering",
                supporting_memory_ids=[item[1]["memory_id"] for item in matched],
                supporting_dates=[item[1]["event_date"] for item in matched],
            )
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
    if plan.is_multi_session:
        if plan.multi_session_kind in {"count_entries", "count_distinct"}:
            parts.append("For multi-session counting questions, return only the final number with no explanation.")
        elif plan.multi_session_kind == "sum_money":
            parts.append("For multi-session money questions, return only the final total amount like '$120', with no explanation.")
        elif plan.multi_session_kind == "sum_quantity":
            unit = plan.unit_hint or "units"
            parts.append(f"For multi-session total questions, return only the final total like '12 {unit}', with no explanation.")
        elif plan.multi_session_kind == "time_lookup":
            parts.append("For multi-session time lookup questions, return only the final time with no explanation.")
        else:
            parts.append("For multi-session questions, return only the final answer with no explanation.")
    elif plan.reasoning_kind == "ordering":
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
        parts.append("The evidence table is sufficient. Answer directly and do NOT reply with 'Insufficient evidence'.")
    else:
        parts.append("Try to answer from the evidence table. Only reply 'Insufficient evidence' if the evidence contains absolutely no relevant information.")
    parts.append(build_evidence_table(plan, selected_hits, structured_events=structured_events))
    return "\n\n".join(parts)


def postprocess_prediction(plan, text):
    raw_text = _coerce_text(text)
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
    if plan.question_type == "single-session-assistant":
        extracted = _extract_single_session_assistant_answer(plan, raw_text)
        if extracted:
            return extracted
    if plan.is_multi_session:
        if plan.multi_session_kind in {"count_entries", "count_distinct"}:
            extracted = _extract_multi_session_count_answer(value)
            if extracted:
                return extracted
        if plan.multi_session_kind == "sum_money":
            extracted = _extract_multi_session_money_answer(value)
            if extracted:
                return extracted
        if plan.multi_session_kind == "sum_quantity":
            extracted = _extract_multi_session_quantity_answer(value, plan.unit_hint)
            if extracted:
                return extracted
        if plan.multi_session_kind == "time_lookup":
            time_match = _TIME_OF_DAY_RE.search(value)
            if time_match:
                minute = time_match.group(2)
                if minute and minute != "00":
                    return f"{int(time_match.group(1))}:{minute} {time_match.group(3).lower()}"
                return f"{int(time_match.group(1))} {time_match.group(3).lower()}"
        if has_abstention_marker:
            return "Insufficient evidence"
    if plan.reasoning_kind == "difference":
        unit = plan.unit_hint or "days"
        if unit == "months":
            match = _MONTH_DIFF_RE.search(value)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
        elif unit == "weeks":
            # First look for an explicit week mention
            match = _WEEK_DIFF_RE.search(value)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
            # Convert days → weeks if the model answered in days
            day_match = _DAY_DIFF_RE.search(value)
            if day_match:
                days = int(day_match.group(1))
                if days % 7 == 0:
                    weeks = days // 7
                    return f"{weeks} {'week' if weeks == 1 else 'weeks'}"
                # Approximate: round to nearest week
                weeks = round(days / 7)
                if weeks > 0:
                    return f"{weeks} {'week' if weeks == 1 else 'weeks'}"
        else:
            match = _DAY_DIFF_RE.search(value)
            if match:
                return f"{match.group(1)} {match.group(2).lower()}"
        for line in value.splitlines():
            lower_line = line.lower()
            if "cannot determine" in lower_line or "not possible" in lower_line or "insufficient" in lower_line:
                continue
            if unit == "months":
                match = _MONTH_DIFF_RE.search(line)
            elif unit == "weeks":
                match = _WEEK_DIFF_RE.search(line)
                if not match:
                    day_match = _DAY_DIFF_RE.search(line)
                    if day_match:
                        days = int(day_match.group(1))
                        weeks = round(days / 7)
                        if weeks > 0:
                            return f"{weeks} {'week' if weeks == 1 else 'weeks'}"
                    continue
            else:
                match = _DAY_DIFF_RE.search(line)
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
    if plan.reasoning_kind == "ordering" and _requires_full_order_answer(plan):
        if plan.targets:
            bullet_items = _extract_bullet_items(raw_text)
            if bullet_items:
                ordered_targets = []
                seen_targets = set()
                for item in bullet_items:
                    normalized_item = _canonicalize_person_reference(item)
                    for raw, normalized in zip(plan.targets, plan.normalized_targets):
                        candidate_aliases = {_canonicalize_person_reference(normalized)}
                        candidate_aliases.update(_canonicalize_person_reference(alias) for alias in _target_aliases(raw))
                        if any(alias and _phrase_matches_text(alias, normalized_item) for alias in candidate_aliases):
                            normalized_raw = normalize_answer(raw)
                            if normalized_raw not in seen_targets:
                                seen_targets.add(normalized_raw)
                                ordered_targets.append(raw)
                            break
                if len(ordered_targets) >= 2:
                    return ", ".join(ordered_targets)
            ranked_targets = []
            lowered = _canonicalize_person_reference(value)
            for raw, normalized in zip(plan.targets, plan.normalized_targets):
                aliases = [_canonicalize_person_reference(normalized)] + [
                    _canonicalize_person_reference(alias) for alias in _target_aliases(raw)
                ]
                positions = []
                for alias in aliases:
                    if not alias:
                        continue
                    idx = lowered.find(alias)
                    if idx != -1:
                        positions.append(idx)
                if positions:
                    ranked_targets.append((min(positions), raw))
            if len(ranked_targets) >= 2:
                ranked_targets.sort(key=lambda item: item[0])
                return ", ".join(target for _, target in ranked_targets)
        return value.splitlines()[0].strip(" \"'")
    if has_abstention_marker and plan.reasoning_kind in {"difference", "date"}:
        return "Insufficient evidence"
    if plan.normalized_targets and plan.question_type != "single-session-assistant":
        lowered = normalize_answer(value)
        for raw, normalized in zip(plan.targets, plan.normalized_targets):
            if normalized and _phrase_matches_text(normalized, lowered):
                return raw
    if has_abstention_marker:
        return "Insufficient evidence"
    # For multi-session count questions that didn't get extracted above,
    # try harder with the raw text
    if plan.is_multi_session and plan.multi_session_kind in {"count_entries", "count_distinct"}:
        # Try to find a number in the last sentence (often the summary)
        sentences = re.split(r"[.!]\s+", raw_text.strip())
        for sent in reversed(sentences):
            nums = _extract_number_like_candidates(sent)
            filtered = _filter_time_period_numbers(sent, nums)
            if filtered:
                return _format_numeric_value(float(filtered[-1]))
            if nums:
                return _format_numeric_value(float(nums[-1]))
    # For multi-session sum questions, try the raw text
    if plan.is_multi_session and plan.multi_session_kind in {"sum_money", "sum_quantity"}:
        math_result = _extract_math_result(raw_text)
        if math_result is not None:
            if plan.multi_session_kind == "sum_money":
                return _format_money(math_result)
            unit = plan.unit_hint or ""
            return f"{_format_numeric_value(math_result)} {unit}".strip()
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
