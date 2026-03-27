import hashlib
import math
import re
from datetime import datetime, timezone


WORD_RE = re.compile(r"[A-Za-z0-9_']+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def utc_now():
    return datetime.now(timezone.utc)


def iso_timestamp(value=None):
    if value is None:
        value = utc_now()
    if isinstance(value, str):
        return parse_timestamp(value).isoformat()
    return value.astimezone(timezone.utc).isoformat()


def parse_timestamp(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def tokenize(text, drop_stopwords=True):
    tokens = [token.lower() for token in WORD_RE.findall(text or "")]
    if not drop_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


def normalize_vector(values):
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return [0.0 for _ in values]
    return [v / norm for v in values]


def cosine_similarity(left, right):
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def stable_hash(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def preview(text, limit=88):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
