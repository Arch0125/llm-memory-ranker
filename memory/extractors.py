"""Benchmark-agnostic extractors for dates, quantities, currency, time-of-day,
and a few other surface signals used by the generic memory layer.

Larger or domain-specific patterns (LongMemEval-style "case competition" lists,
weekday-relative phrasing, etc.) intentionally live in the benchmark adapter,
not here.
"""
from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, datetime, timedelta


MONTH_ALIASES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
ISO_LIKE_DATE_RE = re.compile(r"\b\d{4}[/-]\d{2}[/-]\d{2}\b")
MONTH_DAY_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)
DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+of\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)
SHORT_MONTH_DAY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
MONEY_RE = re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)")
QUANTITY_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(hours?|days?|weeks?|months?|years?|pages?|miles?|km|kg|lbs?|items?)\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")
TIME_OF_DAY_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)>\"]+")
DOMAIN_RE = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")
DAY_DIFF_RE = re.compile(r"\b(\d+)\s*(day|days)\b", re.IGNORECASE)
WEEK_DIFF_RE = re.compile(r"\b(\d+)\s*(week|weeks)\b", re.IGNORECASE)
MONTH_DIFF_RE = re.compile(r"\b(\d+)\s*(month|months)\b", re.IGNORECASE)


def parse_iso_date(value):
    """Parse an ISO date string (YYYY-MM-DD). Returns None on failure."""
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip()
    match = ISO_DATE_RE.search(text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def normalize_iso_date(value):
    """Normalize various forms (yyyy/mm/dd, yyyy-mm-dd) to yyyy-mm-dd. Empty on failure."""
    if not value:
        return ""
    match = ISO_LIKE_DATE_RE.search(str(value))
    if not match:
        return ""
    return match.group(0).replace("/", "-")


def extract_iso_dates(text):
    return [m.group(0).replace("/", "-") for m in ISO_LIKE_DATE_RE.finditer(text or "")]


def extract_money_values(text):
    """Return list of {raw: '$120', amount: 120.0} for every money mention."""
    out = []
    for match in MONEY_RE.finditer(text or ""):
        raw = match.group(0)
        try:
            amount = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        out.append({"raw": raw, "amount": amount})
    return out


def extract_quantities(text):
    """Return list of {raw, value, unit}."""
    out = []
    for match in QUANTITY_RE.finditer(text or ""):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        unit = match.group(2).lower()
        if unit.endswith("s") and unit not in {"miles", "lbs"}:
            unit_singular = unit[:-1]
        else:
            unit_singular = unit
        out.append({"raw": match.group(0), "value": value, "unit": unit_singular})
    return out


def extract_clock_times(text):
    out = []
    for match in TIME_OF_DAY_RE.finditer(text or ""):
        hour = int(match.group(1))
        minute = match.group(2)
        suffix = match.group(3).lower()
        canonical = (
            f"{hour}:{minute} {suffix}" if minute and minute != "00" else f"{hour} {suffix}"
        )
        out.append({"raw": match.group(0), "text": canonical, "hour": hour, "minute": int(minute) if minute else 0, "suffix": suffix})
    return out


def subtract_months(value, months):
    if value is None:
        return None
    year = value.year
    month = value.month - months
    while month <= 0:
        year -= 1
        month += 12
    if not (1 <= year <= 9999):
        return None
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


def subtract_years(value, years):
    if value is None:
        return None
    year = value.year - years
    if not (1 <= year <= 9999):
        return None
    day = min(value.day, monthrange(year, value.month)[1])
    return date(year, value.month, day)


def difference_between_dates(left, right, unit="days"):
    """Absolute difference between two date objects, expressed in `unit`."""
    if left is None or right is None:
        return 0
    delta = abs((left - right).days)
    if unit == "weeks":
        return delta // 7
    if unit == "months":
        return max(0, abs((left.year - right.year) * 12 + (left.month - right.month)))
    if unit == "years":
        return abs(left.year - right.year)
    return delta


def format_month_day(d):
    if d is None:
        return ""
    return f"{MONTH_NAMES[d.month]} {d.day}"


def first_month_day(text):
    """Return ('Month', day) if a month-day phrase is present in `text`, else None."""
    match = MONTH_DAY_RE.search(text or "")
    if match:
        return match.group(1).capitalize(), int(match.group(2))
    match = DAY_MONTH_RE.search(text or "")
    if match:
        return match.group(2).capitalize(), int(match.group(1))
    return None


def relative_anchor_date(name, base_date):
    """Resolve named anchors like 'black friday' / 'cyber monday' to a concrete date.

    Returns None for anchors we don't recognize.
    """
    if not name or base_date is None:
        return None
    anchor = name.strip().lower()
    if anchor not in {"black friday", "cyber monday"}:
        return None
    november_first = date(base_date.year, 11, 1)
    days_until_thursday = (3 - november_first.weekday()) % 7
    thanksgiving = november_first + timedelta(days=days_until_thursday + 21)
    offset = 1 if anchor == "black friday" else 4
    candidate = thanksgiving + timedelta(days=offset)
    if candidate > base_date + timedelta(days=7):
        november_first = date(base_date.year - 1, 11, 1)
        days_until_thursday = (3 - november_first.weekday()) % 7
        thanksgiving = november_first + timedelta(days=days_until_thursday + 21)
        candidate = thanksgiving + timedelta(days=offset)
    return candidate
