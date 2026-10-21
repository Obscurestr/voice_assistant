"""Rule-based slot extraction (times, durations, numbers, places, items).

The classifier decides *what* the user wants; these functions pull out the
parameters needed to actually do it. Input is expected to be raw transcribed
text; each function normalises it itself.
"""

from __future__ import annotations

import datetime as dt
import operator
import re

from .data import normalize_text

WORD_NUMBERS = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "fifty": 50,
    "sixty": 60, "ninety": 90,
}  # fmt: skip
_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, WORD_NUMBERS), key=len, reverse=True)) + r")\b"
)


def words_to_digits(text: str) -> str:
    """'set a timer for five minutes' -> 'set 1 timer for 5 minutes' (only number words)."""
    return _WORD_NUM_RE.sub(lambda m: str(WORD_NUMBERS[m.group(1)]), text)


# --- durations ---------------------------------------------------------------

_UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(second|sec|minute|min|hour|hr)s?\b")


def parse_duration(text: str) -> int | None:
    """Total seconds in phrases like '1 hour 30 minutes', 'half an hour', 'ten seconds'."""
    text = normalize_text(text)
    if "half an hour" in text or "half hour" in text:
        return 1800
    text = words_to_digits(text)
    matches = _DURATION_RE.findall(text)
    if not matches:
        return None
    return int(sum(float(n) * _UNIT_SECONDS[unit] for n, unit in matches))


def format_duration(seconds: int) -> str:
    parts = []
    for name, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        value, seconds = divmod(seconds, size)
        if value:
            parts.append(f"{value} {name}{'s' if value != 1 else ''}")
    return " and ".join(parts) or "0 seconds"


# --- clock times -------------------------------------------------------------

_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\s*m|p\s*m)?\b")


def parse_time(text: str) -> dt.time | None:
    """'7 pm' / '7:30 am' / '19:15' / 'noon' / 'midnight' -> datetime.time."""
    text = normalize_text(text).replace("a.m", "am").replace("p.m", "pm")
    if "noon" in text:
        return dt.time(12, 0)
    if "midnight" in text:
        return dt.time(0, 0)
    for match in _TIME_RE.finditer(words_to_digits(text)):
        hour, minute, period = int(match.group(1)), int(match.group(2) or 0), match.group(3)
        bare = period is None and match.group(2) is None
        # A bare number ("in 5 minutes") is not a time unless preceded by "at".
        if bare and not re.search(rf"\bat\s+{match.group(1)}\b", text):
            continue
        if period:
            period = period.replace(" ", "")
            if not 1 <= hour <= 12:
                continue
            hour = hour % 12 + (12 if period == "pm" else 0)
        if 0 <= hour < 24 and 0 <= minute < 60:
            return dt.time(hour, minute)
    return None


def format_time(value: dt.time) -> str:
    return value.strftime("%I:%M %p").lstrip("0")


# --- arithmetic ----------------------------------------------------------------

_OPERATORS = {
    "plus": operator.add, "+": operator.add, "add": operator.add,
    "minus": operator.sub, "-": operator.sub, "subtract": operator.sub,
    "times": operator.mul, "multiplied by": operator.mul, "x": operator.mul, "*": operator.mul,
    "divided by": operator.truediv, "over": operator.truediv, "/": operator.truediv,
}  # fmt: skip
_OP_PATTERN = "|".join(sorted(map(re.escape, _OPERATORS), key=len, reverse=True))
_ARITH_RE = re.compile(rf"(-?\d+(?:\.\d+)?)\s*({_OP_PATTERN})\s*(-?\d+(?:\.\d+)?)")


def parse_arithmetic(text: str) -> float | None:
    """'what is 12 times 4' -> 48.0. Returns None if no expression is found.

    Raises ZeroDivisionError for division by zero so the skill can say so.
    """
    text = words_to_digits(str(text).lower().replace("?", " "))
    match = _ARITH_RE.search(text)
    if not match:
        return None
    left, op, right = match.groups()
    return _OPERATORS[op](float(left), float(right))


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4g}"


# --- unit conversion -----------------------------------------------------------

_LINEAR_UNITS = {
    # unit: (dimension, factor to base unit)
    "km": ("length", 1000), "kilometer": ("length", 1000), "kilometers": ("length", 1000),
    "m": ("length", 1), "meter": ("length", 1), "meters": ("length", 1),
    "cm": ("length", 0.01), "centimeters": ("length", 0.01),
    "mile": ("length", 1609.344), "miles": ("length", 1609.344),
    "foot": ("length", 0.3048), "feet": ("length", 0.3048),
    "inch": ("length", 0.0254), "inches": ("length", 0.0254),
    "kg": ("mass", 1), "kilogram": ("mass", 1), "kilograms": ("mass", 1),
    "g": ("mass", 0.001), "gram": ("mass", 0.001), "grams": ("mass", 0.001),
    "pound": ("mass", 0.45359237), "pounds": ("mass", 0.45359237), "lbs": ("mass", 0.45359237),
    "ounce": ("mass", 0.0283495), "ounces": ("mass", 0.0283495),
    "liter": ("volume", 1), "liters": ("volume", 1),
    "litre": ("volume", 1), "litres": ("volume", 1),
    "ml": ("volume", 0.001), "milliliters": ("volume", 0.001),
    "gallon": ("volume", 3.78541), "gallons": ("volume", 3.78541),
}  # fmt: skip
_TEMPERATURE = {"celsius": "c", "c": "c", "fahrenheit": "f", "f": "f", "kelvin": "k", "k": "k"}
_CONVERT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:degrees\s+)?([a-z]+)\s+(?:to|in|into)\s+([a-z]+)")


def _temperature(value: float, src: str, dst: str) -> float:
    celsius = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[src]
    return {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[dst]


def parse_conversion(text: str) -> tuple[float, str, float, str] | None:
    """'convert 5 km to miles' -> (5.0, 'km', 3.107, 'miles')."""
    match = _CONVERT_RE.search(words_to_digits(normalize_text(text)))
    if not match:
        return None
    value, src, dst = float(match.group(1)), match.group(2), match.group(3)
    if src in _TEMPERATURE and dst in _TEMPERATURE:
        return value, src, _temperature(value, _TEMPERATURE[src], _TEMPERATURE[dst]), dst
    if src in _LINEAR_UNITS and dst in _LINEAR_UNITS:
        (dim_a, f_a), (dim_b, f_b) = _LINEAR_UNITS[src], _LINEAR_UNITS[dst]
        if dim_a == dim_b:
            return value, src, value * f_a / f_b, dst
    return None


# --- free-text slots ---------------------------------------------------------------

_STOP_TAIL = re.compile(r"\b(today|tonight|tomorrow|right now|now|please|this week)\b.*$")


def extract_place(text: str, prepositions: tuple[str, ...] = ("in", "for", "at")) -> str | None:
    """'what's the weather in new york today' -> 'New York'."""
    text = normalize_text(text)
    pattern = rf"\b(?:{'|'.join(prepositions)})\s+([a-z][a-z\s]*)"
    for match in re.finditer(pattern, text):
        place = _STOP_TAIL.sub("", match.group(1)).strip()
        if place and place not in {"the", "a", "my", "me"}:
            return place.title()
    return None


def extract_destination(text: str) -> str | None:
    """'how do I get to the airport' -> 'The Airport'."""
    text = normalize_text(text)
    match = re.search(r"\b(?:to|towards)\s+(?:the\s+)?([a-z0-9][a-z0-9\s]*)", text)
    if not match:
        return None
    destination = _STOP_TAIL.sub("", match.group(1)).strip()
    return destination.title() or None


KNOWN_SITES = {
    "youtube": "https://www.youtube.com",
    "you tube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "git hub": "https://github.com",
    "gmail": "https://mail.google.com",
    "wikipedia": "https://www.wikipedia.org",
    "linkedin": "https://www.linkedin.com",
    "netflix": "https://www.netflix.com",
    "reddit": "https://www.reddit.com",
    "stack overflow": "https://stackoverflow.com",
    "spotify": "https://open.spotify.com",
}
_DOMAIN_RE = re.compile(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|dev|rs|ai|edu))\b")


def extract_website(text: str) -> tuple[str, str] | None:
    """Return (display name, url) for a known site or a spoken domain name."""
    lowered = str(text).lower()
    domain = _DOMAIN_RE.search(lowered)
    if domain:
        return domain.group(1), f"https://{domain.group(1)}"
    normalized = normalize_text(text)
    for name, url in KNOWN_SITES.items():
        if re.search(rf"\b{name}\b", normalized):
            return name.replace(" ", "").title(), url
    return None


_LIST_ADD_RE = re.compile(
    r"\b(?:add|put|include)\s+(.+?)\s+(?:to|on|onto|in)\s+(?:my\s+|the\s+)?"
    r"(?:shopping\s+|grocery\s+|to do\s+|todo\s+)?(?:list|cart)\b"
)
_NEED_RE = re.compile(r"\b(?:i need(?: to buy)?|buy|we need)\s+(?:some\s+)?(.+)$")


def extract_list_item(text: str) -> str | None:
    """'add milk and eggs to my shopping list' -> 'milk and eggs'. Commas are kept so the
    skill can split 'milk, eggs and bread' into separate items."""
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9:',\s]", " ", str(text).lower())).strip()
    match = _LIST_ADD_RE.search(text) or _NEED_RE.search(text)
    if not match:
        return None
    item = re.sub(r"^(some|a|an|the)\s+", "", match.group(1)).strip()
    return item or None


_ITEM_SEP = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+")


def split_items(text: str) -> list[str]:
    """'milk, eggs and bread' -> ['milk', 'eggs', 'bread']. Expects normalised or raw text."""
    parts = [re.sub(r"^(some|a|an|the)\s+", "", p.strip()) for p in _ITEM_SEP.split(text)]
    return [p for p in parts if p]


def extract_items_after_with(text: str) -> list[str]:
    """Items listed after 'with' / 'containing', keeping the commas Whisper puts in lists."""
    lowered = str(text).lower()
    match = re.search(r"\b(?:with|containing|including)\s+(.+)$", lowered)
    if not match:
        return []
    cleaned = re.sub(r"[^a-z0-9,'\s]", " ", match.group(1))
    cleaned = re.sub(
        r"\b(in|as|to|into)\s+(a\s+)?(word\s+)?(doc|docx|document|file)\b.*$", "", cleaned
    )
    return split_items(cleaned)


def extract_after(text: str, keywords: tuple[str, ...]) -> str | None:
    """Text following the first keyword, e.g. 'remind me to call mom' -> 'call mom'."""
    text = normalize_text(text)
    for keyword in keywords:
        match = re.search(rf"\b{re.escape(keyword)}\s+(.+)$", text)
        if match:
            return match.group(1).strip()
    return None
