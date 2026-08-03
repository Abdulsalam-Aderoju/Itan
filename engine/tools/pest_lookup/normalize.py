"""Pest-name canonicalization, shared by build_db.py and pest_lookup.py.

corpus/05_structure_extract.py's PEST_NAME_RE captures a whole noun phrase
around a pest keyword (e.g. "usually attacked by thrips", "Symptoms of
thrips", "season thrips"), not a clean pest name -- exact-match lookup on
that raw text would fail almost every real query. This module reduces any
such phrase down to the single keyword it was matched against in the first
place, so both the ETL (grouping raw rows) and the query function
(normalizing caller input) agree on the same canonical vocabulary.

PEST_KEYWORDS must be kept in sync with corpus/05_structure_extract.py's
list of the same name -- it cannot be imported directly because that
module's filename starts with a digit, which is not a valid Python import
target.
"""
import re

PEST_KEYWORDS = [
    "weevil", "borer", "aphid", "nematode", "striga", "armyworm", "thrips", "mealybug",
    "mite", "beetle", "caterpillar", "worm", "whitefly", "moth", "blight", "rust",
    "wilt", "mosaic virus", "anthracnose", "smut", "rot", "leaf spot",
]

# Compiled once: word-boundary matching matters here for the same reason
# it matters elsewhere in this corpus (e.g. "rice" inside "price") --
# "\bworm\b" will NOT match the "worm" inside "armyworm" since there is no
# boundary between "army" and "worm", so armyworm and worm stay distinct
# canonical pests without needing special-case ordering.
_KEYWORD_RES = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.I)) for kw in PEST_KEYWORDS]


def canonicalize_pest_name(text: str) -> str | None:
    """Return the canonical pest keyword found in `text`, or None if no
    known keyword is present.

    If more than one keyword matches, the earliest (leftmost) match wins,
    since that is the keyword the original phrase was really centered on.
    """
    if not text:
        return None
    best: tuple[int, str] | None = None
    for kw, regex in _KEYWORD_RES:
        m = regex.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), kw)
    return best[1] if best else None
