#!/usr/bin/env python
"""
Stage 5 (build): deterministic regex/pattern extraction into the six
structured SQLite tables. No LLM calls anywhere in this file.

Every extracted row carries a source_id (foreign key into sources.csv) and a
needs_review flag: low-confidence extractions are NOT discarded, they are
flagged so a human can verify/correct them later.

Input: corpus/clean/<source_id>.txt + corpus/sources.csv
Output: corpus/structured.db (SQLite)

Runnable standalone: python corpus/05_structure_extract.py
"""
from __future__ import annotations

import csv
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS_DIR, CLEAN_DIR, CROPS, ZONES, print_elapsed_checkpoint, progress_line  # noqa: E402

SOURCES_CSV = CORPUS_DIR / "sources.csv"
DB_PATH = CORPUS_DIR / "structured.db"

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
MONTH_RE_FRAG = "|".join(MONTHS)
# A month name, optionally preceded by "early/late/mid" and/or followed by
# a day-of-month number ("January 15", "5th March") -- extension prose and
# academic papers routinely give exact planting/harvest dates this way,
# not just bare month names. One capture group: the month name itself.
MONTH_WITH_DAY = (
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)?\s+)?"
    rf"(?:early|late|mid[\s-]?|(?:first|second|third|fourth|last)\s+week\s+of\s*)?\s*"
    rf"({MONTH_RE_FRAG})(?:\s+\d{{1,2}}(?:st|nd|rd|th)?)?"
)
# Up to 4 words of gap (never crossing a period, so it can't bridge two
# sentences) between an activity verb and "between" -- covers "plant
# soybean between", "planting date was the first day between", etc.,
# without needing to enumerate every phrasing by hand.
WORD_GAP = r"(?:\s+[^\s.]+){0,4}?"

NIGERIAN_STATES = [
    "abia", "adamawa", "akwa ibom", "anambra", "bauchi", "bayelsa", "benue", "borno",
    "cross river", "delta", "ebonyi", "edo", "ekiti", "enugu", "gombe", "imo", "jigawa",
    "kaduna", "kano", "katsina", "kebbi", "kogi", "kwara", "lagos", "nasarawa", "niger",
    "ogun", "ondo", "osun", "oyo", "plateau", "rivers", "sokoto", "taraba", "yobe",
    "zamfara", "fct", "abuja",
]

PEST_KEYWORDS = [
    "weevil", "borer", "aphid", "nematode", "striga", "armyworm", "thrips", "mealybug",
    "mite", "beetle", "caterpillar", "worm", "whitefly", "moth", "blight", "rust",
    "wilt", "mosaic virus", "anthracnose", "smut", "rot", "leaf spot",
]
GROWTH_STAGES = ["seedling", "vegetative", "flowering", "grain filling", "maturity", "germination", "tillering"]

CONFIDENCE_THRESHOLD = 0.7
TIGHT_OVERRIDE_DISTANCE = 60  # a mention this close overrides the doc's primary crop

# Set once per document by main() before its extractors run. Documents in
# this corpus are overwhelmingly single-crop guides/factsheets, so when a
# document's sources.csv row names exactly one crop, that crop is preferred
# over whatever crop happens to sit nearest a given regex match -- otherwise
# an incidental mention (e.g. cowpea suggested as an intercrop in a maize
# guide) can outrank the actual subject crop just by being a few words
# closer. A mention of a DIFFERENT crop within TIGHT_OVERRIDE_DISTANCE chars
# still wins, since that tight a match usually means the sentence really is
# about that other crop (e.g. a per-crop rate table row).
_DOC_CROPS: set[str] = set()

# Minimum total mentions, and minimum lead over the runner-up crop, before
# infer_doc_crop() below trusts a content-derived single-crop guess.
# 20 is not arbitrary: checked mention counts across every document this
# fallback fired on -- every correct single-crop attribution had 37-317
# mentions of the winning crop, but the one false positive found (a
# "Potato Research" journal paper mentioning "maize" only in passing as a
# rotation crop, 367 "potato" mentions vs 12 "maize") had just 12. 20 sits
# in the gap between those two clusters. CROPS only lists the 10 target
# crops, so this fallback has no way to know a document's TRUE subject is
# a non-target crop like potato that outnumbers every target-crop mention
# -- the high absolute floor is what keeps a passing mention of a target
# crop in an off-topic paper from winning by default.
DOC_CROP_MIN_MENTIONS = 20
DOC_CROP_MIN_LEAD_RATIO = 3


def infer_doc_crop(text: str) -> set[str]:
    """Fallback for when sources.csv's crops_covered is empty -- true for
    most of this corpus, since that field is only populated from filename
    matching (common for hand-named extension guides, essentially never
    for the DOI-named academic papers OpenAlex/Semantic Scholar make up
    the bulk of the corpus). Scans the whole document for crop-name
    mentions and, if exactly one crop clearly dominates, returns it as a
    single-element set so it flows into the same _DOC_CROPS single-crop-
    document contract find_window_crop() already trusts -- just derived
    from content instead of the filename. Deliberately conservative (a
    real minimum mention count, not just "most mentions of at least one")
    so a comparative multi-crop study, where several crops are mentioned
    at similar rates, correctly stays ambiguous (empty set) instead of
    getting attributed to whichever crop happens to be named first."""
    lower = text.lower()
    counts = {c: len(re.findall(rf"\b{re.escape(c)}\b", lower)) for c in CROPS}
    nonzero = {c: n for c, n in counts.items() if n > 0}
    if not nonzero:
        return set()
    top_crop, top_count = max(nonzero.items(), key=lambda kv: kv[1])
    if top_count < DOC_CROP_MIN_MENTIONS:
        return set()
    runner_up = max((n for c, n in nonzero.items() if c != top_crop), default=0)
    if runner_up * DOC_CROP_MIN_LEAD_RATIO <= top_count:
        return {top_crop}
    return set()


def _crop_distances(text: str, pos: int, window: int) -> dict[str, int]:
    """Word-boundary matching matters here too: "rice" is a plain substring
    of "price", which shows up constantly in extension text discussing
    market prices, and a naive substring search would misattribute those
    rows to rice."""
    lo, hi = max(0, pos - window), pos + window
    ctx = text[lo:hi].lower()
    rel_pos = pos - lo
    distances: dict[str, int] = {}
    for c in CROPS:
        for m in re.finditer(rf"\b{re.escape(c)}\b", ctx):
            dist = abs(m.start() - rel_pos)
            if c not in distances or dist < distances[c]:
                distances[c] = dist
    return distances


def find_window_crop(text: str, pos: int, window: int = 400) -> str | None:
    """Return the crop mentioned closest to pos within the window, biased
    toward the document's known primary crop (see _DOC_CROPS docstring).

    For single-crop documents the doc-level crop is the default even when
    it doesn't happen to fall inside `window` itself (e.g. it was named in
    the opening paragraph, far from a rate/spacing figure mentioned later)
    -- it only loses to a DIFFERENT crop mentioned within
    TIGHT_OVERRIDE_DISTANCE chars of the match, which is a strong signal
    the sentence is really about that other crop."""
    distances = _crop_distances(text, pos, window)

    if len(_DOC_CROPS) == 1:
        doc_crop = next(iter(_DOC_CROPS))
        competing = {c: d for c, d in distances.items() if c != doc_crop}
        if competing and min(competing.values()) <= TIGHT_OVERRIDE_DISTANCE:
            return min(competing, key=competing.get)
        return doc_crop

    if not distances:
        return None
    return min(distances, key=distances.get)


def find_window_zone_or_state(text: str, pos: int, window: int = 300) -> tuple[str | None, str | None]:
    """Word-boundary matching is required here: "niger" (the state) is a
    plain substring of "nigeria" (the country, mentioned constantly), so a
    naive `in` check on every document flags a nonexistent Niger State
    mention on almost every row."""
    ctx = text[max(0, pos - window):pos + window].lower()
    zone = next((z for z in ZONES if re.search(rf"\b{re.escape(z.lower())}\b", ctx)), None)
    state = next((s for s in NIGERIAN_STATES if re.search(rf"\b{re.escape(s)}\b", ctx)), None)
    return zone, state


def make_conf(has_crop: bool, has_place: bool, strict_pattern: bool) -> float:
    # crop linkage matters most -- a row with no identifiable crop is of
    # limited use downstream even if the pattern match itself was clean, so
    # it is weighted heavily enough to land below CONFIDENCE_THRESHOLD on
    # its own and be routed to the needs_review queue.
    score = 0.3
    if strict_pattern:
        score += 0.2
    if has_crop:
        score += 0.35
    if has_place:
        score += 0.15
    return round(min(score, 1.0), 2)


# ---------------------------------------------------------------------------
# Per-table extractors. Each returns a list of dict rows (without source_id,
# added by caller) for one document's cleaned text.
# ---------------------------------------------------------------------------

def _window_pattern(verb_stem: str) -> str:
    """'[verb] ... between/from X to/and Y', tolerating a few words of gap
    (crop name, aux verbs, 'date was the first day', etc.) between the verb
    and the connector -- shared shape for planting_window/harvest_window
    across every verb stem (plant, sow, harvest) rather than one-off regexes
    per phrasing. Requires the sentence to actually start with the activity
    verb, so e.g. a "sow(?:ing)?" pattern does not match an unrelated
    "harvested between X and Y" sentence a few words later."""
    return rf"{verb_stem}{WORD_GAP}\s+(?:between|from)\s+{MONTH_WITH_DAY}\s*(?:to|and|-|–)\s*{MONTH_WITH_DAY}"


def extract_crop_calendar(text: str) -> list[dict]:
    rows = []
    patterns = [
        (re.compile(rf"plant(?:ing)?\s+in\s+({MONTH_RE_FRAG})", re.I), "plant_in_month", True),
        (re.compile(rf"planting\s+window[:\s]+({MONTH_RE_FRAG})\s*(?:to|-|–)\s*({MONTH_RE_FRAG})", re.I), "planting_window", True),
        # Extension-guide prose rarely says "planting window: X to Y" -- it
        # says "planting is carried out between March and May", "farmers
        # plant soybean between mid-June and July", "the planting date was
        # the first day between May 1 and June 30". WORD_GAP absorbs the
        # aux verbs / crop name / filler words between the activity verb
        # and the connector without needing to enumerate every phrasing by
        # hand, while still requiring the sentence to actually be about
        # planting (starts with "plant..."), so it does NOT match a nearby
        # "land preparation is done between X and Y" sentence about a
        # different, earlier activity.
        (re.compile(_window_pattern(r"plant(?:ing)?"), re.I), "planting_window", True),
        # "Sow/sown/sowing" is the dominant term for planting in academic
        # papers (far more common than "plant" itself in this corpus) --
        # "sown in March", "sowing in early July", "Maize was sown in
        # early October".
        (re.compile(rf"sow(?:n|ing)?\s+in\s+({MONTH_RE_FRAG})", re.I), "plant_in_month", True),
        # "Sowing from June 1 to July 5", "sown between March and April",
        # "Time of sowing: Mid June to July" (table-cell style, no
        # between/from at all -- just a colon or bare space).
        (re.compile(_window_pattern(r"sow(?:n|ing)?"), re.I), "planting_window", True),
        (re.compile(
            rf"(?:time\s+of\s+)?sow(?:ing)?\s*[:\s]\s*{MONTH_WITH_DAY}\s*(?:to|-|–)\s*{MONTH_WITH_DAY}",
            re.I,
        ), "planting_window", True),
        (re.compile(rf"first\s+rains?[:\s]+({MONTH_RE_FRAG})", re.I), "first_rains", True),
        # "with the first rains from late February to early March" --
        # same fact as the colon-form above, common phrasing in
        # land-preparation-timing paragraphs.
        (re.compile(
            rf"first\s+rains?\s+from\s+(?:early|late|mid)?\s*({MONTH_RE_FRAG})\s*"
            rf"(?:to|and|-|–)?\s*(?:early|late|mid)?\s*({MONTH_RE_FRAG})?",
            re.I,
        ), "first_rains", True),
        # Harvest-window extraction did not exist at all before this --
        # "harvest between March and May" mirrors the planting pattern
        # above. crop_calendar tool's _best_window() already looks for
        # this exact activity name.
        (re.compile(_window_pattern(r"harvest(?:ed|ing)?"), re.I), "harvest_window", True),
    ]
    for regex, activity, strict in patterns:
        for m in regex.finditer(text):
            crop = find_window_crop(text, m.start())
            zone, state = find_window_zone_or_state(text, m.start())
            groups = m.groups()
            month_start = groups[0].title() if groups[0] else ""
            month_end = groups[1].title() if len(groups) > 1 and groups[1] else ""
            rows.append({
                "crop": crop or "",
                "zone": zone or "",
                "state": state or "",
                "activity": activity,
                "month_start": month_start,
                "month_end": month_end,
                "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
                "confidence": make_conf(bool(crop), bool(zone or state), strict),
            })
    return rows


def extract_spacing(text: str) -> list[dict]:
    rows = []
    patterns = [
        (re.compile(r"(\d+(?:\.\d+)?)\s*cm\s*[x×X]\s*(\d+(?:\.\d+)?)\s*cm"), "spacing_cm_x_cm"),
        (re.compile(r"(\d[\d,]*)\s*plants?\s*(?:per|/)\s*(?:hectare|ha)\b", re.I), "plants_per_ha"),
        (re.compile(r"row\s+spacing\s+of\s+(\d+(?:\.\d+)?)\s*(cm|m)?", re.I), "row_spacing"),
    ]
    for regex, kind in patterns:
        for m in regex.finditer(text):
            crop = find_window_crop(text, m.start())
            spacing_cm, plants_per_ha, row_spacing_cm = "", "", ""
            if kind == "spacing_cm_x_cm":
                spacing_cm = f"{m.group(1)}x{m.group(2)}"
            elif kind == "plants_per_ha":
                plants_per_ha = m.group(1).replace(",", "")
            elif kind == "row_spacing":
                row_spacing_cm = m.group(1)
            rows.append({
                "crop": crop or "",
                "spacing_cm": spacing_cm,
                "plants_per_ha": plants_per_ha,
                "row_spacing_cm": row_spacing_cm,
                "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
                "confidence": make_conf(bool(crop), False, True),
            })
    return rows


# "kg/ha", "kg ha", "kg ha-1", "kg ha−1" (unicode minus, not a hyphen),
# and "kg ha⁻¹" (true superscript minus-one) are all the same unit --
# extension guides write "kg/ha", academic papers overwhelmingly write one
# of the "kg ha<exponent>" forms instead.
KG_HA_UNIT_RE = r"kg\s*(?:/\s*ha|ha)(?:\s*[-−]\s*1|\s*⁻¹)?"

# Stages whose pattern has no separate fertilizer-type capture group, but
# where the phrasing itself names the nutrient (e.g. "nitrogen rate of ..."
# is unambiguously about nitrogen even with no second group to capture it).
STAGE_IMPLIED_FERT_TYPE = {
    "basal_dressing": "urea",
    "n_application": "nitrogen",
    "nitrogen_rate": "nitrogen",
    "kg_n_per_ha": "nitrogen",
    "kg_n_slash_ha": "nitrogen",
}


def extract_fertilizer_rate(text: str) -> list[dict]:
    rows = []
    patterns = [
        (re.compile(r"apply\s+(\d+(?:\.\d+)?)\s*kg\s*/\s*ha\s+of\s+([A-Za-z0-9\-: ]{2,30})", re.I), "apply_kg_ha"),
        (re.compile(r"basal\s+dressing[:\s]+(\d+(?:\.\d+)?)\s*kg", re.I), "basal_dressing"),
        (re.compile(r"top[\s-]?dress(?:ing)?\s+with\s+(\d+(?:\.\d+)?)\s*kg\s+([a-zA-Z0-9: ]{2,20})", re.I), "top_dress"),
        # Academic-paper phrasings -- extension guides say "apply X kg/ha of
        # Y", papers overwhelmingly say one of these instead.
        (re.compile(rf"N\s+application\s+of\s+(\d+(?:\.\d+)?)\s*{KG_HA_UNIT_RE}", re.I), "n_application"),
        (re.compile(rf"nitrogen\s+rate\s+of\s+(\d+(?:\.\d+)?)\s*{KG_HA_UNIT_RE}", re.I), "nitrogen_rate"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*kg\s*N\s+per\s+hectare", re.I), "kg_n_per_ha"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*kg\s*N\s*/\s*ha\b", re.I), "kg_n_slash_ha"),
        (re.compile(rf"applied\s+at\s+(\d+(?:\.\d+)?)\s*{KG_HA_UNIT_RE}", re.I), "applied_at_kg_ha"),
        # Looser than applied_at_kg_ha above -- "applied at 50 kg" with no
        # per-hectare unit attached at all, still common in results tables.
        (re.compile(r"applied\s+at\s+(\d+(?:\.\d+)?)\s*kg\b", re.I), "applied_at_kg"),
        (re.compile(r"fertili[sz]ed\s+with\s+(\d+(?:\.\d+)?)\s*kg\b", re.I), "fertilized_with"),
        (re.compile(rf"recommended\s+rate\s+of\s+(\d+(?:\.\d+)?)\s*(?:{KG_HA_UNIT_RE})?", re.I), "recommended_rate"),
        (re.compile(r"at\s+a\s+rate\s+of\s+(\d+(?:\.\d+)?)\s*(?:kg|L)\s*/\s*ha\b", re.I), "at_a_rate_of"),
        # Bare "<number> kg/ha"-style unit with no trigger phrase at all --
        # deliberately broad (this is what finally moves fertilizer_rate off
        # near-zero for this corpus), so it's scored lower than the other
        # patterns above: it has no confirming context, and a bare kg/ha
        # figure is just as likely to be a reported yield as an application
        # rate. Low confidence routes it straight to needs_review rather
        # than silently trusting it.
        (re.compile(rf"(\d+(?:\.\d+)?)\s*{KG_HA_UNIT_RE}", re.I), "bare_kg_ha"),
    ]
    for regex, stage in patterns:
        for m in regex.finditer(text):
            crop = find_window_crop(text, m.start())
            rate = m.group(1)
            fert_type = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else STAGE_IMPLIED_FERT_TYPE.get(stage, "")
            rows.append({
                "crop": crop or "",
                "fertilizer_type": fert_type,
                "rate_kg_ha": rate,
                "application_stage": stage,
                "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
                "confidence": make_conf(bool(crop), bool(fert_type), stage != "bare_kg_ha"),
            })
    return rows


def extract_agrochemical(text: str) -> list[dict]:
    rows = []
    # Product names are proper nouns and often multi-word ("Emamectin
    # benzoate"), so the capture allows up to 3 words. The leading
    # [A-Z] is NOT case-insensitive -- capitalization is the signal that
    # distinguishes a product name from an ordinary sentence fragment, so
    # re.I is scoped only to the unit tokens, not the whole pattern.
    rate_re = re.compile(
        r"([A-Z][A-Za-z0-9\-®]*(?:\s+[A-Za-z][A-Za-z0-9\-®]*){0,2})\s+at\s+"
        r"(\d+(?:\.\d+)?)\s*(?i:(ml|g))\s*/\s*(?i:(litre|liter|l))\b"
    )
    litres_ha_re = re.compile(r"apply\s+(\d+(?:\.\d+)?)\s*litres?\s*/\s*ha", re.I)
    # Was requiring "interval" to be followed immediately by ":" or
    # whitespace+digits, which never matched "interval OF 7 days" -- the
    # overwhelmingly common academic phrasing. Now accepts "of", ":", or
    # just whitespace before the number.
    phi_re = re.compile(r"pre[\s-]?harvest\s+interval\s*(?:of|:)?\s*(\d+)\s*days?", re.I)
    # "PHI" abbreviation form ("PHI 7 days" / "PHI: 7 days" / "PHI of 7
    # days"). Deliberately NOT case-insensitive -- "phi" lowercase is the
    # Greek letter and shows up in unrelated statistical/math notation in
    # these papers, but the abbreviation is always written uppercase.
    phi_abbrev_re = re.compile(r"PHI\s*(?:of|:)?\s*(\d+)\s*days?")
    # Academic-paper dose/concentration phrasings, distinct from the
    # extension-guide "[product] at X ml/litre" form rate_re already covers.
    dose_gl_re = re.compile(r"dose\s+of\s+(\d+(?:\.\d+)?)\s*g\s*/\s*L\b", re.I)
    concentration_mll_re = re.compile(r"concentration\s+of\s+(\d+(?:\.\d+)?)\s*ml\s*/\s*L\b", re.I)
    applied_at_lha_re = re.compile(r"applied\s+at\s+(\d+(?:\.\d+)?)\s*L\s*/\s*ha\b", re.I)
    rate_of_lha_re = re.compile(r"at\s+a\s+rate\s+of\s+(\d+(?:\.\d+)?)\s*L\s*/\s*ha\b", re.I)

    for m in rate_re.finditer(text):
        crop = find_window_crop(text, m.start())
        rows.append({
            "product_name": m.group(1).strip(),
            "active_ingredient": "",
            "crop": crop or "",
            "rate": m.group(2),
            "rate_unit": f"{m.group(3)}/{m.group(4)}",
            "pre_harvest_interval_days": "",
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), False, True),
        })
    for m in litres_ha_re.finditer(text):
        crop = find_window_crop(text, m.start())
        rows.append({
            "product_name": "", "active_ingredient": "", "crop": crop or "",
            "rate": m.group(1), "rate_unit": "litres/ha", "pre_harvest_interval_days": "",
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), False, True) - 0.1,
        })
    for m in phi_re.finditer(text):
        crop = find_window_crop(text, m.start())
        rows.append({
            "product_name": "", "active_ingredient": "", "crop": crop or "",
            "rate": "", "rate_unit": "", "pre_harvest_interval_days": m.group(1),
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), False, True),
        })
    for m in phi_abbrev_re.finditer(text):
        crop = find_window_crop(text, m.start())
        rows.append({
            "product_name": "", "active_ingredient": "", "crop": crop or "",
            "rate": "", "rate_unit": "", "pre_harvest_interval_days": m.group(1),
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), False, True),
        })
    for m in dose_gl_re.finditer(text):
        crop = find_window_crop(text, m.start())
        rows.append({
            "product_name": "", "active_ingredient": "", "crop": crop or "",
            "rate": m.group(1), "rate_unit": "g/L", "pre_harvest_interval_days": "",
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), False, True) - 0.1,
        })
    for m in concentration_mll_re.finditer(text):
        crop = find_window_crop(text, m.start())
        rows.append({
            "product_name": "", "active_ingredient": "", "crop": crop or "",
            "rate": m.group(1), "rate_unit": "ml/L", "pre_harvest_interval_days": "",
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), False, True) - 0.1,
        })
    for m in applied_at_lha_re.finditer(text):
        crop = find_window_crop(text, m.start())
        rows.append({
            "product_name": "", "active_ingredient": "", "crop": crop or "",
            "rate": m.group(1), "rate_unit": "L/ha", "pre_harvest_interval_days": "",
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), False, True) - 0.1,
        })
    for m in rate_of_lha_re.finditer(text):
        crop = find_window_crop(text, m.start())
        rows.append({
            "product_name": "", "active_ingredient": "", "crop": crop or "",
            "rate": m.group(1), "rate_unit": "L/ha", "pre_harvest_interval_days": "",
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), False, True) - 0.1,
        })
    return rows


def extract_variety(text: str) -> list[dict]:
    rows = []
    maturity_re = re.compile(r"maturity[:\s]+(\d+)\s*(?:-\s*(\d+))?\s*days?", re.I)
    # Same reasoning as the agrochemical product-name capture: capitalization
    # is what marks this token as a variety code/name, so it must stay
    # case-sensitive even though "released variety" itself is matched
    # case-insensitively via the scoped (?i:...) group.
    released_re = re.compile(r"([A-Z][A-Za-z0-9\-]{1,20})\s+(?:is\s+a\s+)?(?i:released\s+variety)")
    recommended_re = re.compile(r"recommended\s+for\s+([A-Za-z /]{3,40})", re.I)

    for m in maturity_re.finditer(text):
        crop = find_window_crop(text, m.start())
        zone, _state = find_window_zone_or_state(text, m.start())
        days = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
        rows.append({
            "crop": crop or "", "variety_name": "", "maturity_days": days, "zone": zone or "",
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), bool(zone), True),
        })
    for m in released_re.finditer(text):
        crop = find_window_crop(text, m.start())
        zone, _state = find_window_zone_or_state(text, m.start())
        rows.append({
            "crop": crop or "", "variety_name": m.group(1), "maturity_days": "", "zone": zone or "",
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), bool(zone), True),
        })
    for m in recommended_re.finditer(text):
        crop = find_window_crop(text, m.start())
        candidate_zone = m.group(1).strip()
        zone_match = next((z for z in ZONES if z.lower() in candidate_zone.lower()), candidate_zone)
        rows.append({
            "crop": crop or "", "variety_name": "", "maturity_days": "", "zone": zone_match,
            "raw_text": text[max(0, m.start() - 60):m.end() + 60].strip(),
            "confidence": make_conf(bool(crop), True, False),
        })
    return rows


PEST_NAME_RE = re.compile(
    r"([A-Z][a-z]+(?:\s[A-Za-z]+){0,2}\s(?:" + "|".join(PEST_KEYWORDS) + r"))", re.I
)
SECTION_RE = {
    "symptoms": re.compile(r"symptoms?[:\s]+(.{20,300}?)(?=\n[A-Z][a-z]+:|\n\n|$)", re.I | re.S),
    "cultural_control": re.compile(r"cultural\s+control[:\s]+(.{10,300}?)(?=\n[A-Z][a-z]+:|\n\n|$)", re.I | re.S),
    "chemical_control": re.compile(r"chemical\s+control[:\s]+(.{10,300}?)(?=\n[A-Z][a-z]+:|\n\n|$)", re.I | re.S),
}


def extract_pest(text: str) -> list[dict]:
    rows = []
    for m in PEST_NAME_RE.finditer(text):
        window = text[m.start():m.start() + 800]
        crop = find_window_crop(text, m.start())
        symptoms = next(iter(SECTION_RE["symptoms"].findall(window)), "")
        cultural = next(iter(SECTION_RE["cultural_control"].findall(window)), "")
        chemical = next(iter(SECTION_RE["chemical_control"].findall(window)), "")
        stage = next((s for s in GROWTH_STAGES if s in window.lower()), "")

        fields_found = sum(bool(x) for x in (symptoms, cultural, chemical, stage))
        conf = 0.3 + 0.15 * fields_found + (0.1 if crop else 0)
        rows.append({
            "crop": crop or "",
            "pest_name": m.group(1).strip(),
            "symptoms": symptoms.strip().replace("\n", " ")[:400],
            "growth_stage": stage,
            "cultural_control": cultural.strip().replace("\n", " ")[:400],
            "chemical_control": chemical.strip().replace("\n", " ")[:400],
            "raw_text": window[:300].replace("\n", " "),
            "confidence": round(min(conf, 1.0), 2),
        })
    return rows


SCHEMA = {
    "crop_calendar": ["source_id", "crop", "zone", "state", "activity", "month_start", "month_end", "raw_text", "confidence", "needs_review"],
    "spacing": ["source_id", "crop", "spacing_cm", "plants_per_ha", "row_spacing_cm", "raw_text", "confidence", "needs_review"],
    "fertilizer_rate": ["source_id", "crop", "fertilizer_type", "rate_kg_ha", "application_stage", "raw_text", "confidence", "needs_review"],
    "agrochemical": ["source_id", "product_name", "active_ingredient", "crop", "rate", "rate_unit", "pre_harvest_interval_days", "raw_text", "confidence", "needs_review"],
    "variety": ["source_id", "crop", "variety_name", "maturity_days", "zone", "raw_text", "confidence", "needs_review"],
    "pest": ["source_id", "crop", "pest_name", "symptoms", "growth_stage", "cultural_control", "chemical_control", "raw_text", "confidence", "needs_review"],
}

EXTRACTORS = {
    "crop_calendar": extract_crop_calendar,
    "spacing": extract_spacing,
    "fertilizer_rate": extract_fertilizer_rate,
    "agrochemical": extract_agrochemical,
    "variety": extract_variety,
    "pest": extract_pest,
}


def init_db(conn: sqlite3.Connection):
    # This script always processes every file in CLEAN_DIR on every run (not
    # just new ones since the last run), so the tables must be rebuilt from
    # scratch each time -- DROP + CREATE, not CREATE IF NOT EXISTS. Without
    # this, rerunning on top of an existing structured.db silently duplicates
    # every already-processed document's rows (found and fixed 2026-08-02:
    # an earlier double-run had inflated the pest table to ~12,200 rows
    # before this fix, vs. the true ~6,300 from a single clean pass).
    for table in SCHEMA:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    for table, cols in SCHEMA.items():
        col_defs = ", ".join(f'"{c}" TEXT' if c != "confidence" else '"confidence" REAL' for c in cols)
        conn.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})')
    conn.commit()


def main():
    if not SOURCES_CSV.exists():
        print(f"[structure] ERROR: {SOURCES_CSV} not found. Run 01_fetch.py first.")
        sys.exit(1)
    if not any(CLEAN_DIR.glob("*.txt")):
        print(f"[structure] ERROR: no cleaned documents in {CLEAN_DIR}. Run 03_clean.py first.")
        sys.exit(1)

    with open(SOURCES_CSV, newline="", encoding="utf-8") as fh:
        sources = {r["source_id"]: r for r in csv.DictReader(fh)}

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    table_counts = {t: 0 for t in SCHEMA}
    confidences: list[float] = []
    zero_yield_docs = []

    files = sorted(CLEAN_DIR.glob("*.txt"))
    total = len(files)
    print(f"[structure] {total} cleaned documents to parse")
    start_time = time.time()

    for i, f in enumerate(files, start=1):
        global _DOC_CROPS
        sid = f.stem
        if sid not in sources:
            progress_line("structure", i, total, f.name, ok=False, reason="not in sources.csv, refusing to insert rows with no valid source_id")
            print_elapsed_checkpoint("structure", i, total, start_time)
            continue

        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            _DOC_CROPS = {c for c in (sources[sid].get("crops_covered") or "").split(";") if c}
            if not _DOC_CROPS:
                _DOC_CROPS = infer_doc_crop(text)

            doc_row_count = 0
            for table, extractor in EXTRACTORS.items():
                rows = extractor(text)
                cols = SCHEMA[table]
                for r in rows:
                    conf = r.get("confidence", 0.5)
                    confidences.append(conf)
                    r["source_id"] = sid
                    r["needs_review"] = "1" if conf < CONFIDENCE_THRESHOLD else "0"
                    values = [r.get(c, "") for c in cols]
                    placeholders = ", ".join("?" for _ in cols)
                    conn.execute(
                        f'INSERT INTO "{table}" ({", ".join(cols)}) VALUES ({placeholders})', values
                    )
                    table_counts[table] += 1
                    doc_row_count += 1
            conn.commit()

            if doc_row_count == 0:
                zero_yield_docs.append(sid)
            progress_line("structure", i, total, f.name, ok=True, reason=f"{doc_row_count} structured rows")
        except Exception as exc:
            conn.rollback()
            progress_line("structure", i, total, f.name, ok=False, reason=str(exc))

        print_elapsed_checkpoint("structure", i, total, start_time)

    conn.close()

    total_rows = sum(table_counts.values())
    print(f"\n[structure] DONE. {total_rows} rows written to {DB_PATH}")
    print("[structure] Rows extracted per table:")
    for t, n in table_counts.items():
        print(f"    {t:18s} {n}")

    if confidences:
        low = sum(1 for c in confidences if c < CONFIDENCE_THRESHOLD)
        mid = sum(1 for c in confidences if CONFIDENCE_THRESHOLD <= c < 0.9)
        high = sum(1 for c in confidences if c >= 0.9)
        print(f"[structure] Confidence distribution: low(<{CONFIDENCE_THRESHOLD})={low} "
              f"mid({CONFIDENCE_THRESHOLD}-0.9)={mid} high(>=0.9)={high}")
        print(f"[structure] needs_review rows: {low} of {total_rows} "
              f"({(low / total_rows * 100) if total_rows else 0:.1f}%) -- this is the Week 3 manual review queue")
    print(f"[structure] Documents yielding zero structured rows: {len(zero_yield_docs)}")
    for sid in zero_yield_docs[:20]:
        print(f"    - {sid}")


if __name__ == "__main__":
    main()
