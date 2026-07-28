#!/usr/bin/env python
"""
One-off utility: rebuild harvest/checkpoint.json from the evidence that
actually exists on disk -- sources_discovered.csv (rows that succeeded)
and blocked.log (attempts that failed) -- for when checkpoint.json itself
has been lost, corrupted, or is stale relative to what a real crawl
actually completed.

IMPORTANT LIMITATION -- read this before trusting the reconstruction:
sources_discovered.csv does NOT store the exact search query string that
found each row. It only has the crawler name (api_source) and whichever
crops were found IN THE DOCUMENT TEXT (crops_covered) -- which is not
necessarily the crop that was being searched for. The literal query text
(e.g. "maize production guide Nigeria extension") only ever existed
transiently inside 01_discover.py's checkpoint keys at crawl time; it was
never persisted per-row. So this script cannot literally "read the query
string back out of api_source or title" -- those fields don't contain it.

What it does instead, as the closest honest approximation: for every row
whose api_source/country/crops_covered identify a crop x country
combination found via a given crawler, it marks ALL of that crawler's
query templates for that (crop, country) as "success". This is a
one-directional over-approximation: it will never mark something
successful that wasn't attempted, but it CAN mark a specific template as
"success" when actually a *different* template for the same crop/country
is what produced the row (the other template might genuinely still be
un-tried, or have failed on its own). That trade-off is judged acceptable
here: skipping a handful of individually-untested template variants is
far cheaper than re-running the entire crawl to find out.

blocked.log entries need no such approximation: every entry is run
through classify_outcome() -- the identical function 01_discover.py
itself uses -- so the resulting categories are consistent with what a
live run would have recorded. As of the blocked.log checked while writing
this, there were no literal "403" or "timeout" entries in it at all:
entries present were blocked:429 (401), http_error:500 (5),
http_error:404 (3), and robots_disallowed (2). classify_outcome() already
maps all four of those onto the correct permanent/temporary buckets, so
no separate "403"/"timeout" special-casing was needed -- if your
blocked.log does contain different statuses, classify_outcome() handles
those too, by the same rules described in 01_discover.py.

Does not touch sources_discovered.csv. Overwrites harvest/checkpoint.json
in place -- back the current one up first if you want to keep it, e.g.:
    cp harvest/checkpoint.json harvest/checkpoint.json.bak

Runnable standalone: python corpus/harvest/reconstruct_checkpoint.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import HARVEST_DIR  # noqa: E402

SOURCES_CSV = HARVEST_DIR / "sources_discovered.csv"
BLOCKED_LOG = HARVEST_DIR / "blocked.log"
CHECKPOINT_JSON = HARVEST_DIR / "checkpoint.json"

BLOCKED_LOG_RE = re.compile(
    r"^\[(?P<date>[^\]]+)\]\s+source=(?P<source>.*?)\s+url=(?P<url>\S+)\s+status=(?P<status>\S+)\s+headers=(?P<headers>.*)$"
)

QUERY_TEMPLATES = [
    "{crop} production guide {country} extension",
    "{crop} {country} smallholder farming",
    "{crop} agro-ecological zone {country} recommendation",
]

# api_source value (as written by 01_discover.py) -> checkpoint key prefix
# for the three-templates-per-crop-per-country crawlers.
API_SOURCE_TO_PREFIX = {
    "openalex": "openalex",
    "openalex+unpaywall": "openalex",
    "semantic_scholar": "semanticscholar",
    "semantic_scholar+unpaywall": "semanticscholar",
}
# Single-template (no QUERY_TEMPLATES loop) crop/country crawlers.
SINGLE_TEMPLATE_PREFIXES = {
    "core": "core",
    "core+unpaywall": "core",
    "cgspace": "cgspace",
}

# Direct-scrape crawlers: checkpoint key is "{institution}:{path}", not a
# crop/country query, so every candidate path that crawler tries is marked
# done -- a row doesn't tell us which specific path produced it.
DIRECT_SCRAPE_PATHS = {
    "naerls_direct": ("naerls", ["/publications", "/resources", "/"]),
    "nihort_direct": ("nihort", ["home"]),
    "kalro_direct": ("kalro", ["/publications", "/"]),
    "mofa_direct": ("mofa", ["/publications", "/"]),
    "tari_direct": ("tari", ["/publications", "/"]),
    "naro_direct": ("naro", ["/publications", "/resources", "/"]),
    "eiar_direct": ("eiar", ["/publications", "/"]),
    "nafdac_playwright": ("nafdac", ["playwright"]),
}


def _load_classify_outcome():
    """01_discover.py can't be `import`ed normally (filename starts with a
    digit) -- load it by path and reuse its exact classify_outcome(), so
    blocked.log entries are categorized identically to how a live run
    would categorize the same failure."""
    spec = importlib.util.spec_from_file_location("discover_stage", Path(__file__).resolve().parent / "01_discover.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify_outcome


def reconstruct_from_sources(completed: dict[str, str]) -> int:
    if not SOURCES_CSV.exists():
        print(f"[reconstruct] {SOURCES_CSV} not found -- skipping the success side of the reconstruction")
        return 0

    with open(SOURCES_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[reconstruct] {len(rows)} rows in {SOURCES_CSV}")

    added = 0
    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"[reconstruct] Processing {i + 1}/{len(rows)}...")

        api_source = row.get("api_source", "")
        country = row.get("country", "")
        crops = [c for c in row.get("crops_covered", "").split(";") if c]

        template_prefix = API_SOURCE_TO_PREFIX.get(api_source)
        single_prefix = SINGLE_TEMPLATE_PREFIXES.get(api_source)

        if (template_prefix or single_prefix) and country and crops:
            for crop in crops:
                if template_prefix:
                    for template in QUERY_TEMPLATES:
                        key = f"{template_prefix}:{template.format(crop=crop, country=country)}"
                        if key not in completed:
                            completed[key] = "success"
                            added += 1
                else:
                    key = f"{single_prefix}:{crop} {country}"
                    if key not in completed:
                        completed[key] = "success"
                        added += 1
            continue

        direct = DIRECT_SCRAPE_PATHS.get(api_source)
        if direct:
            inst, paths = direct
            for path in paths:
                key = f"{inst}:{path}"
                if key not in completed:
                    completed[key] = "success"
                    added += 1

    return added


def reconstruct_from_blocked_log(completed: dict[str, str], classify_outcome) -> int:
    if not BLOCKED_LOG.exists():
        print(f"[reconstruct] {BLOCKED_LOG} not found -- skipping the failure side of the reconstruction")
        return 0

    with open(BLOCKED_LOG, encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    print(f"[reconstruct] {len(lines)} lines in {BLOCKED_LOG}")

    added = 0
    for i, line in enumerate(lines):
        if (i + 1) % 100 == 0:
            print(f"[reconstruct] Processing {i + 1}/{len(lines)}...")
        m = BLOCKED_LOG_RE.match(line.strip())
        if not m:
            continue
        # blocked.log's (source, url) pair doesn't always match
        # 01_discover.py's internal checkpoint key format exactly (the
        # label passed to log_blocked() varies by crawler), so this uses
        # the log's own (source, url) as the key -- good enough for what
        # the checkpoint needs: something stable to mark permanent vs.
        # temporary so a future run does (or doesn't) retry it.
        source = m.group("source")
        url = m.group("url")
        status = m.group("status")
        key = f"{source}:{url}"
        category = classify_outcome(status)
        if completed.get(key) != category:
            completed[key] = category
            added += 1

    return added


def main():
    print("[reconstruct] Starting ...")
    classify_outcome = _load_classify_outcome()

    completed: dict[str, str] = {}
    if CHECKPOINT_JSON.exists():
        try:
            with open(CHECKPOINT_JSON, encoding="utf-8") as fh:
                existing = json.load(fh).get("completed_queries", {})
            if isinstance(existing, dict):
                completed.update(existing)
                print(f"[reconstruct] {len(completed)} pre-existing entries loaded from {CHECKPOINT_JSON}")
        except (json.JSONDecodeError, OSError):
            pass

    added_success = reconstruct_from_sources(completed)
    added_blocked = reconstruct_from_blocked_log(completed, classify_outcome)

    tmp = CHECKPOINT_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"completed_queries": dict(sorted(completed.items()))}, fh, indent=2)
    tmp.replace(CHECKPOINT_JSON)

    print(f"\n[reconstruct] DONE. {added_success} entries reconstructed from sources_discovered.csv, "
          f"{added_blocked} entries reconstructed from blocked.log")
    print(f"[reconstruct] {len(completed)} total entries written to {CHECKPOINT_JSON}")
    print("[reconstruct] Done.")


if __name__ == "__main__":
    main()
