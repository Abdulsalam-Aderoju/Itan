#!/usr/bin/env python
"""
Stage 1 (harvest): discover and catalogue sources WITHOUT downloading any
file content. Writes one row per discovered source to sources_discovered.csv.

Geographic scope: all of sub-Saharan Africa (see common.COUNTRIES), not just
Nigeria -- Nigeria remains the primary market and is listed first, but every
crop x country combination is queried the same way.

Source strategy (rebuilt from direct site-scraping to APIs):
  - OpenAlex, Semantic Scholar, CORE, and CGSpace are bibliographic/
    full-text APIs, not web pages meant for browsing, so they carry no
    robots.txt crawl restrictions the way plantwise.org/iita.org/fao.org
    search pages do -- that's why the earlier direct-scrape approach kept
    hitting robots_disallowed. Unpaywall enriches any DOI the first three
    surface with the canonical legal open-access PDF location; CGSpace
    doesn't need that -- its DSpace 7 API returns a direct bitstream
    (file) download link itself when the item has one attached.
  - NAERLS (Nigeria) and NIHORT (Nigeria) are kept as direct
    requests+BeautifulSoup scrapes: in practice these two actually returned
    real links, unlike CABI/IITA/FAO/CGIAR/FMARD/NCRI/CRIN, which mostly
    either 404'd, hit robots_disallowed, or (CRIN) returned irrelevant
    institutional PDFs ("Organogram", "Guest House Accommodation") instead
    of crop content.
  - KALRO (Kenya), MOFA (Ghana), TARI (Tanzania), NARO (Uganda), and EIAR
    (Ethiopia) are added as direct-scrape attempts alongside NAERLS/NIHORT,
    using the same defensive pattern. IMPORTANT: their base URLs below are
    my best knowledge of each institution's official domain, NOT verified
    live against the actual sites -- confirm they resolve on your first
    real run and adjust INSTITUTION_TARGETS if a domain has moved. Rwanda,
    Senegal, Mali, and Burkina Faso have no direct-scrape institution here
    (none was specified) and are covered only via the three APIs above.
  - NAFDAC's agrochemical register renders via Playwright because a plain
    requests.get() against it returned nothing usable.

Checkpointing: given the query volume above, this script is built to be
killed and resumed. Every completed query (or scrape path) is recorded in
harvest/checkpoint.json right after it finishes, and sources_discovered.csv
is appended to after every query rather than written once at the end -- so
Ctrl+C (or a crash) loses at most the one in-flight query, not the whole
run. Re-running the script loads checkpoint.json and skips anything already
recorded. Delete checkpoint.json (or just don't have one) to start fresh --
that also starts sources_discovered.csv over rather than appending to
whatever was left from an unrelated earlier run.

Each checkpoint entry also records WHY the query ended (see
classify_outcome() / PERMANENT_OUTCOMES / TEMPORARY_OUTCOMES), not just that
it was attempted. On resume, permanent outcomes (success, robots.txt
disallow, 404) are skipped, but temporary ones (429, 5xx, connection/timeout
errors) are retried automatically -- so re-running the next day after a
burst of Semantic Scholar/OpenAlex 429s picks those specific queries back up
without re-doing everything else or spamming sources that are hard-blocked.

Environment variables (all optional except CORE_API_KEY):
  CORE_API_KEY      required to query CORE at all (free key: core.ac.uk/services/api).
                     Without it, the CORE step is skipped with a warning, not a crash.
  CONTACT_EMAIL      identifies this crawler to OpenAlex's "polite pool" and is the
                     required contact identifier for Unpaywall. Defaults to the
                     operator's own email if not set.
  SEMANTIC_SCHOLAR_API_KEY  optional; unauthenticated requests work but share a very
                     low public rate limit, so expect some 429s without one.

Runnable standalone:  python corpus/harvest/01_discover.py
Runnable via pipeline: invoked by run_pipeline.sh
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus, urlencode, urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    COUNTRIES, CROPS, HARVEST_DIR, REQUEST_DELAY, new_session, polite_get,
    random_ua, robots_allowed, find_crops, find_zones, guess_sql_tables,
)

from bs4 import BeautifulSoup

OUT_CSV = HARVEST_DIR / "sources_discovered.csv"
BLOCKED_LOG = HARVEST_DIR / "blocked.log"
CHECKPOINT_JSON = HARVEST_DIR / "checkpoint.json"

RATE_LIMIT_WAIT_SECONDS = 60

CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "aoalawal2004@gmail.com")
CORE_API_KEY = os.environ.get("CORE_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

FIELDS = [
    "id", "url", "title", "publisher", "country", "year", "format", "page_count_estimate",
    "licence", "open_access", "crops_covered", "zones_mentioned",
    "sql_tables_likely", "abstract_snippet", "blocked", "paywall", "retrieval_date",
    "api_source", "authors", "language",
]

_id_counter = 0


def next_id() -> str:
    global _id_counter
    _id_counter += 1
    return f"SRC{_id_counter:05d}"


def log_blocked(source: str, url: str, status_note: str, resp=None):
    with open(BLOCKED_LOG, "a", encoding="utf-8") as fh:
        headers = dict(resp.headers) if resp is not None else {}
        fh.write(f"[{date.today().isoformat()}] source={source} url={url} status={status_note} headers={headers}\n")


def make_row(url, title, publisher, country, year, fmt, pages, licence, open_access,
             crops, zones, tables, abstract, blocked=False, paywall=False, api_source="",
             authors="", language=""):
    return {
        "id": next_id(),
        "url": url,
        "title": title or "",
        "publisher": publisher,
        "country": country,
        "year": year or "",
        "format": fmt,
        "page_count_estimate": pages or "",
        "licence": licence or "",
        "open_access": bool(open_access),
        "crops_covered": ";".join(crops),
        "zones_mentioned": ";".join(zones),
        "sql_tables_likely": ";".join(tables),
        "abstract_snippet": (abstract or "")[:400].replace("\n", " ").strip(),
        "blocked": bool(blocked),
        "paywall": bool(paywall),
        "retrieval_date": date.today().isoformat(),
        "api_source": api_source,
        "authors": authors,
        "language": language,
    }


def _handle_failure(label: str, url: str, note, resp) -> None:
    print(f"  [{label}]: {note}")
    if note.startswith("blocked") or note == "robots_disallowed" or note.startswith("http_error"):
        log_blocked(label, url, note, resp)


# ---------------------------------------------------------------------------
# Checkpoint / resume: every query (or scrape path) gets a unique key, and
# the OUTCOME of attempting it is classified into one of six categories and
# stored alongside the key -- not just a bare "done" flag. That distinction
# is what lets resume be selective:
#   - PERMANENT_OUTCOMES are skipped on resume: retrying them wastes a
#     request on something that isn't going to change (robots.txt still
#     disallows it, the page still 404s, OpenAlex still has no OA copy).
#   - TEMPORARY_OUTCOMES are NOT skipped: a 429 today may well succeed
#     tomorrow once the rate limit / IP cooldown has passed, a 500 or a
#     connection error is very plausibly transient. is_done() below treats
#     these as "not done" so the next run retries them automatically.
# Saved after every single query, not batched, so a kill at any point loses
# at most the one in-flight query.
# ---------------------------------------------------------------------------
PERMANENT_OUTCOMES = {"success", "blocked_robots", "http_404", "core_disabled"}
TEMPORARY_OUTCOMES = {"http_429", "http_500", "connection_error"}
# "core_disabled": not a real outcome classify_outcome() ever returns --
# it's applied manually (see harvest/checkpoint.json) to CORE queries that
# failed while CORE itself was still enabled, once the crawl_core() call
# in main() gets disabled (see the `if False:` guard around it). Those
# entries would otherwise sit as "connection_error" (temporary) forever,
# which is moot while CORE isn't being called at all, but would cause them
# to all get retried the moment CORE is re-enabled -- explicit
# "core_disabled" makes it clear on inspection why they're marked done and
# keeps that a deliberate choice rather than a side effect of disabling.

_completed_queries: dict[str, str] = {}


def classify_outcome(note: str) -> str:
    """Maps a raw polite_get()/polite_get_with_backoff() status note -- or a
    synthetic one for a non-HTTP failure like a malformed JSON body or a
    Playwright error -- onto one of the six checkpoint categories.
    Unrecognized notes fall back to "connection_error" (temporary): better
    to retry something unfamiliar next run than to silently skip it
    forever because it didn't match a known pattern."""
    if note == "ok":
        return "success"
    if note == "robots_disallowed":
        return "blocked_robots"
    if note in ("timeout", "connection_error") or note.startswith("request_error"):
        return "connection_error"

    code = None
    if ":" in note:
        try:
            code = int(note.rsplit(":", 1)[1])
        except ValueError:
            code = None
    if code == 429:
        return "http_429"
    if code is not None and 500 <= code < 600:
        return "http_500"
    if code is not None and 400 <= code < 500:
        # Any other 4xx (401/403/404/...): a deliberate client-side
        # rejection that waiting a day out won't fix, same bucket as 404.
        return "http_404"
    return "connection_error"


def load_checkpoint() -> None:
    """Populates _completed_queries in place. Also decides, right here,
    whether this is a fresh run or a resume -- see docstring at the top of
    the file."""
    global _completed_queries
    if not CHECKPOINT_JSON.exists():
        # Fresh run (no checkpoint yet): a leftover sources_discovered.csv
        # from an unrelated earlier run would otherwise get silently
        # appended to below -- start it over instead.
        if OUT_CSV.exists():
            OUT_CSV.unlink()
        _completed_queries = {}
        return
    try:
        with open(CHECKPOINT_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        raw = data.get("completed_queries", {})
        if isinstance(raw, list):
            # Pre-refinement checkpoint: a flat list with no failure reason
            # recorded. There's no way to know which of these were really
            # a 429 vs a real success, so treat them all as permanent --
            # the safe direction is to under-retry, not to spam a source
            # that was actually a hard 404 or robots_disallowed.
            print(f"[discover] {CHECKPOINT_JSON} is in the old format (no failure reasons) -- "
                  f"treating all {len(raw)} entries as permanently done")
            _completed_queries = {k: "success" for k in raw}
        else:
            _completed_queries = dict(raw)
    except (json.JSONDecodeError, OSError):
        _completed_queries = {}


def resume_id_counter() -> None:
    """On a resumed run, sources_discovered.csv already has rows from the
    previous run with their own SRCxxxxx ids -- next_id() must continue
    from the highest one seen, not restart at SRC00001 and collide with
    ids already on disk. A no-op if the file doesn't exist (fresh run)."""
    global _id_counter
    if not OUT_CSV.exists():
        return
    try:
        with open(OUT_CSV, newline="", encoding="utf-8") as fh:
            max_n = 0
            for row in csv.DictReader(fh):
                sid = row.get("id", "")
                if sid.startswith("SRC"):
                    try:
                        max_n = max(max_n, int(sid[3:]))
                    except ValueError:
                        pass
            _id_counter = max_n
    except OSError:
        pass


def save_checkpoint() -> None:
    # Write-to-temp-then-replace so a kill mid-write can't leave a corrupt
    # (half-written) checkpoint.json behind.
    tmp = CHECKPOINT_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"completed_queries": dict(sorted(_completed_queries.items()))}, fh, indent=2)
    tmp.replace(CHECKPOINT_JSON)


def mark_done(key: str, note: str) -> None:
    _completed_queries[key] = classify_outcome(note)
    save_checkpoint()


def is_done(key: str) -> bool:
    """Only PERMANENT outcomes count as "done" -- a key recorded with a
    TEMPORARY outcome (429/500/connection error) is treated as not-done so
    it gets retried this run."""
    return _completed_queries.get(key) in PERMANENT_OUTCOMES


# ---------------------------------------------------------------------------
# Incremental CSV writes: append the rows produced by the query that just
# finished, rather than accumulating everything in memory and writing once
# at the very end.
# ---------------------------------------------------------------------------
_last_flushed = 0


def flush_rows(rows: list) -> None:
    global _last_flushed
    new_rows = rows[_last_flushed:]
    if not new_rows:
        return
    file_has_header = OUT_CSV.exists() and OUT_CSV.stat().st_size > 0
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not file_has_header:
            writer.writeheader()
        writer.writerows(new_rows)
    _last_flushed = len(rows)


# ---------------------------------------------------------------------------
# 429 / timeout handling: both are plausibly transient (a rate limit that
# clears, a slow endpoint having a bad moment -- CORE's search API in
# particular was observed to hang past a 10s timeout in testing even though
# its root endpoint and other API hosts responded in under a second), so
# both get the same treatment: wait 60s with a printed countdown, retry
# once, then give up. The caller sees whatever the retry returned --
# success, or still failing.
# ---------------------------------------------------------------------------
RETRYABLE_NOTES = {"blocked:429", "timeout"}


def polite_get_with_backoff(url: str, session=None, **kwargs):
    resp, note = polite_get(url, session=session, **kwargs)
    if note not in RETRYABLE_NOTES:
        return resp, note

    reason = "rate limited (429)" if note == "blocked:429" else "timed out"
    print(f"  [{note}] {reason} on {url[:90]} -- waiting {RATE_LIMIT_WAIT_SECONDS}s before retrying once")
    for remaining in range(RATE_LIMIT_WAIT_SECONDS, 0, -1):
        print(f"\r  [{note}] retrying in {remaining:2d}s...", end="", flush=True)
        time.sleep(1)
    print(f"\r  [{note}] retrying now...                    ")

    resp, note2 = polite_get(url, session=session, **kwargs)
    if note2 in RETRYABLE_NOTES:
        print(f"  [{note2}] retry also failed -- marking as blocked/failed")
    return resp, note2


def reconstruct_abstract(inverted_index: dict | None, max_words: int = 60) -> str:
    """OpenAlex returns abstracts as a word->[positions] inverted index
    instead of plain text (to dodge copyright on the abstract text itself).
    Rebuild a readable snippet from it, truncated to max_words."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    ordered = [positions[i] for i in sorted(positions)[:max_words]]
    return " ".join(ordered)


# ---------------------------------------------------------------------------
# Unpaywall enrichment: given a DOI surfaced by OpenAlex/Semantic Scholar/CORE,
# look up the canonical legal open-access PDF location. Capped globally so a
# result set with hundreds of DOIs can't blow up the run's total time.
# ---------------------------------------------------------------------------
UNPAYWALL_CALL_CAP = 200
# A single query page can return up to 50 results, each with its own DOI --
# enriching all of them inline would make the FIRST query's progress line
# take minutes to print (each lookup is its own rate-limited HTTP request).
# Capping enrichment per query keeps progress visible and total runtime
# predictable; the global cap above is still the hard ceiling.
MAX_ENRICH_PER_QUERY = 5
_unpaywall_cache: dict[str, str | None] = {}
_unpaywall_calls = 0
_unpaywall_cap_warned = False


def unpaywall_lookup(doi: str, session=None) -> str | None:
    global _unpaywall_calls, _unpaywall_cap_warned
    if not doi:
        return None
    doi_clean = doi.strip().lower().replace("https://doi.org/", "")
    if not doi_clean:
        return None
    if doi_clean in _unpaywall_cache:
        return _unpaywall_cache[doi_clean]
    if _unpaywall_calls >= UNPAYWALL_CALL_CAP:
        if not _unpaywall_cap_warned:
            print(f"  [unpaywall] hit the {UNPAYWALL_CALL_CAP}-lookup cap for this run -- "
                  f"remaining rows keep their original API-provided URL")
            _unpaywall_cap_warned = True
        return None

    _unpaywall_calls += 1
    url = f"https://api.unpaywall.org/v2/{quote_plus(doi_clean)}?email={quote_plus(CONTACT_EMAIL)}"
    resp, note = polite_get_with_backoff(url, session=session)
    result = None
    if resp is not None and note == "ok":
        try:
            data = resp.json()
            best = data.get("best_oa_location") or {}
            result = best.get("url_for_pdf") or best.get("url")
        except ValueError:
            result = None
    elif note.startswith("blocked") or note == "robots_disallowed":
        log_blocked("Unpaywall", url, note, resp)
    _unpaywall_cache[doi_clean] = result
    return result


# ---------------------------------------------------------------------------
# Shared query templates -- used by both OpenAlex and Semantic Scholar below.
# {country} makes every combination sub-Saharan-Africa-wide, not Nigeria-only.
# ---------------------------------------------------------------------------
QUERY_TEMPLATES = [
    "{crop} production guide {country} extension",
    "{crop} {country} smallholder farming",
    "{crop} agro-ecological zone {country} recommendation",
]


# ---------------------------------------------------------------------------
# Source 1: OpenAlex
# ---------------------------------------------------------------------------
def crawl_openalex(session, rows: list):
    print("[discover] OpenAlex API ...")
    base = "https://api.openalex.org/works"
    total = 0
    for country in COUNTRIES:
        for crop in CROPS:
            for template in QUERY_TEMPLATES:
                query = template.format(crop=crop, country=country)
                key = f"openalex:{query}"
                if is_done(key):
                    continue
                params = {
                    "search": query,
                    "filter": "open_access.is_oa:true",
                    "per_page": 50,
                    "mailto": CONTACT_EMAIL,
                }
                url = f"{base}?{urlencode(params)}"
                resp, note = polite_get_with_backoff(url, session=session)
                if resp is None or note != "ok":
                    _handle_failure(f"openalex '{query}'", url, note, resp)
                    mark_done(key, note)
                    continue
                try:
                    data = resp.json()
                except ValueError:
                    print(f"  [openalex] '{query}': non-JSON response")
                    mark_done(key, "non_json_response")
                    continue

                found = 0
                enriched_this_query = 0
                for work in data.get("results", []):
                    oa = work.get("open_access") or {}
                    oa_url = oa.get("oa_url")
                    if not oa_url:
                        continue
                    title = work.get("title") or ""
                    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                    doi = (work.get("doi") or "").replace("https://doi.org/", "")
                    upgraded = None
                    if doi and enriched_this_query < MAX_ENRICH_PER_QUERY:
                        upgraded = unpaywall_lookup(doi, session=session)
                        enriched_this_query += 1
                    final_url = upgraded or oa_url

                    rows.append(make_row(
                        url=final_url, title=title, publisher="OpenAlex", country=country,
                        year=work.get("publication_year") or "",
                        fmt="PDF" if final_url.lower().endswith(".pdf") else "HTML",
                        pages="", licence=oa.get("oa_status", ""), open_access=True,
                        # Unlike the institution-specific NAERLS/NIHORT searches, these
                        # are broad full-text queries that can surface completely
                        # off-topic results (e.g. an energy-policy paper matching
                        # "Nigeria" loosely) -- falling back to the searched crop
                        # would falsely tag them as a real source for it, so a row
                        # with no actual crop mention is left uncategorized instead.
                        crops=find_crops(title + " " + abstract),
                        zones=find_zones(title + " " + abstract),
                        tables=guess_sql_tables(title + " " + abstract),
                        abstract=abstract,
                        api_source="openalex+unpaywall" if upgraded else "openalex",
                    ))
                    found += 1
                total += found
                print(f"  [openalex] '{query}': {found} OA results")
                mark_done(key, "ok")
                flush_rows(rows)
    print(f"  [openalex] total: {total}")


# ---------------------------------------------------------------------------
# Source 2: Semantic Scholar (shares QUERY_TEMPLATES with OpenAlex above)
# ---------------------------------------------------------------------------
def crawl_semantic_scholar(session, rows: list):
    print("[discover] Semantic Scholar API ...")
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else None
    total = 0
    for country in COUNTRIES:
        for crop in CROPS:
            for template in QUERY_TEMPLATES:
                query = template.format(crop=crop, country=country)
                key = f"semanticscholar:{query}"
                if is_done(key):
                    continue
                params = {"query": query, "fields": "title,year,openAccessPdf,authors,abstract,externalIds", "limit": 20}
                url = f"{base}?{urlencode(params)}"
                resp, note = polite_get_with_backoff(url, session=session, extra_headers=headers)
                if resp is None or note != "ok":
                    _handle_failure(f"semanticscholar '{query}'", url, note, resp)
                    mark_done(key, note)
                    continue
                try:
                    data = resp.json()
                except ValueError:
                    print(f"  [semanticscholar] '{query}': non-JSON response")
                    mark_done(key, "non_json_response")
                    continue

                found = 0
                enriched_this_query = 0
                for paper in data.get("data", []):
                    pdf_info = paper.get("openAccessPdf")
                    if not pdf_info or not pdf_info.get("url"):
                        continue
                    title = paper.get("title") or ""
                    abstract = paper.get("abstract") or ""
                    doi = (paper.get("externalIds") or {}).get("DOI") or ""
                    upgraded = None
                    if doi and enriched_this_query < MAX_ENRICH_PER_QUERY:
                        upgraded = unpaywall_lookup(doi, session=session)
                        enriched_this_query += 1
                    final_url = upgraded or pdf_info["url"]

                    rows.append(make_row(
                        url=final_url, title=title, publisher="Semantic Scholar", country=country,
                        year=paper.get("year") or "", fmt="PDF", pages="",
                        licence="open access (Semantic Scholar verified)", open_access=True,
                        # Unlike the institution-specific NAERLS/NIHORT searches, these
                        # are broad full-text queries that can surface completely
                        # off-topic results (e.g. an energy-policy paper matching
                        # "Nigeria" loosely) -- falling back to the searched crop
                        # would falsely tag them as a real source for it, so a row
                        # with no actual crop mention is left uncategorized instead.
                        crops=find_crops(title + " " + abstract),
                        zones=find_zones(title + " " + abstract),
                        tables=guess_sql_tables(title + " " + abstract),
                        abstract=abstract,
                        api_source="semantic_scholar+unpaywall" if upgraded else "semantic_scholar",
                    ))
                    found += 1
                total += found
                print(f"  [semanticscholar] '{query}': {found} OA results")
                mark_done(key, "ok")
                flush_rows(rows)
    print(f"  [semanticscholar] total: {total}")


# ---------------------------------------------------------------------------
# Source 3: CORE
# ---------------------------------------------------------------------------
def crawl_core(session, rows: list):
    print("[discover] CORE API ...")
    if not CORE_API_KEY:
        print("  [core] CORE_API_KEY not set -- skipping. Get a free key at "
              "https://core.ac.uk/services/api and set the CORE_API_KEY environment variable.")
        return

    base = "https://api.core.ac.uk/v3/search/works"
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
    total = 0
    for country in COUNTRIES:
        for crop in CROPS:
            query = f"{crop} {country}"
            key = f"core:{query}"
            if is_done(key):
                continue
            url = f"{base}?{urlencode({'q': query, 'limit': 50})}"
            # CORE's search endpoint was observed in testing to hang past the
            # default 10s timeout even when its own root endpoint and every
            # other API host responded in under a second -- 30s gives it a
            # fairer chance, and a timeout here is retried once via
            # polite_get_with_backoff just like a 429 (both temporary).
            # delay=3 (not the global 2s default): CORE's registered-user
            # rate limit is 25 requests/minute (one every 2.4s), so 2s was
            # cutting it close.
            #
            # NOTE: an `_exists_:downloadUrl` filter term was tried in the
            # query (to have CORE itself only return results with a PDF)
            # and confirmed broken against CORE's current backend -- it
            # returns HTTP 500 "Could not find a property named
            # 'downloadUrl' on type 'search.document'" (Azure Cognitive
            # Search error). Filtering for downloadUrl is instead done
            # client-side below, same as before.
            resp, note = polite_get_with_backoff(url, session=session, extra_headers=headers, timeout=30, delay=3)
            if resp is None or note != "ok":
                _handle_failure(f"core '{query}'", url, note, resp)
                mark_done(key, note)
                continue
            try:
                data = resp.json()
            except ValueError:
                print(f"  [core] '{query}': non-JSON response")
                mark_done(key, "non_json_response")
                continue

            found = 0
            enriched_this_query = 0
            for work in data.get("results", []):
                download_url = work.get("downloadUrl") or next(iter(work.get("sourceFulltextUrls") or []), None)
                if not download_url:
                    continue
                title = work.get("title") or ""
                abstract = work.get("abstract") or ""
                doi = work.get("doi") or ""
                authors = "; ".join(a.get("name", "") for a in (work.get("authors") or []) if a.get("name"))
                language = (work.get("language") or {}).get("name") or (work.get("language") or {}).get("code") or ""
                upgraded = None
                if doi and enriched_this_query < MAX_ENRICH_PER_QUERY:
                    upgraded = unpaywall_lookup(doi, session=session)
                    enriched_this_query += 1
                final_url = upgraded or download_url

                rows.append(make_row(
                    url=final_url, title=title, publisher=work.get("publisher") or "CORE", country=country,
                    year=work.get("yearPublished") or "",
                    fmt="PDF" if final_url.lower().endswith(".pdf") else "HTML",
                    pages="", licence="open access (CORE aggregated)", open_access=True,
                    # Same reasoning as OpenAlex/Semantic Scholar above: CORE's search
                    # is broad full-text, so an off-topic hit isn't forced into the
                    # searched crop's bucket -- it's left uncategorized instead.
                    crops=find_crops(title + " " + abstract),
                    zones=find_zones(title + " " + abstract),
                    tables=guess_sql_tables(title + " " + abstract),
                    abstract=abstract,
                    api_source="core+unpaywall" if upgraded else "core",
                    authors=authors, language=language,
                ))
                found += 1
            total += found
            print(f"  [core] '{query}': {found} results")
            mark_done(key, "ok")
            flush_rows(rows)
    print(f"  [core] total: {total}")


# ---------------------------------------------------------------------------
# Source 3b: CGSpace (CGIAR's DSpace 7 repository)
#
# Verified live while writing this (not a guess like the institution URLs
# below): a plain query with no `embed` param returns items with NO bundle
# or bitstream data at all -- DSpace 7 only includes them if explicitly
# requested via `embed=bundles/bitstreams`, which is what actually makes
# "extract the direct PDF link" possible here. With that embed added,
# real Nigeria/Africa extension documents came back with a working direct
# download link at bitstream["_links"]["content"]["href"], e.g.
# https://cgspace.cgiar.org/server/api/core/bitstreams/<uuid>/content.
# Not every item has an ORIGINAL bundle (metadata-only records with no
# attached file do turn up), so a missing PDF link falls back to the
# item's own landing page instead of being dropped.
# ---------------------------------------------------------------------------
def _extract_cgspace_pdf_url(item: dict) -> str | None:
    bundles = item.get("_embedded", {}).get("bundles", {}).get("_embedded", {}).get("bundles", [])
    for bundle in bundles:
        if bundle.get("name") != "ORIGINAL":
            continue
        bitstreams = bundle.get("_embedded", {}).get("bitstreams", {}).get("_embedded", {}).get("bitstreams", [])
        for bitstream in bitstreams:
            content_href = bitstream.get("_links", {}).get("content", {}).get("href")
            if content_href:
                return content_href
    return None


def crawl_cgspace(session, rows: list):
    print("[discover] CGSpace (CGIAR DSpace) API ...")
    base = "https://cgspace.cgiar.org/server/api/discover/search/objects"
    total = 0
    for country in COUNTRIES:
        for crop in CROPS:
            query = f"{crop} {country}"
            key = f"cgspace:{query}"
            if is_done(key):
                continue
            params = {"query": query, "size": 50, "embed": "bundles/bitstreams"}
            url = f"{base}?{urlencode(params)}"
            resp, note = polite_get_with_backoff(url, session=session)
            if resp is None or note != "ok":
                _handle_failure(f"cgspace '{query}'", url, note, resp)
                mark_done(key, note)
                continue
            try:
                data = resp.json()
            except ValueError:
                print(f"  [cgspace] '{query}': non-JSON response")
                mark_done(key, "non_json_response")
                continue

            found = 0
            objects = data.get("_embedded", {}).get("searchResult", {}).get("_embedded", {}).get("objects", [])
            for obj in objects:
                item = obj.get("_embedded", {}).get("indexableObject") or {}
                if not item:
                    continue
                title = item.get("name") or ""
                metadata = item.get("metadata", {})

                year = ""
                for date_field in ("dcterms.issued", "dc.date.issued"):
                    vals = metadata.get(date_field)
                    if vals:
                        year = (vals[0].get("value") or "")[:4]
                        break

                abstract = ""
                for abs_field in ("dcterms.abstract", "dc.description.abstract"):
                    vals = metadata.get(abs_field)
                    if vals:
                        abstract = vals[0].get("value") or ""
                        break

                pdf_url = _extract_cgspace_pdf_url(item)
                # Fallback for the ~80% of results with no attached file
                # (verified: only ~1 in 5 items in a real query had a PDF
                # bitstream): item["_links"]["self"]["href"] looks like a
                # landing page but is actually the JSON API resource for
                # the item, not a browsable page -- 02_extract.py's HTML
                # parser would get raw JSON, not real content. The handle
                # gives the actual public-facing CGSpace page instead.
                handle = item.get("handle") or ""
                landing_url = f"https://cgspace.cgiar.org/handle/{handle}" if handle else ""
                final_url = pdf_url or landing_url
                if not final_url:
                    continue

                rows.append(make_row(
                    url=final_url, title=title, publisher="CGSpace", country=country,
                    year=year, fmt="PDF" if pdf_url else "HTML", pages="",
                    licence="CGIAR open access (CGSpace)", open_access=True,
                    # Same reasoning as OpenAlex/Semantic Scholar/CORE above:
                    # a broad full-text search can surface off-topic hits, so
                    # no fallback to the searched crop if nothing matched.
                    crops=find_crops(title + " " + abstract),
                    zones=find_zones(title + " " + abstract),
                    tables=guess_sql_tables(title + " " + abstract),
                    abstract=abstract, api_source="cgspace",
                ))
                found += 1
            total += found
            print(f"  [cgspace] '{query}': {found} results")
            mark_done(key, "ok")
            flush_rows(rows)
    print(f"  [cgspace] total: {total}")


# ---------------------------------------------------------------------------
# Source 4a: NAERLS and NIHORT -- kept as direct scrapes, these actually work
# ---------------------------------------------------------------------------
def crawl_naerls(session, rows: list):
    print("[discover] NAERLS (direct) ...")
    base = "https://www.naerls.gov.ng"
    total = 0
    for path in ("/publications", "/resources", "/"):
        key = f"naerls:{path}"
        if is_done(key):
            continue
        url = urljoin(base, path)
        resp, note = polite_get_with_backoff(url, session=session)
        if resp is None or note != "ok":
            _handle_failure(f"naerls {path}", url, note, resp)
            mark_done(key, note)
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        links = [a for a in soup.find_all("a", href=True)
                 if re.search(r"bulletin|extension|guide|manual|\.pdf", a.get_text(" ") + a["href"], re.I)]
        for a in links[:20]:
            href = urljoin(base, a["href"])
            title = a.get_text(" ", strip=True) or "NAERLS extension bulletin"
            rows.append(make_row(
                url=href, title=title, publisher="NAERLS", country="Nigeria", year="",
                fmt="PDF" if href.lower().endswith(".pdf") else "HTML", pages="",
                licence="Nigerian government extension material", open_access=True,
                crops=find_crops(title), zones=find_zones(title),
                tables=guess_sql_tables(title), abstract=title, api_source="naerls_direct",
            ))
            total += 1
        if links:
            print(f"  [naerls] {path}: {len(links)} links")
        mark_done(key, "ok")
        flush_rows(rows)
    print(f"  [naerls] total: {total}")


def crawl_nihort(session, rows: list):
    print("[discover] NIHORT (direct) ...")
    key = "nihort:home"
    if is_done(key):
        return
    base = "https://nihort.gov.ng"
    resp, note = polite_get_with_backoff(base, session=session)
    if resp is None or note != "ok":
        _handle_failure("nihort", base, note, resp)
        mark_done(key, note)
        return
    soup = BeautifulSoup(resp.text, "html.parser")
    links = [a for a in soup.find_all("a", href=True)
             if re.search(r"manual|guide|recommend|bulletin|\.pdf|publication", a.get_text(" ") + a["href"], re.I)]
    found = 0
    for a in links[:15]:
        href = urljoin(base, a["href"])
        title = a.get_text(" ", strip=True) or "NIHORT publication"
        crops = find_crops(title) or ["tomato"]  # NIHORT's mandate crops when the title names none
        rows.append(make_row(
            url=href, title=title, publisher="NIHORT", country="Nigeria", year="",
            fmt="PDF" if href.lower().endswith(".pdf") else "HTML", pages="",
            licence="Nigerian federal/state government material", open_access=True,
            crops=crops, zones=find_zones(title), tables=guess_sql_tables(title),
            abstract=title, api_source="nihort_direct",
        ))
        found += 1
    print(f"  [nihort] {found} publications")
    mark_done(key, "ok")
    flush_rows(rows)


# ---------------------------------------------------------------------------
# Source 4b: direct-scrape institutions for the other five SSA countries,
# alongside NAERLS/NIHORT. Same defensive pattern (try a few candidate
# paths, extract publication-looking links, never crash on failure).
#
# IMPORTANT: base_url below is my best knowledge of each institution's
# official domain -- it has NOT been verified live in this environment (per
# instruction, nothing in this file was run while writing it). Confirm each
# one actually resolves on your first real run and fix INSTITUTION_TARGETS
# if a domain has moved; a wrong domain just fails gracefully and logs to
# blocked.log/manual_priority_list.txt like any other unreachable source,
# it won't crash the pipeline.
#
# Rwanda, Senegal, Mali, and Burkina Faso have no entry here (no institution
# was specified for them) -- they're covered only via the three APIs above.
# ---------------------------------------------------------------------------
INSTITUTION_TARGETS = [
    # (institution, country, base_url, candidate_paths)
    ("KALRO", "Kenya", "https://www.kalro.org", ("/publications", "/")),
    ("MOFA", "Ghana", "https://mofa.gov.gh", ("/publications", "/")),
    ("TARI", "Tanzania", "https://www.tari.go.tz", ("/publications", "/")),
    ("NARO", "Uganda", "https://www.naro.go.ug", ("/publications", "/resources", "/")),
    ("EIAR", "Ethiopia", "https://www.eiar.gov.et", ("/publications", "/")),
]


def crawl_institution_direct(session, rows: list, institution: str, country: str,
                              base: str, candidate_paths: tuple[str, ...]):
    print(f"[discover] {institution} {country} (direct) ...")
    total = 0
    for path in candidate_paths:
        key = f"{institution.lower()}:{path}"
        if is_done(key):
            continue
        url = urljoin(base, path)
        resp, note = polite_get_with_backoff(url, session=session)
        if resp is None or note != "ok":
            _handle_failure(f"{institution.lower()} {path}", url, note, resp)
            mark_done(key, note)
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        links = [a for a in soup.find_all("a", href=True)
                 if re.search(r"bulletin|extension|guide|manual|recommend|publication|\.pdf",
                              a.get_text(" ") + a["href"], re.I)]
        for a in links[:15]:
            href = urljoin(base, a["href"])
            title = a.get_text(" ", strip=True) or f"{institution} publication"
            rows.append(make_row(
                url=href, title=title, publisher=institution, country=country, year="",
                fmt="PDF" if href.lower().endswith(".pdf") else "HTML", pages="",
                licence=f"{country} government/research institution material", open_access=True,
                crops=find_crops(title), zones=find_zones(title), tables=guess_sql_tables(title),
                abstract=title, api_source=f"{institution.lower()}_direct",
            ))
            total += 1
        if links:
            print(f"  [{institution.lower()}] {path}: {len(links)} links")
        mark_done(key, "ok")
        flush_rows(rows)
    print(f"  [{institution.lower()}] total: {total}")


# ---------------------------------------------------------------------------
# Source 5: NAFDAC via Playwright (a plain requests.get() returned nothing usable)
# ---------------------------------------------------------------------------
def crawl_nafdac_playwright(rows: list):
    print("[discover] NAFDAC agrochemical register (Playwright) ...")
    key = "nafdac:playwright"
    if is_done(key):
        return
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [nafdac] playwright not installed -- skipping. "
              "Run: pip install playwright && playwright install chromium")
        return  # not marked done: no real attempt was made, worth retrying once playwright is installed

    url = "https://www.nafdac.gov.ng/our-services/registration/agrochemicals/"
    if not robots_allowed(url):
        print("  [nafdac] robots_disallowed")
        log_blocked("NAFDAC", url, "robots_disallowed")
        mark_done(key, "robots_disallowed")
        return

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(user_agent=random_ua())
            page.goto(url, timeout=20000, wait_until="networkidle")
            time.sleep(REQUEST_DELAY)
            html = page.content()
            browser.close()
    except Exception as exc:
        print(f"  [nafdac] Playwright render failed: {exc}")
        log_blocked("NAFDAC", url, f"playwright_error:{exc.__class__.__name__}")
        # Not a recognized HTTP/network note, so classify_outcome() falls
        # back to "connection_error" (temporary) -- a Playwright render
        # failure (page timeout, browser crash) is plausibly transient and
        # worth retrying, not a permanent block.
        mark_done(key, f"playwright_error:{exc.__class__.__name__}")
        return

    soup = BeautifulSoup(html, "html.parser")
    found = 0
    for table in soup.find_all("table"):
        header_cells = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 2:
                continue
            row_dict = dict(zip(header_cells, cells)) if len(header_cells) == len(cells) else {}
            product = row_dict.get("product name") or row_dict.get("product") or cells[0]
            crop_field = row_dict.get("crop") or row_dict.get("indication") or ""
            if not product:
                continue
            rows.append(make_row(
                url=url, title=f"NAFDAC registered agrochemical: {product}", publisher="NAFDAC", country="Nigeria",
                year="", fmt="HTML", pages="", licence="Nigerian regulatory register (public)",
                open_access=True, crops=find_crops(crop_field) or find_crops(product),
                zones=[], tables=["agrochemical"],
                abstract=f"{product} — {crop_field}".rstrip(" —"), api_source="nafdac_playwright",
            ))
            found += 1
    print(f"  [nafdac] {found} registered products extracted via Playwright")
    mark_done(key, "ok")
    flush_rows(rows)


def main():
    load_checkpoint()
    resume_id_counter()
    if _completed_queries:
        permanent = [v for v in _completed_queries.values() if v in PERMANENT_OUTCOMES]
        temporary = [v for v in _completed_queries.values() if v in TEMPORARY_OUTCOMES]
        print(f"[discover] resuming from {CHECKPOINT_JSON}: {len(permanent)} queries permanently "
              f"done (skipped), {len(temporary)} previously-temporary failures will be retried this run")
        if temporary:
            breakdown = Counter(temporary)
            for outcome, n in sorted(breakdown.items()):
                print(f"    retrying {n} queries that previously failed with: {outcome}")

    session = new_session()
    rows: list[dict] = []

    try:
        crawl_openalex(session, rows)
        crawl_semantic_scholar(session, rows)
        if False:  # CORE disabled: timing out on every query even with the 60s backoff
            # retry, adding 3+ hours to the runtime for zero results. Code is left
            # intact below -- flip this to `if True:` (or drop the guard) once
            # CORE's API is reliable again.
            crawl_core(session, rows)
        crawl_cgspace(session, rows)
        crawl_naerls(session, rows)
        crawl_nihort(session, rows)
        for institution, country, base, paths in INSTITUTION_TARGETS:
            crawl_institution_direct(session, rows, institution, country, base, paths)
        crawl_nafdac_playwright(rows)
    except KeyboardInterrupt:
        flush_rows(rows)  # anything appended to `rows` since the last per-query flush
        save_checkpoint()  # redundant with the per-query saves, but explicit and cheap
        print(f"\n[discover] Interrupted. {len(_completed_queries)} queries checkpointed, "
              f"{len(rows)} new rows this run flushed to {OUT_CSV}.")
        print(f"[discover] Re-run this script to resume from {CHECKPOINT_JSON}.")
        sys.exit(130)  # conventional exit code for SIGINT

    total_rows = 0
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="", encoding="utf-8") as fh:
            total_rows = sum(1 for _ in csv.DictReader(fh))

    print(f"\n[discover] DONE. {len(rows)} new sources this run ({total_rows} total in {OUT_CSV})")


if __name__ == "__main__":
    main()
