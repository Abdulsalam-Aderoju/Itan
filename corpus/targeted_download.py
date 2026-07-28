#!/usr/bin/env python
"""
Standalone utility: download a hand-picked list of high-priority URLs
directly to corpus/raw/manual/, bypassing the score/manifest pipeline
entirely (no scoring, no dedup-by-title, no manifest generation -- just
"get these specific documents now").

Two sources feed the download queue:
  1. HARDCODED_URLS below -- specific documents worth getting regardless
     of what the automated crawlers found or how they scored.
  2. Any URL already sitting in harvest/sources_scored.csv (falls back to
     sources_discovered.csv) whose domain matches HIGH_PRIORITY_DOMAINS --
     naerls.gov.ng, iita.org, fmard.gov.ng -- treated as high priority
     even if it never made it into a scored manifest.

Runnable standalone: python corpus/targeted_download.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS_DIR, HARVEST_DIR, new_session, polite_get, safe_filename  # noqa: E402

MANUAL_DIR = CORPUS_DIR / "raw" / "manual"
MANUAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV = CORPUS_DIR / "logs" / "targeted_download_log.csv"

HIGH_PRIORITY_DOMAINS = ["naerls.gov.ng", "iita.org", "fmard.gov.ng"]

# ---------------------------------------------------------------------------
# Hand-picked, hard-coded high-priority URLs.
#
# The NAERLS entries below are real -- pulled directly from this project's
# own harvest/sources_discovered.csv (verified present, not guessed).
#
# The IITA and "agrinigeria" entries are left as explicit TODOs: there is
# no IITA or agrinigeria URL anywhere in sources_discovered.csv, and no
# "search results" from any prior step in this pipeline to pull them from.
# Paste the real URLs in below before running this script -- a fabricated
# placeholder URL would just produce a 404 in targeted_download_log.csv,
# which is worse than an obviously-empty TODO.
# ---------------------------------------------------------------------------
HARDCODED_URLS = [
    # --- NAERLS (verified real, from harvest/sources_discovered.csv) ---
    {"url": "https://naerls.gov.ng/agric-extension/", "title": "Agric Extension", "publisher": "NAERLS"},
    {"url": "https://naerls.gov.ng/e-extension/", "title": "E-Extension", "publisher": "NAERLS"},
    {"url": "https://naerls.gov.ng/bulletins/", "title": "Bulletins", "publisher": "NAERLS"},
    {"url": "https://naerls.gov.ng/njae/", "title": "NJAE Extension Journal", "publisher": "NAERLS"},
    {"url": "http://naerls.gov.ng/", "title": "NAERLS Home", "publisher": "NAERLS"},

    # --- NAFDAC (verified live: HTTP 200, application/pdf, 405903 bytes) ---
    {
        "url": "https://www.nafdac.gov.ng/wp-content/uploads/Files/Resources/Directorate_Resources/R_and_R/"
               "List-of-Registered-Animal-Health-Products-and-Agrochemicals.pdf",
        "title": "List of Registered Animal Health Products and Agrochemicals", "publisher": "NAFDAC",
    },

    # --- CGSpace: real substitutes for the URLs provided, found and verified live ---
    # Every one of the 5 non-NAFDAC URLs originally pasted here (the IITA
    # wp-content path, the agrinigeria blob-storage path, the NAERLS
    # wp-content path, and both CGSpace /bitstream/handle/... paths)
    # returned HTTP 404 when checked -- confirmed against reachable base
    # domains, so it wasn't a network fluke, those specific paths just
    # don't exist. Old-style CGSpace "/bitstream/handle/<id>/<file>.pdf"
    # links in particular don't survive its DSpace 7 migration; the
    # current working form is "/server/api/core/bitstreams/<uuid>/content".
    #
    # Searching CGSpace's real API for the same subjects turned up genuine,
    # live-verified (HTTP 200, real PDF bytes) documents -- including an
    # exact title match for the IITA maize guide that was being requested
    # ("Guide to maize production in northern Nigeria"), just hosted on
    # CGSpace rather than at the fabricated iita.org path:
    {
        "url": "https://cgspace.cgiar.org/server/api/core/bitstreams/cddacd1f-17db-450a-ae1d-14bff35a5754/content",
        "title": "Guide to maize production in northern Nigeria", "publisher": "CGSpace",
    },
    {
        "url": "https://cgspace.cgiar.org/server/api/core/bitstreams/95a959a1-3344-488c-b0a6-5d2994ed25f2/content",
        "title": "Growing cassava commercially in Nigeria: an illustrated guide", "publisher": "CGSpace",
    },
    {
        "url": "https://cgspace.cgiar.org/server/api/core/bitstreams/94bed861-c0a2-4073-a9cf-1909836782d4/content",
        "title": "Guide to cowpea production in northern Nigeria", "publisher": "CGSpace",
    },

    # --- TODO: NAERLS "Maize Value Chain Bulletin" -- the pasted URL 404'd
    # and no working substitute was found; paste a corrected URL here. ---
    # {"url": "https://naerls.gov.ng/...", "title": "...", "publisher": "NAERLS"},

    # --- TODO: "agrinigeria" -- the pasted blob-storage URL 404'd (Azure
    # BlobNotFound) and "agrinigeria" isn't a publisher recognized anywhere
    # else in this pipeline, so no substitute could be found. Paste a
    # corrected URL here if you can confirm the real source. ---
    # {"url": "https://.../...", "title": "...", "publisher": "AgriNigeria"},
]


def discover_high_priority_from_csv() -> list[dict]:
    """Scan sources_scored.csv (falls back to sources_discovered.csv) for
    any URL whose domain matches HIGH_PRIORITY_DOMAINS, regardless of its
    score or manifest status -- these are worth grabbing even if they
    scored too low, or were never scored at all."""
    scored = HARVEST_DIR / "sources_scored.csv"
    discovered = HARVEST_DIR / "sources_discovered.csv"
    path = scored if scored.exists() else discovered
    if not path.exists():
        print(f"[targeted] {path} not found -- skipping CSV-based discovery")
        return []

    found = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            url = row.get("url", "")
            if not url:
                continue
            domain = urlparse(url).netloc.lower()
            if any(hp in domain for hp in HIGH_PRIORITY_DOMAINS):
                found.append({"url": url, "title": row.get("title", ""), "publisher": row.get("publisher", "")})
    print(f"[targeted] {len(found)} high-priority-domain URLs found in {path.name}")
    return found


def dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for item in items:
        key = item["url"].strip().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def download_one(session, item: dict) -> dict:
    url = item["url"]
    title = item.get("title") or ""
    publisher = item.get("publisher") or "manual"

    resp, note = polite_get(url, session=session)
    if resp is None or note != "ok":
        return {"url": url, "title": title, "publisher": publisher, "status": f"failed:{note}", "file_path": ""}

    content = resp.content
    is_pdf = content[:5] == b"%PDF-"
    ext = ".pdf" if is_pdf else ".html"
    base_name = safe_filename(title or url)
    out_path = MANUAL_DIR / f"{base_name}{ext}"
    counter = 1
    while out_path.exists():
        out_path = MANUAL_DIR / f"{base_name}_{counter}{ext}"
        counter += 1

    try:
        out_path.write_bytes(content)
    except OSError as exc:
        return {"url": url, "title": title, "publisher": publisher, "status": f"write_error:{exc}", "file_path": ""}

    return {
        "url": url, "title": title, "publisher": publisher, "status": "ok",
        "file_path": str(out_path.relative_to(CORPUS_DIR)),
    }


def main():
    print("[targeted] Starting ...")
    session = new_session()

    all_items = dedupe(HARDCODED_URLS + discover_high_priority_from_csv())
    total = len(all_items)
    print(f"[targeted] {total} URLs queued for download -> {MANUAL_DIR}")

    if not all_items:
        print("[targeted] Nothing to download. Done.")
        return

    results = []
    ok, failed = 0, 0
    for i, item in enumerate(all_items):
        if (i + 1) % 100 == 0:
            print(f"[targeted] Processing {i + 1}/{total}...")
        result = download_one(session, item)
        results.append(result)
        if result["status"] == "ok":
            ok += 1
            print(f"  [ OK ] {item['url']} -> {result['file_path']}")
        else:
            failed += 1
            print(f"  [FAIL] {item['url']} -> {result['status']}")

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["url", "title", "publisher", "status", "file_path"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[targeted] DONE. downloaded={ok} failed={failed} total={total}")
    print(f"[targeted] Log written to {LOG_CSV}")
    print("[targeted] Done.")


if __name__ == "__main__":
    main()
