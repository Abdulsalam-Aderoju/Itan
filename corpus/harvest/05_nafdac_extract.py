#!/usr/bin/env python
"""
Stage 5 (harvest): NAFDAC agrochemical register special-case handler.
NAFDAC's register is structured HTML (a table), not a document to chunk, so
it is parsed directly into a flat CSV that can be loaded straight into the
`agrochemical` SQLite table later, bypassing 04_chunk.py entirely.

Runnable standalone: python corpus/harvest/05_nafdac_extract.py
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CROPS, HARVEST_DIR, new_session, polite_get  # noqa: E402

from bs4 import BeautifulSoup

OUT_CSV = HARVEST_DIR / "nafdac_agrochemicals.csv"
BASE = "https://www.nafdac.gov.ng"
CANDIDATE_PATHS = [
    "/our-services/registration/agrochemicals/",
    "/our-services/registration-and-regulatory-affairs/agrochemicals-and-pesticides/",
    "/category/agrochemicals/",
]

FIELDS = [
    "product_name", "active_ingredient", "registration_number", "approval_date",
    "crop_indications", "application_rate", "priority", "source_url", "retrieval_date",
]

# Loose row-matching regexes for whatever table/list markup NAFDAC serves.
REG_NO_RE = re.compile(r"\b([A-Z]{2,6}[/-]?\d{2,6}[/-]?\d{0,6})\b")
RATE_RE = re.compile(r"(\d+(?:\.\d+)?\s?(?:ml|l|g|kg)\s?/\s?(?:l|litre|ha|hectare))", re.I)


def parse_table_rows(soup) -> list[dict]:
    rows = []
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if not cells or len(cells) < 2:
                continue
            row = {}
            if headers and len(headers) == len(cells):
                row = dict(zip(headers, cells))
            else:
                # Best-effort positional guess: name, active ingredient, reg no, crop
                row = {
                    "product name": cells[0] if len(cells) > 0 else "",
                    "active ingredient": cells[1] if len(cells) > 1 else "",
                    "registration number": cells[2] if len(cells) > 2 else "",
                    "crop": cells[3] if len(cells) > 3 else "",
                }
            rows.append(row)
    return rows


def normalize_row(raw: dict, source_url: str) -> dict | None:
    def pick(*keys):
        for k in keys:
            for rk, rv in raw.items():
                if k in rk:
                    return rv
        return ""

    product = pick("product", "trade name", "name")
    if not product:
        return None
    active = pick("active ingredient", "active", "ingredient")
    reg_no = pick("registration", "reg no", "reg. no", "number")
    if not reg_no:
        m = REG_NO_RE.search(" ".join(raw.values()))
        reg_no = m.group(1) if m else ""
    approval = pick("approval", "date")
    crop = pick("crop", "indication", "use")
    rate_m = RATE_RE.search(" ".join(raw.values()))
    rate = rate_m.group(1) if rate_m else ""

    crop_l = crop.lower()
    priority = any(c in crop_l for c in CROPS)

    return {
        "product_name": product,
        "active_ingredient": active,
        "registration_number": reg_no,
        "approval_date": approval,
        "crop_indications": crop,
        "application_rate": rate,
        "priority": priority,
        "source_url": source_url,
        "retrieval_date": date.today().isoformat(),
    }


def main():
    print("[nafdac] Starting ...")
    session = new_session()
    all_rows: list[dict] = []

    for path in CANDIDATE_PATHS:
        url = urljoin(BASE, path)
        resp, note = polite_get(url, session=session)
        if resp is None or note != "ok":
            print(f"[nafdac] {path}: {note}")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        raw_rows = parse_table_rows(soup)
        print(f"[nafdac] {path}: {len(raw_rows)} raw table rows found")
        for i, raw in enumerate(raw_rows):
            if (i + 1) % 100 == 0:
                print(f"[nafdac] Processing {i + 1}/{len(raw_rows)}...")
            norm = normalize_row(raw, url)
            if norm:
                all_rows.append(norm)

        # Follow any "next page" / pagination links for this section, capped.
        pag_links = soup.find_all("a", href=re.compile(r"page|paged", re.I))
        for a in pag_links[:5]:
            next_url = urljoin(BASE, a["href"])
            resp2, note2 = polite_get(next_url, session=session)
            if resp2 is None or note2 != "ok":
                continue
            soup2 = BeautifulSoup(resp2.text, "html.parser")
            page_rows = parse_table_rows(soup2)
            for i, raw in enumerate(page_rows):
                if (i + 1) % 100 == 0:
                    print(f"[nafdac] Processing {i + 1}/{len(page_rows)} (pagination)...")
                norm = normalize_row(raw, next_url)
                if norm:
                    all_rows.append(norm)

        if raw_rows:
            break  # found a working path, no need to try the alternates

    print(f"[nafdac] {len(all_rows)} normalized rows -- deduplicating...")

    # de-duplicate by (product_name, registration_number)
    seen = set()
    deduped = []
    for i, r in enumerate(all_rows):
        if (i + 1) % 100 == 0:
            print(f"[nafdac] Processing {i + 1}/{len(all_rows)}...")
        key = (r["product_name"].lower(), r["registration_number"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(deduped)

    priority_count = sum(1 for r in deduped if r["priority"])
    print(f"\n[nafdac] {len(deduped)} agrochemical entries written to {OUT_CSV}")
    print(f"[nafdac] {priority_count} flagged priority (match one of the 10 target crops)")
    if not deduped:
        print("[nafdac] NOTE: 0 rows extracted — NAFDAC's site structure may have changed, "
              "or the register may be JS-rendered. Manual verification recommended; "
              "this does not crash the pipeline.")
    print("[nafdac] Done.")


if __name__ == "__main__":
    main()
