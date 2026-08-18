#!/usr/bin/env python
"""
Stage 9 (harvest): load the NAFDAC animal health products & agrochemicals
register into structured.db's agrochemical table.

Mirrors 07_nafdac_load.py exactly (same target table, same reasoning) for
08_animal_health_pdf_extract.py's output. See that script's docstring for
the full rationale on why this is safe (structured.db's `agrochemical`
table is unrelated to spray_dilution()'s own hand-seeded database) and
what it does/doesn't give you.

Maps this register's COMPOSITION column to `active_ingredient` -- an
imperfect fit (COMPOSITION here is a full feed-ingredient list, e.g. "VIT
A(3,000,000IU),VIT D3(5000IU),..." for a multi-vitamin feed supplement,
not a single active ingredient the way the pesticide register's ACTIVE
INGREDIENT column is), but the closest available column and still
genuinely useful for "what's in this product" questions. No crop, rate,
or pre-harvest-interval data, same as the pesticide register.

MUST run AFTER 05_structure_extract.py, same reason as 07_nafdac_load.py
(that script does DROP TABLE IF EXISTS + CREATE TABLE for agrochemical on
every run). Safe to run in either order relative to 07_nafdac_load.py
itself (both only INSERT, never DROP/recreate).

Runnable standalone: python corpus/harvest/09_animal_health_load.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = CORPUS_DIR / "harvest" / "nafdac_animal_health_register.csv"
DB_PATH = CORPUS_DIR / "structured.db"

# Direct transcription from a structured government register (dedicated
# word-position table parser), same trust level as 07_nafdac_load.py's
# NAFDAC pesticide-register load. See 08_animal_health_pdf_extract.py --
# 0 of 281 rows were flagged needs_review on the final calibrated
# thresholds (no detected merge corruption), but the flagged path is kept
# here in case a future re-parse (e.g. after a PDF font-encoding fix)
# produces some.
CONFIDENCE_CLEAN = 0.95
CONFIDENCE_FLAGGED = 0.5


def main():
    if not CSV_PATH.exists():
        print(f"[animal_health_load] ERROR: {CSV_PATH} not found. Run 08_animal_health_pdf_extract.py first.")
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"[animal_health_load] ERROR: {DB_PATH} not found. Run 05_structure_extract.py first.")
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    conn = sqlite3.connect(DB_PATH)
    inserted = 0
    flagged = 0
    for r in rows:
        product_name = r["product_name"].strip()
        composition = r["composition"].strip()
        if not product_name or not composition:
            continue

        needs_review_src = r.get("needs_review", "").strip() == "True"
        confidence = CONFIDENCE_FLAGGED if needs_review_src else CONFIDENCE_CLEAN
        if needs_review_src:
            flagged += 1

        raw_text = (
            f"{product_name} | composition: {composition} | "
            f"reg. no. {r['registration_number']} | "
            f"approved {r['approval_date']}, expires {r['expiry_date']} | "
            f"applicant: {r['applicant']}"
        )[:400]

        conn.execute(
            "INSERT INTO agrochemical (source_id, product_name, active_ingredient, crop, "
            "rate, rate_unit, pre_harvest_interval_days, raw_text, confidence, needs_review) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r["source_id"], product_name, composition, None, None, None, None,
             raw_text, confidence, "1" if needs_review_src else "0"),
        )
        inserted += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM agrochemical").fetchone()[0]
    conn.close()

    print(f"[animal_health_load] {len(rows)} CSV rows -> {inserted} loaded ({flagged} flagged "
          f"needs_review), {len(rows) - inserted} skipped (missing product_name/composition). "
          f"agrochemical table now has {total} total rows.")


if __name__ == "__main__":
    main()
