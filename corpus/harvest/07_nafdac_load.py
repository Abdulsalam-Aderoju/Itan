#!/usr/bin/env python
"""
Stage 7 (harvest): load the NAFDAC pesticide register into structured.db's
agrochemical table.

06_nafdac_pdf_extract.py already parses the NAFDAC pesticide register PDF
into a clean, structured CSV (corpus/harvest/nafdac_pesticide_register.csv,
409 rows: product name, active ingredient, registration number, dates,
applicant, manufacturer) -- but nothing ever loaded it into structured.db.
05_structure_extract.py's own generic prose-regex extraction produced
exactly 1 agrochemical row from the whole corpus (it never had a chance
against this document -- 02_extract.py's generic pdfplumber text extraction
scrambles this borderless table into unreadable column-major garbage, which
is exactly why 06_nafdac_pdf_extract.py exists as a dedicated parser in the
first place). This script is the missing link between the two.

MUST run AFTER 05_structure_extract.py: that script does
DROP TABLE IF EXISTS + CREATE TABLE for agrochemical on every run, so
running this first would just get wiped on the next full pipeline pass.
See run_pipeline.sh for stage order.

What this does NOT give you: the NAFDAC register is an approval list
(product name, active ingredient, registration/expiry dates, applicant,
manufacturer) -- it has no crop, application rate, or pre-harvest-interval
data. Those columns are left empty for every row loaded here. This answers
"is <product> registered in Nigeria, and what's its active ingredient" --
it does NOT answer "what rate of <product> should I use on <crop>", which
still depends on the corpus's separately-sourced rate/PHI content (thin --
see Itan_ADTC2026_Blueprint_v2.pdf SS1.3).

Runnable standalone: python corpus/harvest/07_nafdac_load.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = CORPUS_DIR / "harvest" / "nafdac_pesticide_register.csv"
DB_PATH = CORPUS_DIR / "structured.db"

# Direct transcription from a structured government register (dedicated
# word-position table parser, not a prose regex guess) -- meaningfully more
# trustworthy than 05_structure_extract.py's label-proximity confidence
# scores (see blueprint SS1.4), but not perfect: 06_nafdac_pdf_extract.py
# already flags rows with jumbled source-PDF font/encoding issues via its
# own needs_review column, which this script respects rather than
# overriding.
CONFIDENCE_CLEAN = 0.95
CONFIDENCE_FLAGGED = 0.5


def main():
    if not CSV_PATH.exists():
        print(f"[nafdac_load] ERROR: {CSV_PATH} not found. Run 06_nafdac_pdf_extract.py first.")
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"[nafdac_load] ERROR: {DB_PATH} not found. Run 05_structure_extract.py first.")
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    conn = sqlite3.connect(DB_PATH)
    inserted = 0
    flagged = 0
    for r in rows:
        product_name = r["product_name"].strip()
        active_ingredient = r["active_ingredient"].strip()
        if not product_name or not active_ingredient:
            continue

        needs_review_src = r.get("needs_review", "").strip() == "True"
        confidence = CONFIDENCE_FLAGGED if needs_review_src else CONFIDENCE_CLEAN
        if needs_review_src:
            flagged += 1

        raw_text = (
            f"{product_name} | active ingredient: {active_ingredient} | "
            f"reg. no. {r['registration_number']} | "
            f"registered {r['date_registered']}, expires {r['reg_expiry']} | "
            f"applicant: {r['applicant']}"
        )[:400]

        conn.execute(
            "INSERT INTO agrochemical (source_id, product_name, active_ingredient, crop, "
            "rate, rate_unit, pre_harvest_interval_days, raw_text, confidence, needs_review) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r["source_id"], product_name, active_ingredient, None, None, None, None,
             raw_text, confidence, "1" if needs_review_src else "0"),
        )
        inserted += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM agrochemical").fetchone()[0]
    conn.close()

    print(f"[nafdac_load] {len(rows)} CSV rows -> {inserted} loaded ({flagged} flagged "
          f"needs_review from source-PDF encoding issues), {len(rows) - inserted} skipped "
          f"(missing product_name/active_ingredient). agrochemical table now has {total} "
          f"total rows.")


if __name__ == "__main__":
    main()
