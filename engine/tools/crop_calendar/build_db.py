#!/usr/bin/env python
"""Builds the crop_calendar module's own SQLite database from
corpus/structured.db's auto-extracted crop_calendar table.

This is a thin ETL: it does not re-derive or reinterpret rows (that is
05_structure_extract.py's job), it only dedupes exact-duplicate matches
(the regex-based extractor can match the same sentence more than once
across a document) and copies rows into a stable, module-local schema so
the crop_calendar() tool function does not need to reach into the corpus
harvest pipeline's internals at call time.

NOTE ON COVERAGE: as of 2026-08-17, corpus/structured.db's crop_calendar
table has 60 rows, covering 8 of the 10 target crops (missing: cowpea,
pepper). Audited the same way pest_lookup.db was audited for phantom
empty-content matches (see engine/tools/pest_lookup/build_db.py) -- no
equivalent bug found here: every row has a non-empty month_start, because
05_structure_extract.py's crop_calendar extraction only creates a row when
it actually captured a month value, unlike pest extraction, which creates
a row on any pest-keyword regex match regardless of whether the nearby
symptom/control sections were found. This script surfaces whatever exists
rather than filling gaps -- see README.md. Zone/state coverage is still
thin (most rows have neither) -- that's a corpus-content gap, not a bug
here; see Itan_ADTC2026_Blueprint_v2.pdf SS2.5 for the retrieval-side
confirmation that zone-specific answers aren't currently supportable.

Runnable standalone: python engine/tools/crop_calendar/build_db.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
STRUCTURED_DB_PATH = REPO_ROOT / "corpus" / "structured.db"
OUT_DB_PATH = MODULE_DIR / "crop_calendar.db"
SCHEMA_PATH = MODULE_DIR / "schema.sql"


def main():
    if not STRUCTURED_DB_PATH.exists():
        print(f"[crop_calendar] ERROR: {STRUCTURED_DB_PATH} not found. "
              f"Run corpus/05_structure_extract.py first.")
        sys.exit(1)

    if OUT_DB_PATH.exists():
        OUT_DB_PATH.unlink()

    src_conn = sqlite3.connect(STRUCTURED_DB_PATH)
    src_conn.row_factory = sqlite3.Row
    out_conn = sqlite3.connect(OUT_DB_PATH)

    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        out_conn.executescript(fh.read())

    rows = src_conn.execute(
        "SELECT crop, zone, state, activity, month_start, month_end, "
        "confidence, needs_review, source_id FROM crop_calendar"
    ).fetchall()

    seen = set()
    inserted = 0
    for r in rows:
        key = (r["crop"], r["zone"], r["state"], r["activity"], r["month_start"], r["month_end"], r["source_id"])
        if key in seen:
            continue
        seen.add(key)
        out_conn.execute(
            "INSERT INTO crop_calendar (crop, zone, state, activity, month_start, month_end, "
            "confidence, needs_review, source_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r["crop"], r["zone"] or None, r["state"] or None, r["activity"],
             r["month_start"], r["month_end"] or None, r["confidence"],
             int(r["needs_review"]), r["source_id"]),
        )
        inserted += 1

    out_conn.commit()

    crops_covered = {r["crop"] for r in rows if r["crop"]}
    print(f"[crop_calendar] {inserted} rows written to {OUT_DB_PATH} "
          f"(from {len(rows)} source rows, {len(rows) - inserted} exact-duplicate)")
    print(f"[crop_calendar] Crops with at least one row: {sorted(crops_covered) or 'NONE'}")

    src_conn.close()
    out_conn.close()


if __name__ == "__main__":
    main()
