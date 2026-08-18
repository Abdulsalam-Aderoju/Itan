#!/usr/bin/env python
"""Builds the fertilizer_rate module's own SQLite database from
corpus/structured.db's auto-extracted fertilizer_rate table.

corpus/05_structure_extract.py's fertilizer regex only reliably captures a
"kg N/ha"-style pattern -- it identifies a numeric rate whenever it finds
one (rate_kg_ha is non-empty on all 3,025 raw rows), but only identifies
WHICH nutrient (fertilizer_type) in 121 of them; the other 2,904 are a bare
number with no identified nutrient, which is not independently useful (a
"30 kg/ha" of what?). This script keeps only rows where BOTH
fertilizer_type and rate_kg_ha are populated, mirroring
engine/tools/pest_lookup/build_db.py's empty-content gate, and additionally
restricts to needs_review='0' (confidence >= 0.7) -- unlike pest_lookup,
which keeps needs_review rows and lets the caller decide, this table is
small enough (121 candidates) that a stricter bar was affordable without
losing meaningful coverage: 108 rows still remain, vs. e.g. the one
needs_review='1' row that would otherwise have surfaced "50100 kg of urea"
(a garbled "50-100" range digit-mashed together, with "of urea" captured
as the "nutrient type" instead of "urea"/"nitrogen") as if it were a clean
fact.

NOTE ON COVERAGE (2026-08-17): every populated fertilizer_type value today
is "nitrogen" (bar the one dropped garbage row above) -- the extraction
pattern has no phosphorus/potassium recognition, so this table cannot
currently answer NPK-blend questions the way engine/tools/agri_calc's own
small hand-seeded fertilizer_rate table (6 rows) can for its covered
crops. This table is additive to that one, not a replacement for it --
see engine/agent.py's query_structured_db(), which queries both.

`application_stage` is carried through as extracted, but in practice often
holds a unit-label artifact of the source regex (e.g. "kg_n_slash_ha")
rather than an actual growth-stage description ("at planting", "top-dress")
-- treat it as informational, not authoritative, until 05_structure_extract.py's
pattern for this field is tightened.

Runnable standalone: python engine/tools/fertilizer_rate/build_db.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
STRUCTURED_DB_PATH = REPO_ROOT / "corpus" / "structured.db"
OUT_DB_PATH = MODULE_DIR / "fertilizer_rate.db"
SCHEMA_PATH = MODULE_DIR / "schema.sql"


def main():
    if not STRUCTURED_DB_PATH.exists():
        print(f"[fertilizer_rate] ERROR: {STRUCTURED_DB_PATH} not found. "
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
        "SELECT crop, fertilizer_type, rate_kg_ha, application_stage, "
        "confidence, needs_review, source_id FROM fertilizer_rate"
    ).fetchall()

    seen = set()
    inserted = 0
    skipped_empty = 0
    skipped_unreviewed = 0
    crops_covered = set()
    for r in rows:
        if not r["fertilizer_type"] or not r["rate_kg_ha"]:
            skipped_empty += 1
            continue
        if str(r["needs_review"]) == "1":
            skipped_unreviewed += 1
            continue

        key = (r["crop"], r["fertilizer_type"], r["rate_kg_ha"], r["application_stage"], r["source_id"])
        if key in seen:
            continue
        seen.add(key)
        if r["crop"]:
            crops_covered.add(r["crop"])

        out_conn.execute(
            "INSERT INTO fertilizer_rate (crop, fertilizer_type, rate_kg_ha, application_stage, "
            "confidence, needs_review, source_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r["crop"] or None, r["fertilizer_type"], r["rate_kg_ha"], r["application_stage"] or None,
             r["confidence"], int(r["needs_review"]), r["source_id"]),
        )
        inserted += 1

    out_conn.commit()

    print(f"[fertilizer_rate] {len(rows)} raw rows -> {skipped_empty} skipped (no fertilizer_type "
          f"or rate), {skipped_unreviewed} skipped (needs_review), {inserted} written to {OUT_DB_PATH}")
    print(f"[fertilizer_rate] Crops with at least one usable row: {sorted(crops_covered) or 'NONE'}")

    src_conn.close()
    out_conn.close()


if __name__ == "__main__":
    main()
