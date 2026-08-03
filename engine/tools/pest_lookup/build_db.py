#!/usr/bin/env python
"""Builds the pest_lookup module's own SQLite database from
corpus/structured.db's auto-extracted pest table.

corpus/05_structure_extract.py's PEST_NAME_RE captures whatever noun
phrase surrounds a pest keyword ("usually attacked by thrips", "Symptoms
of thrips", "season thrips"), so the raw `pest` table has ~12,200 rows
that are not directly exact-matchable. This script canonicalizes each raw
pest_name down to its keyword (see normalize.py), groups rows by
(canonical pest_name, crop), and merges each group into one clean record:

  - confidence: the max confidence among the group's raw rows
  - each of symptoms / growth_stage / cultural_control / chemical_control:
    the longest non-empty value for that field across the group (each
    value comes from a single raw row, so no cross-document text is
    spliced together within one field)
  - source_ids: up to MAX_SOURCE_IDS contributing citations (source_count
    records the true total, even when the stored list is capped)

Runnable standalone: python engine/tools/pest_lookup/build_db.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
STRUCTURED_DB_PATH = REPO_ROOT / "corpus" / "structured.db"
OUT_DB_PATH = MODULE_DIR / "pest_lookup.db"
SCHEMA_PATH = MODULE_DIR / "schema.sql"

sys.path.insert(0, str(REPO_ROOT))
from engine.tools.pest_lookup.normalize import canonicalize_pest_name  # noqa: E402

# Mirrors corpus/05_structure_extract.py's CONFIDENCE_THRESHOLD -- cannot
# import it directly (that module's filename starts with a digit).
CONFIDENCE_THRESHOLD = 0.7
MAX_SOURCE_IDS = 20


def main():
    if not STRUCTURED_DB_PATH.exists():
        print(f"[pest_lookup] ERROR: {STRUCTURED_DB_PATH} not found. "
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
        "SELECT crop, pest_name, symptoms, growth_stage, cultural_control, "
        "chemical_control, confidence, source_id FROM pest"
    ).fetchall()

    groups: dict[tuple[str, str | None], list[sqlite3.Row]] = defaultdict(list)
    skipped_unparseable = 0
    for r in rows:
        canonical = canonicalize_pest_name(r["pest_name"])
        if canonical is None:
            skipped_unparseable += 1
            continue
        crop_key = r["crop"] or None
        groups[(canonical, crop_key)].append(r)

    inserted = 0
    for (canonical, crop_key), group_rows in groups.items():
        confidence = max(r["confidence"] for r in group_rows)

        def longest(field: str) -> str:
            candidates = [r[field] for r in group_rows if r[field]]
            return max(candidates, key=len) if candidates else ""

        symptoms = longest("symptoms")
        growth_stage = longest("growth_stage")
        cultural_control = longest("cultural_control")
        chemical_control = longest("chemical_control")

        all_source_ids = sorted({r["source_id"] for r in group_rows})
        source_ids = all_source_ids[:MAX_SOURCE_IDS]

        out_conn.execute(
            "INSERT INTO pest (pest_name, crop, symptoms, growth_stage, cultural_control, "
            "chemical_control, confidence, needs_review, source_count, source_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (canonical, crop_key, symptoms, growth_stage, cultural_control, chemical_control,
             confidence, int(confidence < CONFIDENCE_THRESHOLD), len(group_rows),
             json.dumps(source_ids)),
        )
        inserted += 1

    out_conn.commit()

    print(f"[pest_lookup] {len(rows)} raw rows -> {skipped_unparseable} unparseable "
          f"(no keyword matched), {inserted} canonical (pest, crop) records written to {OUT_DB_PATH}")
    canonical_pests = sorted({k[0] for k in groups})
    print(f"[pest_lookup] Canonical pests with data: {canonical_pests}")

    src_conn.close()
    out_conn.close()


if __name__ == "__main__":
    main()
