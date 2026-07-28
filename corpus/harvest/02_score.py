#!/usr/bin/env python
"""
Stage 2 (harvest): score every discovered source for corpus value, without
downloading anything. Reads sources_discovered.csv, writes sources_scored.csv.

Runnable standalone: python corpus/harvest/02_score.py
"""
from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CROPS, ZONES, HARVEST_DIR, SQL_TABLE_KEYWORDS  # noqa: E402

IN_CSV = HARVEST_DIR / "sources_discovered.csv"
OUT_CSV = HARVEST_DIR / "sources_scored.csv"

PUBLISHER_TIER = {
    "NAERLS": 30, "NAERLS (via Google Scholar)": 30, "FMARD": 30, "ADP": 30,
    "IITA": 25, "CGIAR": 25, "CIMMYT": 25, "ICRISAT": 25, "AfricaRice": 25,
    "FAO": 20, "CABI PlantWise": 20,
    "ResearchGate": 10, "academia.edu (NAERLS fallback)": 10,
    "NIHORT": 30, "NCRI": 30, "CRIN": 30, "NAFDAC": 25,
}

ZONE_KEYWORDS_LOOSE = ["savanna", "forest zone", "mangrove", "coastal"]


def relevance_score(row) -> tuple[int, dict]:
    crops = [c for c in row["crops_covered"].split(";") if c]
    crop_pts = min(len(crops) * 4, 40)

    text = f"{row['title']} {row['abstract_snippet']}".lower()
    zones = [z for z in row["zones_mentioned"].split(";") if z]
    if zones or any(k in text for k in ZONE_KEYWORDS_LOOSE):
        zone_pts = 10
    elif "nigeria" in text:
        zone_pts = 5
    elif "west africa" in text:
        zone_pts = 2
    else:
        zone_pts = 0

    tables = [t for t in row["sql_tables_likely"].split(";") if t]
    table_pts = min(len(tables) * 5, 30)

    total = min(crop_pts + zone_pts + table_pts, 40)
    return total, {"crop_pts": crop_pts, "zone_pts": zone_pts, "table_pts": table_pts}


def quality_score(row) -> tuple[int, dict]:
    pub_pts = PUBLISHER_TIER.get(row["publisher"], 10)
    pub_pts = min(pub_pts, 30)

    year = row.get("year", "")
    try:
        year_i = int(year)
    except (ValueError, TypeError):
        year_i = None
    if year_i is not None and year_i > 2015:
        year_pts = 10
    elif year_i is not None and 2010 <= year_i <= 2015:
        year_pts = 5
    else:
        year_pts = 0

    fmt = row.get("format", "")
    try:
        pages = int(row.get("page_count_estimate") or 0)
    except ValueError:
        pages = 0
    if fmt == "PDF" and pages > 10:
        fmt_pts = 10
    else:
        fmt_pts = 5

    # Publisher tier alone can hit 30, which would leave no room for the
    # year/format signals within the 30-point quality budget, so it is
    # weighted down to ~2/3 before adding the other two components.
    total = min(round(pub_pts * 0.67) + year_pts + fmt_pts, 30)
    return total, {"publisher_pts_raw": pub_pts, "year_pts": year_pts, "format_pts": fmt_pts}


def accessibility_score(row) -> tuple[int, dict]:
    blocked = str(row.get("blocked", "")).lower() == "true"
    paywall = str(row.get("paywall", "")).lower() == "true"
    open_access = str(row.get("open_access", "")).lower() == "true"
    fmt = row.get("format", "")
    publisher = row.get("publisher", "")

    if blocked:
        pts = 0
    elif paywall:
        pts = 0
    elif "ResearchGate" in publisher and not open_access:
        pts = 10
    elif open_access and fmt == "PDF":
        pts = 30
    elif open_access and fmt == "HTML":
        pts = 20
    elif "ResearchGate" in publisher:
        pts = 10
    else:
        pts = 0
    return pts, {"blocked": blocked, "paywall": paywall}


def main():
    print("[score] Starting ...")
    if not IN_CSV.exists():
        print(f"[score] ERROR: {IN_CSV} not found. Run 01_discover.py first.")
        sys.exit(1)

    with open(IN_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    total_rows = len(rows)
    print(f"[score] Loaded {total_rows} rows from {IN_CSV}")

    if not rows:
        print("[score] No rows to score.")
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        print("[score] Done.")
        return

    # crop/zone combo counts for gap_filler flag
    combo_counts: dict[tuple, int] = {}
    for r in rows:
        crops = [c for c in r["crops_covered"].split(";") if c]
        zones = [z for z in r["zones_mentioned"].split(";") if z] or ["(unspecified)"]
        for c in crops:
            for z in zones:
                combo_counts[(c, z)] = combo_counts.get((c, z), 0) + 1

    titles = [r["title"] for r in rows]
    crop_sets = [set(c for c in r["crops_covered"].split(";") if c) for r in rows]
    scored = []
    for i, r in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"[score] Processing {i + 1}/{total_rows}...")
        rel, rel_detail = relevance_score(r)
        qual, qual_detail = quality_score(r)
        acc, acc_detail = accessibility_score(r)
        final = rel + qual + acc

        crops = [c for c in r["crops_covered"].split(";") if c]
        zones = [z for z in r["zones_mentioned"].split(";") if z] or ["(unspecified)"]
        gap_filler = any(combo_counts.get((c, z), 0) < 3 for c in crops for z in zones)

        text = f"{r['title']} {r['abstract_snippet']}".lower()
        tier_a_feeder = any(
            any(kw in text for kw in kws)
            for table, kws in SQL_TABLE_KEYWORDS.items()
            if table in ("fertilizer_rate", "spacing", "crop_calendar")
        )

        duplicate_risk = False
        row_crops = crop_sets[i]
        for j in range(i):
            # Titles following a shared template ("X Production Guide for
            # Nigeria") can score >0.85 on pure text similarity even when X
            # is a different crop entirely -- so a title match only counts
            # as a real duplicate risk when the two rows also share at
            # least one crop (or neither names a crop, e.g. both unparsed).
            if not (row_crops & crop_sets[j] or (not row_crops and not crop_sets[j])):
                continue
            if titles[j] and difflib.SequenceMatcher(None, r["title"].lower(), titles[j].lower()).ratio() > 0.85:
                duplicate_risk = True
                break

        out = dict(r)
        out.update({
            "relevance_score": rel,
            "quality_score": qual,
            "accessibility_score": acc,
            "score": final,
            "gap_filler": gap_filler,
            "tier_a_feeder": tier_a_feeder,
            "duplicate_risk": duplicate_risk,
        })
        scored.append(out)

    fieldnames = list(scored[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)

    # ---- summary ----
    total = len(scored)
    per_crop = {c: 0 for c in CROPS}
    per_zone = {z: 0 for z in ZONES}
    per_table = {t: 0 for t in SQL_TABLE_KEYWORDS}
    for r in scored:
        for c in r["crops_covered"].split(";"):
            if c in per_crop:
                per_crop[c] += 1
        for z in r["zones_mentioned"].split(";"):
            if z in per_zone:
                per_zone[z] += 1
        for t in r["sql_tables_likely"].split(";"):
            if t in per_table:
                per_table[t] += 1

    high_score = [r for r in scored if r["score"] > 60]
    gap_fillers = [r for r in scored if r["gap_filler"]]

    print(f"\n[score] Total sources scored: {total}")
    print("[score] Per-crop counts:")
    for c, n in per_crop.items():
        print(f"    {c:12s} {n}")
    print("[score] Per-zone counts:")
    for z, n in per_zone.items():
        print(f"    {z:26s} {n}")
    print("[score] Per-SQL-table coverage estimate:")
    for t, n in per_table.items():
        print(f"    {t:18s} {n}")
    print(f"[score] Sources with score > 60: {len(high_score)}")
    print(f"[score] Sources flagged gap_filler: {len(gap_fillers)}")
    print(f"[score] Written to {OUT_CSV}")
    print("[score] Done.")


if __name__ == "__main__":
    main()
