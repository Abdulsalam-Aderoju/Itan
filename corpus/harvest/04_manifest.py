#!/usr/bin/env python
"""
Stage 4 (harvest): turn scored sources into a prioritised, actionable
download manifest. Still zero downloads here — this only writes CSV/MD/SH
files. The automated part of the pipeline (corpus/01_fetch.py) reads
download_manifest.csv next and performs the actual downloads.

Runnable standalone: python corpus/harvest/04_manifest.py
"""
from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import HARVEST_DIR, safe_filename  # noqa: E402

IN_CSV = HARVEST_DIR / "sources_scored.csv"
OUT_CSV = HARVEST_DIR / "download_manifest.csv"
OUT_MD = HARVEST_DIR / "download_manifest.md"
OUT_WGET = HARVEST_DIR / "wget_commands.sh"

MANIFEST_FIELDS = [
    "priority_rank", "id", "url", "title", "publisher", "year", "format",
    "score", "crops_covered", "sql_tables_likely", "gap_filler",
    "tier_a_feeder", "estimated_pages", "download_instructions",
]


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def instructions_for(row) -> str:
    publisher = row["publisher"]
    fmt = row["format"]
    url = row["url"]
    fname = safe_filename(row["title"] or row["id"])

    if publisher == "NAFDAC":
        return f"NAFDAC register page — screenshot or copy table to nafdac_{fname}.txt"
    if publisher == "CABI PlantWise":
        crop = (row["crops_covered"].split(";") or [""])[0]
        return f"PlantWise factsheet — visit URL, select all, copy to {crop}_{fname}_plantwise.txt"
    if fmt == "PDF":
        return f"Direct PDF — right-click and Save As, or wget {url}"
    return f"HTML page — use wget --mirror or copy-paste to {fname}.html"


def estimate_pages(row) -> int:
    try:
        p = int(row.get("page_count_estimate") or 0)
        if p > 0:
            return p
    except ValueError:
        pass
    return 15 if row["format"] == "PDF" else 1


def main():
    print("[manifest] Starting ...")
    if not IN_CSV.exists():
        print(f"[manifest] ERROR: {IN_CSV} not found. Run 02_score.py first.")
        sys.exit(1)

    with open(IN_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[manifest] Loaded {len(rows)} scored rows from {IN_CSV}")

    filtered = [
        r for r in rows
        if truthy(r.get("open_access")) and not truthy(r.get("blocked"))
        and float(r.get("score") or 0) >= 50
    ]
    filtered.sort(key=lambda r: float(r["score"]), reverse=True)
    print(f"[manifest] {len(filtered)} rows pass the score>=50/open/not-blocked filter -- deduplicating...")

    # Dedup: the `duplicate_risk` column from 02_score.py was computed by
    # comparing each row only to earlier rows in DISCOVERY order, which is
    # not the score-sorted order used here -- blindly trusting that flag
    # could drop a high-scoring row just because it happened to resemble
    # some other (possibly excluded/low-scoring) row. Instead, re-check
    # similarity directly against titles already kept in THIS sorted list:
    # a row is only dropped if it actually duplicates something better (or
    # equally) ranked that made it into the manifest.
    #
    # A title match alone is not enough: templated titles ("X Production
    # Guide for Nigeria") score >0.85 on pure text similarity even when X is
    # a different crop, so two rows are only treated as duplicates when they
    # also share at least one crop.
    kept = []
    kept_entries = []  # (title, crop_set) for everything kept so far
    total_filtered = len(filtered)
    for i, r in enumerate(filtered):
        if (i + 1) % 100 == 0:
            print(f"[manifest] Processing {i + 1}/{total_filtered}...")
        row_crops = set(c for c in r["crops_covered"].split(";") if c)
        is_dup_of_kept = any(
            (row_crops & kc or (not row_crops and not kc))
            and difflib.SequenceMatcher(None, r["title"].lower(), kt.lower()).ratio() > 0.85
            for kt, kc in kept_entries
        )
        if is_dup_of_kept:
            continue
        kept.append(r)
        kept_entries.append((r["title"], row_crops))
    print(f"[manifest] Deduplication done: {len(kept)}/{total_filtered} kept.")

    manifest_rows = []
    for i, r in enumerate(kept, start=1):
        if i % 100 == 0:
            print(f"[manifest] Processing {i}/{len(kept)}...")
        manifest_rows.append({
            "priority_rank": i,
            "id": r["id"],
            "url": r["url"],
            "title": r["title"],
            "publisher": r["publisher"],
            "year": r.get("year", ""),
            "format": r["format"],
            "score": r["score"],
            "crops_covered": r["crops_covered"],
            "sql_tables_likely": r["sql_tables_likely"],
            "gap_filler": r.get("gap_filler", False),
            "tier_a_feeder": r.get("tier_a_feeder", False),
            "estimated_pages": estimate_pages(r),
            "download_instructions": instructions_for(r),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # ---- markdown, grouped by publisher ----
    by_publisher: dict[str, list] = {}
    for r in manifest_rows:
        by_publisher.setdefault(r["publisher"], []).append(r)

    md = ["# Download Manifest", "", (
        "Target: 400-600 pages of source material yields approximately "
        "10,000 chunks after extraction and cleaning."
    ), ""]
    running_total = 0
    for publisher, items in sorted(by_publisher.items(), key=lambda kv: -sum(x["estimated_pages"] for x in kv[1])):
        pub_pages = sum(x["estimated_pages"] for x in items)
        running_total += pub_pages
        md.append(f"## {publisher} — {len(items)} sources, ~{pub_pages} pages (running total: {running_total})")
        md.append("")
        for r in sorted(items, key=lambda x: x["priority_rank"]):
            md.append(
                f"- **[{r['priority_rank']}] {r['title']}** ({r['year'] or 'n.d.'}, score {r['score']}) "
                f"— {r['url']}\n  - crops: {r['crops_covered']} | tables: {r['sql_tables_likely']} "
                f"| gap_filler={r['gap_filler']} tier_a_feeder={r['tier_a_feeder']}\n"
                f"  - {r['download_instructions']}"
            )
        md.append("")
    md.append(f"**Total estimated pages across manifest: {running_total}**")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    # ---- wget script for direct PDFs ----
    wget_lines = ["#!/bin/bash", "set -e", "", "# Auto-generated by 04_manifest.py — batch download of direct PDF links.", ""]
    pdf_count = 0
    for r in manifest_rows:
        if r["format"] != "PDF" or not r["url"].lower().startswith("http"):
            continue
        pub_dir = safe_filename(r["publisher"])
        fname = safe_filename(r["title"] or r["id"]) + ".pdf"
        out_path = f"corpus/raw/{pub_dir}/{fname}"
        wget_lines.append(f'mkdir -p "corpus/raw/{pub_dir}"')
        wget_lines.append(f'wget --wait=2 --timeout=10 -U "Mozilla/5.0" --output-document="{out_path}" "{r["url"]}" || echo "FAILED: {r["url"]}"')
        pdf_count += 1

    with open(OUT_WGET, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(wget_lines) + "\n")
    OUT_WGET.chmod(0o755)

    print(f"[manifest] {len(rows)} scored sources -> {len(manifest_rows)} in manifest (score>=50, open, not blocked)")
    print(f"[manifest] {pdf_count} direct PDF wget commands written")
    print(f"[manifest] Written: {OUT_CSV}, {OUT_MD}, {OUT_WGET}")
    print("[manifest] Done.")


if __name__ == "__main__":
    main()
