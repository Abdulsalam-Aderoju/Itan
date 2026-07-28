#!/usr/bin/env python
"""
Stage 0 (harvest): catalogue manually-obtained files sitting in
corpus/raw/manual/ into sources_discovered.csv, so they flow through
scoring/gap-analysis/manifest like any automatically-discovered source.

This only records metadata -- it never reads PDF/text content, and it
never downloads anything (the files are already there, that's the point).

Row fields inferred:
  - publisher: the file's immediate subfolder name under raw/manual/
    (e.g. raw/manual/NAERLS/bulletin.pdf -> publisher "NAERLS"). A file
    sitting directly in raw/manual/ with no subfolder gets publisher
    "manual".
  - crops_covered: common.find_crops() applied to the filename.
  - country: "Nigeria" if any path component is named "crop_calendars"
    (as specified); otherwise common.find_country() applied to the
    subfolder/publisher name as a best-effort fallback, else blank.
  - id: starting from SRC06000 (or higher if sources_discovered.csv
    already has ids past that point) to avoid colliding with ids the
    automated crawlers in 01_discover.py already assigned.

Runs safely more than once: a file already catalogued (matched by its
relative path recorded in the url column) is skipped, not re-added.

Runnable standalone: python corpus/harvest/00_manual_intake.py
"""
from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CORPUS_DIR, HARVEST_DIR, find_country, find_crops, find_zones, guess_sql_tables  # noqa: E402

MANUAL_DIR = CORPUS_DIR / "raw" / "manual"
OUT_CSV = HARVEST_DIR / "sources_discovered.csv"

VALID_EXTENSIONS = {".pdf": "PDF", ".txt": "TEXT"}

DEFAULT_FIELDS = [
    "id", "url", "title", "publisher", "country", "year", "format", "page_count_estimate",
    "licence", "open_access", "crops_covered", "zones_mentioned",
    "sql_tables_likely", "abstract_snippet", "blocked", "paywall", "retrieval_date",
    "api_source", "authors", "language",
]


def get_fieldnames() -> list[str]:
    """Use the existing file's own header if there is one, so a schema
    that has drifted slightly from DEFAULT_FIELDS (e.g. a column added
    since this file was created) doesn't get misaligned by this script."""
    if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
        with open(OUT_CSV, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
        if header:
            return header
    return DEFAULT_FIELDS


def existing_state() -> tuple[set[str], int]:
    """Returns (already-catalogued relative paths, highest numeric SRC id
    seen) so this script can skip files it already added and avoid id
    collisions with anything 01_discover.py (or a prior run of this
    script) already assigned."""
    catalogued: set[str] = set()
    max_id = 0
    if not OUT_CSV.exists():
        return catalogued, max_id
    with open(OUT_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            url = row.get("url", "")
            if url:
                catalogued.add(url)
            sid = row.get("id", "")
            if sid.startswith("SRC"):
                try:
                    max_id = max(max_id, int(sid[3:]))
                except ValueError:
                    pass
    return catalogued, max_id


def scan_manual_files() -> list[Path]:
    if not MANUAL_DIR.exists():
        return []
    return sorted(
        p for p in MANUAL_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def infer_publisher(rel_path: Path) -> str:
    return rel_path.parts[0] if len(rel_path.parts) > 1 else "manual"


def infer_country(rel_path: Path, publisher: str) -> str:
    parts_lower = [p.lower() for p in rel_path.parts]
    if "crop_calendars" in parts_lower:
        return "Nigeria"
    # Best-effort fallback: does the subfolder/publisher name itself
    # mention a known country (e.g. a folder literally named "Kenya" or
    # "KALRO_Kenya")? common.find_country() is the same word-boundary
    # matcher used everywhere else in this pipeline.
    return find_country(" ".join(rel_path.parts) + " " + publisher) or ""


def infer_title(file_path: Path) -> str:
    return file_path.stem.replace("_", " ").replace("-", " ").strip().title()


def main():
    print("[intake] Starting ...")
    files = scan_manual_files()
    print(f"[intake] {len(files)} PDF/text files found under {MANUAL_DIR}")

    if not files:
        print("[intake] Nothing to add. Done.")
        return

    catalogued, max_id = existing_state()
    next_id_n = max(6000, max_id + 1)
    fieldnames = get_fieldnames()
    file_has_header = OUT_CSV.exists() and OUT_CSV.stat().st_size > 0

    added = 0
    skipped = 0
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_has_header:
            writer.writeheader()

        for file_path in files:
            rel_path = file_path.relative_to(CORPUS_DIR)
            url = str(rel_path).replace("\\", "/")
            if url in catalogued:
                print(f"  [SKIP] {url} (already catalogued)")
                skipped += 1
                continue

            rel_to_manual = file_path.relative_to(MANUAL_DIR)
            publisher = infer_publisher(rel_to_manual)
            country = infer_country(rel_to_manual, publisher)
            title = infer_title(file_path)
            fmt = VALID_EXTENSIONS[file_path.suffix.lower()]
            crops = find_crops(file_path.stem.replace("_", " ").replace("-", " "))
            zones = find_zones(title)
            tables = guess_sql_tables(title)

            row = {
                "id": f"SRC{next_id_n:05d}",
                "url": url,
                "title": title,
                "publisher": publisher,
                "country": country,
                "year": "",
                "format": fmt,
                "page_count_estimate": "",
                # Manually obtained -- licence/open-access status hasn't been
                # verified programmatically the way an API response's
                # metadata is, so this is flagged for a human to confirm
                # rather than asserted outright.
                "licence": "unknown - manual intake, verify licence",
                "open_access": True,
                "crops_covered": ";".join(sorted(crops)),
                "zones_mentioned": ";".join(sorted(zones)),
                "sql_tables_likely": ";".join(sorted(tables)),
                "abstract_snippet": "",
                "blocked": False,
                "paywall": False,
                "retrieval_date": date.today().isoformat(),
                "api_source": "manual_intake",
                "authors": "",
                "language": "",
            }
            # Only write fields the file's actual header has (handles an
            # older/newer schema than DEFAULT_FIELDS gracefully).
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            print(f"  [ADD] {row['id']} | {publisher} | {country or '(unknown)'} | {url}")
            added += 1
            next_id_n += 1

    print(f"\n[intake] DONE. added={added} skipped_existing={skipped} total_scanned={len(files)}")
    print(f"[intake] Written to {OUT_CSV}")
    print("[intake] Done.")


if __name__ == "__main__":
    main()
