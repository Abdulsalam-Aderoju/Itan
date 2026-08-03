#!/usr/bin/env python
"""
Stage 1 (build): automated download.

This reads harvest/download_manifest.csv (already filtered to score >= 50,
open_access = True, blocked = False, deduplicated, priority-ranked by
corpus/harvest/04_manifest.py) and downloads every one of those sources
automatically into corpus/raw/<publisher>/. Nothing here is manual: this is
the "pipe the harvest results straight into downloads" stage.

Every attempt (success or failure) is logged. Successes are appended to
sources.csv, the provenance ground truth for every later stage. Failures
(403/404/timeout/blocked) are appended to blocked_sources.txt and the run
continues — a bad source never crashes the pipeline.

Runnable standalone: python corpus/01_fetch.py
"""
from __future__ import annotations

import csv
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS_DIR, HARVEST_DIR, RAW_DIR, SOURCES_FIELDS, new_session, polite_get, safe_filename  # noqa: E402

MANIFEST_CSV = HARVEST_DIR / "download_manifest.csv"
SCORED_CSV = HARVEST_DIR / "sources_scored.csv"
SOURCES_CSV = CORPUS_DIR / "sources.csv"
BLOCKED_TXT = CORPUS_DIR / "blocked_sources.txt"
FETCH_LOG_CSV = CORPUS_DIR / "logs" / "fetch_log.csv"

FETCH_LOG_FIELDS = ["id", "url", "status", "http_code", "file_path", "timestamp"]


def load_manifest() -> list[dict]:
    if not MANIFEST_CSV.exists():
        print(f"[fetch] ERROR: {MANIFEST_CSV} not found.")
        print("[fetch] Run the harvest pipeline first: 01_discover.py -> 02_score.py -> 03_gap_analysis.py -> 04_manifest.py")
        sys.exit(1)
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_licences() -> dict[str, str]:
    if not SCORED_CSV.exists():
        return {}
    with open(SCORED_CSV, newline="", encoding="utf-8") as fh:
        return {r["id"]: r.get("licence", "") for r in csv.DictReader(fh)}


def append_row(path: Path, fieldnames: list[str], row: dict):
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def log_blocked(row: dict, note: str, http_code: str = ""):
    with open(BLOCKED_TXT, "a", encoding="utf-8") as fh:
        fh.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"id={row['id']} status={note} http_code={http_code} url={row['url']}\n"
        )


def is_pdf_bytes(content: bytes) -> bool:
    return content[:5] == b"%PDF-"


def main():
    manifest = load_manifest()
    licences = load_licences()
    session = new_session()

    already_have = set()
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, newline="", encoding="utf-8") as fh:
            already_have = {r["source_id"] for r in csv.DictReader(fh)}

    print(f"[fetch] {len(manifest)} sources in manifest. Beginning automated download...")

    ok, failed, skipped = 0, 0, 0
    for row in manifest:
        sid = row["id"]
        if sid in already_have:
            skipped += 1
            continue

        publisher_dir = RAW_DIR / safe_filename(row["publisher"])
        publisher_dir.mkdir(parents=True, exist_ok=True)
        base_name = safe_filename(row["title"] or sid)
        fmt = (row.get("format") or "HTML").upper()
        ext = ".pdf" if fmt == "PDF" else ".html"
        file_path = publisher_dir / f"{base_name}{ext}"

        resp, note = polite_get(row["url"], session=session)
        http_code = str(getattr(resp, "status_code", "")) if resp is not None else ""

        if resp is None or note not in ("ok",):
            print(f"  [FAIL] {sid} :: {row['title'][:60]!r} -> {note} ({http_code})")
            log_blocked(row, note, http_code)
            append_row(FETCH_LOG_CSV, FETCH_LOG_FIELDS, {
                "id": sid, "url": row["url"], "status": f"failed:{note}",
                "http_code": http_code, "file_path": "", "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            failed += 1
            continue

        content = resp.content
        # The manifest's declared format is only a guess (e.g. every OpenAlex
        # row defaults to HTML regardless of what the URL actually serves) --
        # sniff the real bytes and correct the extension in EITHER direction
        # rather than trusting `fmt`. Originally this only guarded PDF ->
        # HTML (redirected to a landing/paywall page); the missing HTML ->
        # PDF direction let 154 sources that really were PDFs get saved with
        # a .html extension, which then sent them through 02_extract.py's
        # BeautifulSoup path -- decoding raw PDF binary as UTF-8 text
        # produces garbage that still contains literal PDF-syntax fragments
        # (found via corpus/chunks.parquet: 8,517 chunks/154 docs affected).
        actual_ext = ".pdf" if is_pdf_bytes(content) else ".html"
        if actual_ext != ext:
            ext = actual_ext
            file_path = publisher_dir / f"{base_name}{ext}"

        try:
            file_path.write_bytes(content)
        except OSError as exc:
            print(f"  [FAIL] {sid} :: could not write file ({exc})")
            log_blocked(row, f"write_error:{exc}")
            failed += 1
            continue

        append_row(SOURCES_CSV, SOURCES_FIELDS, {
            "source_id": sid,
            "url": row["url"],
            "title": row["title"],
            "publisher": row["publisher"],
            "year": row.get("year", ""),
            "licence": licences.get(sid, ""),
            "crops_covered": row.get("crops_covered", ""),
            "has_tables": "unknown",  # populated by 02_extract.py once the file is parsed
            "file_path": str(file_path.relative_to(CORPUS_DIR)),
            "retrieval_date": date.today().isoformat(),
        })
        append_row(FETCH_LOG_CSV, FETCH_LOG_FIELDS, {
            "id": sid, "url": row["url"], "status": "ok", "http_code": http_code,
            "file_path": str(file_path.relative_to(CORPUS_DIR)),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"  [ OK ] {sid} :: {row['title'][:60]!r} -> {file_path.relative_to(CORPUS_DIR)}")
        ok += 1

    print(f"\n[fetch] DONE. downloaded={ok} failed={failed} skipped_existing={skipped} total_in_manifest={len(manifest)}")
    print(f"[fetch] Provenance ledger: {SOURCES_CSV}")
    print(f"[fetch] Failures logged to: {BLOCKED_TXT}")
    if ok < 30:
        print(f"[fetch] WARNING: only {ok} sources downloaded. Consider working through "
              f"harvest/gap_report.txt for manual sourcing before proceeding to extraction.")


if __name__ == "__main__":
    main()
