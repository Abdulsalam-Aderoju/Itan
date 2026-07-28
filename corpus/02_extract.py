#!/usr/bin/env python
"""
Stage 2 (build): extract clean(ish) text from every downloaded source.

- PDFs -> pdfplumber, page by page. Prose and tables are extracted
  separately per page; tables are never discarded, they are converted to
  markdown and inlined as a fenced block so 04_chunk.py can find and
  protect them with a simple regex.
- HTML -> BeautifulSoup, nav/footer/aside/script/style stripped, main
  content kept.

Page boundaries are preserved as `[[PAGE:n]]` markers so later stages can
compute page_start/page_end per chunk.

Output: one .txt file per source in corpus/raw_text/<source_id>.txt
(kept separate from corpus/raw/<publisher>/ which holds the original
downloaded PDFs/HTML so nothing gets overwritten).

sources.csv is normally written by 01_fetch.py, but files sometimes land in
corpus/raw/ through a side channel (e.g. a manual wget batch, or dropping a
PDF straight into corpus/raw/manual/) without ever going through 01_fetch.py.
So on every run this script also walks corpus/raw/ recursively for PDFs and
HTML files, and for any file with no matching sources.csv row (matched by
file_path), synthesizes a minimal provenance row -- publisher from the
subdirectory name, title from the filename, url/year/licence left blank --
and appends it to sources.csv so nothing already downloaded gets silently
skipped. Fill in the real url/year/licence by hand afterwards if needed.

Runnable standalone: python corpus/02_extract.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
import traceback
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CORPUS_DIR, RAW_DIR, SOURCES_FIELDS, find_crops, print_elapsed_checkpoint, progress_line, safe_filename,
)

import pdfplumber
from bs4 import BeautifulSoup

SOURCES_CSV = CORPUS_DIR / "sources.csv"
RAW_TEXT_DIR = CORPUS_DIR / "raw_text"
ERROR_LOG = CORPUS_DIR / "extraction_errors.log"
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE_EXTS = {".pdf", ".html", ".htm"}

STRIP_TAGS = ["nav", "footer", "aside", "script", "style", "header", "form", "iframe"]
STRIP_CLASS_RE = ["nav", "footer", "sidebar", "menu", "breadcrumb", "cookie", "banner", "advert"]


def table_to_markdown(table: list[list]) -> str:
    rows = [[(c or "").strip().replace("\n", " ") for c in row] for row in table if row]
    if not rows:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in body:
        r = r + [""] * (len(header) - len(r)) if len(r) < len(header) else r[:len(header)]
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def extract_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            parts.append(f"\n\n[[PAGE:{i}]]\n\n")
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            parts.append(text)

            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for table in tables or []:
                md = table_to_markdown(table)
                if md:
                    parts.append(f"\n\n[[TABLE_START]]\n{md}\n[[TABLE_END]]\n\n")
    return "".join(parts)


def extract_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in STRIP_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    for pattern in STRIP_CLASS_RE:
        for el in soup.find_all(class_=lambda c: c and pattern in " ".join(c if isinstance(c, list) else [c]).lower()):
            el.decompose()
        for el in soup.find_all(id=lambda i: i and pattern in i.lower()):
            el.decompose()

    main = soup.find("main") or soup.find("article") or soup.find(class_=lambda c: c and "content" in " ".join(c if isinstance(c, list) else [c]).lower()) or soup.body or soup
    text = main.get_text("\n", strip=True)
    return f"\n\n[[PAGE:1]]\n\n{text}"


def has_table_markers(text: str) -> bool:
    return "[[TABLE_START]]" in text


def discover_raw_files() -> list[Path]:
    return sorted(p for p in RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() in RAW_FILE_EXTS)


def make_source_id(rel_path: Path) -> str:
    """Deterministic id from the file's path so reruns produce the same
    source_id (and therefore hit the same raw_text/<id>.txt on skip-check)."""
    digest = hashlib.sha1(rel_path.as_posix().encode("utf-8")).hexdigest()[:8]
    stem = safe_filename(rel_path.stem, maxlen=40)
    return f"RAW_{stem}_{digest}"


def synthesize_row(file_path: Path) -> dict:
    rel = file_path.relative_to(CORPUS_DIR)
    # corpus/raw/<publisher>/... -> parts = ("raw", "<publisher>", ...)
    publisher = rel.parts[1] if len(rel.parts) > 2 else "unknown"
    title = file_path.stem.replace("_", " ").replace("-", " ").strip()
    return {
        "source_id": make_source_id(rel),
        "url": "",
        "title": title,
        "publisher": publisher,
        "year": "",
        "licence": "",
        "crops_covered": ";".join(find_crops(title)),
        "has_tables": "unknown",
        "file_path": str(rel),
        "retrieval_date": date.today().isoformat(),
    }


def load_and_reconcile_sources() -> list[dict]:
    """sources.csv rows (if any) + a synthesized row for every file sitting
    in corpus/raw/ that sources.csv doesn't already know about."""
    rows: list[dict] = []
    known_paths: set[str] = set()
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        known_paths = {r["file_path"] for r in rows}

    new_rows = []
    for f in discover_raw_files():
        rel = str(f.relative_to(CORPUS_DIR))
        if rel not in known_paths:
            new_rows.append(synthesize_row(f))
            known_paths.add(rel)

    if new_rows:
        print(f"[extract] {len(new_rows)} file(s) in {RAW_DIR} had no sources.csv entry -- "
              f"synthesized minimal provenance rows (publisher=folder name, title=filename; "
              f"url/year/licence left blank, backfill by hand if needed)")
        rows.extend(new_rows)

    return rows


def main():
    rows = load_and_reconcile_sources()
    if not rows:
        print(f"[extract] ERROR: no sources in {SOURCES_CSV} and no PDF/HTML files under {RAW_DIR}. "
              f"Run 01_fetch.py or add files to {RAW_DIR} first.")
        sys.exit(1)

    total = len(rows)
    print(f"[extract] {total} sources to process")
    ok, failed, skipped = 0, 0, 0
    updated_rows = []
    start_time = time.time()

    for i, row in enumerate(rows, start=1):
        sid = row["source_id"]
        file_path = CORPUS_DIR / row["file_path"]
        out_path = RAW_TEXT_DIR / f"{sid}.txt"
        fname = Path(row["file_path"]).name

        if not file_path.exists():
            msg = f"{sid}: source file missing at {file_path}"
            progress_line("extract", i, total, fname, ok=False, reason="source file missing")
            with open(ERROR_LOG, "a", encoding="utf-8") as elog:
                elog.write(msg + "\n")
            failed += 1
            updated_rows.append(row)
        elif out_path.exists():
            progress_line("extract", i, total, fname, ok=True, reason="skipped, already extracted")
            skipped += 1
            updated_rows.append(row)
        else:
            try:
                if file_path.suffix.lower() == ".pdf":
                    text = extract_pdf(file_path)
                else:
                    text = extract_html(file_path)
                out_path.write_text(text, encoding="utf-8")
                row["has_tables"] = str(has_table_markers(text))
                ok += 1
                progress_line("extract", i, total, fname, ok=True)
            except Exception as exc:
                with open(ERROR_LOG, "a", encoding="utf-8") as elog:
                    elog.write(f"{sid}: {file_path} :: {exc}\n{traceback.format_exc()}\n")
                progress_line("extract", i, total, fname, ok=False, reason=str(exc))
                failed += 1
            updated_rows.append(row)

        print_elapsed_checkpoint("extract", i, total, start_time)

    # Rewrite sources.csv with the now-known has_tables values and any
    # newly synthesized rows.
    if updated_rows:
        with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SOURCES_FIELDS)
            writer.writeheader()
            writer.writerows(updated_rows)

    print(f"\n[extract] DONE. ok={ok} failed={failed} skipped_already_extracted={skipped}")
    print(f"[extract] Text output: {RAW_TEXT_DIR}")
    if failed:
        print(f"[extract] Failures logged to {ERROR_LOG}")


if __name__ == "__main__":
    main()
