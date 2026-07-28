#!/usr/bin/env python
"""
Stage 3 (build): clean extracted text.

- Dehyphenate line-break hyphens ("small-\nholder" -> "smallholder")
- Strip repeated headers/footers: a line that recurs on more than 30% of a
  document's pages is treated as running header/footer noise and dropped
- Remove table-of-contents pages (dot-leader / "Contents" heuristics)
- Normalize whitespace
- Table blocks ([[TABLE_START]]..[[TABLE_END]]) are pulled out before any
  of the above and reinserted verbatim afterwards -- cleaning must never
  touch table content.

Input: corpus/raw_text/<source_id>.txt (from 02_extract.py)
Output: corpus/clean/<source_id>.txt

Runnable standalone: python corpus/03_clean.py
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS_DIR, CLEAN_DIR, print_elapsed_checkpoint, progress_line  # noqa: E402

RAW_TEXT_DIR = CORPUS_DIR / "raw_text"

PAGE_RE = re.compile(r"\[\[PAGE:(\d+)\]\]")
TABLE_RE = re.compile(r"\[\[TABLE_START\]\](.*?)\[\[TABLE_END\]\]", re.DOTALL)
HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def split_pages(text: str) -> list[str]:
    """Split into page contents, dropping anything before the first marker."""
    parts = PAGE_RE.split(text)
    # parts = [prefix, num1, content1, num2, content2, ...]
    pages = []
    for i in range(1, len(parts), 2):
        pages.append(parts[i + 1] if i + 1 < len(parts) else "")
    return pages or [text]


def extract_tables(page_text: str) -> tuple[str, dict[str, str]]:
    tables = {}

    def repl(m):
        key = f"\x00TABLE{len(tables)}\x00"
        tables[key] = m.group(0)  # keep full [[TABLE_START]]...[[TABLE_END]] wrapper
        return key

    stripped = TABLE_RE.sub(repl, page_text)
    return stripped, tables


def dehyphenate(text: str) -> str:
    return HYPHEN_RE.sub(r"\1\2", text)


def is_toc_page(page_text_no_tables: str) -> bool:
    lines = [l for l in page_text_no_tables.splitlines() if l.strip()]
    if not lines:
        return False
    if re.search(r"table of contents|^contents$", page_text_no_tables, re.I | re.M):
        return True
    dot_leader_lines = sum(1 for l in lines if DOT_LEADER_RE.search(l))
    return len(lines) > 3 and (dot_leader_lines / len(lines)) > 0.3


def find_repeated_lines(pages_no_tables: list[str], threshold: float = 0.3) -> set[str]:
    if len(pages_no_tables) < 2:
        return set()
    page_line_sets = []
    for p in pages_no_tables:
        lines = {l.strip() for l in p.splitlines() if l.strip() and len(l.strip()) > 2}
        page_line_sets.append(lines)
    counter = Counter()
    for lines in page_line_sets:
        counter.update(lines)
    n_pages = len(pages_no_tables)
    # A line must recur on at least 2 distinct pages to be a header/footer
    # candidate at all -- otherwise the >threshold fraction alone falsely
    # flags ordinary unique body text on short documents (e.g. a line that
    # appears once out of 3 pages is already "33%").
    return {line for line, count in counter.items() if count >= 2 and count / n_pages > threshold}


def normalize_whitespace(text: str) -> str:
    lines = [MULTI_SPACE_RE.sub(" ", l).rstrip() for l in text.splitlines()]
    text = "\n".join(lines)
    return MULTI_BLANK_RE.sub("\n\n", text).strip()


def clean_document(raw_text: str) -> str:
    pages = split_pages(raw_text)
    tables_by_page = []
    pages_no_tables = []
    for p in pages:
        stripped, tables = extract_tables(p)
        pages_no_tables.append(stripped)
        tables_by_page.append(tables)

    repeated = find_repeated_lines(pages_no_tables)

    cleaned_pages = []
    for i, page_text in enumerate(pages_no_tables, start=1):
        if is_toc_page(page_text):
            continue  # drop entire TOC page (tables on a TOC page are noise too)

        lines = page_text.splitlines()
        kept = [l for l in lines if l.strip() not in repeated]
        page_text = "\n".join(kept)
        page_text = dehyphenate(page_text)

        # restore this page's tables
        for key, table_block in tables_by_page[i - 1].items():
            page_text = page_text.replace(key, f"\n\n{table_block}\n\n")

        cleaned_pages.append(f"[[PAGE:{i}]]\n{page_text}")

    return normalize_whitespace("\n\n".join(cleaned_pages))


def main():
    if not RAW_TEXT_DIR.exists() or not any(RAW_TEXT_DIR.glob("*.txt")):
        print(f"[clean] ERROR: no extracted text found in {RAW_TEXT_DIR}. Run 02_extract.py first.")
        sys.exit(1)

    files = sorted(RAW_TEXT_DIR.glob("*.txt"))
    total = len(files)
    print(f"[clean] {total} documents to clean")
    ok, skipped = 0, 0
    start_time = time.time()
    for i, f in enumerate(files, start=1):
        out_path = CLEAN_DIR / f.name
        if out_path.exists():
            skipped += 1
            progress_line("clean", i, total, f.name, ok=True, reason="skipped, already cleaned")
        else:
            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
                cleaned = clean_document(raw)
                out_path.write_text(cleaned, encoding="utf-8")
                ok += 1
                progress_line("clean", i, total, f.name, ok=True)
            except Exception as exc:
                progress_line("clean", i, total, f.name, ok=False, reason=str(exc))

        print_elapsed_checkpoint("clean", i, total, start_time)

    print(f"\n[clean] DONE. {ok}/{total} documents cleaned (skipped_already_done={skipped}) -> {CLEAN_DIR}")


if __name__ == "__main__":
    main()
