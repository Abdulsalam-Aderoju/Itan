#!/usr/bin/env python
"""
Stage 8 (harvest): special-case handler for the second static NAFDAC
register PDF -- "List of Registered Animal Health Products & Agrochemicals
(AHFC) 2016-2018" -- mirroring 06_nafdac_pdf_extract.py's approach (word-
level bounding boxes, fixed column x-bands, no ruling lines in the source
table) for this document's own column layout, which differs from the
pesticide register's.

WHY A DEDICATED PARSER. Same root cause as the pesticide register:
corpus/02_extract.py's generic pdfplumber extract_text() scrambles this
borderless table (no ruling lines -- extract_tables() finds nothing;
sources.csv has this source's has_tables as "unknown", i.e. never
verified, consistent with nobody having looked closely at it before).

THE FIX: read word-level bounding boxes (page.extract_words) and
reconstruct rows using fixed column x-bands, empirically measured from
this PDF's own header row (page 1, tops 214.4/226.1/237.7) and confirmed
stable on pages 1, 2, 40, 76, 77 (of 77 total):

    S/N                                x in [0, 76)
    NAME OF PRODUCT                    x in [76, 133)
    NAFDAC REG. NO                     x in [133, 176)
    COMPOSITION                        x in [176, 262)
    DOSAGE FORM, PRESENTATION          x in [262, 343)
    NAME & ADDRESS OF APPLICANT        x in [343, 431)
    NAME AND ADDRESS OF MANUFACTURER   x in [431, 519)
    APPROVAL DATE                      x in [519, 607)
    EXPIRY DATE                        x in [607, ...)

DIFFERENCES FROM THE PESTICIDE-REGISTER PARSER, both confirmed by direct
inspection of pages 1/2/40/76/77:
  - No footer/page-number text anywhere in this document (the pesticide
    register had "Page N of 56" at a fixed top on every page) -- so no
    FOOTER_TOP_CUTOFF is needed here.
  - Only page 1 carries the title banner + column-header rows (tops
    158.7-237.7); every later page starts directly with data at top=84.1.
    A single fixed HEADER_TOP_CUTOFF would wrongly discard the first row
    of every page after page 1. Instead of a top-based cutoff, rows are
    found purely by REG_NO_RE-matching anchors in the reg-no column band --
    the banner/header text never matches that pattern (confirmed: no
    reg-no-shaped strings in the header words), so it's naturally excluded
    without needing a cutoff at all.
  - This document's registration numbers look like "A9-0249" / "A10-0154"
    -- still matched by the same REG_NO_RE used for the pesticide
    register (`[A-Z]?\\d{1,2}-\\d{3,6}`), no change needed.

Row boundaries are anchored on the REG NO column exactly as in
06_nafdac_pdf_extract.py: a row spans from its own reg-no line up to (not
including) the next reg-no line, so wrapped COMPOSITION / APPLICANT /
MANUFACTURER text on the lines in between gets correctly attached.

SCOPE, HONESTLY STATED. Like the pesticide register, this is a
*registration* record -- COMPOSITION here is closer to an active-ingredient
list than a crop-specific dosage rate, and there is no crop or
pre-harvest-interval column. Same treatment as the pesticide register:
parse to a clean CSV here, load into structured.db's `agrochemical` table
via a follow-up loader (mirroring corpus/harvest/07_nafdac_load.py), not a
crop/rate source.

Output: corpus/harvest/nafdac_animal_health_register.csv

Runnable standalone: python corpus/harvest/08_animal_health_pdf_extract.py
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CORPUS_DIR, HARVEST_DIR  # noqa: E402

import pdfplumber

PDF_PATH = CORPUS_DIR / "raw" / "manual" / "List-of-Registered-Animal-Health-Products-and-Agrochemicals.pdf"
OUT_CSV = HARVEST_DIR / "nafdac_animal_health_register.csv"
SOURCE_ID = "RAW_List-of-Registered-Animal-Health-Product_4bb95898"

# Empirically measured column x-bands (points), stable across pages 1/2/40/76/77.
COLUMNS = [
    ("serial_no", 0, 76),
    ("product_name", 76, 133),
    ("registration_number", 133, 176),
    ("composition", 176, 262),
    ("presentation", 262, 343),
    ("applicant", 343, 431),
    ("manufacturer", 431, 519),
    ("approval_date", 519, 607),
    ("expiry_date", 607, 9999),
]

REG_NO_RE = re.compile(r"^[A-Z]?\d{1,2}-\d{3,6}$")
REG_NO_FRAGMENT_RE = re.compile(r"[A-Z]?\d{1,2}-\d{3,6}")
# Page-1-only title banner ("Registration & Regulatory Affairs (R&R)
# Directorate", "NAFDAC Office Complex...", "List of Registered Animal
# Health Products & Agrochemicals (AHFC)-2016-2018") and repeated column
# headers land above the first real data row (top=248.3) but can share an
# x-band with real columns (e.g. "NAME"/"OF"/"PRODUCT" sit in the
# product_name band). None of these tokens are reg-no-shaped, so they
# never anchor a row and are excluded by construction -- no explicit
# header-token skip list needed here, unlike the pesticide-register parser's
# LABEL_TOKENS (that one had a label bleeding into the anchor row itself;
# this document's banner sits well above the first anchor).

FIELDS = [
    "source_id", "serial_no", "product_name", "composition", "registration_number",
    "approval_date", "expiry_date", "presentation", "applicant",
    "manufacturer", "needs_review", "source_file", "retrieval_date",
]


def column_for(x0: float) -> str | None:
    for name, lo, hi in COLUMNS:
        if lo <= x0 < hi:
            return name
    return None


def parse_page(words: list[dict]) -> list[dict]:
    """Reconstruct product rows from one page's word list."""
    anchors = sorted(
        {w["top"] for w in words if column_for(w["x0"]) == "registration_number" and REG_NO_RE.match(w["text"])}
    )
    if not anchors:
        return []

    rows = []
    for i, row_top in enumerate(anchors):
        row_end = anchors[i + 1] if i + 1 < len(anchors) else max(w["top"] for w in words) + 1
        row_words = [w for w in words if row_top <= w["top"] < row_end]

        by_col: dict[str, list[dict]] = {name: [] for name, _, _ in COLUMNS}
        for w in row_words:
            col = column_for(w["x0"])
            if col is None:
                continue
            by_col[col].append(w)

        row = {}
        for name, _, _ in COLUMNS:
            col_words = sorted(by_col[name], key=lambda w: (round(w["top"], 1), w["x0"]))
            row[name] = " ".join(w["text"] for w in col_words).strip()

        # Same two imperfect corruption signals as the pesticide-register
        # parser (see that script's docstring for the full reasoning):
        # a merged row's registration_number containing more than one
        # reg-no-shaped fragment, or a field far longer than any genuine
        # single-product value seen in this document. Thresholds here are
        # NOT copied from the pesticide-register parser -- that document's
        # ACTIVE INGREDIENT field is normally 1-2 chemical names, while this
        # document's COMPOSITION is a full feed-ingredient list, legitimately
        # up to 256 chars on inspection with zero rows showing actual
        # merge corruption (multi_reg == 0 across the full 281-row run).
        # Thresholds set with headroom above the observed clean-data max
        # (product_name p max=57, composition max=256) rather than reused
        # from the other document, which was flagging ~27% of rows as
        # needs_review purely on length despite them being clean.
        multi_reg = len(REG_NO_FRAGMENT_RE.findall(row["registration_number"])) > 1
        oversized = len(row["product_name"]) > 90 or len(row["composition"]) > 320
        row["needs_review"] = str(multi_reg or oversized)

        if row["product_name"] and row["registration_number"]:
            rows.append(row)

    return rows


def main():
    if not PDF_PATH.exists():
        print(f"[animal_health_pdf] ERROR: {PDF_PATH} not found.")
        sys.exit(1)

    all_rows: list[dict] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        total = len(pdf.pages)
        print(f"[animal_health_pdf] {total} pages to parse")
        for i, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            page_rows = parse_page(words)
            for r in page_rows:
                r["source_id"] = SOURCE_ID
                r["source_file"] = "raw/manual/List-of-Registered-Animal-Health-Products-and-Agrochemicals.pdf"
                r["retrieval_date"] = date.today().isoformat()
            all_rows.extend(page_rows)
            print(f"  page {i}/{total}: {len(page_rows)} rows")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[animal_health_pdf] {len(all_rows)} product rows written to {OUT_CSV}")
    no_composition = sum(1 for r in all_rows if not r["composition"])
    no_manufacturer = sum(1 for r in all_rows if not r["manufacturer"])
    flagged = sum(1 for r in all_rows if r["needs_review"] == "True")
    print(f"[animal_health_pdf] rows missing composition: {no_composition}")
    print(f"[animal_health_pdf] rows missing manufacturer: {no_manufacturer}")
    print(f"[animal_health_pdf] rows flagged needs_review (source PDF corruption merged/scrambled "
          f"a row -- signal is not exhaustive, see docstring): {flagged} of {len(all_rows)}")
    print("[animal_health_pdf] NOTE: this is a registration record, not a crop/rate dosage guide.")


if __name__ == "__main__":
    main()
