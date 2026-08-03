#!/usr/bin/env python
"""
Stage 6 (harvest): special-case handler for the static NAFDAC pesticide
register PDF, mirroring 05_nafdac_extract.py's intent (structured source,
not prose -- parse directly into a flat CSV rather than relying on
04_chunk.py + 05_structure_extract.py's prose regexes) but for a PDF file
instead of a live HTML table.

WHY A DEDICATED PARSER. corpus/02_extract.py's generic pdfplumber
extract_text() scrambles this document badly: it's a borderless table (no
ruling lines, so pdfplumber's extract_tables() finds zero tables in it --
confirmed via sources.csv's has_tables=False for this source), and rows
with long wrapped applicant/manufacturer addresses get their columns
interleaved into unreadable character soup by pdfplumber's default
top-to-bottom/left-to-right text-flow ordering.

THE FIX: read word-level bounding boxes directly (page.extract_words) and
reconstruct rows using fixed column x-bands, empirically measured from
this PDF's header row and confirmed stable across pages 1/2/6/31:

    PRODUCT NAME        x in [0, 83)
    ACTIVE INGREDIENT    x in [83, 165)
    APPLICANT (name+addr) x in [165, 249)
    MANUFACTURER (name+addr) x in [249, 322)
    REG NO               x in [322, 361)
    DATE REGISTERED      x in [361, 400)
    REG EXPIRY            x in [400, 438)
    PRESENTATION          x in [438, 488)
    COUNTRY                x in [488, 522)
    STATE (Nigeria only)    x in [522, ...)

Row boundaries are anchored on the REG NO column: every product's
registration number, both dates, presentation and country always land on
that row's first line (confirmed by direct inspection -- even in rows
where the wrapped applicant/manufacturer text is corrupted in the source
PDF itself, e.g. "ACTION 40" / reg 04-6213, the anchor-line fields stay
clean). A row therefore spans from its own reg-no line up to (not
including) the next reg-no line; only PRODUCT NAME / ACTIVE INGREDIENT /
APPLICANT / MANUFACTURER can wrap onto the lines in between.

SCOPE, HONESTLY STATED. This PDF is a *registration* record -- it has no
crop, application rate, or pre-harvest-interval columns at all (that data
was never in NAFDAC's own register; the previous session's assumption
that it would appear here once the table was parsed cleanly was wrong).
So this script's output is NOT loaded into structured.db's `agrochemical`
table, which exists to serve spray_dilution() and needs a real
rate_per_ha -- inserting rows with a blank rate would just make that
table's crop-specific matching silently pick a useless row. What this
DOES produce is a clean, citable ground-truth list of real registered
product names + active ingredients + registration numbers, useful for
validating/enriching product names mentioned elsewhere in the corpus, or
for hand-curating a few verified entries into engine/tools/agri_calc's
agrochemical table the same way its existing example rows were seeded.

Output: corpus/harvest/nafdac_pesticide_register.csv

Runnable standalone: python corpus/harvest/06_nafdac_pdf_extract.py
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

PDF_PATH = CORPUS_DIR / "raw" / "manual" / "nafdac-approved-pesticides-in-nigeria.pdf"
OUT_CSV = HARVEST_DIR / "nafdac_pesticide_register.csv"
SOURCE_ID = "RAW_nafdac-approved-pesticides-in-nigeria_cf7c320e"

# Empirically measured column x-bands (points), stable across pages 1/2/6/31.
COLUMNS = [
    ("product_name", 0, 83),
    ("active_ingredient", 83, 165),
    ("applicant", 165, 249),
    ("manufacturer", 249, 322),
    ("registration_number", 322, 361),
    ("date_registered", 361, 400),
    ("reg_expiry", 400, 438),
    ("presentation", 438, 488),
    ("country", 488, 522),
    ("state", 522, 9999),
]

# Header banner + repeated column headers sit at top < 100 on every page
# (confirmed tops: 19.0, 33.0, 76.8, 82.2, 83.9, 96.2). The footer
# "Page N of 56" sits at exactly top=346.0 on every page checked. Data
# rows live strictly between the two.
HEADER_TOP_CUTOFF = 99.5
FOOTER_TOP_CUTOFF = 345.5

REG_NO_RE = re.compile(r"^[A-Z]?\d{1,2}-\d{3,6}$")
REG_NO_FRAGMENT_RE = re.compile(r"[A-Z]?\d{1,2}-\d{3,6}")
# The "PESTICIDE PRODUCTS" section label (page 1 only) lands in the
# PRODUCT NAME column band just above the first real row and would
# otherwise get glued onto it.
LABEL_TOKENS = {"PESTICIDE", "PRODUCTS", "PRODUCT", "S"}

FIELDS = [
    "source_id", "product_name", "active_ingredient", "registration_number",
    "date_registered", "reg_expiry", "presentation", "country", "state",
    "applicant", "manufacturer", "needs_review", "source_file", "retrieval_date",
]


def column_for(x0: float) -> str | None:
    for name, lo, hi in COLUMNS:
        if lo <= x0 < hi:
            return name
    return None


def parse_page(words: list[dict]) -> list[dict]:
    """Reconstruct product rows from one page's word list."""
    data_words = [
        w for w in words
        if HEADER_TOP_CUTOFF < w["top"] < FOOTER_TOP_CUTOFF
    ]

    anchors = sorted(
        {w["top"] for w in data_words if column_for(w["x0"]) == "registration_number" and REG_NO_RE.match(w["text"])}
    )
    if not anchors:
        return []

    rows = []
    for i, row_top in enumerate(anchors):
        row_end = anchors[i + 1] if i + 1 < len(anchors) else FOOTER_TOP_CUTOFF
        row_words = [w for w in data_words if row_top <= w["top"] < row_end]

        by_col: dict[str, list[dict]] = {name: [] for name, _, _ in COLUMNS}
        for w in row_words:
            col = column_for(w["x0"])
            if col is None:
                continue
            text = w["text"]
            if col == "product_name" and text.upper() in LABEL_TOKENS and w["top"] < row_top + 5:
                continue  # drop the page-1 "PESTICIDE PRODUCTS" label bleed
            by_col[col].append(w)

        row = {}
        for name, _, _ in COLUMNS:
            col_words = sorted(by_col[name], key=lambda w: (round(w["top"], 1), w["x0"]))
            row[name] = " ".join(w["text"] for w in col_words).strip()

        # Some spans of this PDF are corrupted at the source (glyphs out of
        # order in the content stream itself, not a text-flow-ordering
        # artifact) badly enough that no reg-no anchor is detectable for one
        # or more products in between two good anchors -- their content gets
        # swept into whichever row anchored the span. Two independent,
        # imperfect signals for this: (a) the merged row's
        # registration_number field contains more than one reg-no-shaped
        # fragment, when the swallowed row's own reg-no survived intact
        # enough to still match; (b) product_name/active_ingredient is far
        # longer than any genuine single-product value seen in this
        # document, which catches merges where the swallowed reg-no was
        # itself fragmented into unmatchable pieces (e.g. "A5-0012 A A5 5-
        # -0 00 20 66 0"). Neither signal is exhaustive -- this flags the
        # clear-cut cases, it does not guarantee every corrupted row is
        # caught (see e.g. JUSTOXIN 480 TABLETS: single clean reg-no, but
        # its active_ingredient field is scrambled at the source and stays
        # under the length threshold). Spot-check before treating
        # needs_review=False as a clean-data guarantee.
        multi_reg = len(REG_NO_FRAGMENT_RE.findall(row["registration_number"])) > 1
        oversized = len(row["product_name"]) > 50 or len(row["active_ingredient"]) > 60
        row["needs_review"] = str(multi_reg or oversized)

        if row["product_name"] and row["registration_number"]:
            rows.append(row)

    return rows


def main():
    if not PDF_PATH.exists():
        print(f"[nafdac_pdf] ERROR: {PDF_PATH} not found.")
        sys.exit(1)

    all_rows: list[dict] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        total = len(pdf.pages)
        print(f"[nafdac_pdf] {total} pages to parse")
        for i, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            page_rows = parse_page(words)
            for r in page_rows:
                r["source_id"] = SOURCE_ID
                r["source_file"] = "raw/manual/nafdac-approved-pesticides-in-nigeria.pdf"
                r["retrieval_date"] = date.today().isoformat()
            all_rows.extend(page_rows)
            print(f"  page {i}/{total}: {len(page_rows)} rows")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[nafdac_pdf] {len(all_rows)} product rows written to {OUT_CSV}")
    no_active = sum(1 for r in all_rows if not r["active_ingredient"])
    no_manufacturer = sum(1 for r in all_rows if not r["manufacturer"])
    flagged = sum(1 for r in all_rows if r["needs_review"] == "True")
    print(f"[nafdac_pdf] rows missing active_ingredient: {no_active}")
    print(f"[nafdac_pdf] rows missing manufacturer (source PDF corruption or wrap edge case): {no_manufacturer}")
    print(f"[nafdac_pdf] rows flagged needs_review (source PDF corruption merged/scrambled a row -- see "
          f"docstring, this signal is not exhaustive): {flagged} of {len(all_rows)}")
    print("[nafdac_pdf] NOTE: this file has no crop/rate/PHI columns -- it is a registration")
    print("[nafdac_pdf] record, not a dosage guide. Not loaded into structured.db's agrochemical")
    print("[nafdac_pdf] table; see this script's docstring for intended use.")


if __name__ == "__main__":
    main()
