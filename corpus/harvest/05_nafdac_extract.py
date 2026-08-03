#!/usr/bin/env python
"""
Stage 5 (harvest): NAFDAC agrochemical register special-case handler.

NAFDAC's own website (nafdac.gov.ng) does not expose the pesticide register as
scrapeable HTML -- confirmed by hand before writing this version of the
script: the "our-services/registration/agrochemicals" style paths this file
used to guess all 404, the site's press release on pesticide regulation
lists no register link, and the Greenbook search portal
(greenbook.nafdac.gov.ng) only covers drugs/vaccines/medical devices/
veterinary/herbals/disinfectants -- pesticides are not one of its product
categories. So this does NOT hit the live NAFDAC site at all.

Instead it downloads a direct mirror of NAFDAC's own "Directorate of
Registration & Regulatory Affairs" pesticide-products register PDF -- the
same file already anticipated in sources.csv as
RAW_nafdac-approved-pesticides-in-nigeria_cf7c320e, but never actually
fetched/parsed (raw/ is gitignored and the file was never present in this
checkout). The PDF itself is the genuine NAFDAC register: page 1 opens with
"NATIONAL AGENCY FOR FOOD & DRUG ADMINISTRATION AND CONTROL / DIRECTORATE OF
REGISTRATION & REGULATORY AFFAIRS" and a "PESTICIDE PRODUCTS" table with
columns product name / active ingredient / applicant / manufacturer /
NAFDAC reg. no / date registered / reg. expiry / presentation / country --
467 products across all 56 pages, each with a real NAFDAC registration
number and registration/expiry date.

The PDF's table has no ruling lines, so pdfplumber's own extract_tables()
finds nothing, and naive text extraction reads column-major garbage (every
product name, then every active ingredient, then every applicant address,
all concatenated). This script instead reads word-level (x0, top) positions
and reconstructs true rows: the 9 columns sit at fixed x-ranges that are
byte-identical across all 56 pages (verified against pages 1, 2, 30, 56), and
each product's NAFDAC reg. no. anchors one row -- the vertical band from one
reg. no.'s `top` to the next `top` is that product's row, regardless of how
many lines its applicant/manufacturer address wraps onto.

Even with correct row reconstruction, some entries come out with jumbled
characters in product_name/active_ingredient (a genuine PDF font/encoding
quirk in the source document, not a parsing bug -- verified by comparing
against a non-layout dump of the same rows). Those are flagged
needs_review=1 via a garbling heuristic rather than silently trusted or
discarded; the raw registration number and dates are always taken verbatim
regardless, since they are the only fields load-bearing enough to matter for
"grounding". Nothing in this file invents a registration number, active
ingredient, or date -- every value is a substring of the fetched PDF's text.

Writes corpus/harvest/nafdac_agrochemicals.csv, then loads it into
corpus/structured.db's `agrochemical` table (replacing any prior rows from
this source), matching the existing 6-table schema convention: every row
carries source_id + raw_text (provenance) + confidence + needs_review.

Runnable standalone: python corpus/harvest/05_nafdac_extract.py
"""
from __future__ import annotations

import csv
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CORPUS_DIR, HARVEST_DIR, new_session, polite_get  # noqa: E402

import pdfplumber

OUT_CSV = HARVEST_DIR / "nafdac_agrochemicals.csv"
DB_PATH = CORPUS_DIR / "structured.db"
MANUAL_DIR = CORPUS_DIR / "raw" / "manual"
PDF_PATH = MANUAL_DIR / "nafdac-approved-pesticides-in-nigeria.pdf"

# Mirror of the NAFDAC "Directorate of Registration & Regulatory Affairs"
# pesticide register. This exact file/URL was already catalogued in
# sources.csv (id RAW_nafdac-approved-pesticides-in-nigeria_cf7c320e) before
# this script was fixed; robots.txt on this host allows fetching it.
PDF_URL = "https://fnr.ecp.mybluehost.me/wp-content/uploads/2019/07/nafdac-approved-pesticides-in-nigeria.pdf"
SOURCE_ID = "RAW_nafdac-approved-pesticides-in-nigeria_cf7c320e"

FIELDS = [
    "source_id", "product_name", "active_ingredient", "crop", "rate", "rate_unit",
    "pre_harvest_interval_days", "raw_text", "confidence", "needs_review",
]

# Fixed-column x0 ranges (PDF points), read off the header row and cross-checked
# against pages 1, 2, 30 and 56 -- identical on every page, confirming this is a
# genuine fixed-width table export rather than something that drifts per page.
COLS = {
    "product_name": (0, 84.5),
    "active_ingredient": (84.5, 164.5),
    "applicant": (164.5, 250),
    "manufacturer": (250, 325),
    "reg_no": (325, 360),
    "date_registered": (360, 403),
    "reg_expiry": (403, 438),
    "presentation": (438, 489),
    "country": (489, 524),
}
REGNO_RE = re.compile(r"^([0-9]{2}-[0-9]{3,6}[A-Z]?|A[0-9]-[0-9]{3,6}[A-Z]?)$")
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def col_of(x0: float) -> str | None:
    for name, (lo, hi) in COLS.items():
        if lo <= x0 < hi:
            return name
    return None


def is_garbled(text: str) -> bool:
    """Heuristic for the PDF's character-jumbling quirk: a real product name
    or active-ingredient phrase is mostly words of 3+ letters; the jumbled
    ones come out as a long run of 1-2 character fragments."""
    tokens = text.split()
    if not tokens:
        return True
    short = sum(1 for t in tokens if len(t) <= 2)
    return len(tokens) >= 4 and (short / len(tokens)) > 0.35


def parse_page(words: list[dict]) -> list[dict]:
    anchors = sorted(
        (w for w in words if COLS["reg_no"][0] <= w["x0"] < COLS["reg_no"][1] and REGNO_RE.match(w["text"])),
        key=lambda w: w["top"],
    )
    rows = []
    for i, a in enumerate(anchors):
        top_start = a["top"] - 0.5
        top_end = anchors[i + 1]["top"] - 0.5 if i + 1 < len(anchors) else 10_000
        band = [w for w in words if top_start <= w["top"] < top_end]
        cells: dict[str, list[dict]] = {name: [] for name in COLS}
        for w in band:
            c = col_of(w["x0"])
            if c:
                cells[c].append(w)
        for name in cells:
            cells[name].sort(key=lambda w: (round(w["top"], 1), w["x0"]))

        def joined(name: str) -> str:
            return " ".join(w["text"] for w in cells[name]).strip()

        date_reg = next((w["text"] for w in cells["date_registered"] if DATE_RE.match(w["text"])), "")
        date_exp = next((w["text"] for w in cells["reg_expiry"] if DATE_RE.match(w["text"])), "")
        rows.append({
            "product_name": joined("product_name"),
            "active_ingredient": joined("active_ingredient"),
            "reg_no": a["text"],
            "date_registered": date_reg,
            "reg_expiry": date_exp,
            "presentation": joined("presentation"),
            "country": joined("country"),
        })
    return rows


def to_agrochemical_row(r: dict, page: int) -> dict:
    pn_bad = is_garbled(r["product_name"])
    ai_bad = not r["active_ingredient"] or is_garbled(r["active_ingredient"])

    confidence = 0.4  # reg_no + dates are a solid, unambiguous regex anchor
    if r["product_name"] and not pn_bad and len(r["product_name"]) <= 60:
        confidence += 0.3
    if r["active_ingredient"] and not ai_bad and len(r["active_ingredient"]) <= 60:
        confidence += 0.3
    confidence = round(min(confidence, 1.0), 2)

    raw_text = (
        f"NAFDAC Reg. No. {r['reg_no']} | Registered {r['date_registered']}, "
        f"Expires {r['reg_expiry']} | Presentation: {r['presentation']} | "
        f"Country of origin: {r['country']} | Source: NAFDAC pesticide products "
        f"register (mirrored copy), {PDF_URL}, p.{page}"
    )[:500]

    return {
        "source_id": SOURCE_ID,
        "product_name": r["product_name"],
        "active_ingredient": r["active_ingredient"],
        # The register is not crop-specific (a NAFDAC pesticide registration
        # covers the product/active ingredient generally, not a per-crop
        # label claim), so left blank rather than guessed -- see file
        # docstring / no crop indication anywhere in the source table.
        "crop": "",
        # "presentation" in this register is packaging size (e.g. "1L/BOTTLE"),
        # not an application rate -- deliberately NOT stuffed into rate/
        # rate_unit, which would misrepresent packaging as a dosage.
        "rate": "",
        "rate_unit": "",
        "pre_harvest_interval_days": "",
        "raw_text": raw_text,
        "confidence": confidence,
        "needs_review": "1" if confidence < 0.7 else "0",
    }


def fetch_pdf() -> bool:
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0:
        print(f"[nafdac] Using cached PDF at {PDF_PATH}")
        return True
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    session = new_session()
    print(f"[nafdac] Fetching {PDF_URL} ...")
    resp, note = polite_get(PDF_URL, session=session, timeout=30)
    if resp is None or note != "ok":
        print(f"[nafdac] FAILED to fetch PDF: {note}")
        return False
    PDF_PATH.write_bytes(resp.content)
    print(f"[nafdac] Saved {len(resp.content)} bytes to {PDF_PATH}")
    return True


def extract_rows() -> list[dict]:
    all_rows: list[dict] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"[nafdac] {len(pdf.pages)} pages in PDF")
        for i, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            for r in parse_page(words):
                all_rows.append(to_agrochemical_row(r, i))
            if i % 10 == 0:
                print(f"[nafdac] parsed {i}/{len(pdf.pages)} pages, {len(all_rows)} rows so far")
    return all_rows


def write_csv(rows: list[dict]) -> None:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[nafdac] {len(rows)} rows written to {OUT_CSV}")


def load_into_db(rows: list[dict]) -> None:
    if not DB_PATH.exists():
        print(f"[nafdac] {DB_PATH} does not exist yet -- skipping DB load (CSV is still written).")
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS agrochemical (id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'source_id TEXT, product_name TEXT, active_ingredient TEXT, crop TEXT, rate TEXT, '
            'rate_unit TEXT, pre_harvest_interval_days TEXT, raw_text TEXT, confidence REAL, needs_review TEXT)'
        )
        # This script is the sole populator of the agrochemical table -- the
        # only pre-existing content was a single garbage regex-extraction
        # artifact row (source_id RAW_s44447-025-00017-8_c1234e1f, a mangled
        # academic-paper sentence, not real NAFDAC data). Clearing the whole
        # table keeps re-runs idempotent and guarantees that garbage row
        # never survives alongside the real register data.
        before = conn.execute("SELECT COUNT(*) FROM agrochemical").fetchone()[0]
        conn.execute("DELETE FROM agrochemical")
        cols = FIELDS
        placeholders = ", ".join("?" for _ in cols)
        conn.executemany(
            f'INSERT INTO agrochemical ({", ".join(cols)}) VALUES ({placeholders})',
            [[r[c] for c in cols] for r in rows],
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM agrochemical").fetchone()[0]
        print(f"[nafdac] agrochemical table: {before} rows -> {after} rows")
    finally:
        conn.close()


def main():
    print("[nafdac] Starting ...")

    if not fetch_pdf():
        print("[nafdac] NOTE: could not obtain the NAFDAC pesticide register PDF this run. "
              "Not writing a CSV / touching the DB -- leaving whatever is already there rather "
              "than silently truncating good data.")
        return

    rows = extract_rows()
    if not rows:
        print("[nafdac] NOTE: 0 rows extracted from the PDF -- its layout may have changed. "
              "Manual verification recommended; this does not crash the pipeline.")
        return

    write_csv(rows)

    needs_review = sum(1 for r in rows if r["needs_review"] == "1")
    print(f"[nafdac] {needs_review}/{len(rows)} rows flagged needs_review "
          f"({needs_review / len(rows) * 100:.1f}%)")

    load_into_db(rows)
    print("[nafdac] Done.")


if __name__ == "__main__":
    main()
