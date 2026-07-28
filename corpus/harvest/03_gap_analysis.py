#!/usr/bin/env python
"""
Stage 3 (harvest): coverage matrices + plain-English gap report.
Reads sources_scored.csv (falls back to sources_discovered.csv), writes
gap_report.txt. No downloads, no network calls.

Coverage is reported two ways:
  - Crop x Country (all of sub-Saharan Africa, see common.COUNTRIES) is the
    PRIMARY matrix now that 01_discover.py covers more than Nigeria.
  - Crop x Nigerian agro-ecological Zone is kept as its own SEPARATE
    section, since Nigeria remains the primary market and the zone detail
    doesn't apply to the other nine countries.

Runnable standalone: python corpus/harvest/03_gap_analysis.py
"""
from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path
from urllib.parse import quote_plus, unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    COUNTRIES, CROPS, ZONES, SQL_TABLE_KEYWORDS, HARVEST_DIR,
    find_country, find_crops, find_zones, guess_sql_tables,
)

SCORED_CSV = HARVEST_DIR / "sources_scored.csv"
DISCOVERED_CSV = HARVEST_DIR / "sources_discovered.csv"
OUT_TXT = HARVEST_DIR / "gap_report.txt"
BLOCKED_LOG = HARVEST_DIR / "blocked.log"
MANUAL_PRIORITY_TXT = HARVEST_DIR / "manual_priority_list.txt"

BLOCKED_LOG_RE = re.compile(
    r"^\[(?P<date>[^\]]+)\]\s+source=(?P<source>.*?)\s+url=(?P<url>\S+)\s+status=(?P<status>\S+)\s+headers=(?P<headers>.*)$"
)

STATUS_LABELS = {
    "robots_disallowed": "blocked by robots.txt",
    "blocked:403": "blocked (HTTP 403 Forbidden)",
    "blocked:429": "blocked (HTTP 429 Too Many Requests)",
    "blocked:503": "blocked (HTTP 503 Service Unavailable)",
}

INSTITUTION_HINTS = {
    "maize": "IITA / NAERLS / NCRI",
    "cassava": "IITA",
    "rice": "AfricaRice / NCRI",
    "cowpea": "IITA",
    "yam": "IITA (yam is a flagship IITA crop)",
    "tomato": "NIHORT",
    "sorghum": "ICRISAT / NCRI",
    "groundnut": "ICRISAT / IITA",
    "pepper": "NIHORT",
    "soybean": "IITA",
}

# Per-country institution to suggest for a manual crop/country gap. The five
# with a direct scraper in 01_discover.py are exact; the rest (no scraper
# wired up, API coverage only) are my best-effort knowledge of the relevant
# national institute, not independently verified -- treat as a starting
# point for the operator's own search, same caveat as INSTITUTION_TARGETS
# in 01_discover.py.
COUNTRY_INSTITUTION_HINTS = {
    "Nigeria": "NAERLS / NIHORT / IITA",
    "Kenya": "KALRO",
    "Ghana": "MOFA",
    "Tanzania": "TARI",
    "Uganda": "NARO",
    "Ethiopia": "EIAR",
    "Rwanda": "Rwanda Agriculture and Animal Resources Development Board (RAB)",
    "Senegal": "Institut Senegalais de Recherches Agricoles (ISRA)",
    "Mali": "Institut d'Economie Rurale (IER)",
    "Burkina Faso": "Institut de l'Environnement et de Recherches Agricoles (INERA)",
}


def load_rows():
    path = SCORED_CSV if SCORED_CSV.exists() else DISCOVERED_CSV
    if not path.exists():
        print("[gap] ERROR: no sources_scored.csv or sources_discovered.csv found. Run 01_discover.py first.")
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh)), path


def build_crop_country_matrix(rows):
    total = len(rows)
    print(f"[gap] Building crop x country matrix ({total} rows)...")
    matrix = {c: {country: 0 for country in COUNTRIES} for c in CROPS}
    for i, r in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"[gap] Processing {i + 1}/{total}...")
        crops = [c for c in r.get("crops_covered", "").split(";") if c]
        country = r.get("country", "")
        if country not in COUNTRIES:
            continue
        for c in crops:
            if c in matrix:
                matrix[c][country] += 1
    print("[gap] Crop x country matrix done.")
    return matrix


def build_crop_zone_matrix(rows):
    """Nigeria-only by construction: ZONES are Nigerian agro-ecological
    zones, so the caller is expected to pre-filter `rows` to Nigeria (see
    main() -- rows without a country column at all, from before this field
    existed, are treated as Nigeria for backward compatibility)."""
    total = len(rows)
    print(f"[gap] Building crop x zone matrix ({total} Nigeria rows)...")
    matrix = {c: {z: 0 for z in ZONES} for c in CROPS}
    for i, r in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"[gap] Processing {i + 1}/{total}...")
        crops = [c for c in r.get("crops_covered", "").split(";") if c]
        zones = [z for z in r.get("zones_mentioned", "").split(";") if z]
        for c in crops:
            if c not in matrix:
                continue
            for z in zones:
                if z in matrix[c]:
                    matrix[c][z] += 1
    print("[gap] Crop x zone matrix done.")
    return matrix


def build_table_crop_matrix(rows):
    total = len(rows)
    print(f"[gap] Building SQL table x crop matrix ({total} rows)...")
    matrix = {t: {c: 0 for c in CROPS} for t in SQL_TABLE_KEYWORDS}
    for i, r in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"[gap] Processing {i + 1}/{total}...")
        crops = [c for c in r.get("crops_covered", "").split(";") if c]
        tables = [t for t in r.get("sql_tables_likely", "").split(";") if t]
        for t in tables:
            if t not in matrix:
                continue
            for c in crops:
                if c in matrix[t]:
                    matrix[t][c] += 1
    print("[gap] SQL table x crop matrix done.")
    return matrix


def _load_score_functions():
    """02_score.py can't be `import`ed normally (its filename starts with a
    digit), so load it by path and reuse its exact relevance/quality/
    accessibility scoring functions -- a blocked source gets scored with
    the identical formula used for every other source, not an ad hoc one."""
    spec = importlib.util.spec_from_file_location("score_stage", Path(__file__).resolve().parent / "02_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.relevance_score, mod.quality_score, mod.accessibility_score


def _parse_blocked_log() -> list[dict]:
    if not BLOCKED_LOG.exists():
        return []
    entries: dict[tuple[str, str], dict] = {}
    with open(BLOCKED_LOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = BLOCKED_LOG_RE.match(line.strip())
            if not m:
                continue
            status = m.group("status")
            if status != "robots_disallowed" and not status.startswith("blocked:"):
                continue  # not what was asked for: skip 404s, timeouts, connection errors
            entry = {"date": m.group("date"), "source": m.group("source"), "url": m.group("url"), "status": status}
            entries[(entry["source"], entry["url"])] = entry  # last occurrence wins
    return list(entries.values())


def build_manual_priority_list():
    """harvest/manual_priority_list.txt: every source blocked by robots.txt
    or site-level blocking (403/429/503), scored with 02_score.py's own
    formula (accessibility forced to 0, since these were never confirmed
    reachable) and sorted so the operator knows exactly which ones are
    worth manually chasing down first."""
    blocked_entries = _parse_blocked_log()

    if not blocked_entries:
        with open(MANUAL_PRIORITY_TXT, "w", encoding="utf-8") as fh:
            fh.write(
                "MANUAL PRIORITY LIST\n" + "=" * 60 + "\n"
                f"No blocked/robots-disallowed attempts found in {BLOCKED_LOG.name}.\n"
                "Run harvest/01_discover.py first, or nothing was blocked this run.\n"
            )
        print(f"[gap] no blocked/robots_disallowed entries found -> {MANUAL_PRIORITY_TXT} written (empty)")
        return

    relevance_score, quality_score, accessibility_score = _load_score_functions()

    total_blocked = len(blocked_entries)
    print(f"[gap] Scoring {total_blocked} blocked/robots_disallowed entries...")
    scored_entries = []
    for i, entry in enumerate(blocked_entries):
        if (i + 1) % 100 == 0:
            print(f"[gap] Processing {i + 1}/{total_blocked}...")
        decoded = unquote(entry["url"])
        crops = find_crops(decoded)
        zones = find_zones(decoded)
        tables = guess_sql_tables(decoded)
        country = find_country(decoded)

        row = {
            "title": f"[blocked] {entry['source']} query",
            "abstract_snippet": decoded,
            "crops_covered": ";".join(crops),
            "zones_mentioned": ";".join(zones),
            "sql_tables_likely": ";".join(tables),
            "publisher": entry["source"],
            "year": "",
            "format": "",
            "page_count_estimate": "",
            "blocked": "True",
            "paywall": "False",
            "open_access": "False",
        }
        rel, _ = relevance_score(row)
        qual, _ = quality_score(row)
        acc, _ = accessibility_score(row)
        score = rel + qual + acc

        description = f"crops: {', '.join(crops) or 'unspecified'}"
        if country:
            description += f" | country: {country}"
        if zones:
            description += f" | zone: {', '.join(zones)}"
        if tables:
            description += f" | likely feeds: {', '.join(tables)}"

        scored_entries.append({
            **entry,
            "score": score,
            "status_label": STATUS_LABELS.get(entry["status"], entry["status"]),
            "description": description,
        })

    scored_entries.sort(key=lambda e: -e["score"])

    lines = [
        "MANUAL PRIORITY LIST",
        "=" * 60,
        f"{len(scored_entries)} sources blocked by robots.txt or site-level blocking, "
        f"scored with 02_score.py's formula (accessibility forced to 0).",
        "",
        "These could not be fetched automatically. Work through them top to",
        "bottom: open the URL in a browser, and if it's a search/listing page,",
        "run the search manually and grab whatever documents look relevant.",
        "",
        "-" * 60,
    ]
    for i, e in enumerate(scored_entries, start=1):
        lines.append(f"[{i}] score={e['score']} | {e['source']} | {e['status_label']}")
        lines.append(f"    {e['url']}")
        lines.append(f"    Likely contains: {e['description']}")
        lines.append("")

    with open(MANUAL_PRIORITY_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[gap] {len(scored_entries)} blocked/robots_disallowed sources -> {MANUAL_PRIORITY_TXT}")


def render_ascii_matrix(row_labels, col_labels, cell_fn, row_header="", col_width=10):
    lines = []
    header = row_header.ljust(24) + "".join(str(c)[:col_width].ljust(col_width) for c in col_labels)
    lines.append(header)
    lines.append("-" * len(header))
    for rl in row_labels:
        line = str(rl).ljust(24) + "".join(str(cell_fn(rl, cl)).ljust(col_width) for cl in col_labels)
        lines.append(line)
    return "\n".join(lines)


def main():
    print("[gap] Starting ...")
    rows, source_path = load_rows()
    print(f"[gap] loaded {len(rows)} rows from {source_path}")

    # Rows from before the "country" column existed have no country field at
    # all -- everything discovered back then was Nigeria-only, so treat a
    # missing/blank country as Nigeria for backward compatibility.
    nigeria_rows = [r for r in rows if r.get("country", "Nigeria") in ("Nigeria", "")]

    crop_country = build_crop_country_matrix(rows)
    crop_zone = build_crop_zone_matrix(nigeria_rows)
    table_crop = build_table_crop_matrix(rows)

    print("\n[gap] Crop x Country coverage matrix (primary -- all of sub-Saharan Africa):")
    ascii_country = render_ascii_matrix(
        CROPS, [c[:9] for c in COUNTRIES],
        lambda crop, c: crop_country[crop][[cc for cc in COUNTRIES if cc.startswith(c)][0]],
        row_header="crop \\ country",
    )
    print(ascii_country)

    print("\n[gap] Crop x Zone coverage matrix (Nigeria only -- primary market, kept separate):")
    ascii_zone = render_ascii_matrix(
        CROPS, [z.split()[0][:9] for z in ZONES],
        lambda c, z: crop_zone[c][[zz for zz in ZONES if zz.startswith(z)][0]],
        row_header="crop \\ zone",
    )
    print(ascii_zone)

    print("\n[gap] SQL table x Crop coverage matrix:")
    ascii_table = render_ascii_matrix(
        list(SQL_TABLE_KEYWORDS.keys()), CROPS,
        lambda t, c: table_crop[t][c],
        row_header="table \\ crop",
    )
    print(ascii_table)

    # ---- hard gaps ----
    crop_country_gaps = [(c, country) for c in CROPS for country in COUNTRIES if crop_country[c][country] == 0]
    crop_zone_gaps = [(c, z) for c in CROPS for z in ZONES if crop_zone[c][z] == 0]
    table_crop_gaps = [(t, c) for t in SQL_TABLE_KEYWORDS for c in CROPS if table_crop[t][c] == 0]

    lines = []
    lines.append("ITAN CORPUS — GAP REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated from: {source_path.name} ({len(rows)} sources)")
    lines.append("")
    lines.append("This file is plain English. No code needed to read it.")
    lines.append("")
    lines.append("SECTION 1: CROP x COUNTRY COVERAGE MATRIX (primary -- all of sub-Saharan Africa)")
    lines.append("-" * 60)
    lines.append(ascii_country)
    lines.append("")
    lines.append("SECTION 2: CROP x ZONE COVERAGE MATRIX (Nigeria only -- primary market, separate from the above)")
    lines.append("-" * 60)
    lines.append(ascii_zone)
    lines.append("")
    lines.append("SECTION 3: SQL TABLE x CROP COVERAGE MATRIX")
    lines.append("-" * 60)
    lines.append(ascii_table)
    lines.append("")
    lines.append("SECTION 4: HARD GAPS — crop/country combinations with ZERO sources")
    lines.append("-" * 60)
    if not crop_country_gaps:
        lines.append("None. Every crop/country combination has at least one source.")
    for crop, country in crop_country_gaps:
        lines.append(f"\n* {crop.upper()} in {country}")
        g_query = f'"{crop}" "{country}" "planting calendar" OR "fertiliser rate" extension filetype:pdf'
        s_query = f"{crop} {country} extension recommendation smallholder"
        lines.append(f"    Google:         {g_query}")
        lines.append(f"    Google (link):  https://www.google.com/search?q={quote_plus(g_query)}")
        lines.append(f"    Google Scholar: {s_query}")
        lines.append(f"    Scholar (link): https://scholar.google.com/scholar?q={quote_plus(s_query)}")
        lines.append(f"    Best institution to try directly: {COUNTRY_INSTITUTION_HINTS.get(country, 'national ministry of agriculture / agricultural research institute')}")

    lines.append("")
    lines.append("SECTION 5: HARD GAPS — Nigeria crop/zone combinations with ZERO sources")
    lines.append("-" * 60)
    if not crop_zone_gaps:
        lines.append("None. Every Nigerian crop/zone combination has at least one source.")
    for crop, zone in crop_zone_gaps:
        lines.append(f"\n* {crop.upper()} in {zone}")
        g_query = f'"{crop}" "{zone}" "planting calendar" OR "fertiliser rate" Nigeria extension filetype:pdf'
        s_query = f"{crop} {zone} Nigeria extension recommendation"
        lines.append(f"    Google:         {g_query}")
        lines.append(f"    Google (link):  https://www.google.com/search?q={quote_plus(g_query)}")
        lines.append(f"    Google Scholar: {s_query}")
        lines.append(f"    Scholar (link): https://scholar.google.com/scholar?q={quote_plus(s_query)}")
        lines.append(f"    Best institution to try directly: {INSTITUTION_HINTS.get(crop, 'IITA')}")

    lines.append("")
    lines.append("SECTION 6: HARD GAPS — SQL table x crop combinations with ZERO likely-feeding sources")
    lines.append("-" * 60)
    if not table_crop_gaps:
        lines.append("None. Every SQL table has at least one likely source per crop.")
    for table, crop in table_crop_gaps:
        lines.append(f"\n* Table '{table}' has no source for {crop.upper()}")
        kw = SQL_TABLE_KEYWORDS[table][0]
        g_query = f'"{crop}" Nigeria "{kw}" extension filetype:pdf'
        lines.append(f"    Google: {g_query}")
        lines.append(f"    Google (link): https://www.google.com/search?q={quote_plus(g_query)}")
        if table == "agrochemical":
            lines.append("    Direct institution: NAFDAC agrochemical register (see 05_nafdac_extract.py)")
        elif table in ("fertilizer_rate", "spacing", "crop_calendar"):
            lines.append(f"    Direct institution: NAERLS / FMARD extension bulletins for {crop}")
        elif table == "variety":
            lines.append(f"    Direct institution: {INSTITUTION_HINTS.get(crop, 'IITA')} variety release catalogues")
        elif table == "pest":
            lines.append(f"    Direct institution: CABI PlantWise factsheets for {crop}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"TOTAL HARD GAPS: {len(crop_country_gaps)} crop/country + {len(crop_zone_gaps)} Nigeria crop/zone "
                  f"+ {len(table_crop_gaps)} table/crop")
    lines.append("Next: work through 04_manifest.py's download_manifest.md for what IS")
    lines.append("available, then manually search the gaps listed above.")

    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"\n[gap] {len(crop_country_gaps)} crop/country hard gaps, {len(crop_zone_gaps)} Nigeria crop/zone hard gaps, "
          f"{len(table_crop_gaps)} table/crop hard gaps")
    print(f"[gap] Gap report written to {OUT_TXT}")

    build_manual_priority_list()
    print("[gap] Done.")


if __name__ == "__main__":
    main()
