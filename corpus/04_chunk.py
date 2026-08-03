#!/usr/bin/env python
"""
Stage 4 (build): semantic chunking. The most important stage in the
pipeline -- this is what actually becomes retrievable corpus content.

Rules:
  - Target 300-450 tokens per chunk (tiktoken cl100k_base), 60-token overlap
    between consecutive non-table chunks.
  - Prefer splitting at, in priority order: section headings (ALL CAPS
    lines / numbered headings like "3.1" / lines ending in ":"), paragraph
    breaks, numbered list items.
  - HARD RULE: a markdown table block ([[TABLE_START]]..[[TABLE_END]]) is
    never split. It becomes its own chunk even if it exceeds 450 tokens.
    Overlap is not carried across a table boundary (duplicating a table
    fragment into the next text chunk would corrupt structured content).
  - Every chunk gets full provenance + retrieval metadata.

Input: corpus/clean/<source_id>.txt + corpus/sources.csv
Output: corpus/chunks.parquet

Runnable standalone: python corpus/04_chunk.py
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS_DIR, CLEAN_DIR, CROPS, find_crops, find_zones, print_elapsed_checkpoint, progress_line  # noqa: E402

import pandas as pd
import tiktoken

SOURCES_CSV = CORPUS_DIR / "sources.csv"
OUT_PARQUET = CORPUS_DIR / "chunks.parquet"

TARGET_MIN = 300
TARGET_MAX = 450
OVERLAP_TOKENS = 60
MIN_CHUNK_TOKENS = 50  # small trailing fragments get merged into the previous chunk

ENC = tiktoken.get_encoding("cl100k_base")

PAGE_RE = re.compile(r"\[\[PAGE:(\d+)\]\]")
TABLE_RE = re.compile(r"\[\[TABLE_START\]\](.*?)\[\[TABLE_END\]\]", re.DOTALL)

ALL_CAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,.\-/&()]{2,79}$")
NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S.{0,78}$")
COLON_HEADING_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 ,\-/&()]{2,78}:$")
LIST_ITEM_RE = re.compile(r"^(\d+[.)]|\(\d+\)|[-*•])\s+\S")

TOPIC_KEYWORDS = {
    "planting": ["planting", "sowing", "seed rate", "planting date", "transplanting", "nursery", "seedbed"],
    "fertilisation": ["fertiliser", "fertilizer", "npk", "urea", "manure", "basal dressing", "top dress", "nutrient", "nitrogen"],
    "pest_management": ["pest", "insect", "infestation", "insecticide", "weevil", "borer", "aphid", "striga", "parasitic weed"],
    "disease_management": ["disease", "fungus", "fungicide", "blight", "wilt", "rot", "virus", "pathogen"],
    "harvest": ["harvest", "harvesting", "maturity", "threshing", "reaping"],
    "storage": ["storage", "storing", "warehouse", "silo", "post-harvest loss", "drying"],
    "variety_selection": ["variety", "cultivar", "released variety", "hybrid", "seed selection"],
    "soil_management": ["soil", "tillage", "ploughing", "erosion", "soil fertility", "organic matter"],
    "irrigation": ["irrigation", "watering", "drought", "water management", "rainfall"],
    "market": ["market", "price", "sale", "marketing", "value chain", "profit", "income"],
}


def count_tokens(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def decode_last_n_tokens(text: str, n: int) -> str:
    ids = ENC.encode(text, disallowed_special=())
    if len(ids) <= n:
        return text
    return ENC.decode(ids[-n:])


# ---------------------------------------------------------------------------
# Parse a cleaned document into an ordered list of atomic units.
# Each unit: {"type": "table"|"heading"|"paragraph"|"list_item", "text": str, "page": int}
# ---------------------------------------------------------------------------

def classify_paragraph(paragraph: str) -> str:
    lines = [l for l in paragraph.splitlines() if l.strip()]
    if len(lines) == 1:
        line = lines[0].strip()
        if ALL_CAPS_RE.match(line) or NUMBERED_HEADING_RE.match(line) or COLON_HEADING_RE.match(line):
            return "heading"
    return "paragraph"


def split_into_units(page_text: str, page_num: int) -> list[dict]:
    units = []
    for para in re.split(r"\n\s*\n", page_text):
        para = para.strip("\n")
        if not para.strip():
            continue
        kind = classify_paragraph(para)
        if kind == "paragraph":
            lines = [l for l in para.splitlines() if l.strip()]
            if len(lines) > 1 and all(LIST_ITEM_RE.match(l.strip()) for l in lines):
                for line in lines:
                    units.append({"type": "list_item", "text": line.strip(), "page": page_num})
                continue
        units.append({"type": kind, "text": para.strip(), "page": page_num})
    return units


def parse_document(text: str) -> list[dict]:
    units: list[dict] = []
    page_chunks = PAGE_RE.split(text)
    for i in range(1, len(page_chunks), 2):
        page_num = int(page_chunks[i])
        page_content = page_chunks[i + 1] if i + 1 < len(page_chunks) else ""
        # walk table markers within the page in order
        pos = 0
        for m in TABLE_RE.finditer(page_content):
            before = page_content[pos:m.start()]
            units.extend(split_into_units(before, page_num))
            units.append({"type": "table", "text": m.group(0), "page": page_num})
            pos = m.end()
        units.extend(split_into_units(page_content[pos:], page_num))
    return units


# ---------------------------------------------------------------------------
# Pack units into chunks
# ---------------------------------------------------------------------------

def pack_units(units: list[dict]) -> list[dict]:
    chunks = []
    buffer_units: list[dict] = []
    buffer_text = ""
    buffer_tokens = 0
    prev_chunk_was_table = False

    def flush(is_table=False):
        nonlocal buffer_units, buffer_text, buffer_tokens, prev_chunk_was_table
        if not buffer_units:
            return
        pages = [u["page"] for u in buffer_units]
        chunks.append({
            "text": buffer_text.strip(),
            "page_start": min(pages),
            "page_end": max(pages),
            "is_table": is_table,
        })
        buffer_units, buffer_text, buffer_tokens = [], "", 0
        prev_chunk_was_table = is_table

    for unit in units:
        if unit["type"] == "table":
            flush()  # flush whatever prose was accumulating
            buffer_units = [unit]
            buffer_text = unit["text"]
            flush(is_table=True)  # overlap is never carried across a table boundary
            continue

        new_text = (buffer_text + "\n\n" + unit["text"]).strip() if buffer_text else unit["text"]
        new_tokens = count_tokens(new_text)

        if new_tokens > TARGET_MAX and buffer_tokens >= TARGET_MIN:
            # current buffer is already a good size -> flush, start fresh
            flush()
            overlap_text = decode_last_n_tokens(chunks[-1]["text"], OVERLAP_TOKENS) if chunks and not prev_chunk_was_table else ""
            buffer_text = (overlap_text + "\n\n" + unit["text"]).strip() if overlap_text else unit["text"]
            buffer_tokens = count_tokens(buffer_text)
            buffer_units = [unit]
        elif new_tokens > TARGET_MAX and buffer_tokens < TARGET_MIN:
            # buffer alone is under-target; accept the overflow rather than
            # splitting the unit itself, then flush immediately.
            buffer_text, buffer_tokens = new_text, new_tokens
            buffer_units.append(unit)
            flush()
        else:
            buffer_text, buffer_tokens = new_text, new_tokens
            buffer_units.append(unit)

    flush()

    # merge a too-small trailing chunk into its predecessor (never into/out of a table chunk)
    merged = []
    for c in chunks:
        if (merged and not merged[-1]["is_table"] and not c["is_table"]
                and count_tokens(c["text"]) < MIN_CHUNK_TOKENS):
            merged[-1]["text"] = (merged[-1]["text"] + "\n\n" + c["text"]).strip()
            merged[-1]["page_end"] = max(merged[-1]["page_end"], c["page_end"])
        else:
            merged.append(c)
    return merged


TOPIC_INCLUDE_MIN_RATIO = 0.5   # a secondary topic must be at least half as well
                                  # evidenced (by raw keyword-hit count) as the winner
TOPIC_INCLUDE_MIN_ABS = 2        # ...and have at least 2 raw keyword hits itself, so a
                                  # chunk whose winning topic only has 1 hit can never
                                  # spawn a secondary label from a spurious 1-vs-1 "tie"


def classify_topics(text: str) -> tuple[str, str]:
    """Returns (topic, topics).

    A single ~300-450 token chunk often spans more than one of TOPIC_KEYWORDS'
    subject areas (e.g. a book chapter chunk covering both a Striga/pest
    paragraph and general harvest content, or literal adjacent "4.1 Threshing"
    / "4.2 Storage" subsections in the same packed chunk) -- picking only the
    single highest-count topic silently discards a real, keyword-supported
    secondary topic and can make 08_validate.py's strict topic == equality
    miss a chunk that was correctly retrieved. `topic` keeps the exact old
    single-winner behaviour (tie-broken by TOPIC_KEYWORDS dict order) for any
    future single-value use (e.g. a retrieval-layer topic facet). `topics` is
    a semicolon-joined set -- mirrors the existing `crops`/`zone` convention
    below -- of every topic within TOPIC_INCLUDE_MIN_RATIO of the winner's
    count, subject to the TOPIC_INCLUDE_MIN_ABS floor. Both thresholds were
    picked from corpus-wide percentile breaks in how often a runner-up topic
    is well-evidenced, not fit to any single example chunk.
    """
    text_l = text.lower()
    # "post-harvest"/"post harvest" is extremely common phrasing for storage,
    # drying and post-harvest-loss content -- but as a plain substring it
    # contains "harvest", which inflated the harvest bucket's count and stole
    # content that's actually about what happens AFTER harvest (see corpus
    # validation Q042: a chunk headed "4.2. Storage:" still lost to
    # "harvest" because "post-harvest handling" appeared twice). Masked out
    # before counting the harvest bucket only -- other buckets' keywords
    # don't contain "harvest" as a substring, so they're unaffected.
    harvest_source = text_l.replace("post-harvest", " ").replace("post harvest", " ")
    counts = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        source = harvest_source if topic == "harvest" else text_l
        counts[topic] = sum(source.count(kw) for kw in keywords)

    best_topic, best_count = "unclassified", 0
    for topic, count in counts.items():
        if count > best_count:
            best_topic, best_count = topic, count

    if best_count == 0:
        return "unclassified", ""

    included = {best_topic}
    for topic, count in counts.items():
        if topic == best_topic:
            continue
        if count >= TOPIC_INCLUDE_MIN_ABS and count >= TOPIC_INCLUDE_MIN_RATIO * best_count:
            included.add(topic)
    return best_topic, ";".join(sorted(included))


def main():
    if not SOURCES_CSV.exists():
        print(f"[chunk] ERROR: {SOURCES_CSV} not found. Run 01_fetch.py first.")
        sys.exit(1)
    if not any(CLEAN_DIR.glob("*.txt")):
        print(f"[chunk] ERROR: no cleaned documents in {CLEAN_DIR}. Run 03_clean.py first.")
        sys.exit(1)

    with open(SOURCES_CSV, newline="", encoding="utf-8") as fh:
        sources = {r["source_id"]: r for r in csv.DictReader(fh)}

    all_chunks = []
    files = sorted(CLEAN_DIR.glob("*.txt"))
    total = len(files)
    print(f"[chunk] {total} cleaned documents to chunk")
    start_time = time.time()

    for i, f in enumerate(files, start=1):
        sid = f.stem
        src = sources.get(sid)
        if src is None:
            progress_line("chunk", i, total, f.name, ok=False, reason="not found in sources.csv (provenance would be broken)")
            print_elapsed_checkpoint("chunk", i, total, start_time)
            continue

        text = f.read_text(encoding="utf-8", errors="ignore")
        units = parse_document(text)
        if not units:
            progress_line("chunk", i, total, f.name, ok=False, reason="no parseable content")
            print_elapsed_checkpoint("chunk", i, total, start_time)
            continue
        packed = pack_units(units)

        doc_crops = set(c for c in (src.get("crops_covered") or "").split(";") if c)
        doc_zones = find_zones(text)

        for idx, c in enumerate(packed):
            chunk_crops = set(find_crops(c["text"])) | doc_crops
            chunk_zones = find_zones(c["text"]) or doc_zones
            topic, topics = classify_topics(c["text"])
            all_chunks.append({
                "chunk_id": f"{sid}_{idx:04d}",
                "source_id": sid,
                "source_url": src.get("url", ""),
                "title": src.get("title", ""),
                "publisher": src.get("publisher", ""),
                "year": src.get("year", ""),
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "crops": ";".join(sorted(chunk_crops)) if chunk_crops else "",
                "zone": ";".join(sorted(chunk_zones)) if chunk_zones else "",
                "topic": topic,
                "topics": topics,
                "licence": src.get("licence", ""),
                "is_table": c["is_table"],
                "token_count": count_tokens(c["text"]),
                "text": c["text"],
            })
        progress_line("chunk", i, total, f.name, ok=True, reason=f"{len(packed)} chunks")
        print_elapsed_checkpoint("chunk", i, total, start_time)

    if not all_chunks:
        print("[chunk] ERROR: 0 chunks produced across all documents.")
        sys.exit(1)

    df = pd.DataFrame(all_chunks)
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"\n[chunk] DONE. {len(df)} chunks from {len(files)} documents -> {OUT_PARQUET}")
    print(f"[chunk] token_count stats: min={df['token_count'].min()} "
          f"median={int(df['token_count'].median())} max={df['token_count'].max()}")
    print("[chunk] Per-crop chunk counts:")
    for crop in CROPS:
        n = df["crops"].str.contains(crop, na=False).sum()
        print(f"    {crop:12s} {n}")
    print(f"[chunk] Table chunks (never split): {int(df['is_table'].sum())}")
    print(f"[chunk] Topic distribution (single best label):\n{df['topic'].value_counts().to_string()}")
    print("[chunk] Per-topic chunk counts (multi-label 'topics' field, a chunk can count toward more than one):")
    for topic in TOPIC_KEYWORDS:
        n = df["topics"].str.contains(topic, na=False).sum()
        print(f"    {topic:20s} {n}")


if __name__ == "__main__":
    main()
