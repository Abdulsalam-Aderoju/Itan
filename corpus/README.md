# Corpus Pipeline

Harvest + build pipeline that turns agricultural extension literature and research
papers into a RAG-ready corpus: chunked text, dense + sparse retrieval indices, and
a structured SQLite knowledge base (crop calendars, spacing, fertilizer rates,
agrochemicals, varieties, pests).

Target coverage: 10 crops (maize, cassava, rice, cowpea, yam, tomato, sorghum,
groundnut, pepper, soybean) across 10 countries (Nigeria primary market, plus Kenya,
Ghana, Tanzania, Uganda, Ethiopia, Rwanda, Senegal, Mali, Burkina Faso). Nigeria's
agro-ecological zones (Sudan Savanna, Northern/Southern Guinea Savanna, Derived
Savanna, Forest Zone, Mangrove/Coastal) are tracked as Nigeria-specific detail.

## Architecture

**Harvest stage** (`harvest/`) — discovers and scores candidate sources:
`00_manual_intake.py` (catalogues hand-obtained files) → `01_discover.py` (OpenAlex,
Semantic Scholar, CGSpace, national ag-research institutions) → `02_score.py` →
`03_gap_analysis.py` → `04_manifest.py` → `05_nafdac_extract.py`.

**Build stage** (root) — turns downloaded sources into the retrievable corpus:

| stage | script | does |
|---|---|---|
| 1 | `01_fetch.py` | downloads sources from the harvest manifest into `raw/` |
| 2 | `02_extract.py` | PDF/HTML → text via pdfplumber/BeautifulSoup; tables kept as markdown blocks; also picks up any file dropped into `raw/` outside the manifest |
| 3 | `03_clean.py` | dehyphenation, header/footer stripping, TOC removal |
| 4 | `04_chunk.py` | semantic chunking (300–450 tokens, tables never split), crop/zone/topic tagging |
| 5 | `05_structure_extract.py` | regex-based extraction into the 6 SQLite tables |
| 6 | `06_embed.py` | dense embeddings via quantized ONNX `bge-small-en-v1.5` |
| 7 | `07_bm25.py` | sparse BM25 index |
| 8 | `08_validate.py` | retrieval smoke test against 50 hand-curated gold questions |

Raw downloads, extracted/cleaned text, and generated artifacts (`chunks.parquet`,
`vectors.npy`, `bm25.pkl`, `structured.db`, `sources.csv`, etc.) are gitignored —
regenerate them by running the pipeline, not by pulling them from git.

## Current status

- **704 source documents** harvested (650 OpenAlex, 50 Semantic Scholar, 4 manual —
  including the NAFDAC registered-agrochemicals list)
- **16,181 semantic chunks** across 637 successfully extracted/cleaned documents
- **Structured DB**: 6,407 rows across the 6 tables — `pest` (6,100), `spacing` (181),
  `variety` (114), `fertilizer_rate` (10), `crop_calendar` (2), `agrochemical` (0)
- **Dense + sparse indices built**: `vectors.npy` (16,181 embeddings) and `bm25.pkl`
- **Retrieval validated**: Hit Rate@5 = **66.0%** (33/50) against 50 gold questions
  curated to match what this corpus actually contains (pest/disease diagnosis,
  fertiliser rate, post-harvest storage, improved varieties)

## What's left

- **`agrochemical` table is still empty (0 rows)** and `fertilizer_rate` is thin (10
  rows) — the regex patterns catch more academic phrasing now than they used to, but
  this corpus is research-paper-heavy and pesticide/dose phrasing in particular hasn't
  been cracked yet
- **`pest_disease_diagnosis` hit rate is 55%**, `tomato`/`pepper`/`soybean` are below
  the 60% per-crop gap threshold — worth another look at retrieval quality or corpus
  coverage for those
- **Crop coverage is imbalanced** (11.3× ratio between best- and worst-covered crop)
  and 1,084 chunks are under the 50-token minimum
- **97%+ of `pest` table rows are flagged `needs_review`** — the regex extraction is
  broad-net/low-confidence by design for that table; a manual review pass is the
  actual accuracy gate, not the extraction step itself
- **No query-normalization lexicon yet**: this corpus talks about varieties as
  "NASC-released cultivar" / "IITA-released genotype", not "recommended for
  smallholder farmers" — confirmed empirically when 5 gold questions went from 0% to
  60% hit rate purely from rephrasing, no corpus changes. A production RAG system
  needs a farmer-language → corpus-vocabulary lexicon in the query layer to avoid
  this same miss on real user queries.
- **Per-zone validation isn't meaningful right now** — the current 50 gold questions
  intentionally aren't tied to specific Nigerian agro-ecological zones, so
  `per_zone_hit_rate` in `validation_report.json` reads 0% across all 6 zones. That's
  a stale-reporting artifact, not a corpus gap; worth revisiting once zone-specific
  content is actually tested for.

## Running it

```
pip install -r requirements.txt
python 02_extract.py   # 01_fetch.py only needed if you have a harvest manifest to pull from
python 03_clean.py
python 04_chunk.py
python 05_structure_extract.py
python 06_embed.py
python 07_bm25.py
python 08_validate.py
```

Every stage skips already-processed files/artifacts on rerun and prints per-item
progress, so it's safe to interrupt and resume.
