#!/usr/bin/env python
"""
Stage 8 (build): smoke-test the corpus before it is used for anything real.

Runs 50 hard-coded gold questions (defined below, in this file) through
dense retrieval, BM25, and Reciprocal Rank Fusion, and checks whether a
correct-looking chunk shows up in the fused top 5. It also runs structural
sanity checks (no tiny chunks, no split tables, crop balance).

The 50 gold questions below are hand-curated against this corpus's actual
composition (16,181 chunks, heavily research-paper-weighted: 6,100
structured pest rows, strong fertiliser-rate coverage) rather than generic
extension-literature questions. An earlier placeholder set tested exact
planting-spacing facts and zone-specific planting calendars and scored 0%
on that tier, because this corpus barely contains that content -- these 50
instead ask what the corpus can actually answer: pest/disease diagnosis,
fertiliser rate, post-harvest storage, and improved varieties. Each
question is matched against chunk *metadata* (crop + topic) rather than an
exact source_id -- that's a proxy for "a relevant chunk", not a guarantee
of the single correct document. Add an `expected_source_id` for a stricter
check once specific source documents are known to contain the answer.

Runnable standalone: python corpus/08_validate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS_DIR, CROPS, ZONES, print_elapsed_checkpoint, progress_line  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `engine`

# torch must be imported before numpy/pandas, not after -- see the matching
# comment in 06_embed.py/engine/retrieval.py. pandas-before-torch reliably
# crashes the whole process here with STATUS_ACCESS_VIOLATION (0xC0000005),
# a raw native crash with no Python traceback at all, not a catchable
# exception -- engine.retrieval (imported lazily inside run_retrieval_eval())
# also imports torch first for the same reason, but that's too late if
# pandas below has already claimed the DLL first.
import torch  # noqa: F401,E402

import pandas as pd

CHUNKS_PARQUET = CORPUS_DIR / "chunks.parquet"
REPORT_JSON = CORPUS_DIR / "validation_report.json"

TOP_K = 5
HIT_RATE_GAP_THRESHOLD = 0.60
MIN_CHUNK_TOKENS = 50

TIERS = ["pest_disease_diagnosis", "fertiliser_rate", "crop_management_storage", "variety_selection"]

# 20 pest/disease diagnosis questions -- (crop, symptom, expected_topic).
# Half insect-pest symptoms (-> pest_management), half fungal/viral/weed
# symptoms (-> disease_management); "pest or disease" in the query wording
# below covers both without presupposing which one it is.
PEST_DISEASE_SYMPTOMS = [
    ("maize", "extensive tunnelling and holes bored into the stem, with visible frass", "pest_management"),
    ("maize", "narrow yellow-to-white streaks running parallel along the leaves", "disease_management"),
    ("cassava", "white, waxy, powdery insects clustered on the undersides of leaves and stunted shoot growth", "pest_management"),
    ("cassava", "mottled yellow-and-green mosaic patterns and distorted, misshapen leaves", "disease_management"),
    ("rice", "leaves folded and webbed together with white streaks running along the fold", "pest_management"),
    ("rice", "diamond-shaped grey lesions with brown borders on the leaves", "disease_management"),
    ("cowpea", "holes bored into the flower buds and developing pods", "pest_management"),
    ("cowpea", "mottled yellow mosaic patterns on young leaves and stunted growth", "disease_management"),
    ("yam", "tunnels bored into the tuber surface and internal galleries", "pest_management"),
    ("yam", "dark, sunken leaf spots that expand and cause premature defoliation", "disease_management"),
    ("tomato", "tiny insects clustered under the leaves with a sticky honeydew residue and sooty mould", "pest_management"),
    ("tomato", "sudden wilting of the whole plant despite adequate soil moisture", "disease_management"),
    ("sorghum", "stunted, yellowing plants with a parasitic weed attached to the roots", "pest_management"),
    ("sorghum", "reddish-brown lesions on the leaves and lodging of the stem", "disease_management"),
    ("groundnut", "holes bored into the pods and shrivelled, damaged kernels", "pest_management"),
    ("groundnut", "severe stunting, yellow mottling, and bunched leaves", "disease_management"),
    ("pepper", "curling, yellowing leaves and clusters of tiny insects on the underside", "pest_management"),
    ("pepper", "dark, sunken lesions on the ripening fruit", "disease_management"),
    ("soybean", "extensive leaf defoliation and damaged pods from chewing insects", "pest_management"),
    ("soybean", "small rust-coloured pustules on the underside of the leaves", "disease_management"),
]

# 15 fertiliser rate questions: one per crop asking about Nigeria, plus 5
# more (the crops most likely to have dedicated nitrogen-rate trials in an
# academic corpus) asking about Africa more broadly for phrasing variety.
FERTILISER_CROPS_NIGERIA = ["maize", "cassava", "rice", "cowpea", "yam", "tomato", "sorghum", "groundnut", "pepper", "soybean"]
FERTILISER_CROPS_AFRICA = ["maize", "cassava", "rice", "cowpea", "groundnut"]

# 10 crop management / post-harvest storage questions, one per crop.
STORAGE_CROPS = ["maize", "cassava", "rice", "cowpea", "yam", "tomato", "sorghum", "groundnut", "pepper", "soybean"]

# 5 improved-variety questions. First attempt at these ("What improved
# varieties of {crop} are recommended for smallholder farmers in West
# Africa?") scored 0/5 despite 701 topic=variety_selection chunks existing
# in the corpus (178 maize, 95 rice, 70 cassava, 55 soybean, 44 cowpea) --
# a retrieval/phrasing miss, not a content gap. This corpus talks about
# varieties as "NASC-released cultivar", "IITA-released genotype",
# "improved cultivar" -- not "recommended for smallholder farmers", so each
# question below is rephrased individually to match that vocabulary rather
# than reusing one generic template. (crop, query, zone) tuples since the
# phrasing genuinely varies per question now, not just the crop name.
VARIETY_QUESTIONS = [
    ("maize", "What IITA-released maize varieties are recommended for Nigeria?", "Nigeria"),
    ("cassava", "Which cassava cultivars have been released by IITA for West Africa?", "West Africa"),
    ("cowpea", "What are the characteristics of improved cowpea genotypes for Nigerian farmers?", "Nigeria"),
    ("soybean", "Which soybean varieties are recommended for the Guinea Savanna zone?", "Guinea Savanna"),
    ("yam", "What improved yam varieties have been released for smallholder production in Nigeria?", "Nigeria"),
]


def _build_gold_questions() -> list[dict]:
    """50 gold questions curated against this corpus's real composition:
    20 pest/disease diagnosis + 15 fertiliser rate + 10 storage/crop
    management + 5 improved variety. zone is left "" for question types
    that aren't tied to a specific Nigerian agro-ecological zone -- as a
    consequence per_zone_hit_rate below reads 0% across the board, which
    is expected (none of these 50 are zone-specific), not a corpus gap."""
    questions = []
    qid = 0

    for crop, symptom, topic in PEST_DISEASE_SYMPTOMS:
        qid += 1
        questions.append({
            "id": f"Q{qid:03d}",
            "query": f"A farmer's {crop} crop shows {symptom} -- what pest or disease is this and how should it be controlled?",
            "crop": crop,
            "zone": "",
            "tier": "pest_disease_diagnosis",
            "expected_topic": topic,
        })

    for crop in FERTILISER_CROPS_NIGERIA:
        qid += 1
        questions.append({
            "id": f"Q{qid:03d}",
            "query": f"What nitrogen rate is recommended for {crop} in Nigeria?",
            "crop": crop,
            "zone": "Nigeria",
            "tier": "fertiliser_rate",
            "expected_topic": "fertilisation",
        })
    for crop in FERTILISER_CROPS_AFRICA:
        qid += 1
        questions.append({
            "id": f"Q{qid:03d}",
            "query": f"What nitrogen rate is recommended for {crop} production in Africa?",
            "crop": crop,
            "zone": "Africa",
            "tier": "fertiliser_rate",
            "expected_topic": "fertilisation",
        })

    for crop in STORAGE_CROPS:
        qid += 1
        questions.append({
            "id": f"Q{qid:03d}",
            "query": f"What are the main post-harvest storage challenges for {crop}?",
            "crop": crop,
            "zone": "",
            "tier": "crop_management_storage",
            "expected_topic": "storage",
        })

    for crop, query, zone in VARIETY_QUESTIONS:
        qid += 1
        questions.append({
            "id": f"Q{qid:03d}",
            "query": query,
            "crop": crop,
            "zone": zone,
            "tier": "variety_selection",
            "expected_topic": "variety_selection",
        })

    return questions


GOLD_QUESTIONS = _build_gold_questions()
assert len(GOLD_QUESTIONS) == 50, f"expected 50 gold questions, got {len(GOLD_QUESTIONS)}"


def load_corpus_artifacts():
    if not CHUNKS_PARQUET.exists():
        print(f"[validate] ERROR: {CHUNKS_PARQUET} not found. Run 04_chunk.py first.")
        sys.exit(1)
    print(f"[validate] loading {CHUNKS_PARQUET} ...")
    df = pd.read_parquet(CHUNKS_PARQUET)
    df = df.set_index("chunk_id", drop=False)
    return df


def is_hit(chunk_ids: list[str], df: pd.DataFrame, question: dict) -> bool:
    for cid in chunk_ids:
        if cid not in df.index:
            continue
        row = df.loc[cid]
        crops = str(row.get("crops", "")).split(";")
        # `topics` (multi-label, semicolon-joined, mirrors `crops`) replaces
        # the old single-label `topic` here: a chunk spanning more than one
        # TOPIC_KEYWORDS subject (common -- e.g. a "4.1 Threshing" / "4.2
        # Storage" pair in the same packed chunk) can legitimately match more
        # than one expected_topic. See corpus/04_chunk.py's classify_topics().
        topics = str(row.get("topics", "")).split(";")
        if question["crop"] in crops and question["expected_topic"] in topics:
            return True
    return False


def run_retrieval_eval(df) -> dict:
    # Uses the actual production RetrievalEngine (engine/retrieval.py) --
    # this used to be a separate hand-rolled dense+BM25+RRF implementation
    # duplicated here, which meant a real fix to the production retrieval
    # path (e.g. the crop-aware reranking added 2026-08-04) silently didn't
    # show up in this validation harness's numbers. One engine, one set of
    # numbers, always in sync by construction.
    from engine.retrieval import RetrievalEngine
    print("[validate] loading RetrievalEngine (embed model + BM25 + vectors), reused for all 50 questions ...")
    engine = RetrievalEngine(corpus_dir=CORPUS_DIR)
    if engine.vectors is None:
        print("[validate] WARNING: vectors.npy/chunk_ids.json missing -- dense retrieval will be skipped "
              "(run 06_embed.py). Falling back to BM25-only evaluation.")
    if engine.bm25_data is None:
        print("[validate] WARNING: bm25.pkl missing -- BM25 retrieval will be skipped (run 07_bm25.py).")

    results = []
    total = len(GOLD_QUESTIONS)
    start_time = time.time()
    for i, q in enumerate(GOLD_QUESTIONS, start=1):
        try:
            # Call the single production retrieve() path end-to-end (candidate
            # pool depth, fusion, crop rerank all live there -- see
            # RRF_CANDIDATE_POOL_MULT in engine/retrieval.py) rather than
            # re-composing dense_search/bm25_search/rrf_fuse by hand here with
            # our own pool-size choice, which is exactly the kind of drift
            # this file's own docstring says it was rewritten to avoid.
            query_vec = engine.embed_query(q["query"]) if engine.vectors is not None else None
            chunks = engine.retrieve(q["query"], top_k=TOP_K, query_vec=query_vec)
            fused = [c["chunk_id"] for c in chunks]
            hit = is_hit(fused, df, q) if fused else False
            results.append({**q, "hit": hit, "fused_top5": fused})
            progress_line("validate", i, total, q["id"], ok=True, reason="hit" if hit else "miss")
        except Exception as exc:
            results.append({**q, "hit": False, "fused_top5": []})
            progress_line("validate", i, total, q["id"], ok=False, reason=str(exc))
        print_elapsed_checkpoint("validate", i, total, start_time)

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    hit_rate = hits / total if total else 0.0

    per_crop = {}
    for crop in CROPS:
        crop_results = [r for r in results if r["crop"] == crop]
        per_crop[crop] = sum(1 for r in crop_results if r["hit"]) / len(crop_results) if crop_results else 0.0

    per_zone = {}
    for zone in ZONES:
        zone_results = [r for r in results if r["zone"] == zone]
        per_zone[zone] = sum(1 for r in zone_results if r["hit"]) / len(zone_results) if zone_results else 0.0

    per_tier = {}
    for tier in TIERS:
        tier_results = [r for r in results if r["tier"] == tier]
        per_tier[tier] = sum(1 for r in tier_results if r["hit"]) / len(tier_results) if tier_results else 0.0

    gaps = {
        "crops": [c for c, r in per_crop.items() if r < HIT_RATE_GAP_THRESHOLD],
        "zones": [z for z, r in per_zone.items() if r < HIT_RATE_GAP_THRESHOLD],
    }

    return {
        "overall_hit_rate_at_5": round(hit_rate, 3),
        "total_questions": total,
        "hits": hits,
        "per_crop_hit_rate": {k: round(v, 3) for k, v in per_crop.items()},
        "per_zone_hit_rate": {k: round(v, 3) for k, v in per_zone.items()},
        "per_tier_hit_rate": {k: round(v, 3) for k, v in per_tier.items()},
        "gaps_below_threshold": gaps,
        "question_results": results,
    }


def run_structural_checks(df: pd.DataFrame) -> dict:
    tiny_chunks = df[df["token_count"] < MIN_CHUNK_TOKENS]["chunk_id"].tolist()

    split_table_chunks = []
    for _, row in df.iterrows():
        text = row["text"]
        starts, ends = text.count("[[TABLE_START]]"), text.count("[[TABLE_END]]")
        if starts != ends:
            split_table_chunks.append(row["chunk_id"])

    per_crop_counts = {}
    for crop in CROPS:
        per_crop_counts[crop] = int(df["crops"].fillna("").str.contains(crop).sum())
    nonzero = [v for v in per_crop_counts.values() if v > 0]
    balance_ratio = (max(nonzero) / min(nonzero)) if nonzero and min(nonzero) > 0 else float("inf")

    return {
        "chunks_under_min_tokens": {"count": len(tiny_chunks), "chunk_ids": tiny_chunks[:50]},
        "chunks_with_split_tables": {"count": len(split_table_chunks), "chunk_ids": split_table_chunks[:50]},
        "per_crop_chunk_counts": per_crop_counts,
        "crop_balance_max_over_min_ratio": None if balance_ratio == float("inf") else round(balance_ratio, 2),
        "crop_balance_flagged": balance_ratio > 3.0 if balance_ratio != float("inf") else True,
    }


def main():
    df = load_corpus_artifacts()
    print(f"[validate] {len(df)} chunks loaded. Running {len(GOLD_QUESTIONS)} gold questions ...")

    retrieval_report = run_retrieval_eval(df)
    structural_report = run_structural_checks(df)

    report = {"retrieval": retrieval_report, "structural": structural_report}
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Overall Hit Rate@5: {retrieval_report['overall_hit_rate_at_5']:.1%} "
          f"({retrieval_report['hits']}/{retrieval_report['total_questions']})")

    print("\nPer-crop Hit Rate@5:")
    for crop, rate in retrieval_report["per_crop_hit_rate"].items():
        flag = " <-- GAP" if rate < HIT_RATE_GAP_THRESHOLD else ""
        print(f"    {crop:12s} {rate:.1%}{flag}")

    print("\nPer-tier Hit Rate@5:")
    for tier, rate in retrieval_report["per_tier_hit_rate"].items():
        print(f"    {tier:28s} {rate:.1%}")

    print("\nPer-zone Hit Rate@5:")
    for zone, rate in retrieval_report["per_zone_hit_rate"].items():
        flag = " <-- GAP" if rate < HIT_RATE_GAP_THRESHOLD else ""
        print(f"    {zone:26s} {rate:.1%}{flag}")

    print(f"\nCorpus gaps (Hit Rate@5 < {HIT_RATE_GAP_THRESHOLD:.0%}):")
    print(f"    crops: {retrieval_report['gaps_below_threshold']['crops'] or 'none'}")
    print(f"    zones: {retrieval_report['gaps_below_threshold']['zones'] or 'none'}")

    print("\nStructural checks:")
    print(f"    chunks under {MIN_CHUNK_TOKENS} tokens: {structural_report['chunks_under_min_tokens']['count']}")
    print(f"    chunks with an unbalanced/split table marker: {structural_report['chunks_with_split_tables']['count']}")
    print(f"    crop balance (max/min chunk count ratio): {structural_report['crop_balance_max_over_min_ratio']}"
          f"{' <-- FLAGGED (imbalanced)' if structural_report['crop_balance_flagged'] else ''}")

    print(f"\n[validate] Full report written to {REPORT_JSON}")


if __name__ == "__main__":
    main()
