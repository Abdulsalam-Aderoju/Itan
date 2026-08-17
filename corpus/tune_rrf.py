#!/usr/bin/env python
"""RRF candidate-pool-depth and fusion-weight sweep against the 50-question
corpus validation set (the same GOLD_QUESTIONS as corpus/08_validate.py).

Not a pipeline stage (not numbered 01-08) -- a standalone diagnostic used
once (2026-08-17) to find RRF_CANDIDATE_POOL_MULT in engine/retrieval.py.
Re-run this after any corpus rebuild that changes chunk count/composition
significantly, to confirm the pool-depth constant is still near-optimal --
it's cheap (~1 minute) since dense/BM25 candidates are embedded/searched
once and cached, then re-fused many times over in memory.

Finding from the last run: fusion WEIGHTING (dense vs BM25 contribution)
barely moved the needle once the corpus was fully re-embedded (all configs
landed at 0.76-0.82 hit-rate@5); fusion DEPTH was the real lever -- the old
top_k*2 candidate pool was cutting off the correct chunk before RRF ever
saw it. top_k*8 got to 0.92 on the full 72,452-chunk corpus; deeper pools
(top_k*10, *12) gave no further improvement. See engine/retrieval.py's
RRF_CANDIDATE_POOL_MULT for the constant this produced.

Usage: python corpus/tune_rrf.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "corpus"))

from engine.retrieval import RetrievalEngine, CROP_MATCH_BOOST, CROP_MISMATCH_PENALTY  # noqa: E402
import importlib  # noqa: E402
val = importlib.import_module("08_validate")

TOP_K = val.TOP_K


def weighted_rrf(eng, dense_ids, bm25_ids, dense_weight, bm25_weight, query_crops, k=60, top_k=TOP_K):
    """Standalone RRF with per-list weights, for sweeping only -- the shipped
    engine.rrf_fuse() doesn't take weights since the sweep found equal
    weighting already optimal (see module docstring). Mirrors rrf_fuse()'s
    crop-aware rerank exactly (dropping it here previously understated every
    result -- crop rerank is not optional, it's load-bearing)."""
    scores: dict[str, float] = {}
    for cid_list, weight in ((dense_ids, dense_weight), (bm25_ids, bm25_weight)):
        for rank, cid in enumerate(cid_list):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)

    if query_crops and eng.chunks_df is not None:
        query_crop_set = set(query_crops)
        for cid in scores:
            if cid not in eng.chunks_df.index:
                continue
            row = eng.chunks_df.loc[cid]
            chunk_crops = {c for c in str(row.get("crops", "")).split(";") if c}
            if not chunk_crops:
                continue
            scores[cid] *= CROP_MATCH_BOOST if (chunk_crops & query_crop_set) else CROP_MISMATCH_PENALTY

    sorted_cids = sorted(scores.items(), key=lambda kv: -kv[1])
    return [cid for cid, _ in sorted_cids[:top_k]]


def main():
    print("[tune_rrf] loading RetrievalEngine ...")
    eng = RetrievalEngine()

    pool = TOP_K * 15  # fetch a deep pool once; slice smaller for the sweeps below
    cache = []
    t0 = time.time()
    for i, q in enumerate(val.GOLD_QUESTIONS, start=1):
        qvec = eng.embed_query(q["query"])
        dense_ids = eng.dense_search(qvec, top_k=pool)
        bm25_ids = eng.bm25_search(q["query"], top_k=pool)
        query_crops = eng.detect_query_crops(q["query"])
        cache.append((q, dense_ids, bm25_ids, query_crops))
        print(f"  cached {i}/{len(val.GOLD_QUESTIONS)}", end="\r")
    print(f"\n[tune_rrf] cached all candidate lists in {time.time()-t0:.1f}s")

    def hit_rate(dense_weight, bm25_weight, sub_pool):
        hits = 0
        for q, dense_ids, bm25_ids, query_crops in cache:
            fused = weighted_rrf(eng, dense_ids[:sub_pool], bm25_ids[:sub_pool], dense_weight, bm25_weight, query_crops)
            if val.is_hit(fused, eng.chunks_df, q):
                hits += 1
        return hits / len(cache), hits

    print("\n[tune_rrf] candidate pool depth (equal weights)")
    print("-" * 45)
    for pool_mult in [1, 2, 3, 4, 6, 8, 10, 12]:
        hr, hits = hit_rate(1.0, 1.0, TOP_K * pool_mult)
        print(f"pool=top_k*{pool_mult:<3} ({TOP_K*pool_mult:>3}) | hit_rate={hr:.3f} | {hits}/{len(cache)}")

    print("\n[tune_rrf] fusion weight (at pool=top_k*8)")
    print("-" * 45)
    for dw, bw in [(1.0, 0.0), (0.0, 1.0), (1.0, 0.5), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0)]:
        hr, hits = hit_rate(dw, bw, TOP_K * 8)
        print(f"dense={dw} bm25={bw} | hit_rate={hr:.3f} | {hits}/{len(cache)}")


if __name__ == "__main__":
    main()
