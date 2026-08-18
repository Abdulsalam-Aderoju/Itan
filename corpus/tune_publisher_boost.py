#!/usr/bin/env python
"""Sweep for EXTENSION_PUBLISHER_BOOST (engine/retrieval.py) against the
50-question corpus validation set, plus a direct count of how often a
manual/extension-guide chunk gets promoted into the top-4 that wasn't
there under plain (unweighted, no publisher boost) fusion.

Not a pipeline stage -- a standalone diagnostic, same pattern as
tune_rrf.py. Re-run after any corpus rebuild that changes the academic/
manual ratio meaningfully.

Usage: python corpus/tune_publisher_boost.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "corpus"))

from engine.retrieval import RetrievalEngine, CROP_MATCH_BOOST, CROP_MISMATCH_PENALTY, EXTENSION_PUBLISHERS  # noqa: E402
import importlib  # noqa: E402
val = importlib.import_module("08_validate")

TOP_K = val.TOP_K
POOL = TOP_K * 8  # matches RRF_CANDIDATE_POOL_MULT


def fuse(eng, dense_ids, bm25_ids, query_crops, publisher_boost, k=60, top_k=TOP_K):
    scores: dict[str, float] = {}
    for ranked in (dense_ids, bm25_ids):
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    if query_crops:
        query_crop_set = set(query_crops)
        for cid in scores:
            if cid not in eng.chunks_df.index:
                continue
            row = eng.chunks_df.loc[cid]
            chunk_crops = {c for c in str(row.get("crops", "")).split(";") if c}
            if not chunk_crops:
                continue
            scores[cid] *= CROP_MATCH_BOOST if (chunk_crops & query_crop_set) else CROP_MISMATCH_PENALTY

    if publisher_boost != 1.0:
        for cid in scores:
            if cid not in eng.chunks_df.index:
                continue
            row = eng.chunks_df.loc[cid]
            if str(row.get("publisher", "")) in EXTENSION_PUBLISHERS:
                scores[cid] *= publisher_boost

    sorted_cids = sorted(scores.items(), key=lambda kv: -kv[1])
    return [cid for cid, _ in sorted_cids[:top_k]]


def main():
    print("[tune_publisher] loading RetrievalEngine ...")
    eng = RetrievalEngine()

    cache = []
    t0 = time.time()
    for i, q in enumerate(val.GOLD_QUESTIONS, start=1):
        qvec = eng.embed_query(q["query"])
        dense_ids = eng.dense_search(qvec, top_k=POOL)
        bm25_ids = eng.bm25_search(q["query"], top_k=POOL)
        query_crops = eng.detect_query_crops(q["query"])
        cache.append((q, dense_ids, bm25_ids, query_crops))
        print(f"  cached {i}/{len(val.GOLD_QUESTIONS)}", end="\r")
    print(f"\n[tune_publisher] cached all candidate lists in {time.time()-t0:.1f}s")

    def hit_rate(publisher_boost):
        hits = 0
        for q, dense_ids, bm25_ids, query_crops in cache:
            fused = fuse(eng, dense_ids, bm25_ids, query_crops, publisher_boost)
            if val.is_hit(fused, eng.chunks_df, q):
                hits += 1
        return hits / len(cache), hits

    def manual_promotions(publisher_boost):
        """How many questions get at least one manual-publisher chunk in
        their top-4 under this boost that had NONE under boost=1.0."""
        promoted = 0
        for q, dense_ids, bm25_ids, query_crops in cache:
            baseline = fuse(eng, dense_ids, bm25_ids, query_crops, 1.0)
            boosted = fuse(eng, dense_ids, bm25_ids, query_crops, publisher_boost)

            def has_manual(cids):
                for cid in cids:
                    if cid in eng.chunks_df.index and str(eng.chunks_df.loc[cid].get("publisher", "")) in EXTENSION_PUBLISHERS:
                        return True
                return False

            if has_manual(boosted) and not has_manual(baseline):
                promoted += 1
        return promoted

    print("\n[tune_publisher] boost | hit_rate@5 | questions gaining a manual chunk in top-4")
    print("-" * 65)
    for boost in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        hr, hits = hit_rate(boost)
        promoted = manual_promotions(boost) if boost != 1.0 else 0
        print(f"{boost:>5.2f} | {hr:.3f} ({hits}/{len(cache)}) | {promoted}/{len(cache)}")


if __name__ == "__main__":
    main()
