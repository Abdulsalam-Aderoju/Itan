#!/usr/bin/env python
"""
Stage 7 (build): BM25 sparse index over all chunks.

Tokenization is deliberately dumb: lowercase + whitespace/punctuation split,
NO stemming or lemmatization. Agricultural proper nouns and chemical/variety
names must match exactly, so "maize", "maize's", and "maizes" are kept as
three distinct tokens rather than being collapsed to one stem.

Output: corpus/bm25.pkl (contains the BM25Okapi index, the tokenized
corpus, and the chunk_id list so 08_validate.py can map hits back to
chunks without re-reading the parquet file).

Runnable standalone: python corpus/07_bm25.py
"""
from __future__ import annotations

import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS_DIR, print_elapsed_checkpoint, progress_line  # noqa: E402

import pandas as pd
from rank_bm25 import BM25Okapi

CHUNKS_PARQUET = CORPUS_DIR / "chunks.parquet"
BM25_PKL = CORPUS_DIR / "bm25.pkl"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Whitespace/punctuation tokenizer, lowercase, no stemming. Internal
    apostrophes are kept attached so "maize's" stays distinct from "maize"."""
    return TOKEN_RE.findall(text.lower())


def main():
    if not CHUNKS_PARQUET.exists():
        print(f"[bm25] ERROR: {CHUNKS_PARQUET} not found. Run 04_chunk.py first.")
        sys.exit(1)

    print(f"[bm25] loading {CHUNKS_PARQUET} ...")
    df = pd.read_parquet(CHUNKS_PARQUET)
    chunk_ids = df["chunk_id"].tolist()
    texts = df["text"].tolist()
    total = len(texts)
    print(f"[bm25] tokenizing {total} chunks ...")

    tokenized_corpus = []
    start_time = time.time()
    for i, (cid, t) in enumerate(zip(chunk_ids, texts), start=1):
        try:
            tokenized_corpus.append(tokenize(t))
            progress_line("bm25", i, total, cid, ok=True)
        except Exception as exc:
            tokenized_corpus.append([])
            progress_line("bm25", i, total, cid, ok=False, reason=str(exc))
        print_elapsed_checkpoint("bm25", i, total, start_time)

    print("[bm25] building BM25Okapi index (single atomic call, no sub-progress available) ...")
    index_start = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"[bm25] index build done in {time.time() - index_start:.1f}s")

    print(f"[bm25] writing index to {BM25_PKL} ...")
    with open(BM25_PKL, "wb") as fh:
        pickle.dump({
            "bm25": bm25,
            "tokenized_corpus": tokenized_corpus,
            "chunk_ids": chunk_ids,
        }, fh)
    print(f"[bm25] write done")

    avg_len = sum(len(t) for t in tokenized_corpus) / max(len(tokenized_corpus), 1)
    print(f"\n[bm25] DONE. {len(chunk_ids)} documents indexed, avg tokens/doc={avg_len:.1f}")
    print(f"[bm25] Written: {BM25_PKL}")


if __name__ == "__main__":
    main()
