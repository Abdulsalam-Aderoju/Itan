"""Hybrid Retrieval Engine for Ìtàn (Blueprint Pillar 1).

Combines dense vector search (bge-small-en-v1.5) with sparse BM25 keyword search
fused via Reciprocal Rank Fusion (RRF). Hard-caps output to top-4 chunks
(max ~1,400 tokens) to maintain strict RAM and latency performance budgets.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

# Ensure torch is imported before numpy/pandas to prevent DLL collision on Windows
import torch  # noqa: F401
import numpy as np
import pandas as pd

# Paths setup
REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
CHUNKS_PARQUET = CORPUS_DIR / "chunks.parquet"
VECTORS_NPY = CORPUS_DIR / "vectors.npy"
CHUNK_IDS_JSON = CORPUS_DIR / "chunk_ids.json"
BM25_PKL = CORPUS_DIR / "bm25.pkl"
ONNX_DIR = CORPUS_DIR / "models" / "bge-small-en-v1.5-onnx"

class RetrievalEngine:
    """Thread-safe hybrid dense+sparse retrieval engine."""

    def __init__(self, corpus_dir: Path | str | None = None):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else CORPUS_DIR
        self.chunks_df: pd.DataFrame | None = None
        self.vectors: np.ndarray | None = None
        self.vec_chunk_ids: list[str] | None = None
        self.bm25_data: dict | None = None
        self.model = None
        self.tokenizer = None
        self.bm25_tokenizer = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Lazy load pre-built corpus indices."""
        chunks_path = self.corpus_dir / "chunks.parquet"
        vectors_path = self.corpus_dir / "vectors.npy"
        chunk_ids_path = self.corpus_dir / "chunk_ids.json"
        bm25_path = self.corpus_dir / "bm25.pkl"

        if chunks_path.exists():
            df = pd.read_parquet(chunks_path)
            if "chunk_id" in df.columns:
                df = df.set_index("chunk_id", drop=False)
            self.chunks_df = df

        if vectors_path.exists() and chunk_ids_path.exists():
            self.vectors = np.load(vectors_path).astype(np.float32)
            with open(chunk_ids_path, encoding="utf-8") as f:
                self.vec_chunk_ids = json.load(f)

        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25_data = pickle.load(f)

    def _init_embed_model(self):
        """Initialize the ONNX quantized bge-small embedder lazily."""
        if self.model is not None and self.tokenizer is not None:
            return
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        quantized_path = ONNX_DIR / "model_quantized.onnx"
        if not quantized_path.exists():
            raise FileNotFoundError(f"Quantized ONNX model not found at {quantized_path}. Run corpus/06_embed.py first.")

        self.tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
        self.model = ORTModelForFeatureExtraction.from_pretrained(ONNX_DIR, file_name="model_quantized.onnx")

    def _init_bm25_tokenizer(self):
        """Initialize BM25 tokenization module lazily."""
        if self.bm25_tokenizer is not None:
            return
        import importlib
        sys.path.insert(0, str(self.corpus_dir))
        try:
            bm25_mod = importlib.import_module("07_bm25")
            self.bm25_tokenizer = bm25_mod.tokenize
        except (ImportError, Exception):
            # Fallback regex tokenizer if module import is hindered
            import re
            self.bm25_tokenizer = lambda text: re.findall(r"\w+", text.lower())

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query into a normalized float32 vector."""
        self._init_embed_model()
        inputs = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="np")
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state
        cls_vec = last_hidden[:, 0, :]
        norm = np.linalg.norm(cls_vec, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        return (cls_vec / norm).astype(np.float32)[0]

    def dense_search(self, query_vec: np.ndarray, top_k: int = 10) -> list[str]:
        """Dense similarity search via brute-force dot product."""
        if self.vectors is None or self.vec_chunk_ids is None:
            return []
        scores = np.dot(self.vectors, query_vec)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [self.vec_chunk_ids[idx] for idx in top_indices]

    def bm25_search(self, query: str, top_k: int = 10) -> list[str]:
        """Sparse BM25 search."""
        if self.bm25_data is None:
            return []
        self._init_bm25_tokenizer()
        bm25_obj = self.bm25_data.get("bm25")
        doc_ids = self.bm25_data.get("doc_ids", [])
        if not bm25_obj or not doc_ids:
            return []
        query_tokens = self.bm25_tokenizer(query)
        scores = bm25_obj.get_scores(query_tokens)
        top_indices = np.argsort(-scores)[:top_k]
        return [doc_ids[idx] for idx in top_indices if scores[idx] > 0]

    def rrf_fuse(self, rank_lists: list[list[str]], k: int = 60, top_k: int = 4) -> list[str]:
        """Reciprocal Rank Fusion (RRF) over dense and sparse rank lists."""
        scores: dict[str, float] = {}
        for ranked in rank_lists:
            for rank, cid in enumerate(ranked):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        sorted_cids = sorted(scores.items(), key=lambda kv: -kv[1])
        return [cid for cid, _ in sorted_cids[:top_k]]

    def retrieve(self, query: str, top_k: int = 4, query_vec: np.ndarray | None = None) -> list[dict[str, Any]]:
        """Run end-to-end hybrid retrieval and return structured chunk metadata."""
        if self.chunks_df is None:
            return []

        if query_vec is None and self.vectors is not None:
            try:
                query_vec = self.embed_query(query)
            except Exception:
                query_vec = None

        dense_ids = self.dense_search(query_vec, top_k=top_k * 2) if query_vec is not None else []
        bm25_ids = self.bm25_search(query, top_k=top_k * 2)

        rank_lists = [l for l in (dense_ids, bm25_ids) if l]
        fused_cids = self.rrf_fuse(rank_lists, top_k=top_k) if rank_lists else []

        results = []
        for cid in fused_cids:
            if cid in self.chunks_df.index:
                row = self.chunks_df.loc[cid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                results.append({
                    "chunk_id": str(cid),
                    "text": str(row.get("text", "")),
                    "source_id": str(row.get("source_id", "")),
                    "crop": str(row.get("crops", "")),
                    "topic": str(row.get("topic", "")),
                    "doc_title": str(row.get("doc_title", "Extension Bulletin")),
                })
        return results
