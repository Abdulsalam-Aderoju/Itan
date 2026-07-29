# -*- coding: utf-8 -*-
"""
Compute the production tier centroids shipped in engine/router/centroids.json.

Unlike build_router.py (which cross-validates on held-out folds to measure
expected accuracy), this script uses ALL 200 gold questions to build the
final centroids -- once we trust the method (validated in build_router.py),
there's no reason to hold data back from the shipped classifier.

Re-run this whenever agbe_eval_questions.jsonl / gold_questions_200_pest_diagnosis.jsonl
gets more/better-labeled examples.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

GOLD_PATH = Path(__file__).parent / "gold_questions_200_pest_diagnosis.jsonl"
CENTROIDS_OUT = Path(__file__).parent.parent.parent / "engine" / "router" / "centroids.json"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TIERS = ["A", "B", "C", "D"]

# From build_router.py's 5-fold stratified cross-validation (see calibration_report.json
# for the full sweep). Re-run build_router.py and update these if the gold set changes
# enough to shift the numbers.
THRESHOLD = 0.74
CV_OVERALL_ACCURACY = 0.95
CV_PER_TIER_RECALL = {"A": 0.9, "B": 0.97, "C": 1.0, "D": 0.8667}


def main():
    rows = [json.loads(l) for l in open(GOLD_PATH, encoding="utf-8")]
    model = SentenceTransformer(MODEL_NAME)
    texts = [r["question"] for r in rows]
    emb = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    tiers = [r["tier"] for r in rows]

    centroids = {}
    for tier in TIERS:
        idx = [i for i, t in enumerate(tiers) if t == tier]
        c = emb[idx].mean(axis=0)
        centroids[tier] = (c / np.linalg.norm(c)).tolist()

    out = {
        "model": MODEL_NAME,
        "embedding_dim": emb.shape[1],
        "method": "centroid",
        "threshold": THRESHOLD,
        "calibrated_on": f"{GOLD_PATH.name} ({len(rows)} questions, pest-diagnosis niche), 5-fold stratified CV",
        "cv_overall_accuracy": CV_OVERALL_ACCURACY,
        "cv_per_tier_recall": CV_PER_TIER_RECALL,
        "centroids": centroids,
    }
    CENTROIDS_OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {CENTROIDS_OUT}")


if __name__ == "__main__":
    main()
