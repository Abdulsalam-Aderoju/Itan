# -*- coding: utf-8 -*-
"""
Compute the production tier centroids shipped in engine/router/centroids.json.

Unlike build_router.py (which cross-validates on held-out folds to measure
expected accuracy), this script uses ALL 200 gold questions to build the
final centroids -- once we trust the method (validated in build_router.py),
there's no reason to hold data back from the shipped classifier.

IMPORTANT: must embed with the exact same pipeline engine.agent uses at
inference time (RetrievalEngine.embed_query(), quantized ONNX bge-small),
not a separately-loaded sentence-transformers model -- see build_router.py's
docstring for why a ~0.97-0.99 cosine-similarity drift between the two
pipelines was enough to systematically misroute in-scope questions to
Tier D in production despite centroids.json passing its own calibration.

Re-run this whenever agbe_eval_questions.jsonl / gold_questions_200_pest_diagnosis.jsonl
gets more/better-labeled examples.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from engine.retrieval import RetrievalEngine  # noqa: E402

GOLD_PATH = Path(__file__).resolve().parents[1] / "datasets" / "gold_questions_200_pest_diagnosis.jsonl"
CENTROIDS_OUT = Path(__file__).resolve().parents[2] / "engine" / "router" / "centroids.json"
MODEL_NAME = "BAAI/bge-small-en-v1.5-onnx-quantized (production pipeline, via RetrievalEngine.embed_query)"
TIERS = ["A", "B", "C", "D"]

# From build_router.py's 5-fold stratified cross-validation (see calibration_report.json
# for the full sweep). Re-run build_router.py and update these if the gold set changes
# enough to shift the numbers.
#
# NOTE: these are the CENTROID method's own numbers, not the overall best method
# in calibration_report.json. build_router.py's cross-validation started favoring
# k-NN (k=1, 94.6%) over centroid (92.5%) once measured with the real production
# ONNX embedder on the expanded 240-question set -- but engine/router/router.py
# only implements centroid comparison, so the number that actually describes what's
# shipped is centroid's, not whichever method currently tops the comparison table.
# If we ever want the k-NN edge, Router itself needs a k-NN mode first.
THRESHOLD = 0.74
CV_OVERALL_ACCURACY = 0.925
CV_PER_TIER_RECALL = {"A": 0.9494, "B": 0.9223, "C": 1.0, "D": 0.6471}


def main():
    rows = [json.loads(l) for l in open(GOLD_PATH, encoding="utf-8")]
    engine = RetrievalEngine()
    texts = [r["question"] for r in rows]
    emb = np.stack([engine.embed_query(t) for t in texts]).astype(np.float32)
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
