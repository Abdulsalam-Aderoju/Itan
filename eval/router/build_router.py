# -*- coding: utf-8 -*-
"""
Build + evaluate the Tier A/B/C/D router (blueprint Sec 3.3) and calibrate
the refusal confidence threshold (blueprint Sec 5.4).

Design: embedding-similarity lookup against the 200 labeled gold questions,
using bge-small-en-v1.5 (matches the corpus pipeline's embedding model
choice). No LLM call -- must be fast (<5ms/query per the blueprint's own
budget).

IMPORTANT: embeddings here MUST come from the exact same pipeline used at
inference time (engine.retrieval.RetrievalEngine.embed_query(), the
quantized ONNX bge-small used for retrieval) rather than a separately-loaded
raw sentence-transformers model. An earlier version of this script used
plain SentenceTransformer(...).encode() -- cosine similarity to the
production ONNX embedder for the same text was only ~0.97-0.99 (quantization
+ pooling-path drift, not a bug in either embedder individually), and the
router's tier-decision margins are frequently much narrower than that (e.g.
0.03-0.05 between the top two tiers), so that drift alone was enough to
systematically misroute clearly in-scope questions to Tier D. Centroids/
thresholds calibrated on the wrong embedding space silently degrade routing
in production even though every unit test passes.

Method selection is itself an ablation, not an assumption: we cross-validate
k-NN (k=1,3,5) against a per-tier centroid classifier and report numbers for
both before picking one.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from engine.retrieval import RetrievalEngine  # noqa: E402

GOLD_PATH = Path(__file__).resolve().parents[1] / "datasets" / "gold_questions_200_pest_diagnosis.jsonl"
MODEL_NAME = "BAAI/bge-small-en-v1.5-onnx-quantized (production pipeline, via RetrievalEngine.embed_query)"
SEED = 42
N_FOLDS = 5

TIERS = ["A", "B", "C", "D"]


def load_gold():
    rows = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_folds(rows, n_folds, seed):
    """Stratified k-fold split indices, grouped per tier so each fold has a
    proportional slice of every tier (important -- Tier A/D only have 30
    examples each, a plain shuffle-split could starve a fold of a tier)."""
    rng = np.random.default_rng(seed)
    by_tier = defaultdict(list)
    for i, r in enumerate(rows):
        by_tier[r["tier"]].append(i)
    folds = [[] for _ in range(n_folds)]
    for tier, idxs in by_tier.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % n_folds].append(int(idx))
    return folds


def cosine_sim_matrix(a, b):
    """a: (n,d), b: (m,d), both L2-normalized -> (n,m) cosine similarities."""
    return a @ b.T


def knn_predict(query_emb, ref_emb, ref_tiers, k):
    sims = cosine_sim_matrix(query_emb, ref_emb)  # (n_query, n_ref)
    top_k_idx = np.argsort(-sims, axis=1)[:, :k]
    preds, confs = [], []
    for row, idxs in zip(sims, top_k_idx):
        neighbor_tiers = [ref_tiers[i] for i in idxs]
        vote = Counter(neighbor_tiers).most_common(1)[0][0]
        # confidence = mean similarity of the neighbors that voted for the winning tier
        winning_sims = [row[i] for i in idxs if ref_tiers[i] == vote]
        confs.append(float(np.mean(winning_sims)))
        preds.append(vote)
    return preds, confs


def centroid_predict(query_emb, centroids_by_tier):
    tiers = list(centroids_by_tier.keys())
    centroid_mat = np.stack([centroids_by_tier[t] for t in tiers])  # (4, d)
    sims = cosine_sim_matrix(query_emb, centroid_mat)  # (n_query, 4)
    preds, confs, margins = [], [], []
    order = np.argsort(-sims, axis=1)
    for row, ranked in zip(sims, order):
        best_i, second_i = ranked[0], ranked[1]
        preds.append(tiers[best_i])
        confs.append(float(row[best_i]))
        margins.append(float(row[best_i] - row[second_i]))
    return preds, confs, margins


def evaluate_method(name, rows, embeddings, folds, predict_fn):
    """predict_fn(train_idx, test_idx) -> (preds, confs) for the test fold."""
    all_true, all_pred, all_conf = [], [], []
    for fi in range(len(folds)):
        test_idx = folds[fi]
        train_idx = [i for j, f in enumerate(folds) if j != fi for i in f]
        preds, confs = predict_fn(train_idx, test_idx)
        all_true.extend(rows[i]["tier"] for i in test_idx)
        all_pred.extend(preds)
        all_conf.extend(confs)

    acc = np.mean([t == p for t, p in zip(all_true, all_pred)])
    per_tier = {}
    for tier in TIERS:
        idxs = [i for i, t in enumerate(all_true) if t == tier]
        tier_acc = np.mean([all_true[i] == all_pred[i] for i in idxs]) if idxs else float("nan")
        per_tier[tier] = round(float(tier_acc), 4)

    print(f"\n=== {name} ===")
    print(f"overall accuracy: {acc:.4f}  (n={len(all_true)})")
    print(f"per-tier recall: {per_tier}")
    return {
        "name": name, "overall_acc": float(acc), "per_tier_recall": per_tier,
        "true": all_true, "pred": all_pred, "conf": all_conf,
    }


def main():
    print("Loading gold questions...")
    rows = load_gold()
    print(f"{len(rows)} questions loaded, tiers: {Counter(r['tier'] for r in rows)}")

    print(f"Loading production retrieval engine ({MODEL_NAME})...")
    engine = RetrievalEngine()

    print("Embedding all questions (via the same embed_query() path used at inference time)...")
    t0 = time.time()
    texts = [r["question"] for r in rows]
    embeddings = np.stack([engine.embed_query(t) for t in texts]).astype(np.float32)
    print(f"Embedded {len(texts)} questions in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/len(texts)*1000:.2f} ms/question)")

    folds = stratified_folds(rows, N_FOLDS, SEED)
    tiers_arr = [r["tier"] for r in rows]

    results = {}

    for k in (1, 3, 5):
        def predict_fn(train_idx, test_idx, k=k):
            ref_emb = embeddings[train_idx]
            ref_tiers = [tiers_arr[i] for i in train_idx]
            query_emb = embeddings[test_idx]
            preds, confs = knn_predict(query_emb, ref_emb, ref_tiers, k)
            return preds, confs
        results[f"knn_k{k}"] = evaluate_method(f"k-NN (k={k})", rows, embeddings, folds, predict_fn)

    def centroid_fn(train_idx, test_idx):
        ref_emb = embeddings[train_idx]
        ref_tiers = [tiers_arr[i] for i in train_idx]
        centroids = {}
        for tier in TIERS:
            tier_idx = [i for i, t in enumerate(ref_tiers) if t == tier]
            c = ref_emb[tier_idx].mean(axis=0)
            centroids[tier] = c / np.linalg.norm(c)
        query_emb = embeddings[test_idx]
        preds, confs, margins = centroid_predict(query_emb, centroids)
        return preds, confs
    results["centroid"] = evaluate_method("Centroid", rows, embeddings, folds, centroid_fn)

    # pick best method by overall CV accuracy
    best_name = max(results, key=lambda k: results[k]["overall_acc"])
    print(f"\n>>> Best method by cross-validated accuracy: {best_name} "
          f"({results[best_name]['overall_acc']:.4f})")

    # --- confidence threshold calibration on the best method ---
    best = results[best_name]
    correct_conf = [c for t, p, c in zip(best["true"], best["pred"], best["conf"]) if t == p]
    wrong_conf = [c for t, p, c in zip(best["true"], best["pred"], best["conf"]) if t != p]
    print(f"\nConfidence stats for {best_name}:")
    print(f"  correct predictions: n={len(correct_conf)}, mean={np.mean(correct_conf):.4f}, "
          f"p10={np.percentile(correct_conf,10):.4f}")
    print(f"  wrong predictions:   n={len(wrong_conf)}, mean={np.mean(wrong_conf) if wrong_conf else float('nan'):.4f}, "
          f"p90={np.percentile(wrong_conf,90) if wrong_conf else float('nan'):.4f}")

    # sweep thresholds: for each candidate, what fraction of WRONG predictions
    # would have been correctly refused, vs what fraction of CORRECT
    # predictions would be wrongly refused (the real cost)
    print("\nThreshold sweep:")
    print(f"{'threshold':<10}{'wrong_caught_%':<16}{'correct_lost_%':<16}")
    candidates = np.arange(0.50, 0.96, 0.02)
    sweep_rows = []
    for th in candidates:
        wrong_caught = np.mean([c < th for c in wrong_conf]) * 100 if wrong_conf else 0.0
        correct_lost = np.mean([c < th for c in correct_conf]) * 100
        sweep_rows.append((float(th), float(wrong_caught), float(correct_lost)))
        print(f"{th:<10.2f}{wrong_caught:<16.1f}{correct_lost:<16.1f}")

    out = {
        "model": MODEL_NAME,
        "n_gold_questions": len(rows),
        "n_folds": N_FOLDS,
        "methods": {k: {"overall_acc": v["overall_acc"], "per_tier_recall": v["per_tier_recall"]} for k, v in results.items()},
        "best_method": best_name,
        "confidence_stats": {
            "correct_mean": float(np.mean(correct_conf)),
            "correct_p10": float(np.percentile(correct_conf, 10)),
            "wrong_mean": float(np.mean(wrong_conf)) if wrong_conf else None,
            "wrong_p90": float(np.percentile(wrong_conf, 90)) if wrong_conf else None,
            "n_correct": len(correct_conf),
            "n_wrong": len(wrong_conf),
        },
        "threshold_sweep": sweep_rows,
    }
    out_path = Path(__file__).parent / "calibration_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(
        "\nNote: this script reports cross-validated accuracy/threshold numbers only. "
        "The actual production centroids shipped in engine/router/centroids.json are "
        "computed from ALL 200 gold questions (not a CV fold) -- see build_centroids.py."
    )


if __name__ == "__main__":
    main()
