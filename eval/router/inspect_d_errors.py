import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

GOLD_PATH = str(Path(__file__).parent / "gold_questions_200_pest_diagnosis.jsonl")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
SEED = 42
N_FOLDS = 5
TIERS = ["A", "B", "C", "D"]

rows = [json.loads(l) for l in open(GOLD_PATH, encoding="utf-8")]
model = SentenceTransformer(MODEL_NAME)
texts = [r["question"] for r in rows]
embeddings = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

rng = np.random.default_rng(SEED)
by_tier = defaultdict(list)
for i, r in enumerate(rows):
    by_tier[r["tier"]].append(i)
folds = [[] for _ in range(N_FOLDS)]
for tier, idxs in by_tier.items():
    idxs = np.array(idxs)
    rng.shuffle(idxs)
    for j, idx in enumerate(idxs):
        folds[j % N_FOLDS].append(int(idx))

results = []
for fi in range(N_FOLDS):
    test_idx = folds[fi]
    train_idx = [i for j, f in enumerate(folds) if j != fi for i in f]
    ref_emb = embeddings[train_idx]
    ref_tiers = [rows[i]["tier"] for i in train_idx]
    centroids = {}
    for tier in TIERS:
        tidx = [i for i, t in enumerate(ref_tiers) if t == tier]
        c = ref_emb[tidx].mean(axis=0)
        centroids[tier] = c / np.linalg.norm(c)
    centroid_mat = np.stack([centroids[t] for t in TIERS])
    query_emb = embeddings[test_idx]
    sims = query_emb @ centroid_mat.T
    order = np.argsort(-sims, axis=1)
    for qi, ranked in zip(test_idx, order):
        pred = TIERS[ranked[0]]
        conf = float(sims[list(test_idx).index(qi)][ranked[0]])
        true = rows[qi]["tier"]
        results.append((rows[qi]["id"], rows[qi]["question"], true, pred, conf))

print(f"{'id':<8}{'true':<6}{'pred':<6}{'conf':<8}question")
for rid, q, true, pred, conf in results:
    if true == "D" and pred != "D":
        print(f"{rid:<8}{true:<6}{pred:<6}{conf:<8.4f}{q[:90]}")

print()
print("Would threshold=0.74 catch these (refuse due to low confidence)?")
for rid, q, true, pred, conf in results:
    if true == "D" and pred != "D":
        caught = conf < 0.74
        print(f"  {rid}: conf={conf:.4f} -> {'CAUGHT (refused anyway)' if caught else 'MISSED (confidently wrong)'}")
