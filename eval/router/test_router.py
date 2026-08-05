# -*- coding: utf-8 -*-
"""Fast unit tests for engine/router -- no embedding model needed.

Covers the parts of the router that don't require re-running bge-small:
centroids.json integrity, safety-keyword overrides, and threshold behavior
given a synthetic embedding. The actual classification-accuracy numbers
live in calibration_report.json (produced by build_router.py, which does
need the embedding model).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.router import Router, TIERS

passed = 0
failed = 0


def check(label, condition):
    global passed, failed
    if condition:
        print(f"PASS | {label}")
        passed += 1
    else:
        print(f"FAIL | {label}")
        failed += 1


router = Router()

# --- centroids.json integrity ---
check("all 4 tiers present in centroids", all(t in TIERS for t in ("A", "B", "C", "D")))
check("embedding_dim matches bge-small-en-v1.5 (384)", router.embedding_dim == 384)
check("threshold is a sane probability-like value", 0.0 < router.threshold < 1.0)

# --- safety keyword overrides (no embedding needed -- short-circuits before use) ---
dummy_emb = np.zeros(router.embedding_dim, dtype=np.float32)
dummy_emb[0] = 1.0  # arbitrary unit vector so norm > 0

safety_cases = [
    "What loan options are available for buying a tractor?",
    "My goat has stopped eating, what medicine should I give it?",
    "I have a fever and body pain after spraying my farm.",
    "Can you recommend a bank for an agricultural loan?",
]
# note: weather-forecast questions ("will it rain next week") ARE in
# SAFETY_KEYWORDS (as a narrow "will it rain" / "weather forecast" pattern) --
# but a plain "\bweather\b" was here originally and false-triggered on
# ordinary plant-pathology phrasing like "...fuzzy in humid weather"
# (found via full-scale testing against the real production ONNX embedder,
# eval/router/gold_questions_200_pest_diagnosis.jsonl row B058). Regression
# test for that below: a legitimate disease-symptom question mentioning
# weather as a description, not a forecast request, must NOT be refused.
for q in safety_cases:
    result = router.classify(q, dummy_emb)
    check(f"safety override catches: {q[:50]!r}", result.tier == "D" and result.reason == "safety_keyword_override")

not_safety_cases = [
    "My onion shows pale yellow patches turning greyish-purple and fuzzy in humid weather. What is the problem?",
    "My tomato wilts more in hot, dry weather than usual -- is that a disease symptom?",
]
for q in not_safety_cases:
    result = router.classify(q, dummy_emb)
    check(f"NOT falsely caught by safety override: {q[:50]!r}", result.reason != "safety_keyword_override")

# --- confidence threshold behavior with synthetic embeddings ---
# a vector identical to the A centroid should classify as A with confidence ~1.0
import json
centroids_data = json.loads((Path(__file__).resolve().parents[2] / "engine" / "router" / "centroids.json").read_text())
a_centroid = np.array(centroids_data["centroids"]["A"], dtype=np.float32)
result = router.classify("some neutral non-safety-keyword text", a_centroid)
check("embedding identical to A centroid classifies as A", result.tier == "A")
check("confidence near 1.0 for exact centroid match", result.confidence > 0.99)

# a low-magnitude random vector orthogonal-ish to all centroids should get refused
rng = np.random.default_rng(0)
random_emb = rng.normal(size=router.embedding_dim).astype(np.float32)
result = router.classify("some neutral non-safety-keyword text", random_emb)
check(
    f"random unrelated embedding gets refused or low confidence (got tier={result.tier}, conf={result.confidence:.3f})",
    result.refused or result.confidence < router.threshold + 0.1,
)

# --- dimension mismatch should raise, not silently misbehave ---
try:
    router.classify("test", np.zeros(10, dtype=np.float32))
    check("wrong-dim embedding raises ValueError", False)
except ValueError:
    check("wrong-dim embedding raises ValueError", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
