# Router evaluation — Tier A/B/C/D classifier + refusal threshold

Covers blueprint Section 3.3 (Four-Tier Taxonomy) and Section 5.4 (Refusal Policy).

## What it does

Classifies each incoming question into Tier A (exact fact), B (explanation/
diagnosis), C (calculation), or D (refusal) using embedding similarity —
no LLM call. Anything the classifier isn't confident about also gets
routed to D rather than guessed.

## Method

Compared per-tier centroid classification against k-NN (k=1, 3, 5) via
5-fold stratified cross-validation on the 200-question pest-diagnosis gold
set (`gold_questions_200_pest_diagnosis.jsonl`), using bge-small-en-v1.5
embeddings (same model the corpus pipeline uses for retrieval).

| Method | Overall accuracy | A recall | B recall | C recall | D recall |
|---|---|---|---|---|---|
| k-NN (k=1) | 94.5% | 96.7% | 99.0% | 100% | 70.0% |
| k-NN (k=3) | 93.5% | 96.7% | 99.0% | 100% | 63.3% |
| k-NN (k=5) | 93.5% | 96.7% | 100% | 100% | 60.0% |
| **Centroid** | **95.0%** | 90.0% | 97.0% | 100% | **86.7%** |

Centroid won, especially on Tier D (the tier that matters most to get
right) — shipped as `engine/router/centroids.json`.

## Confidence threshold: 0.74

Chosen by sweeping thresholds against cross-validated confidence scores
(full sweep in `calibration_report.json`):

- Catches 70% of the classifier's own cross-validation errors, at a 10%
  false-refusal cost on questions it would have answered correctly.
- Validated against 15 synthetic out-of-domain questions (sports scores,
  recipes, general trivia, a prompt-injection attempt) — **100% correctly
  refused** (max confidence among them was 0.65, comfortably under 0.74).

Going higher (e.g. 0.76) catches every CV error but starts costing more
correct classifications (13.2%); going lower misses more errors. 0.74 was
the point right before the cost curve steepens — reasonable default, not
a law of nature. Revisit if real usage data suggests otherwise.

## Known limitation: Tier D is the weakest tier (86.7% recall)

Four D-tier questions get misclassified in cross-validation:

| id | true tier | misrouted to | confidence | caught by threshold? |
|---|---|---|---|---|
| D004 (goat won't eat) | D | B | 0.66 | yes |
| D005 (fever after spraying) | D | B | 0.73 | yes |
| D011 (child pesticide poisoning) | D | B | 0.73 | yes |
| D029 (calc income from yield) | D | C | 0.75 | **no** |

Three of four are medical/veterinary emergencies phrased as symptom
descriptions — they surface-resemble Tier B diagnosis questions to a
generic embedding model, which has no notion that "my goat" and "my maize"
belong in different universes of consequence. The threshold catches most
of these anyway, but not reliably enough on its own for genuinely
high-stakes content — hence `SAFETY_KEYWORDS` in `router.py`: a keyword
override for health/veterinary/legal/financial terms that forces Tier D
regardless of embedding confidence. Defense in depth, not a replacement
for the embedding classifier.

D029 is arguably a labeling call I got wrong when building the gold set,
not a real router failure — "how do I calculate my expected income from
projected yield" genuinely resembles the gross_margin tool's job. Worth
someone double-checking that gold-set entry rather than tuning the router
around it.

## Reproducing

```
pip install sentence-transformers   # pulls in torch CPU + transformers
python eval/router/build_router.py       # cross-validation + threshold sweep -> calibration_report.json
python eval/router/inspect_d_errors.py   # which D-tier questions get misrouted, and at what confidence
python eval/router/ood_stress_test.py    # out-of-domain stress test
python eval/router/build_centroids.py    # regenerates engine/router/centroids.json from all 200 questions
python -m engine.router.standalone       # quick end-to-end demo
```

## Using it in the real pipeline

```python
from engine.router import Router

router = Router()  # loads engine/router/centroids.json
result = router.classify(question_text, embedding)  # embedding: reuse retrieval's bge-small embedding, don't load a second copy
# result.tier -> "A" | "B" | "C" | "D"
# result.refused -> True if tier == "D"
# result.reason -> "classified" | "low_confidence" | "safety_keyword_override"
```

`engine/router/standalone.py` has a `StandaloneRouter` that loads its own
sentence-transformers model, for local testing only — not for production,
since the real pipeline already has bge-small loaded for retrieval and
shouldn't load it twice.
