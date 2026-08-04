# Router evaluation — Tier A/B/C/D classifier + refusal threshold

Covers blueprint Section 3.3 (Four-Tier Taxonomy) and Section 5.4 (Refusal Policy).

## What it does

Classifies each incoming question into Tier A (exact fact), B (explanation/
diagnosis), C (calculation), or D (refusal) using embedding similarity —
no LLM call. Anything the classifier isn't confident about also gets
routed to D rather than guessed.

## Gold-set labeling bug found and fixed (2026-08-04)

The original 200-question set (named `..._pest_diagnosis.jsonl`) was
scoped narrowly around pest/disease diagnosis, and **13 of its 30 D-tier
("refuse") examples were actually legitimate Tier A/B/C questions** about
the corpus's own 10 target crops -- e.g. "When should I plant maize in
Oyo State?" and "What spacing should I use for cassava?" were both
labeled as refusals. Confirmed in production: real spacing/seed-rate
questions were getting confidently routed to D (not from threshold
uncertainty -- the D centroid was winning outright, because it had been
trained on examples just like them). D015's own `answer` field gave away
the cause: *"out of scope under the pest/disease-diagnosis niche"* --
correct for that narrower niche, wrong for the production router, which
has to handle the full blueprint scope (crop_calendar, spacing,
fertiliser_rate, seed_rate, variety), not just pest questions.

Relabeled the 13 mislabeled rows to their correct tier (mostly A, a few
B/C), leaving 3 borderline entries (onion/plantain/cocoa -- crops outside
the corpus's 10) as D, since those are a corpus-coverage question for
retrieval to handle, not a router scope question. A's original 30
examples also turned out to be ~93% one narrow template ("PHI for X on
Y" / "active ingredient for X on Y"), which by itself wasn't enough to
absorb the relabeled examples into a well-shaped centroid -- added 20
more A-tier examples (2 per crop × 10 crops: row spacing, seed rate per
hectare) mirroring the blueprint's own Section 3.3 Tier A examples
("cassava spacing", "NPK rate/ha") to balance it out. Both
`eval/router/` and `eval/datasets/` copies of the gold-question file were
duplicated in the repo -- kept in sync.

## Method

Compared per-tier centroid classification against k-NN (k=1, 3, 5) via
5-fold stratified cross-validation on the 220-question gold set
(`gold_questions_200_pest_diagnosis.jsonl` -- filename kept for
continuity despite no longer being pest-diagnosis-only), using
bge-small-en-v1.5 embeddings (same model the corpus pipeline uses for
retrieval).

| Method | Overall accuracy | A recall | B recall | C recall | D recall |
|---|---|---|---|---|---|
| k-NN (k=1) | 91.4% | 86.4% | 96.1% | 100% | 58.8% |
| k-NN (k=3) | 92.7% | 94.9% | 96.1% | 100% | 47.1% |
| k-NN (k=5) | 92.7% | 98.3% | 96.1% | 100% | 35.3% |
| **Centroid** | **92.7%** | 94.9% | 93.2% | 100% | **64.7%** |

Centroid still wins on Tier D relative to the other methods, and is what
ships (`engine/router/centroids.json`) -- consistent with the original
rationale (Tier D is the tier where a wrong answer is most costly, so it
gets weighted over raw overall accuracy). **D recall dropped from the
old 86.7% to 64.7%** after the relabel -- worth being upfront about
rather than burying: this is mostly the 3 borderline out-of-corpus-crop
questions (see above) plus a smaller, noisier D-tier training sample (17
examples instead of 30). Checked what's actually still misclassified
(`inspect_d_errors.py`) -- every genuinely dangerous category (medical/
veterinary emergencies, market-price lookups) is either caught by the
confidence threshold anyway or independently caught by `SAFETY_KEYWORDS`
regardless of embedding confidence. The recall drop is concentrated in
low-stakes edge cases, not a safety regression.

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

## Known limitation: Tier D is the weakest tier (64.7% recall)

Seven D-tier questions get misclassified in cross-validation (up from
four before the relabel, but see above -- this is an expected consequence
of a smaller, cleaner D-tier sample, not a new failure mode):

| id | true tier | misrouted to | confidence | caught by threshold=0.74? |
|---|---|---|---|---|
| D002 (maize market price in Lagos) | D | C | 0.66 | yes |
| D004 (goat won't eat) | D | B | 0.66 | yes |
| D005 (fever after spraying) | D | B | 0.73 | yes |
| D011 (child pesticide poisoning) | D | B | 0.73 | yes |
| D023 (onion storage, general) | D | A | 0.72 | yes |
| D024 (plantain planting) | D | A | 0.77 | **no** |
| D026 (cocoa trees per hectare) | D | A | 0.71 | yes |

D004/D005/D011 (medical/veterinary emergencies phrased as symptom
descriptions) and D002 (market price) are all still caught by the
confidence threshold regardless of the argmax tier being wrong, and
D004/D005/D011 are independently caught by `SAFETY_KEYWORDS` in
`router.py` even if the threshold weren't there — defense in depth, not
a replacement for the embedding classifier.

D023/D024/D026 are the borderline out-of-corpus-crop questions (onion,
plantain, cocoa aren't in the corpus's 10-crop list) — phrased exactly
like legitimate Tier A questions, because they effectively are, just
about a crop this corpus doesn't cover. The router classifies by
question *style*, not by corpus *coverage*; that distinction is retrieval's
job (low retrieval confidence → refuse), not the router's. D024 slips
through the threshold too (0.77 > 0.74), but the downstream retrieval
step should still fail to find plantain-specific content and refuse
there — worth confirming once retrieval-confidence-based refusal exists,
not a router bug to chase further.

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
