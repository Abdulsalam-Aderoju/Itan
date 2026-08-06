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

## Widened A-tier coverage: planting-time + variety questions (2026-08-05)

Verifying the fix above (live-tested against the *real* production
embedder, not just the reference one -- see the embedder note below)
surfaced an adjacent gap: the relabel fixed spacing/seed-rate questions,
but **planting-time** ("When should I plant maize in Oyo State?") and
**variety-selection** ("Which rice variety should I plant?") phrasings
were *still* misrouted to D, because nothing in Tier A's training data
resembled them either -- same root cause as the original bug, just a
different pair of phrasings nobody had added examples for yet.

Added 20 more Tier A examples (2 per crop × the same 10 crops as the
spacing/seed-rate additions: "When should I plant X?" / "Which X variety
should I plant?"), taking the gold set to 240 questions. All five
originally-failing planting-time/variety queries now correctly route to
A, including one ("When is the best time to harvest yam?") that wasn't
even an explicit training example -- reasonable generalization from a
better-shaped centroid rather than rote memorization of the added
phrasings.

Also fixed while investigating: two of the eval scripts
(`inspect_d_errors.py`, `ood_stress_test.py`) were still using a
standalone `sentence-transformers` reference embedder instead of the real
`RetrievalEngine.embed_query()` production path that `build_router.py`
and `build_centroids.py` had already switched to -- meaning their output
no longer matched what's actually shipped. Both now use the real embedder
too, so every script in this directory measures the same thing production
does.

## Embedder: real production path, not a standalone reference model

`centroids.json` and every script in this directory embed questions via
`RetrievalEngine.embed_query()` (quantized ONNX bge-small, the same
embedder retrieval uses), not a separately-loaded `sentence-transformers`
model. This matters more than it sounds: full-precision and quantized
embeddings of the same text aren't identical, and that ~0.97-0.99
cosine-similarity drift was large enough on its own to occasionally push
a question's confidence across the 0.74 threshold in either direction.
Calibrating against one embedder and deploying against another is a real
way to silently ship a miscalibrated threshold -- worth remembering if
either the embedding model or its quantization ever changes.
`engine/router/standalone.py` still uses plain `sentence-transformers` by
design, but only for casual local testing, never for calibration.

## Method

Compared per-tier centroid classification against k-NN (k=1, 3, 5) via
5-fold stratified cross-validation on the 240-question gold set
(`gold_questions_200_pest_diagnosis.jsonl` -- filename kept for
continuity despite no longer being pest-diagnosis-only), using the real
production embedder described above.

| Method | Overall accuracy | A recall | B recall | C recall | D recall |
|---|---|---|---|---|---|
| k-NN (k=1) | 94.6% | 97.5% | 96.1% | 100% | 58.8% |
| k-NN (k=3) | 94.6% | 98.7% | 96.1% | 100% | 52.9% |
| k-NN (k=5) | 93.3% | 96.2% | 97.1% | 100% | 41.2% |
| **Centroid** | 92.5% | 94.9% | 92.2% | 100% | **64.7%** |

**k-NN (k=1/k=3) now edges out centroid on overall accuracy** (94.6% vs
92.5%) at this gold-set size -- but `engine/router/router.py` only
implements centroid comparison, so centroid's number is what actually
describes the shipped classifier, not whichever method currently tops
this table. Centroid still wins by a wide margin on Tier D specifically
(64.7% vs 41-59%), which is the tier a wrong answer costs the most on --
that's the number that decided the original method choice and it still
holds. If the k-NN edge looks worth chasing later, `Router` needs a k-NN
mode added first; swapping the method without updating the shipped code
would silently ship numbers nobody's classifier actually produces.

D recall (64.7%) is unchanged from the 220-question relabel -- the 20 new
examples target Tier A, not D, so this wasn't expected to move. See
"Known limitation" below for what's actually still misclassified.

## Confidence threshold: 0.74

Chosen by sweeping thresholds against cross-validated confidence scores
(full sweep in `calibration_report.json`):

- Catches 61% of the centroid classifier's own cross-validation errors, at
  a 9.5% false-refusal cost on questions it would have answered correctly.
- Validated against 15 synthetic out-of-domain questions (sports scores,
  recipes, general trivia, a prompt-injection attempt) — **100% correctly
  refused** (max confidence among them was 0.67, comfortably under 0.74).

Going higher (e.g. 0.82) catches nearly every CV error but starts costing
a lot more correct classifications (~53%); going lower misses more
errors. 0.74 remains a reasonable default given the tradeoff curve, not a
law of nature — re-run `build_router.py` and revisit if real usage data
suggests otherwise.

## Known limitation: Tier D is the weakest tier (64.7% recall)

Six D-tier questions get misclassified in cross-validation (down from
seven at 220 questions -- one, the market-price question, now classifies
correctly, likely a side effect of the sharper Tier A centroid):

| id | true tier | misrouted to | confidence | caught by threshold=0.74? |
|---|---|---|---|---|
| D026 (cocoa trees per hectare) | D | A | 0.72 | yes |
| D004 (goat won't eat) | D | B | 0.67 | yes |
| D024 (plantain planting) | D | A | 0.81 | **no** |
| D023 (onion storage, general) | D | A | 0.71 | yes |
| D011 (child pesticide poisoning) | D | B | 0.73 | yes |
| D005 (fever after spraying) | D | B | 0.73 | yes |

D004/D005/D011 (medical/veterinary emergencies phrased as symptom
descriptions) are all still caught by the confidence threshold regardless
of the argmax tier being wrong, and independently caught by
`SAFETY_KEYWORDS` in `router.py` even if the threshold weren't there —
defense in depth, not a replacement for the embedding classifier.

D023/D024/D026 are the borderline out-of-corpus-crop questions (onion,
plantain, cocoa aren't in the corpus's 10-crop list) — phrased exactly
like legitimate Tier A questions, because they effectively are, just
about a crop this corpus doesn't cover. The router classifies by
question *style*, not by corpus *coverage*; that distinction is
retrieval's job (low retrieval confidence → refuse), not the router's.
D024's confidence actually got worse after the planting-time examples
were added (0.77 → 0.81) — the broader, better-shaped A-tier "planting
timing" centroid also pulled the plantain-planting question closer,
since it's phrased identically to the maize/rice/etc. examples that now
anchor that centroid more strongly. Expected tradeoff, not a bug: fixing
the false-refusal on 10 real corpus crops was worth one out-of-corpus
crop drifting further from the threshold. The downstream retrieval step
should still fail to find plantain-specific content and refuse there —
worth confirming once retrieval-confidence-based refusal exists.

## Reproducing

```
pip install sentence-transformers "optimum[onnxruntime]"   # sentence-transformers for standalone.py only;
                                                             # optimum/onnxruntime for the real embed_query() path
python eval/router/build_router.py       # cross-validation + threshold sweep -> calibration_report.json
python eval/router/inspect_d_errors.py   # which D-tier questions get misrouted, and at what confidence
python eval/router/ood_stress_test.py    # out-of-domain stress test
python eval/router/build_centroids.py    # regenerates engine/router/centroids.json from all 240 questions
python -m engine.router.standalone       # quick end-to-end demo (sentence-transformers, local testing only)
```

All scripts except `standalone.py` need `corpus/models/bge-small-en-v1.5-onnx/model_quantized.onnx`
to exist (run `corpus/06_embed.py` first, or export+quantize it directly --
see that script's `get_quantized_model_and_tokenizer()`).

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
