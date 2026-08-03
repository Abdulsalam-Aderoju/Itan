# Technical Report — Ìtàn: An Offline Agricultural Extension Agent

**Team ID:** itan
**Domain:** agriculture
**Model:** Qwen2.5-1.5B-Instruct-Q4_K_M

---

## Problem

Ìtàn is a fully offline advisory agent that answers real farming questions — pest identification,
fertiliser dosing, planting calendars, spray dilution, margins — in English, Hausa, Yorùbá and
Nigerian Pidgin, with a citation on every fact, on a commodity Ubuntu laptop with zero network
dependency.

Sub-Saharan Africa runs roughly one extension officer per 1,000–3,000 farming households against
an FAO-recommended ratio near 1:400. Where advice does not reach farmers, the failures are
concrete and expensive: misidentified pests treated with the wrong pesticide, fertiliser at the
wrong rate or growth stage, planting windows missed by weeks. Cloud AI does not solve this —
connectivity is intermittent, per-query data cost is real against thin margins, and frontier
models are weak in Hausa, Yorùbá and agricultural Pidgin register. Ìtàn puts a verified, citable
knowledge base plus a language interface onto a machine that already exists in a local-government
agriculture office, a cooperative, or an agro-dealer's shop.

---

## Design Decisions

- **Base model:** Qwen2.5-1.5B-Instruct, selected over 7B/8B alternatives specifically because
  model size dominates the non-accuracy 50% of the ADTC score (S_perf, S_eff, thermal risk), and a
  well-built small-model RAG system on a domain-narrow task closes most of the accuracy gap to a
  7B model that a 7B cannot close back on speed/memory. Full comparison table in the project
  blueprint (`Itan_ADTC2026_Blueprint.pdf`, §3.4).
- **Quantization:** Q4_K_M — TODO once quantization sweep (Q4_K_M vs IQ4_XS vs Q3_K_M) is run.
- **Architecture:** retrieval-grounded (dense + BM25 hybrid, RRF fusion), tool-calling for all
  arithmetic (fertiliser dosing, seed rate, spray dilution, margins run as deterministic Python,
  not generated tokens), and a 4-tier query router (exact-fact SQL / RAG explanation / tool-call /
  refusal) that controls exactly when the model is allowed to see or invent a numeric value.
- **Alternatives considered:** see blueprint §3.4 model-selection table (Llama-3.1-8B,
  Mistral-7B, Qwen2.5-3B, Gemma-2-2B, SmolLM2-1.7B, Qwen2.5-0.5B — all rejected for either
  S_perf/S_eff cost or weaker tool-calling/multilingual coverage relative to Qwen2.5-1.5B).

---

## Constraints

- Target: 8 GB RAM, integrated GPU only, Ubuntu 22.04, no network at evaluation time.
- All inference via llama.cpp on CPU; all arithmetic via deterministic Python tools, never
  generated tokens.
- Corpus limited to public/openly-licensed African agricultural extension sources (FAO, IITA,
  CGIAR, CABI PlantWise, NAERLS, national extension bodies) — licence and URL logged per document.
- 5-hour GPU credit budget (Udutech) for any fine-tuning — constrains fine-tuning to LoRA, if
  attempted at all; fine-tune only ships if it beats the base model on the gold set.

---

## Benchmarks

TODO — to be filled in from the ADTC profiler's `submission.json` once a live model + server run
is available (Phase 1/2 in progress). Self-reported development numbers go here first; official
scores are measured by the ADTC profiler on the standard evaluation machine.

| Metric | Value |
|---|---|
| Machine | TODO |
| RAM at peak | TODO |
| Time to first token | TODO |
| Generation speed (TPS) | TODO |
| Thermal throttling | TODO |
| Retrieval Hit Rate@5 | 0.70 (50-question corpus validation set, see `corpus/validation_report.json`) |

---

## Evaluation Methodology

- **Retrieval quality:** validated against a hand-curated 50-question set per crop/topic
  (`corpus/08_validate.py` → `corpus/validation_report.json`).
- **Tool correctness:** 120 unit test cases against the four `agri_calc` functions
  (`eval/agri_calc/run_tests.py`).
- **Router calibration:** 5-fold cross-validated centroid classifier over 200 labeled questions,
  95% overall accuracy (`eval/router/calibration_report.json`).
- **End-to-end system accuracy:** TODO — internal proxy scoring against
  `eval/datasets/gold_questions_200_pest_diagnosis.jsonl` is being wired up
  (`eval/run_eval.py`). Note this is a development-time proxy only; the ADTC panel's own scoring
  for S_acc is based on responses to submitted, domain, and hidden judge prompts, not this
  self-authored gold set.
