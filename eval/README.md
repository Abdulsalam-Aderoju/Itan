# Evaluation & Benchmarking Harness (`eval/`)

Consolidated test suites, benchmark datasets, baseline runners, and competition profiler artifacts.

## Folder Layout

```
eval/
├── datasets/
│   ├── gold_questions_200_pest_diagnosis.jsonl  # 200 gold Q&A benchmark items across Tier A-D
│   └── known_issues_e2e_probes.jsonl            # real end-to-end model answers with known
│                                                  # problems (wrong species, hallucinated
│                                                  # citation, category error) -- see below
├── results/
│   └── baseline_results_raw_model_no_rag.jsonl   # Baseline raw model responses (no RAG/tools)
├── benchmarks/
│   └── submission_profiler_output.json          # Official adtc-profiler output log
├── agri_calc/                                    # Unit tests for the 4 Python math tools (120 cases)
├── router/                                       # Unit tests & calibration for the 4-tier Router
└── run_baseline.py                               # Runs gold questions through local LLM server
```

## Running Tests

- **Router Tests:** `python eval/router/test_router.py`
- **Calculator Tests:** `python eval/agri_calc/run_tests.py`
- **Baseline LLM QA:** `python eval/run_baseline.py`

## `known_issues_e2e_probes.jsonl`

Six real questions run end-to-end (full model + retrieval, not a retrieval-only smoke test) by
Abdulsalam Aderoju on 2026-08-18, against a 168,869-chunk corpus pull that is LARGER than this
repo's own local corpus (72,452 chunks as of the same date) -- corpus artifacts are gitignored
and pulled independently per-machine from the shared Drive folder, so there is currently no
single canonical corpus across the team. Re-check these against whichever corpus is authoritative
before trusting a "fixed" verdict.

Of the five answers actually captured (a sixth question was asked but its answer wasn't pasted):
two look plausibly correct (cassava whitefly, cowpea pod borer -- kept as regression checks, not
just failures), one has a right diagnosis undermined by a hallucinated citation not present in
its own `source_ids`, one names a New World pest for a Nigeria-context maize question, and one
names a fungal disease for symptoms that describe insect boring damage. Full reasoning per
question is in the JSONL's `issue_note` field -- not independently domain-verified by an
agricultural expert, treat the `issue_type` classifications as informed flags, not ground truth.

This file exists so that as retrieval/corpus/prompt changes land, these specific real failures
can be re-run and checked rather than re-litigated from scratch each time. It is not part of the
router/retrieval gold-question sets and is not scored by any existing harness -- there's no
runner for it yet (would need a live model server; see `eval/run_baseline.py` for the pattern to
follow).
