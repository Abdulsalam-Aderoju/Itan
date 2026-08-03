# Evaluation & Benchmarking Harness (`eval/`)

Consolidated test suites, benchmark datasets, baseline runners, and competition profiler artifacts.

## Folder Layout

```
eval/
├── datasets/
│   └── gold_questions_200_pest_diagnosis.jsonl  # 200 gold Q&A benchmark items across Tier A-D
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
