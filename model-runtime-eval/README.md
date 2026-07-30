# model-runtime-eval — model runtime setup + evaluation baseline

Contents from getting the raw model running locally and establishing the
"embarrassing baseline" (Section 9.1 of the blueprint: raw model, no RAG,
no tools) plus running the official ADTC profiler.

## Files

- `metadata.json` — filled-in submission metadata. Test prompts and the
  `cross_disciplinary_pairing` description are scoped to the **pest &
  disease diagnosis niche** (see below) — `team_id` and `github_handle`
  are still placeholders pending DevPost registration.
- `download_model.sh` — downloads Qwen2.5-1.5B-Instruct Q4_K_M (the
  current baseline model) from the official Qwen GGUF repo on Hugging Face.
- `submission_profiler_output.json` — real output from the **official**
  `adtc-profiler` tool (installed from
  github.com/Africa-Deep-Tech-Foundation/adtc-profiler), run with
  `--skip-accuracy`. Measured on this dev machine (an older Haswell-class
  CPU, weaker than ADTC's stated 10th-12th gen reference spec, so treat
  these as a conservative lower bound):
  - throughput: 8.73 tok/s → S_perf = 58.2/100
  - peak RAM: 1,720 MB → S_eff = 76.0/100
  - not throttled
  - accuracy: not run — see note below on why the profiler's accuracy
    path is currently broken.
- `eval/gold_questions_200_pest_diagnosis.jsonl` — 200-question gold set,
  re-scoped around **pest/disease diagnosis** rather than general
  agricultural advice (30 Tier-A exact facts, 100 Tier-B diagnosis
  questions incl. Pidgin/Hausa/Yorùbá symptom descriptions, 40 Tier-C
  diagnosis+calculation, 30 Tier-D refusals — including refusals for
  now-out-of-scope categories like planting calendars/variety selection).
- `eval/run_baseline.py` — runs a question set against a local
  llama-server instance (OpenAI-compatible `/v1/chat/completions`) and
  records the raw answer + latency/tokens per question.
- `eval/baseline_results_raw_model_no_rag.jsonl` — the 200 gold questions
  run through the bare Qwen2.5-1.5B model, **no retrieval, no tools, no
  fine-tune**. This is the reference point every later improvement should
  be measured against. Notable failure: the flagship fall-armyworm test
  prompt gets misdiagnosed as a fungal infection with two fabricated
  product names — good evidence for the report's ablation section.

## Why pest/disease diagnosis, not general advisory

The corpus pipeline (in progress, not yet in this repo) is strong on pest
management (6,100+ pest rows, ~70% retrieval hit-rate) but weak on
planting calendars, variety catalogues, and zone-specific content.
Rather than build for the corpus we wish we had, this narrows the product
to what it can already do well — matching the organizers' explicit
"pick a niche and go deep" guidance.

## Known issue: official profiler's accuracy path is currently broken

`adtc-profiler run` (without `--skip-accuracy`) crashes — it passes
`pretrained=<path>` to lm-eval-harness's `gguf` backend, which actually
requires `base_url=<running llama-server>`. Even fixed, lm-eval's `gguf`
adapter expects a legacy logprobs shape that current `llama-server`
doesn't return. Flagged to organizers on Discord — not our repo's bug.

## Not included here (see .gitignore)

The downloaded `.gguf` model weights and the llama.cpp runtime binaries
are not committed — too large, and the submission rules require weights
to be fetched fresh via `download_model.sh`, not stored in git.
