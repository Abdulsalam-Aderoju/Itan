"""
Phase 0 "embarrassing baseline" runner (proposal Section 9.1):
raw Qwen2.5-1.5B-Instruct, NO retrieval, NO tools, NO fine-tune —
just the bare model answering each gold question directly. Every later
RAG/tool/fine-tune improvement is measured as a delta from these numbers.

Prereqs:
    llama-server.exe already running locally, e.g.:
        runtime/llama-b10153-bin-win-cpu-x64/llama-server.exe \
            -m model/qwen2.5-1.5b-instruct-q4_k_m.gguf -c 4096 -t 3 \
            --host 127.0.0.1 --port 8080

Usage:
    python eval/run_baseline.py --questions eval/datasets/gold_questions_200_pest_diagnosis.jsonl --out eval/results/baseline_results_raw_model_no_rag.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

SYSTEM_PROMPT = (
    "You are a helpful assistant answering a farmer's question. "
    "Answer directly and concisely using your own knowledge."
)


def load_questions(path: Path) -> list[dict]:
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def call_model(base_url: str, question: str, max_tokens: int = 256, temperature: float = 0.2) -> dict:
    payload = {
        "model": "qwen2.5-1.5b-instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "answer": text,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(tps, 2),
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_questions = repo_root / "eval" / "datasets" / "gold_questions_200_pest_diagnosis.jsonl"
    default_out = repo_root / "eval" / "results" / "baseline_results_raw_model_no_rag.jsonl"

    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=default_questions)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="only run first N questions (for a quick smoke test)")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    print(f"Loaded {len(questions)} questions from {args.questions}")

    # health check
    try:
        requests.get(f"{args.base_url}/health", timeout=5)
    except Exception as e:
        print(f"WARNING: could not reach {args.base_url}/health ({e}). Is llama-server running?")

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions, 1):
            try:
                result = call_model(args.base_url, q["question"], max_tokens=args.max_tokens)
                record = {
                    "id": q.get("id"),
                    "tier": q.get("tier"),
                    "crop": q.get("crop"),
                    "question": q["question"],
                    "gold_answer": q.get("answer"),
                    "baseline_model_answer": result["answer"],
                    "elapsed_s": result["elapsed_s"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "error": None,
                }
            except Exception as e:
                record = {
                    "id": q.get("id"), "tier": q.get("tier"), "crop": q.get("crop"),
                    "question": q["question"], "gold_answer": q.get("answer"),
                    "baseline_model_answer": None, "elapsed_s": None,
                    "prompt_tokens": None, "completion_tokens": None,
                    "tokens_per_sec": None, "error": str(e),
                }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            status = "OK" if record["error"] is None else f"ERROR: {record['error']}"
            print(f"[{i}/{len(questions)}] {record['id']} ({record['tier']}) {status}"
                  + (f" — {record['tokens_per_sec']} tok/s, {record['elapsed_s']}s" if record['error'] is None else ""))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
