"""Unified Benchmark Evaluation Runner for Ìtàn (Blueprint Section 9.1).

Runs the 200 Gold Questions through the complete grounded Ìtàn pipeline:
Language Normalisation (Pillar 3) -> 4-Tier Router -> Retrieval / SQL / Tools -> LLM Phrasing + Citations.

This harness REQUIRES a live llama-server (see model/README.md + engine/agent.py's
LLAMA_SERVER_URL). It will not fabricate answers if the server is unreachable --
earlier versions of this script silently fell back to a hardcoded mock response
that got returned for most questions regardless of content, which produced a
results file that looked like a real benchmark but wasn't one. If the server is
down, this script now refuses to run rather than repeat that mistake.

The accuracy figure this script reports is an internal development proxy
(token-overlap F1 against the hand-written gold answers), NOT the official ADTC
S_acc -- that is judge-panel scored against participant/domain/hidden prompts
per the ADTC profiler and submission template. Use this number to catch
obviously wrong or hallucinated answers during development, not as a
leaderboard prediction.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --limit 20        # smoke test on a subset
    python eval/run_eval.py --server-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import engine.agent as agent_mod
from engine.agent import answer_question, Response
from engine.language import normalise_query
from engine.retrieval import RetrievalEngine

GOLD_PATH = REPO_ROOT / "eval" / "datasets" / "gold_questions_200_pest_diagnosis.jsonl"
OUT_PATH = REPO_ROOT / "eval" / "results" / "itan_system_eval_results.jsonl"

# Reference-answer scoring is a proxy metric only; this stopword list keeps it
# from rewarding overlap on filler words instead of actual agronomic content.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "and", "or", "with", "as", "by", "it",
    "this", "that", "these", "those", "i", "you", "your", "should", "would",
    "will", "can", "could", "and/or", "into", "from", "how", "what", "when",
    "consult", "recommend", "recommended", "use", "apply", "per",
}


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def score_accuracy_proxy(answer_text: str, gold_answer: str) -> float:
    """Token-overlap F1 between a generated answer and the gold reference answer,
    scaled 0-100. This is a cheap, deterministic, dependency-free DEVELOPMENT
    PROXY for catching hallucinated/off-topic answers -- it is not the ADTC's
    actual accuracy score (judge-panel assessed, per adtc-profiler README)."""
    ans_tokens = _content_tokens(answer_text or "")
    gold_tokens = _content_tokens(gold_answer or "")
    if not ans_tokens or not gold_tokens:
        return 0.0
    overlap = len(ans_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ans_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1 * 100, 2)


def check_llama_server(url: str) -> bool:
    """Verify llama-server is actually reachable and answering before running
    200 queries against it. Fail fast and loud instead of quietly degrading."""
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "qwen2.5-1.5b-instruct",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4,
            },
            timeout=10,
        )
        resp.raise_for_status()
        resp.json()["choices"][0]["message"]["content"]
        return True
    except Exception as exc:
        print(f"[FATAL] llama-server at {url} is not reachable or not responding correctly: {exc}")
        return False


def run_evaluation(limit: int | None = None, server_url: str | None = None):
    print("=" * 80)
    print("      ÌTÀN UNIFIED SYSTEM BENCHMARK EVALUATION (200 GOLD QUESTIONS)")
    print("=" * 80)

    if not GOLD_PATH.exists():
        print(f"[FATAL] Gold question dataset not found at {GOLD_PATH}")
        sys.exit(1)

    import os
    url = server_url or os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
    os.environ["LLAMA_SERVER_URL"] = url

    print(f"\n[0/3] Checking llama-server at {url} ...")
    if not check_llama_server(url):
        print(
            "\n[FATAL] This harness requires a live, responding llama-server. "
            "Start one first, e.g.:\n"
            "    llama-server -m model/qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080\n"
            "Refusing to run -- an offline/mocked run would silently produce a results "
            "file that looks like real benchmark data but isn't."
        )
        sys.exit(1)
    print("[0/3] llama-server is live and responding.")

    print("\n[1/3] Loading Retrieval Engine & Vector Embedder ...")
    retrieval_engine = RetrievalEngine()

    print(f"[2/3] Loading 200 Gold Questions from {GOLD_PATH.name} ...")
    questions = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if limit:
        questions = questions[:limit]

    print(f"[3/3] Running {len(questions)} evaluation queries ...\n")

    results = []
    tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    lang_counts = {"en": 0, "ha": 0, "yo": 0, "pcm": 0, "ig": 0}
    cited_count = 0
    embed_failures = 0
    accuracy_scores: list[float] = []
    accuracy_by_tier: dict[str, list[float]] = {"A": [], "B": [], "C": [], "D": []}
    t0 = time.time()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q_item in enumerate(questions, 1):
            q_id = q_item.get("id", f"Q{i:03d}")
            q_text = q_item.get("question", "")
            gold_answer = q_item.get("answer", "")

            # 1. Embed query -- a failure here is fatal for this question, not
            # something to paper over with a fabricated tier/answer.
            try:
                query_vec = retrieval_engine.embed_query(q_text)
            except Exception as exc:
                embed_failures += 1
                print(f"  [WARN] {q_id}: embedding failed ({exc}); skipping question")
                record = {
                    "id": q_id, "question": q_text, "error": f"embedding failed: {exc}",
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # 2. Run agent pipeline
            t_start = time.time()
            res: Response = answer_question(q_text, query_vec)
            elapsed = time.time() - t_start

            norm_q, lang = normalise_query(q_text)
            tier_counts[res.tier] = tier_counts.get(res.tier, 0) + 1
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            if res.source_ids:
                cited_count += 1

            acc_score = score_accuracy_proxy(res.answer_text, gold_answer)
            accuracy_scores.append(acc_score)
            accuracy_by_tier.setdefault(res.tier, []).append(acc_score)

            record = {
                "id": q_id,
                "question": q_text,
                "normalised_query": norm_q,
                "detected_language": lang,
                "gold_tier": q_item.get("tier"),
                "routed_tier": res.tier,
                "refused": res.refused,
                "source_ids": res.source_ids,
                "elapsed_s": round(elapsed, 3),
                "answer_text": res.answer_text,
                "gold_answer": gold_answer,
                "proxy_accuracy_score": acc_score,
            }
            results.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i % 25 == 0 or i == len(questions):
                print(f"  Processed [{i}/{len(questions)}] questions... (Last: {q_id} Tier={res.tier} Lang={lang} ProxyAcc={acc_score:.0f})")

    total_time = time.time() - t0
    n_scored = len(questions) - embed_failures
    avg_latency = total_time / n_scored if n_scored else 0.0
    citation_rate = (cited_count / n_scored) * 100 if n_scored else 0.0
    mean_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0

    print("\n" + "=" * 80)
    print("                         SYSTEM EVALUATION SCORECARD")
    print("=" * 80)
    print(f"Total Questions            : {len(questions)}")
    print(f"Embedding Failures (skipped): {embed_failures}")
    print(f"Total Elapsed Time         : {total_time:.2f}s (Avg: {avg_latency:.3f}s per query)")
    print(f"Citation Rate              : {citation_rate:.1f}% ({cited_count}/{n_scored} responses cited sources)")
    print(f"\n[Proxy Accuracy -- DEVELOPMENT METRIC ONLY, not official S_acc]")
    print(f"  Mean token-overlap F1 vs gold answers: {mean_accuracy:.1f} / 100")
    for t in ["A", "B", "C", "D"]:
        scores = accuracy_by_tier.get(t, [])
        if scores:
            print(f"  Tier {t}: {sum(scores) / len(scores):5.1f} / 100  (n={len(scores)})")

    print("\n[Tier Distribution (as routed)]")
    for t in ["A", "B", "C", "D"]:
        count = tier_counts.get(t, 0)
        pct = (count / n_scored) * 100 if n_scored else 0
        print(f"  Tier {t}: {count:3d} ({pct:5.1f}%)")

    print("\n[Language Detection Distribution (Pillar 3)]")
    for l, count in lang_counts.items():
        if count > 0:
            pct = (count / n_scored) * 100
            print(f"  Language '{l}': {count:3d} ({pct:5.1f}%)")

    print("=" * 80)
    print(f"Full results log written to: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N gold questions (smoke test)")
    parser.add_argument("--server-url", type=str, default=None, help="Override LLAMA_SERVER_URL")
    args = parser.parse_args()
    run_evaluation(limit=args.limit, server_url=args.server_url)
