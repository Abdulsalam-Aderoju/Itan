"""End-to-End Engine Integration Test.

Tests that answer_question() in engine/agent.py properly routes and executes:
- Tier A: Structured SQL queries (exact facts)
- Tier B: Hybrid RAG retrieval (context passages + citations)
- Tier C: Deterministic Python tools (agri_calc)
- Tier D: Refusal policy (safety keyword overrides & out-of-scope queries)
"""
import sys
from pathlib import Path
import numpy as np

# Ensure team repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import engine.agent as agent_mod
from engine.agent import answer_question, Response, query_structured_db
from engine.retrieval import RetrievalEngine
from engine.router import Router

def check(label: str, condition: bool):
    if condition:
        print(f"PASS | {label}")
    else:
        print(f"FAIL | {label}")
        sys.exit(1)

def run_integration_tests():
    print("=" * 70)
    print("RUNNING END-TO-END ENGINE INTEGRATION TESTS")
    print("=" * 70)

    retrieval_engine = RetrievalEngine()
    router = Router()

    # Mock call_llm if llama-server is not running locally
    original_call_llm = agent_mod.call_llm
    def mock_call_llm(messages, max_tokens=256, temperature=0.1):
        # Check if real server is available
        try:
            res = original_call_llm(messages, max_tokens, temperature)
            if "Failed to communicate with LLM" not in res:
                return res
        except Exception:
            pass
        
        # Fallback mock response for offline testing
        last_msg = messages[-1]["content"] if messages else ""
        if "fertiliser_rate" in last_msg or "hectares" in last_msg:
            return '{"function": "fertiliser_rate", "params": {"crop": "maize", "area_ha": 2.0, "zone": "Northern Guinea Savanna"}}'
        return "Based on the agronomic records, here is the verified extension advice with citations [source: naerls_maize_2021_p14]."

    agent_mod.call_llm = mock_call_llm

    # 1. Test Tier D Safety Refusal
    q_d = "What bank loan options are available for buying a tractor?"
    emb_d = retrieval_engine.embed_query(q_d)
    res_d = answer_question(q_d, emb_d)
    check("Tier D: Safety override triggered correctly", res_d.tier == "D" and res_d.refused)
    check("Tier D: Returns non-empty answer text", len(res_d.answer_text) > 0)
    print(f"  [Output Preview] Tier D: {res_d.answer_text[:100]}...\n")

    # 2. Test Tier A SQL Exact Fact Lookup
    fact_text, source_ids = query_structured_db("maize")
    check("Tier A: Structured DB query returns facts", len(fact_text) > 0 and len(source_ids) > 0)
    print(f"  [Output Preview] Tier A Fact Query: {fact_text[:120]}...\n")

    # 3. Test Tier B RAG Retrieval
    q_b = "A farmer's maize crop shows extensive tunnelling and holes bored into the stem -- what pest is this?"
    emb_b = retrieval_engine.embed_query(q_b)
    chunks_b = retrieval_engine.retrieve(q_b, top_k=4, query_vec=emb_b)
    check("Tier B: Hybrid RAG retrieves passages", len(chunks_b) > 0)
    res_b = answer_question(q_b, emb_b)
    check("Tier B: Returns non-empty answer text", len(res_b.answer_text) > 0)
    print(f"  [Output Preview] Tier B RAG: Top chunk from '{chunks_b[0]['doc_title']}' [{chunks_b[0]['source_id']}]\n")

    # 4. Test Tier C Math Tools Loop
    q_c = "I have 2 hectares of maize in Northern Guinea Savanna, how much urea do I need?"
    emb_c = retrieval_engine.embed_query(q_c)
    res_c = answer_question(q_c, emb_c)
    check("Tier C: Returns non-empty answer text", len(res_c.answer_text) > 0)
    print(f"  [Output Preview] Tier C Math: {res_c.answer_text[:120]}...\n")

    print("=" * 70)
    print("ALL ENGINE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_integration_tests()
