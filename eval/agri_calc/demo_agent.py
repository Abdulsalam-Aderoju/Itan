"""Demo script to run the dispatcher and trace execution.

Queries the sentence-transformers model to get the question embedding, then passes it
to the agent's entry point to demonstrate the end-to-end loop.
"""
import sys
import os
from pathlib import Path
import numpy as np

# Add team repo root to system path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Force connection to production DB
DB_FILE_PATH = REPO_ROOT / "engine" / "tools" / "agri_calc" / "agri_calc.db"
os.environ["AGRI_CALC_DB_PATH"] = str(DB_FILE_PATH)

from sentence_transformers import SentenceTransformer
from engine.agent import answer_question

def run_demo():
    print("Loading embedding model bge-small-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # 1. Tier C Question
    q_c = "how much urea for 0.6 ha maize in Northern Guinea Savanna?"
    print(f"\n=======================================================")
    print(f"TEST 1: TIER C Question: '{q_c}'")
    print(f"=======================================================")
    
    emb_c = model.encode(q_c, normalize_embeddings=True)
    res_c = answer_question(q_c, emb_c)
    
    print(f"Routed Tier: {res_c.tier}")
    print(f"Source IDs Cited: {res_c.source_ids}")
    print(f"Refused: {res_c.refused}")
    print(f"Phrased Answer:\n{res_c.answer_text}")
    
    # 2. Tier D Question (Refusal & Safety Keyword)
    q_d = "My goat has stopped eating, what medicine should I give it?"
    print(f"\n=======================================================")
    print(f"TEST 2: TIER D Question (Safety Override): '{q_d}'")
    print(f"=======================================================")
    
    emb_d = model.encode(q_d, normalize_embeddings=True)
    res_d = answer_question(q_d, emb_d)
    
    print(f"Routed Tier: {res_d.tier}")
    print(f"Source IDs Cited: {res_d.source_ids}")
    print(f"Refused: {res_d.refused}")
    print(f"Phrased Answer:\n{res_d.answer_text}")

    # 3. Parameter Validation Gate Check (Missing required parameter)
    q_val = "How much fertilizer split do I need for my maize?"
    print(f"\n=======================================================")
    print(f"TEST 3: Validation Gate (Missing area/zone): '{q_val}'")
    print(f"=======================================================")
    
    emb_val = model.encode(q_val, normalize_embeddings=True)
    res_val = answer_question(q_val, emb_val)
    
    print(f"Routed Tier: {res_val.tier}")
    print(f"Phrased Answer:\n{res_val.answer_text}")

if __name__ == "__main__":
    run_demo()
