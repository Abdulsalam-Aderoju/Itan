"""Bounded agent loop dispatcher (blueprint Sec 5.2, Tier C path Sec 3.3).

Orchestrates the routing, parameter extraction, validation gate, deterministic execution,
and final phrasing of agricultural questions.
"""
import os
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence
import requests
import numpy as np

import sqlite3
from pathlib import Path
from engine.router.router import Router
from engine.retrieval import RetrievalEngine
from engine.language import normalise_query
from engine.tools.agri_calc import (
    fertiliser_rate,
    seed_rate,
    spray_dilution,
    gross_margin,
    CostItem,
    AgriCalcError,
    InvalidInputError,
    MissingConstantError,
)

# Constant Qwen 2.5-1.5B Tool Extraction System Prompt
SYSTEM_TOOL_PROMPT = (
    "You are an agricultural assistant. Your task is to extract parameters from a farmer's question "
    "and format them into a JSON tool call. You must output ONLY valid JSON, with no markdown fences, "
    "no explanation, and no extra text. Do not make up any numbers; extract them exactly as provided.\n\n"
    "The available tools are:\n\n"
    "1. fertiliser_rate(crop: str, area_ha: float, zone: str, soil_class: str = None, target_yield: float = None)\n"
    "   Calculates fertiliser requirements. crop must be one of: maize, cassava, rice, yam, cowpea, tomato.\n"
    "   zone is the agro-ecological zone (e.g. Northern Guinea Savanna, Sudan Savanna, Southern Guinea Savanna).\n\n"
    "2. seed_rate(crop: str, area_ha: float, spacing_cm: list[float, float], germination_pct: float, seeds_per_stand: int = 1)\n"
    "   Calculates required seed weight or cutting count. spacing_cm is a list of [row_cm, within_row_cm].\n"
    "   germination_pct is a percentage between 0 and 100.\n\n"
    "3. spray_dilution(product_name: str, tank_litres: float, rate_per_ha: float = None, spray_volume_l_per_ha: float = 200.0, crop: str = None)\n"
    "   Calculates chemical mixing per tank. product_name is the agrochemical name.\n\n"
    "4. gross_margin(yield_kg: float, price_per_kg: float, input_costs: list[dict])\n"
    "   Calculates financial returns. input_costs is a list of objects like: {\"label\": str, \"amount\": float}.\n\n"
    "Output JSON format:\n"
    "{\n"
    "  \"function\": \"function_name\",\n"
    "  \"params\": {\"param1\": value1, ...}\n"
    "}\n\n"
    "If required parameters are missing or the question cannot be answered by these tools, "
    "output a JSON object with a clarifying question or error, like:\n"
    "{\n"
    "  \"error\": \"I need to know the spacing and germination rate to calculate the seed rate. Could you provide those?\"\n"
    "}"
)

# System prompt for phrasing results to the user
SYSTEM_PHRASING_PROMPT = (
    "You are a helpful agricultural extension officer speaking to a farmer. "
    "You are given a query and the exact, mathematically correct results of a calculation. "
    "Your job is to phrase these results clearly and concisely. "
    "Treat the calculation numbers as absolute facts. Do NOT recalculate, modify, or estimate the numbers. "
    "Cite the source IDs provided in the calculation results in your answer (e.g., 'according to naerls_maize_2021_p14')."
)

# System prompt for refusing Tier D/out-of-scope questions
SYSTEM_REFUSAL_PROMPT = (
    "You are a helpful agricultural extension officer speaking to a farmer. "
    "The farmer has asked an out-of-scope question (such as human health, animal/livestock medicine, loans, elections, or general trivia) "
    "or the database does not contain sufficient information to answer it.\n\n"
    "Politely refuse to answer, explain that the topic is outside the scope of your system, "
    "and recommend that they consult a human agricultural extension officer or relevant local expert."
)

@dataclass(frozen=True)
class Response:
    answer_text: str
    tier: str
    source_ids: list[str]
    refused: bool

# Initialize Router once at module level to avoid re-reading centroids.json
_router = Router()
_retrieval_engine: RetrievalEngine | None = None

def get_retrieval_engine() -> RetrievalEngine:
    global _retrieval_engine
    if _retrieval_engine is None:
        _retrieval_engine = RetrievalEngine()
    return _retrieval_engine

def query_structured_db(question: str) -> tuple[str, list[str]]:
    """Query structured DBs (agri_calc.db and corpus/structured.db) for exact Tier A facts."""
    agri_db = Path(__file__).resolve().parent / "tools" / "agri_calc" / "agri_calc.db"
    corpus_db = Path(__file__).resolve().parent.parent / "corpus" / "structured.db"
    
    q_lower = question.lower()
    crops = ["maize", "cassava", "rice", "yam", "cowpea", "tomato", "sorghum", "groundnut", "pepper", "soybean"]
    target_crop = next((c for c in crops if c in q_lower), "maize")

    facts = []
    source_ids = []

    # 1. Query agri_calc.db (clean hand-populated tables)
    if agri_db.exists():
        try:
            conn = sqlite3.connect(agri_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Spacing
            rows = cursor.execute("SELECT * FROM spacing WHERE crop = ?", (target_crop,)).fetchall()
            for r in rows:
                facts.append(f"Spacing for {r['crop']} ({r['zone']}): {r['row_cm']} cm row x {r['within_row_cm']} cm plant. Source: {r['source_id']}")
                if r['source_id']:
                    source_ids.append(str(r['source_id']))

            # Fertilizer rate
            rows = cursor.execute("SELECT * FROM fertilizer_rate WHERE crop = ?", (target_crop,)).fetchall()
            for r in rows:
                facts.append(f"Fertilizer rate for {r['crop']} ({r['zone']}): N={r['n_rate_kg_ha']} kg/ha, P2O5={r['p2o5_rate_kg_ha']} kg/ha, K2O={r['k2o_rate_kg_ha']} kg/ha. Source: {r['source_id']}")
                if r['source_id']:
                    source_ids.append(str(r['source_id']))

            conn.close()
        except Exception:
            logging.getLogger("itan.agent").exception(
                "query_structured_db: agri_calc.db lookup failed for crop=%s", target_crop
            )

    # 2. Query corpus/structured.db (pests, varieties, calendars)
    if corpus_db.exists():
        try:
            conn = sqlite3.connect(corpus_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            rows = cursor.execute("SELECT * FROM pest WHERE crop = ? AND symptoms != '' LIMIT 3", (target_crop,)).fetchall()
            for r in rows:
                p_name = r['pest_name'] or 'Pest'
                control = r['chemical_control'] or r['cultural_control'] or 'Standard management'
                facts.append(f"{p_name} on {r['crop']}: {r['symptoms']}. Control: {control}. Source: {r['source_id']}")
                if r['source_id']:
                    source_ids.append(str(r['source_id']))

            conn.close()
        except Exception:
            logging.getLogger("itan.agent").exception(
                "query_structured_db: corpus/structured.db lookup failed for crop=%s", target_crop
            )

    if not facts:
        # Honest miss, not a fabricated fact: the caller (Tier A prompt) must
        # be told plainly that no structured data was found rather than being
        # handed an invented "default spacing" fact with a fake source_id --
        # that pattern silently produced a wrong, falsely-cited answer for
        # ANY unmatched question (e.g. a fertiliser question about a crop
        # with no fertilizer_rate row got told a fabricated *spacing* fact).
        logging.getLogger("itan.agent").warning(
            "query_structured_db: no structured facts found for crop=%s, question=%r", target_crop, question
        )
        facts.append(f"No structured database entry found for '{target_crop}' matching this question.")

    unique_sources = list(dict.fromkeys(source_ids))
    return "\n".join(facts), unique_sources

def call_llm(messages: list[dict], max_tokens: int = 256, temperature: float = 0.1) -> str:
    """Call the running llama-server instance."""
    url = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
    payload = {
        "model": "qwen2.5-1.5b-instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        # 60s was too short on dev hardware: a full ~1400-token RAG context
        # (blueprint's own cap) plus a 256-300 token generation measures out
        # to ~80s at this machine's roofline (~29 tok/s prefill, ~9 tok/s
        # decode -- see eval/benchmarks/submission_profiler_output.json),
        # so most Tier B/C calls were hitting the client timeout before the
        # server even finished, not because anything was hung. Confirmed via
        # eval/results/itan_system_eval_results.jsonl: 5 of 8 questions in
        # the last real run errored out this way. Target ADTC hardware
        # (10th-12th gen i5) should be meaningfully faster per the
        # blueprint's own roofline analysis, so this is a dev-hardware
        # accommodation, not evidence the real submission needs 180s -- but
        # until lookup decoding / phrase-bank speedups (blueprint Sec 8.2)
        # are implemented, this is what makes eval runs on this hardware
        # produce real answers instead of timeout errors.
        timeout_s = int(os.getenv("LLAMA_CLIENT_TIMEOUT_S", "180"))
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return json.dumps({"error": f"Failed to communicate with LLM: {str(e)}"})

def parse_json_safely(text: str) -> dict | None:
    """Strip markdown blocks and parse JSON safely."""
    text = text.strip()
    if text.startswith("```"):
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Search for curly brace substring
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

def validate_and_execute_tool(func_name: str, params: dict) -> tuple[dict, list[str]] | str:
    """Validate parameters against target signatures and execute the tool.
    Returns either (result_dict, source_ids) or a string error/clarification message.
    """
    valid_crops = {"maize", "cassava", "rice", "yam", "cowpea", "tomato"}

    if func_name == "fertiliser_rate":
        crop = params.get("crop")
        area_ha = params.get("area_ha")
        zone = params.get("zone")

        if not crop or not zone or area_ha is None:
            return "I need the crop, area in hectares, and the agro-ecological zone to calculate the fertilizer rate."
        
        crop_clean = str(crop).lower().strip()
        if crop_clean not in valid_crops:
            return f"The crop '{crop}' is not supported. Supported crops are: {', '.join(sorted(valid_crops))}."
        
        try:
            area_ha_val = float(area_ha)
            if area_ha_val <= 0:
                return "Area must be greater than zero."
        except ValueError:
            return "Area must be a valid number."

        try:
            res = fertiliser_rate(
                crop=crop_clean,
                area_ha=area_ha_val,
                zone=str(zone).strip(),
                soil_class=params.get("soil_class"),
                target_yield=params.get("target_yield"),
            )
            return res.to_dict(), res.source_ids
        except AgriCalcError as e:
            return str(e)

    elif func_name == "seed_rate":
        crop = params.get("crop")
        area_ha = params.get("area_ha")
        spacing_cm = params.get("spacing_cm")
        germination_pct = params.get("germination_pct")

        if not crop or area_ha is None or spacing_cm is None or germination_pct is None:
            return "I need the crop, area in hectares, row & within-row spacing in cm, and expected germination percentage."

        crop_clean = str(crop).lower().strip()
        if crop_clean not in valid_crops:
            return f"The crop '{crop}' is not supported. Supported crops are: {', '.join(sorted(valid_crops))}."

        try:
            area_ha_val = float(area_ha)
            if area_ha_val <= 0:
                return "Area must be greater than zero."
        except ValueError:
            return "Area must be a valid number."

        try:
            if not isinstance(spacing_cm, (list, tuple)) or len(spacing_cm) != 2:
                return "Spacing must be a list of two positive numbers (e.g. [75, 25])."
            spacing_cm_val = (float(spacing_cm[0]), float(spacing_cm[1]))
            if spacing_cm_val[0] <= 0 or spacing_cm_val[1] <= 0:
                return "Spacing values must be greater than zero."
        except (ValueError, TypeError):
            return "Spacing must contain valid numbers."

        try:
            germination_pct_val = float(germination_pct)
            if germination_pct_val <= 0 or germination_pct_val > 100:
                return "Germination percentage must be between 0 and 100."
        except ValueError:
            return "Germination percentage must be a valid number."

        seeds_per_stand = params.get("seeds_per_stand", 1)
        try:
            seeds_per_stand_val = int(seeds_per_stand)
        except (ValueError, TypeError):
            seeds_per_stand_val = 1

        try:
            res = seed_rate(
                crop=crop_clean,
                area_ha=area_ha_val,
                spacing_cm=spacing_cm_val,
                germination_pct=germination_pct_val,
                seeds_per_stand=seeds_per_stand_val,
            )
            return res.to_dict(), res.source_ids
        except AgriCalcError as e:
            return str(e)

    elif func_name == "spray_dilution":
        product_name = params.get("product_name")
        tank_litres = params.get("tank_litres")

        if not product_name or tank_litres is None:
            return "I need the product name and tank volume in litres to calculate the spray dilution."

        try:
            tank_litres_val = float(tank_litres)
            if tank_litres_val <= 0:
                return "Tank volume must be greater than zero."
        except ValueError:
            return "Tank volume must be a valid number."

        rate_per_ha = params.get("rate_per_ha")
        if rate_per_ha is not None:
            try:
                rate_per_ha = float(rate_per_ha)
                if rate_per_ha <= 0:
                    return "Chemical rate per hectare must be greater than zero."
            except ValueError:
                return "Rate per hectare must be a valid number."

        spray_volume_l_per_ha = params.get("spray_volume_l_per_ha", 200.0)
        try:
            spray_volume_l_per_ha_val = float(spray_volume_l_per_ha)
            if spray_volume_l_per_ha_val <= 0:
                return "Spray volume per hectare must be greater than zero."
        except ValueError:
            spray_volume_l_per_ha_val = 200.0

        try:
            res = spray_dilution(
                product_name=str(product_name).strip(),
                tank_litres=tank_litres_val,
                rate_per_ha=rate_per_ha,
                spray_volume_l_per_ha=spray_volume_l_per_ha_val,
                crop=params.get("crop"),
            )
            return res.to_dict(), res.source_ids
        except AgriCalcError as e:
            return str(e)

    elif func_name == "gross_margin":
        yield_kg = params.get("yield_kg")
        price_per_kg = params.get("price_per_kg")
        input_costs = params.get("input_costs")

        if yield_kg is None or price_per_kg is None or input_costs is None:
            return "I need the yield in kg, price per kg, and list of input costs to calculate the gross margin."

        try:
            yield_kg_val = float(yield_kg)
            if yield_kg_val < 0:
                return "Yield cannot be negative."
        except ValueError:
            return "Yield must be a valid number."

        try:
            price_per_kg_val = float(price_per_kg)
            if price_per_kg_val < 0:
                return "Price per kg cannot be negative."
        except ValueError:
            return "Price per kg must be a valid number."

        if not isinstance(input_costs, list):
            return "Input costs must be a list of objects with a label and amount."

        cost_items = []
        for item in input_costs:
            if not isinstance(item, dict) or "label" not in item or "amount" not in item:
                return "Each input cost must have a label and an amount."
            try:
                amt = float(item["amount"])
                if amt < 0:
                    return f"Cost amount for '{item['label']}' cannot be negative."
                cost_items.append(CostItem(label=str(item["label"]), amount=amt))
            except ValueError:
                return f"Cost amount for '{item['label']}' must be a valid number."

        try:
            res = gross_margin(
                yield_kg=yield_kg_val,
                price_per_kg=price_per_kg_val,
                input_costs=cost_items,
            )
            # GrossMarginResult doesn't have source_ids, but return empty list
            return res.to_dict(), []
        except AgriCalcError as e:
            return str(e)

    else:
        return f"Unknown function name '{func_name}'."

def answer_question(question: str, embedding: np.ndarray) -> Response:
    """Classify the question, execute matching logic paths, and return a Response."""
    # 0. Language Normalisation (Pillar 3)
    norm_q, lang = normalise_query(question)

    # 1. Routing step
    route_res = _router.classify(norm_q, embedding)

    if route_res.refused:
        # Tier D Refusal
        refusal_msg = call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_REFUSAL_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        return Response(answer_text=refusal_msg, tier="D", source_ids=[], refused=True)

    if route_res.tier == "A":
        # Tier A: Exact facts SQL lookup
        fact_text, source_ids = query_structured_db(question)
        system_prompt = (
            "You are an agricultural extension officer. Answer the farmer's question using ONLY the exact factual "
            "database facts provided below. Do NOT invent, recalculate, or alter any numbers. "
            "Cite the source IDs in your answer (e.g. 'according to naerls_maize_2021_p14').\n\n"
            f"Database Facts:\n{fact_text}"
        )
        answer = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=256,
            temperature=0.1,
        )
        return Response(answer_text=answer, tier="A", source_ids=source_ids, refused=False)

    elif route_res.tier == "B":
        # Tier B: Explanation & Diagnosis RAG Retrieval
        retrieval_engine = get_retrieval_engine()
        chunks = retrieval_engine.retrieve(question, top_k=4, query_vec=embedding)

        # Blueprint Sec 3.4: "Hard 1,400-token context cap -- prevents the RAM
        # and latency blow-ups that sink S_eff and S_perf." This was never
        # actually enforced: corpus chunking leaves a long tail of oversized
        # chunks (some 1000x the intended 300-450 token target), and without
        # a cap here, top-4 retrieval could hand the LLM a 5000+ token prompt
        # -- observed in practice as llama-server 400s ("exceeds the available
        # context size") and 60s client timeouts on ordinary questions. ~4
        # chars/token is a rough but safe proxy; we don't have the model's
        # tokenizer available cheaply in this process.
        MAX_CONTEXT_CHARS = 5600  # ~1400 tokens
        context_blocks = []
        source_ids = []
        budget = MAX_CONTEXT_CHARS
        for c in chunks:
            if budget <= 0:
                break
            cid = c.get("chunk_id", "")
            sid = c.get("source_id", "")
            text = c.get("text", "")
            if len(text) > budget:
                text = text[:budget].rsplit(" ", 1)[0] + " [...truncated]"
            context_blocks.append(f"--- Document Chunk [{sid or cid}] ---\n{text}")
            if sid:
                source_ids.append(sid)
            budget -= len(text)

        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant document passages found."
        unique_sources = list(dict.fromkeys(source_ids))

        system_prompt = (
            "You are an agricultural extension officer. Answer the farmer's question using the reference text passages below. "
            "Be clear, helpful, and concise. Cite the source IDs in brackets (e.g. [naerls_maize_2021_p14]). "
            "If the provided documents do not contain enough information to answer the question, state that clearly.\n\n"
            f"Reference Passages:\n{context_str}"
        )
        answer = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return Response(answer_text=answer, tier="B", source_ids=unique_sources, refused=False)

    elif route_res.tier == "C":
        # Tier C: Multi-step calculated (Tool Call Loop)
        messages = [
            {"role": "system", "content": SYSTEM_TOOL_PROMPT},
            {"role": "user", "content": question},
        ]
        
        # Bounded loop: cap tool-call rounds at max 2
        last_sources = []
        for round_idx in range(1, 3):
            llm_output = call_llm(messages, max_tokens=256, temperature=0.1)
            parsed_call = parse_json_safely(llm_output)

            # If parser fails or LLM outputs normal text/clarifying error
            if not parsed_call or "function" not in parsed_call:
                # If the LLM returned a structured error/clarification JSON
                if parsed_call and "error" in parsed_call:
                    return Response(
                        answer_text=parsed_call["error"],
                        tier="C",
                        source_ids=[],
                        refused=False
                    )
                # Treat as a direct answer from the model (e.g. clarification)
                return Response(
                    answer_text=llm_output,
                    tier="C",
                    source_ids=[],
                    refused=False
                )

            func_name = parsed_call["function"]
            params = parsed_call.get("params", {})

            # Validation Gate
            exec_res = validate_and_execute_tool(func_name, params)
            if isinstance(exec_res, str):
                # Validation failed or error raised - return as clarification
                return Response(
                    answer_text=exec_res,
                    tier="C",
                    source_ids=[],
                    refused=False,
                )

            # Execution succeeded
            result_dict, source_ids = exec_res
            last_sources = source_ids

            # Feed observation back into LLM
            messages.append({"role": "assistant", "content": llm_output})
            messages.append(
                {"role": "user", "content": f"Observation from {func_name}: {json.dumps(result_dict)}"}
            )

        # Force a final phrasing round after max 2 tool calls
        messages.append({
            "role": "system",
            "content": SYSTEM_PHRASING_PROMPT
        })
        final_answer = call_llm(messages, max_tokens=256, temperature=0.2)
        return Response(
            answer_text=final_answer,
            tier="C",
            source_ids=last_sources,
            refused=False,
        )

    # Fallback default
    return Response(
        answer_text="I am sorry, I am unable to process that question.",
        tier="D",
        source_ids=[],
        refused=True,
    )
