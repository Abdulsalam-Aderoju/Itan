"""Orchestration layer for Ìtàn (blueprint Sec 3.1).

Sits between the UI and the engine: normalises + embeds the incoming question,
runs it through the 4-tier router/agent loop in engine.agent, and returns a
structured, cited response. Binds to 127.0.0.1 only -- no external exposure,
matching the "zero egress" design decision in the blueprint.

This is NOT the model server. Qwen2.5-1.5B runs behind llama.cpp's own
OpenAI-compatible /v1/chat/completions endpoint (LLAMA_SERVER_URL, default
http://127.0.0.1:8080); engine.agent.call_llm() talks to that directly. This
FastAPI app is the layer above it that a UI actually talks to.

Run:
    uvicorn engine.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine.agent import answer_question
from engine.retrieval import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("itan.server")

app = FastAPI(title="Ìtàn Orchestration Layer", version="0.1.0")

_retrieval_engine: RetrievalEngine | None = None


def get_retrieval_engine() -> RetrievalEngine:
    global _retrieval_engine
    if _retrieval_engine is None:
        logger.info("Loading retrieval engine (corpus indices + embedder)...")
        _retrieval_engine = RetrievalEngine()
    return _retrieval_engine


@app.on_event("startup")
def warm_up() -> None:
    # Pre-load corpus indices + embedder at startup, not on first request,
    # so latency budgets (blueprint Sec 8.4) are measured on a warm system.
    get_retrieval_engine()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    tier: str
    refused: bool
    source_ids: list[str]
    latency_s: float


@app.get("/health")
def health() -> dict:
    engine_ready = _retrieval_engine is not None and _retrieval_engine.chunks_df is not None
    return {
        "status": "ok",
        "retrieval_index_loaded": engine_ready,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.time()
    engine = get_retrieval_engine()

    try:
        query_vec = engine.embed_query(req.question)
    except Exception as exc:
        logger.exception("Embedding failed -- cannot route or retrieve without it")
        return JSONResponse(
            status_code=503,
            content={"error": f"embedder unavailable: {exc}"},
        )

    response = answer_question(req.question, query_vec)
    elapsed = time.time() - t0

    return QueryResponse(
        answer=response.answer_text,
        tier=response.tier,
        refused=response.refused,
        source_ids=response.source_ids,
        latency_s=round(elapsed, 3),
    )


@app.exception_handler(Exception)
def unhandled_exception_handler(request, exc):  # noqa: ANN001, ARG001
    logger.exception("Unhandled error while serving %s", request.url)
    return JSONResponse(status_code=500, content={"error": str(exc)})
