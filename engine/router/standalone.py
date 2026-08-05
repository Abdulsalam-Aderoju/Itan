"""Convenience wrapper for local testing ONLY.

Loads bge-small-en-v1.5 directly via sentence-transformers so you can call
the router without wiring up the real retrieval pipeline's embedder yet.

In production, do NOT use this -- reuse the embedding already computed by
the retrieval layer instead (see router.py's module docstring for why).
"""
from __future__ import annotations

from pathlib import Path

from engine.router.router import Router, RouteResult

# Full-precision reference model for local testing. Production uses a
# quantized ONNX export of the same base model via RetrievalEngine.embed_query()
# (see corpus/06_embed.py) -- centroids.json's "model" field now describes that
# production path in prose rather than naming a loadable HF repo ID, so this
# constant intentionally doesn't read from it.
_STANDALONE_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class StandaloneRouter:
    def __init__(self, centroids_path: Path | str | None = None):
        from sentence_transformers import SentenceTransformer  # local import: heavy, optional dep

        kwargs = {}
        if centroids_path is not None:
            kwargs["centroids_path"] = centroids_path
        self._router = Router(**kwargs)
        self._model = SentenceTransformer(_STANDALONE_MODEL_NAME)

    def classify(self, question_text: str) -> RouteResult:
        embedding = self._model.encode(question_text, normalize_embeddings=True)
        return self._router.classify(question_text, embedding)


if __name__ == "__main__":
    import sys

    router = StandaloneRouter()
    examples = sys.argv[1:] or [
        "My maize leaves have ragged holes and sawdust-like frass in the whorl.",
        "What is the pre-harvest interval for the product used against fall armyworm on maize?",
        "I have 2 hectares of maize with fall armyworm. How much product do I need?",
        "What loan options are available for buying a tractor?",
        "My goat has stopped eating, what medicine should I give it?",
        "What's the score of the Arsenal match last night?",
    ]
    for q in examples:
        result = router.classify(q)
        print(f"[{result.tier}] conf={result.confidence:.3f} reason={result.reason:<24} {q}")
