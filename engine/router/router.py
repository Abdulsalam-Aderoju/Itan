"""Tier A/B/C/D router + refusal policy (blueprint Sec 3.3, 5.4).

Classifies each incoming question into one of four tiers by comparing its
embedding against four precomputed centroids (one per tier), then applies a
confidence threshold: anything below it is refused rather than guessed,
regardless of which tier scored highest.

This module deliberately does NOT load an embedding model itself. The
blueprint's design point is that bge-small is already loaded once for
retrieval -- the router should reuse that same embedding, not load a second
copy of the model ("zero added RAM" in the blueprint's own words). Callers
pass in a precomputed embedding vector; see `standalone.py` for a
convenience wrapper that loads sentence-transformers directly, for
local testing only.

Calibration methodology (see eval/router/README.md for full numbers):
  - Centroids computed as the mean of bge-small-en-v1.5 embeddings of all
    200 labeled gold questions (agbe_eval_questions.jsonl) per tier,
    L2-normalized.
  - Centroid classifier beat k-NN (k=1/3/5) in 5-fold stratified
    cross-validation: 95.0% overall accuracy vs 93.5-94.5% for k-NN.
  - Threshold=0.74 chosen by sweeping candidate thresholds against
    cross-validated confidence scores: catches 70% of the classifier's own
    cross-validation errors at a 10% false-refusal cost on correct
    classifications, and separately catches 15/15 (100%) of a synthetic
    out-of-domain stress test (unrelated topics: sports scores, recipes,
    general trivia) whose confidence never exceeded 0.65.
  - Known gap: Tier D has the weakest recall (86.7%) because a handful of
    D-tier questions (medical/veterinary emergencies phrased as symptom
    descriptions, e.g. "my goat has stopped eating") surface-resemble
    Tier B diagnosis questions. The confidence threshold catches most of
    these anyway (3 of 4 in cross-validation), but not all -- hence the
    SAFETY_KEYWORDS override below as a defense-in-depth layer that does
    not depend on embedding similarity at all.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

TIERS = ("A", "B", "C", "D")
DEFAULT_CENTROIDS_PATH = Path(__file__).parent / "centroids.json"

# Defense-in-depth: these topics must never be answered even if the
# embedding classifier is confident about a different tier. Human health
# and veterinary questions are out of scope for an agricultural pest/disease
# tool, and a false "diagnosis" here is a materially different kind of harm
# than a wrong fertiliser rate.
SAFETY_KEYWORDS = (
    r"\bfever\b", r"\bmedicine\b", r"\bdoctor\b", r"\bhospital\b",
    r"\bvet(erinary|erinarian)?\b", r"\bmy (goat|cow|chicken|sheep|dog|cat|animal)s?\b.*\b(sick|ill|stopped eating|dying|medicine)\b",
    r"\bpoison(ing|ed)?\b", r"\bstomach pain", r"\bbody pain",
    r"\bloan\b", r"\bbank\b.*\bloan\b", r"\binsurance\b",
    r"\bweather\b", r"\brain(fall)? forecast\b",
    r"\bpolitical party\b", r"\belection\b",
    r"\blegal\b", r"\blawyer\b", r"\bland (dispute|ownership)\b", r"\bcontract\b",
)
_SAFETY_PATTERN = re.compile("|".join(SAFETY_KEYWORDS), re.IGNORECASE)


@dataclass(frozen=True)
class RouteResult:
    tier: str                  # one of "A", "B", "C", "D"
    confidence: float          # cosine similarity to the winning centroid (1.0 for keyword override)
    refused: bool              # True if tier == "D"
    reason: str                # "classified" | "low_confidence" | "safety_keyword_override"
    tier_scores: dict[str, float]  # similarity to every tier's centroid, for debugging/logging


class Router:
    """Loads precomputed tier centroids and classifies embedded questions."""

    def __init__(self, centroids_path: Path | str = DEFAULT_CENTROIDS_PATH, threshold: float | None = None):
        data = json.loads(Path(centroids_path).read_text(encoding="utf-8"))
        self.model_name: str = data["model"]
        self.embedding_dim: int = data["embedding_dim"]
        self.threshold: float = threshold if threshold is not None else data["threshold"]
        self._centroid_mat = np.array([data["centroids"][t] for t in TIERS], dtype=np.float32)  # (4, d)
        # centroids.json stores already-normalized vectors, but re-normalize
        # defensively in case a future calibration run forgets to.
        norms = np.linalg.norm(self._centroid_mat, axis=1, keepdims=True)
        self._centroid_mat = self._centroid_mat / norms

    def classify(self, question_text: str, embedding: Sequence[float]) -> RouteResult:
        """Classify one question. `embedding` must come from the same model
        named in centroids.json (bge-small-en-v1.5), L2-normalized or not
        (normalized here defensively)."""
        if _SAFETY_PATTERN.search(question_text):
            return RouteResult(
                tier="D", confidence=1.0, refused=True,
                reason="safety_keyword_override",
                tier_scores={t: None for t in TIERS},  # not computed -- keyword short-circuit
            )

        if embedding is None:
            raise ValueError(
                "Router.classify() received embedding=None -- the caller's embed_query() "
                "must have failed upstream. Refusing to guess a tier from no embedding; "
                "fix the embedding failure rather than routing on garbage."
            )
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim != 1:
            raise ValueError(
                f"Router.classify() received an embedding with shape {emb.shape}, expected a "
                f"1-D vector of length {self.embedding_dim}."
            )
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        if emb.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding has dim {emb.shape[0]}, expected {self.embedding_dim} "
                f"(centroids were built with {self.model_name}) -- did you pass an embedding from a different model?"
            )

        sims = self._centroid_mat @ emb  # (4,)
        tier_scores = {t: float(s) for t, s in zip(TIERS, sims)}
        best_i = int(np.argmax(sims))
        best_tier = TIERS[best_i]
        best_conf = float(sims[best_i])

        if best_conf < self.threshold:
            return RouteResult(
                tier="D", confidence=best_conf, refused=True,
                reason="low_confidence", tier_scores=tier_scores,
            )

        return RouteResult(
            tier=best_tier, confidence=best_conf, refused=(best_tier == "D"),
            reason="classified", tier_scores=tier_scores,
        )
