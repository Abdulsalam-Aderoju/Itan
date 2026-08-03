"""Pest lookup tool: symptom -> ranked pest -> treatment resolution.

Primary export matches the blueprint's Section 5.1 TOOLS contract exactly:

    'pest_lookup': (crop, symptom_terms[]) -> ranked [{pest, control, source}]

This is the tool Section 5.3's "guided pest triage" calls: once the agent
has narrowed a vague complaint ("my maize is dying") down to a crop plus a
short list of discriminating symptom terms (via one structured follow-up
question), `pest_lookup(crop, symptom_terms)` deterministically ranks
candidate pests by how many of those terms appear in their documented
symptoms -- no model call, no fuzzy guessing, matching Pillar 2
("tools, not arithmetic" / deterministic Python).

`pest_lookup_by_name` is a second, Ìtàn-specific capability kept alongside
the blueprint's primary tool: an exact lookup by canonical pest name (the
original interface built before the blueprint was available). It isn't
part of the registered TOOLS dict, but it's useful internally and for
direct programmatic use (e.g. once triage has already identified "thrips"
by name, skip re-ranking and fetch its record directly).

Input pest names / symptom terms are matched against text canonicalized
the same way build_db.py canonicalized the raw corpus rows (see
normalize.py), so "thrips", "Thrips damage", and "usually attacked by
thrips" all resolve consistently.
"""
from dataclasses import dataclass, field
import json
from typing import Any

from engine.tools.pest_lookup.db import get_connection
from engine.tools.pest_lookup.errors import InvalidInputError
from engine.tools.pest_lookup.normalize import canonicalize_pest_name, PEST_KEYWORDS
from corpus.common import CROPS


# ---------------------------------------------------------------------------
# Primary tool: pest_lookup(crop, symptom_terms) -> ranked matches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PestMatch:
    """One ranked candidate pest for a (crop, symptom_terms) query."""
    pest: str                        # canonical pest keyword
    match_score: int                 # how many symptom_terms matched
    matched_terms: list[str]
    symptoms: str
    growth_stage: str
    cultural_control: str
    chemical_control: str
    confidence: float
    needs_review: bool
    source_count: int
    source_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pest": self.pest,
            "match_score": self.match_score,
            "matched_terms": self.matched_terms,
            "symptoms": self.symptoms,
            "growth_stage": self.growth_stage,
            "cultural_control": self.cultural_control,
            "chemical_control": self.chemical_control,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "source_count": self.source_count,
            "source_ids": self.source_ids,
        }


@dataclass(frozen=True)
class PestLookupResult:
    """Result of a pest_lookup(crop, symptom_terms) query."""
    crop: str
    symptom_terms: list[str]
    matches: list[PestMatch] = field(default_factory=list)

    @property
    def data_available(self) -> bool:
        return len(self.matches) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop": self.crop,
            "symptom_terms": self.symptom_terms,
            "data_available": self.data_available,
            "matches": [m.to_dict() for m in self.matches],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def pest_lookup(crop: str, symptom_terms: list[str]) -> PestLookupResult:
    """Rank candidate pests for `crop` by how many `symptom_terms` appear
    in their documented symptoms/growth stage/name.

    Matching is deterministic case-insensitive substring matching against
    each record's `pest_name + symptoms + growth_stage` (chemical/cultural
    control text is excluded from matching -- it describes treatment, not
    symptoms, and would produce false positives like "spray" matching
    almost every record's chemical_control text).

    Candidates are drawn from crop-specific records AND crop-generic
    records (crop is NULL in the source text) for `crop`, since a
    generic record can still be the best symptom match even without
    crop-specific confirmation. Only records with at least one matched
    term are returned; the result list is empty (data_available=False)
    if nothing matches, which is a legitimate outcome (wrong symptom
    vocabulary, or a genuine corpus gap), not an error.

    Raises InvalidInputError if `crop` is not one of the 10 target crops,
    or if `symptom_terms` is empty / contains no non-empty strings.
    """
    if not crop:
        raise InvalidInputError("Crop name cannot be empty.")
    clean_crop = crop.lower().strip()
    if clean_crop not in CROPS:
        raise InvalidInputError(
            f"Unrecognized crop '{crop}'. Must be one of: {', '.join(CROPS)}"
        )

    clean_terms = [t.lower().strip() for t in (symptom_terms or []) if t and t.strip()]
    if not clean_terms:
        raise InvalidInputError("symptom_terms must contain at least one non-empty term.")

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM pest WHERE crop = ? OR crop IS NULL ORDER BY id",
            (clean_crop,),
        ).fetchall()

        matches = []
        for r in rows:
            searchable = f"{r['pest_name']} {r['symptoms']} {r['growth_stage']}".lower()
            matched_terms = [t for t in clean_terms if t in searchable]
            if not matched_terms:
                continue
            matches.append(PestMatch(
                pest=r["pest_name"],
                match_score=len(matched_terms),
                matched_terms=matched_terms,
                symptoms=r["symptoms"] or "",
                growth_stage=r["growth_stage"] or "",
                cultural_control=r["cultural_control"] or "",
                chemical_control=r["chemical_control"] or "",
                confidence=r["confidence"],
                needs_review=bool(r["needs_review"]),
                source_count=r["source_count"],
                source_ids=json.loads(r["source_ids"]),
            ))

        matches.sort(key=lambda m: (-m.match_score, -m.confidence))

        return PestLookupResult(
            crop=clean_crop,
            symptom_terms=clean_terms,
            matches=matches,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Secondary capability: exact lookup by canonical pest name
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PestRecord:
    """One pest/crop resolution: symptoms + controls for a single crop."""
    crop: str | None                # None means the source text wasn't tied to a specific crop
    symptoms: str
    growth_stage: str
    cultural_control: str
    chemical_control: str
    confidence: float
    needs_review: bool
    source_count: int
    source_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop": self.crop,
            "symptoms": self.symptoms,
            "growth_stage": self.growth_stage,
            "cultural_control": self.cultural_control,
            "chemical_control": self.chemical_control,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "source_count": self.source_count,
            "source_ids": self.source_ids,
        }


@dataclass(frozen=True)
class PestNameLookupResult:
    """Result of a pest_lookup_by_name query."""
    pest_name: str                   # canonical keyword resolved from the input
    crop_filter: str | None
    records: list[PestRecord] = field(default_factory=list)

    @property
    def data_available(self) -> bool:
        return len(self.records) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pest_name": self.pest_name,
            "crop_filter": self.crop_filter,
            "data_available": self.data_available,
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def pest_lookup_by_name(pest_name: str, crop: str | None = None) -> PestNameLookupResult:
    """Resolve a pest name (and optional crop) to symptoms/treatment.

    Behavior with `crop` given: returns the crop-specific record if one
    exists; otherwise falls back to a crop-generic record (crop is NULL in
    the source text) if one exists; otherwise returns data_available=False
    even if OTHER crops have data for this pest -- treatment advice from an
    unrelated crop is not substituted silently.

    Behavior with `crop` omitted: returns every record on file for this
    pest, one per crop that has data (plus the generic record if any).
    """
    if not pest_name:
        raise InvalidInputError("Pest name cannot be empty.")

    canonical = canonicalize_pest_name(pest_name)
    if canonical is None:
        raise InvalidInputError(
            f"Unrecognized pest '{pest_name}'. Must contain one of: {', '.join(PEST_KEYWORDS)}"
        )

    conn = get_connection()
    try:
        # ORDER BY id: without it, SQLite serves this query from the
        # UNIQUE(pest_name, crop) index, which sorts by crop -- NULL first
        # -- rather than insertion order, making the no-crop-filter result
        # list's order silently depend on SQLite's query plan.
        rows = conn.execute(
            "SELECT * FROM pest WHERE pest_name = ? ORDER BY id", (canonical,)
        ).fetchall()

        if crop is not None:
            clean_crop = crop.lower().strip()
            crop_row = next((r for r in rows if r["crop"] == clean_crop), None)
            generic_row = next((r for r in rows if r["crop"] is None), None)
            matched = crop_row or generic_row
            selected_rows = [matched] if matched else []
        else:
            selected_rows = list(rows)

        records = [
            PestRecord(
                crop=r["crop"],
                symptoms=r["symptoms"] or "",
                growth_stage=r["growth_stage"] or "",
                cultural_control=r["cultural_control"] or "",
                chemical_control=r["chemical_control"] or "",
                confidence=r["confidence"],
                needs_review=bool(r["needs_review"]),
                source_count=r["source_count"],
                source_ids=json.loads(r["source_ids"]),
            )
            for r in selected_rows
        ]

        return PestNameLookupResult(
            pest_name=canonical,
            crop_filter=crop.lower().strip() if crop else None,
            records=records,
        )
    finally:
        conn.close()
