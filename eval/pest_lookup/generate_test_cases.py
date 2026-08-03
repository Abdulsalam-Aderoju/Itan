"""Script to generate test_cases.json for the pest_lookup module.

15 cases (PL-001 to PL-015) covering the blueprint's primary contract
(Section 5.1): (crop, symptom_terms[]) -> ranked [{pest, control, source}].
Covers: single-term and multi-term matching, match-count + confidence
ranking, the crop-specific + crop-generic candidate pool, no-match
(empty, not an error), case/whitespace normalization, and
InvalidInputError for unrecognized crop / empty symptom_terms. The final
two cases (PL-014, PL-015) cover `pest_lookup_by_name`, the secondary
exact-match-by-name capability kept alongside the blueprint's primary
tool.
"""
import json
from pathlib import Path

ROW1 = {  # thrips / groundnut
    "pest": "thrips", "symptoms": "Yellowish-green patches on the upper leaf surface, dwarfing and malformation of leaves.",
    "growth_stage": "seedling", "cultural_control": "", "chemical_control": "Apply approved insecticide at first sign of damage.",
    "confidence": 0.85, "needs_review": False, "source_count": 5, "source_ids": ["groundnut_thrips_ref"]
}
ROW2 = {  # thrips / generic
    "pest": "thrips", "symptoms": "General thrips damage across host crops, yellow streaks visible.",
    "growth_stage": "", "cultural_control": "Remove crop debris after harvest.", "chemical_control": "",
    "confidence": 0.5, "needs_review": True, "source_count": 2, "source_ids": ["generic_thrips_ref"]
}
ROW3 = {  # rust / maize
    "pest": "rust", "symptoms": "Orange pustules on the underside of leaves, yellow halo around lesions.",
    "growth_stage": "vegetative", "cultural_control": "Use resistant varieties.", "chemical_control": "Mancozeb spray at first sign of pustules.",
    "confidence": 0.9, "needs_review": False, "source_count": 8, "source_ids": ["maize_rust_ref"]
}
ROW4 = {  # blight / maize
    "pest": "blight", "symptoms": "Grayish-green lesions on leaves, wilting under humid conditions.",
    "growth_stage": "", "cultural_control": "", "chemical_control": "Copper-based fungicide.",
    "confidence": 0.6, "needs_review": True, "source_count": 3, "source_ids": ["maize_blight_ref"]
}


def generate():
    cases = []

    # PL-001: single term, single match
    cases.append({
        "id": "PL-001",
        "function": "pest_lookup",
        "inputs": {"crop": "groundnut", "symptom_terms": ["dwarfing"]},
        "expected": {
            "crop": "groundnut", "symptom_terms": ["dwarfing"], "data_available": True,
            "matches": [{**ROW1, "match_score": 1, "matched_terms": ["dwarfing"]}]
        },
        "source_id": "groundnut_thrips_ref",
        "notes": "baseline single-term match against the crop-specific record"
    })

    # PL-002: term matches both a crop-specific and a crop-generic record;
    # ranked by confidence since match_score ties
    cases.append({
        "id": "PL-002",
        "function": "pest_lookup",
        "inputs": {"crop": "groundnut", "symptom_terms": ["yellow"]},
        "expected": {
            "crop": "groundnut", "symptom_terms": ["yellow"], "data_available": True,
            "matches": [
                {**ROW1, "match_score": 1, "matched_terms": ["yellow"]},
                {**ROW2, "match_score": 1, "matched_terms": ["yellow"]}
            ]
        },
        "source_id": "groundnut_thrips_ref",
        "notes": "tie on match_score broken by confidence descending (0.85 before 0.5)"
    })

    # PL-003: same term against a different crop -- crop-specific rust record
    # plus the crop-generic thrips record both qualify (candidate pool is
    # crop-specific UNION crop-generic)
    cases.append({
        "id": "PL-003",
        "function": "pest_lookup",
        "inputs": {"crop": "maize", "symptom_terms": ["yellow"]},
        "expected": {
            "crop": "maize", "symptom_terms": ["yellow"], "data_available": True,
            "matches": [
                {**ROW3, "match_score": 1, "matched_terms": ["yellow"]},
                {**ROW2, "match_score": 1, "matched_terms": ["yellow"]}
            ]
        },
        "source_id": "maize_rust_ref",
        "notes": "generic (crop=None) records are candidates for every crop query, not just their own"
    })

    # PL-004: two terms -- match_score ranks a 2-term match above a 1-term match
    cases.append({
        "id": "PL-004",
        "function": "pest_lookup",
        "inputs": {"crop": "maize", "symptom_terms": ["wilting", "lesions"]},
        "expected": {
            "crop": "maize", "symptom_terms": ["wilting", "lesions"], "data_available": True,
            "matches": [
                {**ROW4, "match_score": 2, "matched_terms": ["wilting", "lesions"]},
                {**ROW3, "match_score": 1, "matched_terms": ["lesions"]}
            ]
        },
        "source_id": "maize_blight_ref",
        "notes": "match_score (terms matched), not confidence, is the primary sort key"
    })

    # PL-005: no term matches anything -> empty, not an error
    cases.append({
        "id": "PL-005",
        "function": "pest_lookup",
        "inputs": {"crop": "maize", "symptom_terms": ["nonexistent symptom xyz"]},
        "expected": {
            "crop": "maize", "symptom_terms": ["nonexistent symptom xyz"],
            "data_available": False, "matches": []
        },
        "source_id": "system",
        "notes": "zero matches is a legitimate outcome, not InvalidInputError"
    })

    # PL-006: unrecognized crop -> InvalidInputError
    cases.append({
        "id": "PL-006",
        "function": "pest_lookup",
        "inputs": {"crop": "banana", "symptom_terms": ["yellow"]},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "crop outside the 10 target crops is a caller mistake"
    })

    # PL-007: empty crop -> InvalidInputError
    cases.append({
        "id": "PL-007",
        "function": "pest_lookup",
        "inputs": {"crop": "", "symptom_terms": ["yellow"]},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "empty crop name rejected before hitting the DB"
    })

    # PL-008: empty symptom_terms list -> InvalidInputError
    cases.append({
        "id": "PL-008",
        "function": "pest_lookup",
        "inputs": {"crop": "maize", "symptom_terms": []},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "guided triage must supply at least one discriminating term"
    })

    # PL-009: symptom_terms containing only blank/whitespace strings -> InvalidInputError
    cases.append({
        "id": "PL-009",
        "function": "pest_lookup",
        "inputs": {"crop": "maize", "symptom_terms": ["   ", ""]},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "blank-only terms are stripped and treated as no terms supplied"
    })

    # PL-010: case-insensitive crop and term input normalizes the same as PL-003
    cases.append({
        "id": "PL-010",
        "function": "pest_lookup",
        "inputs": {"crop": "MAIZE", "symptom_terms": ["YELLOW"]},
        "expected": {
            "crop": "maize", "symptom_terms": ["yellow"], "data_available": True,
            "matches": [
                {**ROW3, "match_score": 1, "matched_terms": ["yellow"]},
                {**ROW2, "match_score": 1, "matched_terms": ["yellow"]}
            ]
        },
        "source_id": "maize_rust_ref",
        "notes": "uppercase crop and term input are both normalized before matching"
    })

    # PL-011: crop with zero crop-specific rows, but a crop-generic record still qualifies
    cases.append({
        "id": "PL-011",
        "function": "pest_lookup",
        "inputs": {"crop": "pepper", "symptom_terms": ["yellow"]},
        "expected": {
            "crop": "pepper", "symptom_terms": ["yellow"], "data_available": True,
            "matches": [{**ROW2, "match_score": 1, "matched_terms": ["yellow"]}]
        },
        "source_id": "generic_thrips_ref",
        "notes": "pepper has no crop-specific rows in the seed data, but the generic thrips record still surfaces"
    })

    # PL-012: crop with zero crop-specific rows AND no generic match either -> empty
    cases.append({
        "id": "PL-012",
        "function": "pest_lookup",
        "inputs": {"crop": "pepper", "symptom_terms": ["nonexistent symptom xyz"]},
        "expected": {
            "crop": "pepper", "symptom_terms": ["nonexistent symptom xyz"],
            "data_available": False, "matches": []
        },
        "source_id": "system",
        "notes": "no crop-specific data and no generic match -> empty, not an error"
    })

    # PL-013: whitespace-padded symptom term normalizes the same as PL-002
    cases.append({
        "id": "PL-013",
        "function": "pest_lookup",
        "inputs": {"crop": "groundnut", "symptom_terms": ["  yellow  "]},
        "expected": {
            "crop": "groundnut", "symptom_terms": ["yellow"], "data_available": True,
            "matches": [
                {**ROW1, "match_score": 1, "matched_terms": ["yellow"]},
                {**ROW2, "match_score": 1, "matched_terms": ["yellow"]}
            ]
        },
        "source_id": "groundnut_thrips_ref",
        "notes": "leading/trailing whitespace in a symptom term is stripped before matching"
    })

    # PL-014: pest_lookup_by_name -- the secondary exact-match-by-name capability
    cases.append({
        "id": "PL-014",
        "function": "pest_lookup_by_name",
        "inputs": {"pest_name": "thrips", "crop": "groundnut"},
        "expected": {
            "pest_name": "thrips", "crop_filter": "groundnut", "data_available": True,
            "records": [{
                "crop": "groundnut",
                "symptoms": ROW1["symptoms"], "growth_stage": ROW1["growth_stage"],
                "cultural_control": ROW1["cultural_control"], "chemical_control": ROW1["chemical_control"],
                "confidence": ROW1["confidence"], "needs_review": ROW1["needs_review"],
                "source_count": ROW1["source_count"], "source_ids": ROW1["source_ids"]
            }]
        },
        "source_id": "groundnut_thrips_ref",
        "notes": "secondary capability retained alongside the blueprint's primary pest_lookup tool"
    })

    # PL-015: pest_lookup_by_name -- unrecognized pest keyword -> InvalidInputError
    cases.append({
        "id": "PL-015",
        "function": "pest_lookup_by_name",
        "inputs": {"pest_name": "rodent"},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "pest category outside the known keyword vocabulary is a caller mistake"
    })

    assert len(cases) == 15, f"Expected 15 test cases, generated {len(cases)}"

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_cases.json"
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Successfully generated {len(cases)} test cases at {output_path}")

if __name__ == "__main__":
    generate()
