"""Script to generate test_cases.json for the crop_calendar module.

16 cases (CC-001 to CC-016) covering: the blueprint's (crop, agro_zone,
year) -> {plant_window, harvest_window} contract (Section 5.1), the
plant_window resolution priority (planting_window over plant_in_month,
and first_rains never resolving to a plant_window on its own), zone/state
filtering, the no-op `year` passthrough, case-insensitive crop input, the
"valid crop but no data" empty-result path, and InvalidInputError for
unrecognized crop/zone.
"""
import json
from pathlib import Path

MAIZE_ROW_1 = {
    "activity": "plant_in_month", "month_start": "May", "month_end": None,
    "zone": "Northern Guinea Savanna", "state": None,
    "confidence": 0.85, "needs_review": False, "source_id": "maize_calendar_ref"
}
MAIZE_ROW_2 = {
    "activity": "planting_window", "month_start": "June", "month_end": "July",
    "zone": "Sudan Savanna", "state": None,
    "confidence": 0.9, "needs_review": False, "source_id": "maize_calendar_ref2"
}
YAM_ROW_1 = {
    "activity": "plant_in_month", "month_start": "December", "month_end": None,
    "zone": None, "state": None,
    "confidence": 0.85, "needs_review": False, "source_id": "yam_calendar_ref"
}
YAM_ROW_2 = {
    "activity": "first_rains", "month_start": "April", "month_end": None,
    "zone": None, "state": "benue",
    "confidence": 0.55, "needs_review": True, "source_id": "yam_calendar_low_conf_ref"
}


def generate():
    cases = []

    # CC-001: maize, no filters -> planting_window activity wins over plant_in_month
    cases.append({
        "id": "CC-001",
        "function": "crop_calendar",
        "inputs": {"crop": "maize"},
        "expected": {
            "crop": "maize", "agro_zone": None, "year": None, "state": None,
            "data_available": True,
            "plant_window": {
                "start_month": "June", "end_month": "July",
                "confidence": 0.9, "needs_review": False, "source_id": "maize_calendar_ref2"
            },
            "harvest_window": None,
            "all_signals": [MAIZE_ROW_1, MAIZE_ROW_2]
        },
        "source_id": "maize_calendar_ref2",
        "notes": "planting_window (has both months) is preferred over plant_in_month when both exist"
    })

    # CC-002: maize filtered to the zone that only has the plant_in_month row
    cases.append({
        "id": "CC-002",
        "function": "crop_calendar",
        "inputs": {"crop": "maize", "agro_zone": "Northern Guinea Savanna"},
        "expected": {
            "crop": "maize", "agro_zone": "Northern Guinea Savanna", "year": None, "state": None,
            "data_available": True,
            "plant_window": {
                "start_month": "May", "end_month": None,
                "confidence": 0.85, "needs_review": False, "source_id": "maize_calendar_ref"
            },
            "harvest_window": None,
            "all_signals": [MAIZE_ROW_1]
        },
        "source_id": "maize_calendar_ref",
        "notes": "with only a plant_in_month row in scope, that becomes the resolved plant_window"
    })

    # CC-003: yam, no filters -> plant_in_month wins (no planting_window row exists);
    # first_rains is NOT treated as a plant_window candidate
    cases.append({
        "id": "CC-003",
        "function": "crop_calendar",
        "inputs": {"crop": "yam"},
        "expected": {
            "crop": "yam", "agro_zone": None, "year": None, "state": None,
            "data_available": True,
            "plant_window": {
                "start_month": "December", "end_month": None,
                "confidence": 0.85, "needs_review": False, "source_id": "yam_calendar_ref"
            },
            "harvest_window": None,
            "all_signals": [YAM_ROW_1, YAM_ROW_2]
        },
        "source_id": "yam_calendar_ref",
        "notes": "first_rains is a supporting signal in all_signals, not a plant_window source"
    })

    # CC-004: yam filtered to state='benue' -> only the first_rains row is in scope,
    # so plant_window resolves to None even though data exists (all_signals non-empty)
    cases.append({
        "id": "CC-004",
        "function": "crop_calendar",
        "inputs": {"crop": "yam", "state": "benue"},
        "expected": {
            "crop": "yam", "agro_zone": None, "year": None, "state": "benue",
            "data_available": False,
            "plant_window": None,
            "harvest_window": None,
            "all_signals": [YAM_ROW_2]
        },
        "source_id": "system",
        "notes": "first_rains-only scope has no plant_window/harvest_window despite all_signals being non-empty"
    })

    # CC-005: cassava (valid crop, zero seeded rows) -> fully empty, not an error
    cases.append({
        "id": "CC-005",
        "function": "crop_calendar",
        "inputs": {"crop": "cassava"},
        "expected": {
            "crop": "cassava", "agro_zone": None, "year": None, "state": None,
            "data_available": False, "plant_window": None, "harvest_window": None, "all_signals": []
        },
        "source_id": "system",
        "notes": "valid crop with no corpus coverage returns empty, not InvalidInputError"
    })

    # CC-006: unrecognized crop -> InvalidInputError
    cases.append({
        "id": "CC-006",
        "function": "crop_calendar",
        "inputs": {"crop": "banana"},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "crop outside the 10 target crops is a caller mistake"
    })

    # CC-007: empty crop string -> InvalidInputError
    cases.append({
        "id": "CC-007",
        "function": "crop_calendar",
        "inputs": {"crop": ""},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "empty crop name rejected before hitting the DB"
    })

    # CC-008: unrecognized agro_zone -> InvalidInputError
    cases.append({
        "id": "CC-008",
        "function": "crop_calendar",
        "inputs": {"crop": "maize", "agro_zone": "Wet Guinea Savanna"},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "zone outside the 6 recognized agro-ecological zones is rejected"
    })

    # CC-009: case-insensitive crop input normalizes the same as CC-001
    cases.append({
        "id": "CC-009",
        "function": "crop_calendar",
        "inputs": {"crop": "MAIZE"},
        "expected": {
            "crop": "maize", "agro_zone": None, "year": None, "state": None,
            "data_available": True,
            "plant_window": {
                "start_month": "June", "end_month": "July",
                "confidence": 0.9, "needs_review": False, "source_id": "maize_calendar_ref2"
            },
            "harvest_window": None,
            "all_signals": [MAIZE_ROW_1, MAIZE_ROW_2]
        },
        "source_id": "maize_calendar_ref2",
        "notes": "uppercase crop input is normalized before lookup"
    })

    # CC-010: maize filtered to the zone that only has the planting_window row
    cases.append({
        "id": "CC-010",
        "function": "crop_calendar",
        "inputs": {"crop": "maize", "agro_zone": "Sudan Savanna"},
        "expected": {
            "crop": "maize", "agro_zone": "Sudan Savanna", "year": None, "state": None,
            "data_available": True,
            "plant_window": {
                "start_month": "June", "end_month": "July",
                "confidence": 0.9, "needs_review": False, "source_id": "maize_calendar_ref2"
            },
            "harvest_window": None,
            "all_signals": [MAIZE_ROW_2]
        },
        "source_id": "maize_calendar_ref2",
        "notes": "confirms the two seeded maize rows are independently addressable by zone"
    })

    # CC-011: sorghum (valid crop, zero seeded rows) -> empty
    cases.append({
        "id": "CC-011",
        "function": "crop_calendar",
        "inputs": {"crop": "sorghum"},
        "expected": {
            "crop": "sorghum", "agro_zone": None, "year": None, "state": None,
            "data_available": False, "plant_window": None, "harvest_window": None, "all_signals": []
        },
        "source_id": "system",
        "notes": "second uncovered crop, confirms CC-005 isn't a one-off"
    })

    # CC-012: whitespace-padded crop input normalizes
    cases.append({
        "id": "CC-012",
        "function": "crop_calendar",
        "inputs": {"crop": "  yam  "},
        "expected": {
            "crop": "yam", "agro_zone": None, "year": None, "state": None,
            "data_available": True,
            "plant_window": {
                "start_month": "December", "end_month": None,
                "confidence": 0.85, "needs_review": False, "source_id": "yam_calendar_ref"
            },
            "harvest_window": None,
            "all_signals": [YAM_ROW_1, YAM_ROW_2]
        },
        "source_id": "yam_calendar_ref",
        "notes": "leading/trailing whitespace in crop input is stripped"
    })

    # CC-013: state filter with no match on an existing crop -> empty
    cases.append({
        "id": "CC-013",
        "function": "crop_calendar",
        "inputs": {"crop": "yam", "state": "kano"},
        "expected": {
            "crop": "yam", "agro_zone": None, "year": None, "state": "kano",
            "data_available": False, "plant_window": None, "harvest_window": None, "all_signals": []
        },
        "source_id": "system",
        "notes": "state filter with zero matches returns empty, not an error"
    })

    # CC-014: zone filter with no matching rows for that crop -> empty
    cases.append({
        "id": "CC-014",
        "function": "crop_calendar",
        "inputs": {"crop": "maize", "agro_zone": "Southern Guinea Savanna"},
        "expected": {
            "crop": "maize", "agro_zone": "Southern Guinea Savanna", "year": None, "state": None,
            "data_available": False, "plant_window": None, "harvest_window": None, "all_signals": []
        },
        "source_id": "system",
        "notes": "zone with zero matching rows returns empty, not a fallback to another zone"
    })

    # CC-015: whitespace-only crop -> InvalidInputError
    cases.append({
        "id": "CC-015",
        "function": "crop_calendar",
        "inputs": {"crop": "   "},
        "expected": {"error": "InvalidInputError"},
        "source_id": "system",
        "notes": "whitespace-only crop name is empty after stripping, rejected"
    })

    # CC-016: year is accepted per the blueprint signature but is a no-op today --
    # same result as CC-001, just echoed back in the output
    cases.append({
        "id": "CC-016",
        "function": "crop_calendar",
        "inputs": {"crop": "maize", "year": 2026},
        "expected": {
            "crop": "maize", "agro_zone": None, "year": 2026, "state": None,
            "data_available": True,
            "plant_window": {
                "start_month": "June", "end_month": "July",
                "confidence": 0.9, "needs_review": False, "source_id": "maize_calendar_ref2"
            },
            "harvest_window": None,
            "all_signals": [MAIZE_ROW_1, MAIZE_ROW_2]
        },
        "source_id": "maize_calendar_ref2",
        "notes": "year is echoed back but does not filter -- no row in the corpus carries a year yet"
    })

    assert len(cases) == 16, f"Expected 16 test cases, generated {len(cases)}"

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_cases.json"
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Successfully generated {len(cases)} test cases at {output_path}")

if __name__ == "__main__":
    generate()
