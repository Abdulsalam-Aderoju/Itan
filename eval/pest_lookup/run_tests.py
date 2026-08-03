"""Test runner for the pest_lookup module.

Loads test cases from test_cases.json, sets up a temporary SQLite database
seeded with known synthetic (pest, crop) records (not the real corpus-built
data -- see pest_lookup/README.md for actual coverage), runs pest_lookup(),
and reports details of any failures.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

# Set up environment variable for database path BEFORE importing module
DB_FILE_PATH = Path(__file__).parent / "test_pest_lookup.db"
os.environ["PEST_LOOKUP_DB_PATH"] = str(DB_FILE_PATH)

# Add team repo root to system path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Now import the public interface
from engine.tools.pest_lookup import (
    pest_lookup,
    pest_lookup_by_name,
    PestLookupError,
    InvalidInputError,
)

def init_test_db():
    """Create the SQLite database schema and seed it with test records."""
    if DB_FILE_PATH.exists():
        DB_FILE_PATH.unlink()

    schema_path = REPO_ROOT / "engine" / "tools" / "pest_lookup" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(DB_FILE_PATH)
    try:
        cursor = conn.cursor()
        cursor.executescript(schema_sql)

        rows = [
            ("thrips", "groundnut",
             "Yellowish-green patches on the upper leaf surface, dwarfing and malformation of leaves.",
             "seedling", "", "Apply approved insecticide at first sign of damage.",
             0.85, 0, 5, json.dumps(["groundnut_thrips_ref"])),
            ("thrips", None,
             "General thrips damage across host crops, yellow streaks visible.",
             "", "Remove crop debris after harvest.", "",
             0.5, 1, 2, json.dumps(["generic_thrips_ref"])),
            ("rust", "maize",
             "Orange pustules on the underside of leaves, yellow halo around lesions.",
             "vegetative", "Use resistant varieties.", "Mancozeb spray at first sign of pustules.",
             0.9, 0, 8, json.dumps(["maize_rust_ref"])),
            ("blight", "maize",
             "Grayish-green lesions on leaves, wilting under humid conditions.",
             "", "", "Copper-based fungicide.",
             0.6, 1, 3, json.dumps(["maize_blight_ref"])),
            ("mealybug", None,
             "", "", "", "",
             0.3, 1, 1, json.dumps(["mealybug_generic_ref"])),
        ]
        cursor.executemany(
            "INSERT INTO pest (pest_name, crop, symptoms, growth_stage, cultural_control, "
            "chemical_control, confidence, needs_review, source_count, source_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        conn.commit()
    finally:
        conn.close()

def diff_dicts(expected, actual):
    """Find differences between expected and actual dicts."""
    diffs = {}
    for k, v in expected.items():
        if k not in actual:
            diffs[k] = f"Expected {v}, but key is missing in actual"
        elif actual[k] != v:
            diffs[k] = f"Expected {v} (type: {type(v).__name__}), got {actual[k]} (type: {type(actual[k]).__name__})"
    for k, v in actual.items():
        if k not in expected:
            diffs[k] = f"Extra key in actual: {v}"
    return diffs

def run_tests():
    """Load test cases and run them against the seeded database."""
    print("Initializing test database...")
    init_test_db()

    test_cases_path = Path(__file__).parent / "test_cases.json"
    if not test_cases_path.exists():
        print(f"Error: {test_cases_path} does not exist. Run generate_test_cases.py first.")
        sys.exit(1)

    with open(test_cases_path, "r") as f:
        cases = json.load(f)

    passed_count = 0
    failed_count = 0

    print(f"{'ID':<8} | {'Function':<16} | {'Status':<6} | {'Details / Mismatches'}")
    print("-" * 100)

    for case in cases:
        case_id = case["id"]
        func_name = case["function"]
        inputs = case["inputs"]
        expected = case["expected"]

        raised_error = None
        result_dict = None

        try:
            if func_name == "pest_lookup":
                result = pest_lookup(
                    crop=inputs["crop"],
                    symptom_terms=inputs["symptom_terms"],
                )
                result_dict = result.to_dict()
            elif func_name == "pest_lookup_by_name":
                result = pest_lookup_by_name(
                    pest_name=inputs["pest_name"],
                    crop=inputs.get("crop"),
                )
                result_dict = result.to_dict()
            else:
                print(f"{case_id:<8} | {func_name:<16} | FAILED | Unknown function '{func_name}'")
                failed_count += 1
                continue
        except Exception as e:
            raised_error = e

        if expected is None:
            if raised_error and not isinstance(raised_error, PestLookupError):
                print(f"{case_id:<8} | {func_name:<16} | FAILED | Crashed with unexpected error: {raised_error}")
                failed_count += 1
            else:
                print(f"{case_id:<8} | {func_name:<16} | PASSED | Scaffold run succeeded (no check)")
                passed_count += 1
        elif "error" in expected:
            expected_err = expected["error"]
            if raised_error is None:
                print(f"{case_id:<8} | {func_name:<16} | FAILED | Expected {expected_err}, but function returned successfully")
                failed_count += 1
            elif raised_error.__class__.__name__ == expected_err:
                print(f"{case_id:<8} | {func_name:<16} | PASSED | Raised {expected_err} as expected: {raised_error}")
                passed_count += 1
            else:
                print(f"{case_id:<8} | {func_name:<16} | FAILED | Expected {expected_err}, but raised {raised_error.__class__.__name__}: {raised_error}")
                failed_count += 1
        else:
            if raised_error:
                print(f"{case_id:<8} | {func_name:<16} | FAILED | Raised exception: {raised_error.__class__.__name__}: {raised_error}")
                failed_count += 1
            else:
                diffs = diff_dicts(expected, result_dict)
                if not diffs:
                    print(f"{case_id:<8} | {func_name:<16} | PASSED | Exact Match")
                    passed_count += 1
                else:
                    diff_str = ", ".join([f"{k}: {v}" for k, v in diffs.items()])
                    print(f"{case_id:<8} | {func_name:<16} | FAILED | Mismatches: {diff_str}")
                    failed_count += 1

    if DB_FILE_PATH.exists():
        try:
            DB_FILE_PATH.unlink()
        except OSError:
            pass

    print("-" * 100)
    print(f"Summary: Total Ran = {passed_count + failed_count} | Passed = {passed_count} | Failed = {failed_count}")

    if failed_count > 0:
        print("Result: FAIL")
        sys.exit(1)
    else:
        print("Result: PASS (All test cases passed exact-match validation)")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
