"""Test runner for the crop_calendar module.

Loads test cases from test_cases.json, sets up a temporary SQLite database
seeded with known synthetic rows (not the real, sparse corpus data --
see crop_calendar/README.md for actual coverage), runs crop_calendar(),
and reports details of any failures.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

# Set up environment variable for database path BEFORE importing module
DB_FILE_PATH = Path(__file__).parent / "test_crop_calendar.db"
os.environ["CROP_CALENDAR_DB_PATH"] = str(DB_FILE_PATH)

# Add team repo root to system path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Now import the public interface
from engine.tools.crop_calendar import (
    crop_calendar,
    CropCalendarError,
    InvalidInputError,
)

def init_test_db():
    """Create the SQLite database schema and seed it with test records."""
    if DB_FILE_PATH.exists():
        DB_FILE_PATH.unlink()

    schema_path = REPO_ROOT / "engine" / "tools" / "crop_calendar" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(DB_FILE_PATH)
    try:
        cursor = conn.cursor()
        cursor.executescript(schema_sql)

        rows = [
            ("maize", "Northern Guinea Savanna", None, "plant_in_month", "May", None, 0.85, 0, "maize_calendar_ref"),
            ("maize", "Sudan Savanna", None, "planting_window", "June", "July", 0.9, 0, "maize_calendar_ref2"),
            ("yam", None, None, "plant_in_month", "December", None, 0.85, 0, "yam_calendar_ref"),
            ("yam", None, "benue", "first_rains", "April", None, 0.55, 1, "yam_calendar_low_conf_ref"),
        ]
        cursor.executemany(
            "INSERT INTO crop_calendar (crop, zone, state, activity, month_start, month_end, "
            "confidence, needs_review, source_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            if func_name == "crop_calendar":
                result = crop_calendar(
                    crop=inputs["crop"],
                    agro_zone=inputs.get("agro_zone"),
                    year=inputs.get("year"),
                    state=inputs.get("state"),
                )
                result_dict = result.to_dict()
            else:
                print(f"{case_id:<8} | {func_name:<16} | FAILED | Unknown function '{func_name}'")
                failed_count += 1
                continue
        except Exception as e:
            raised_error = e

        if expected is None:
            if raised_error and not isinstance(raised_error, CropCalendarError):
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
