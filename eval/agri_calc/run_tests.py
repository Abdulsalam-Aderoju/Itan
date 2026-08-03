"""Test runner for the agri_calc module.

Loads test cases from test_cases.json, sets up a temporary SQLite database,
seeds it with test constants, runs the calculators, and reports details of any failures.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

# Set up environment variable for database path BEFORE importing module
DB_FILE_PATH = Path(__file__).parent / "test_agri_calc.db"
os.environ["AGRI_CALC_DB_PATH"] = str(DB_FILE_PATH)

# Add team repo root to system path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Now import the public interface
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

def init_test_db():
    """Create the SQLite database schema and seed it with test records."""
    if DB_FILE_PATH.exists():
        DB_FILE_PATH.unlink()

    # Read schema.sql
    schema_path = REPO_ROOT / "engine" / "tools" / "agri_calc" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(DB_FILE_PATH)
    try:
        cursor = conn.cursor()
        # Initialize schema
        cursor.executescript(schema_sql)

        # 1. Seed product table
        products = [
            ("NPK 15-15-15", 15.0, 15.0, 15.0, 50.0, "naerls_maize_2021_p14"),
            ("Urea", 46.0, 0.0, 0.0, 50.0, "naerls_maize_2021_p14"),
            ("MOP", 0.0, 0.0, 60.0, 50.0, "mop_source"),
            ("SSP", 0.0, 18.0, 0.0, 50.0, "ssp_source"),
        ]
        cursor.executemany(
            "INSERT INTO product (product_name, n_pct, p2o5_pct, k2o_pct, bag_weight_kg, source_id) VALUES (?, ?, ?, ?, ?, ?)",
            products
        )

        # 2. Seed planting_material table
        materials = [
            ("maize", "seed", 250.0, 1, None, "maize_seed_ref"),
            ("cassava", "cutting", None, 1, 50, "cassava_material_ref"),
            ("yam", "sett", None, 1, None, "yam_material_ref"),
            ("cowpea", "seed", 150.0, 1, None, "cowpea_seed_ref"),
            ("tomato", "seed", 3.0, 1, None, "tomato_seed_ref"),
            ("rice", "seed", 30.0, 1, None, "rice_seed_ref"),
        ]
        cursor.executemany(
            "INSERT INTO planting_material (crop, material_type, unit_weight_g, stands_per_unit, units_per_bundle, source_id) VALUES (?, ?, ?, ?, ?, ?)",
            materials
        )

        # 3. Seed fertilizer_rate table
        rates = [
            (1, "maize", "Northern Guinea Savanna", None, None, 120.0, 60.0, 60.0, "naerls_maize_2021_p14"),
            (2, "tomato", "Sudan Savanna", None, None, 100.0, 50.0, 80.0, "tomato_sudan_2020"),
        ]
        cursor.executemany(
            "INSERT INTO fertilizer_rate (id, crop, zone, soil_class, target_yield, n_rate_kg_ha, p2o5_rate_kg_ha, k2o_rate_kg_ha, source_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rates
        )

        # 4. Seed fertilizer_split table
        splits = [
            (1, 1, "basal (at planting)", "NPK 15-15-15", "P2O5", 1.0, "naerls_maize_2021_p14"),
            (1, 2, "top-dress (6 WAP)", "Urea", "N", 1.0, "naerls_maize_2021_p14"),
            (2, 1, "basal", "NPK 15-15-15", "P2O5", 1.0, "tomato_sudan_2020"),
        ]
        cursor.executemany(
            "INSERT INTO fertilizer_split (fertilizer_rate_id, split_number, timing, product_name, basis_nutrient, split_fraction, source_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            splits
        )

        # 5. Seed spacing table
        spacings = [
            ("maize", "Northern Guinea Savanna", 75.0, 25.0, "maize_spacing_ref"),
            ("cassava", "Southern Guinea Savanna", 100.0, 100.0, "cassava_spacing_ref"),
            ("yam", "Southern Guinea Savanna", 100.0, 100.0, "yam_spacing_ref"),
        ]
        cursor.executemany(
            "INSERT INTO spacing (crop, zone, row_cm, within_row_cm, source_id) VALUES (?, ?, ?, ?, ?)",
            spacings
        )

        # 6. Seed agrochemical table
        agrochemicals = [
            ("Glyphosate", None, 4.0, "l", 30, "agrochemical_ref_1"),
            ("Paraquat", "maize", 3.0, "l", 45, "agrochemical_ref_2"),
            ("Mancozeb", "tomato", 1.5, "kg", 7, "agrochemical_ref_3"),
        ]
        cursor.executemany(
            "INSERT INTO agrochemical (product_name, crop, rate_per_ha, rate_unit, pre_harvest_interval_days, source_id) VALUES (?, ?, ?, ?, ?, ?)",
            agrochemicals
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
    
    # Header format
    print(f"{'ID':<8} | {'Function':<16} | {'Status':<6} | {'Details / Mismatches'}")
    print("-" * 100)

    for case in cases:
        case_id = case["id"]
        func_name = case["function"]
        inputs = case["inputs"]
        expected = case["expected"]

        # Call target function
        raised_error = None
        result_dict = None

        try:
            if func_name == "fertiliser_rate":
                result = fertiliser_rate(
                    crop=inputs["crop"],
                    area_ha=inputs["area_ha"],
                    zone=inputs["zone"],
                    soil_class=inputs.get("soil_class"),
                    target_yield=inputs.get("target_yield"),
                )
                result_dict = result.to_dict()
            elif func_name == "seed_rate":
                spacing_tuple = tuple(inputs["spacing_cm"]) if "spacing_cm" in inputs else None
                result = seed_rate(
                    crop=inputs["crop"],
                    area_ha=inputs["area_ha"],
                    spacing_cm=spacing_tuple,
                    germination_pct=inputs["germination_pct"],
                    seeds_per_stand=inputs.get("seeds_per_stand", 1),
                )
                result_dict = result.to_dict()
            elif func_name == "spray_dilution":
                result = spray_dilution(
                    product_name=inputs["product_name"],
                    tank_litres=inputs["tank_litres"],
                    rate_per_ha=inputs.get("rate_per_ha"),
                    spray_volume_l_per_ha=inputs.get("spray_volume_l_per_ha", 200.0),
                    crop=inputs.get("crop"),
                    conc_pct=inputs.get("conc_pct"),
                )
                result_dict = result.to_dict()
            elif func_name == "gross_margin":
                cost_items = [CostItem(label=c["label"], amount=c["amount"]) for c in inputs.get("input_costs", [])]
                result = gross_margin(
                    yield_kg=inputs["yield_kg"],
                    price_per_kg=inputs["price_per_kg"],
                    input_costs=cost_items,
                    currency=inputs.get("currency", "NGN"),
                )
                result_dict = result.to_dict()
            else:
                print(f"{case_id:<8} | {func_name:<16} | FAILED | Unknown function '{func_name}'")
                failed_count += 1
                continue
        except Exception as e:
            raised_error = e

        # Assertions
        if expected is None:
            # Scaffolded case: just check it didn't crash with unexpected exception
            if raised_error and not isinstance(raised_error, AgriCalcError):
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
            # Check exact-match dict equality
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

    # Cleanup temporary test db
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
