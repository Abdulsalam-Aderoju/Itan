"""Database seeding script for the agri_calc module.

Initializes the SQLite database from schema.sql and populates it with real,
traceable agronomic constants for the six prioritized crops.
"""
import sqlite3
from pathlib import Path

DB_FILE_PATH = Path(__file__).resolve().parent / "agri_calc.db"
SCHEMA_FILE_PATH = Path(__file__).resolve().parent / "schema.sql"

def seed_db():
    print(f"Initializing database at: {DB_FILE_PATH}")
    if DB_FILE_PATH.exists():
        DB_FILE_PATH.unlink()

    # Read schema.sql
    with open(SCHEMA_FILE_PATH, "r") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(DB_FILE_PATH)
    try:
        cursor = conn.cursor()
        
        # 1. Initialize schema
        cursor.executescript(schema_sql)
        print("Schema initialized.")

        # 2. Seed product table
        products = [
            ("NPK 15-15-15", 15.0, 15.0, 15.0, 50.0, "naerls_maize_2021_p14"),
            ("Urea", 46.0, 0.0, 0.0, 50.0, "naerls_maize_2021_p14"),
            ("MOP", 0.0, 0.0, 60.0, 50.0, "naerls_maize_2021_p14"),
            ("SSP", 0.0, 18.0, 0.0, 50.0, "naerls_maize_2021_p14"),
        ]
        cursor.executemany(
            "INSERT INTO product (product_name, n_pct, p2o5_pct, k2o_pct, bag_weight_kg, source_id) VALUES (?, ?, ?, ?, ?, ?)",
            products
        )
        print("Seeded product table.")

        # 3. Seed planting_material table
        materials = [
            ("maize", "seed", 250.0, 1, None, "naerls_maize_2021_p14"),
            ("cassava", "cutting", None, 1, 50, "cgspace:95a959a1-3344-488c-b0a6-5d2994ed25f2"),
            ("yam", "sett", None, 1, None, "cgspace:10568/80404"),
            ("cowpea", "seed", 150.0, 1, None, "cgspace:94bed861-c0a2-4073-a9cf-1909836782d4"),
            ("tomato", "seed", 3.0, 1, None, "tomato_sudan_2020"),
            ("rice", "seed", 30.0, 1, None, "cgspace:10568/108804"),
        ]
        cursor.executemany(
            "INSERT INTO planting_material (crop, material_type, unit_weight_g, stands_per_unit, units_per_bundle, source_id) VALUES (?, ?, ?, ?, ?, ?)",
            materials
        )
        print("Seeded planting_material table.")

        # 4. Seed fertilizer_rate table
        rates = [
            (1, "maize", "Northern Guinea Savanna", None, None, 120.0, 60.0, 60.0, "naerls_maize_2021_p14"),
            (2, "tomato", "Sudan Savanna", None, None, 100.0, 50.0, 80.0, "tomato_sudan_2020"),
            (3, "cassava", "Southern Guinea Savanna", None, None, 90.0, 30.0, 90.0, "cgspace:95a959a1-3344-488c-b0a6-5d2994ed25f2"),
            (4, "yam", "Southern Guinea Savanna", None, None, 90.0, 50.0, 75.0, "cgspace:10568/80404"),
            (5, "cowpea", "Northern Guinea Savanna", None, None, 20.0, 40.0, 20.0, "cgspace:94bed861-c0a2-4073-a9cf-1909836782d4"),
            (6, "rice", "Sudan Savanna", None, None, 80.0, 40.0, 40.0, "cgspace:10568/108804"),
        ]
        cursor.executemany(
            "INSERT INTO fertilizer_rate (id, crop, zone, soil_class, target_yield, n_rate_kg_ha, p2o5_rate_kg_ha, k2o_rate_kg_ha, source_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rates
        )
        print("Seeded fertilizer_rate table.")

        # 5. Seed fertilizer_split table
        splits = [
            (1, 1, "basal (at planting)", "NPK 15-15-15", "P2O5", 1.0, "naerls_maize_2021_p14"),
            (1, 2, "top-dress (6 WAP)", "Urea", "N", 1.0, "naerls_maize_2021_p14"),
            (2, 1, "basal", "NPK 15-15-15", "P2O5", 1.0, "tomato_sudan_2020"),
            (3, 1, "basal (at planting)", "NPK 15-15-15", "P2O5", 1.0, "cgspace:95a959a1-3344-488c-b0a6-5d2994ed25f2"),
            (3, 2, "top-dress (8 WAP)", "Urea", "N", 1.0, "cgspace:95a959a1-3344-488c-b0a6-5d2994ed25f2"),
            (3, 3, "top-dress (8 WAP)", "MOP", "K2O", 1.0, "cgspace:95a959a1-3344-488c-b0a6-5d2994ed25f2"),
            (4, 1, "basal", "NPK 15-15-15", "P2O5", 1.0, "cgspace:10568/80404"),
            (4, 2, "top-dress", "Urea", "N", 1.0, "cgspace:10568/80404"),
            (4, 3, "top-dress", "MOP", "K2O", 1.0, "cgspace:10568/80404"),
            (5, 1, "basal", "NPK 15-15-15", "P2O5", 1.0, "cgspace:94bed861-c0a2-4073-a9cf-1909836782d4"),
            (6, 1, "basal", "NPK 15-15-15", "P2O5", 1.0, "cgspace:10568/108804"),
            (6, 2, "top-dress", "Urea", "N", 1.0, "cgspace:10568/108804"),
        ]
        cursor.executemany(
            "INSERT INTO fertilizer_split (fertilizer_rate_id, split_number, timing, product_name, basis_nutrient, split_fraction, source_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            splits
        )
        print("Seeded fertilizer_split table.")

        # 6. Seed spacing table
        spacings = [
            ("maize", "Northern Guinea Savanna", 75.0, 25.0, "naerls_maize_2021_p14"),
            ("cassava", "Southern Guinea Savanna", 100.0, 100.0, "cgspace:95a959a1-3344-488c-b0a6-5d2994ed25f2"),
            ("yam", "Southern Guinea Savanna", 100.0, 100.0, "cgspace:10568/80404"),
            ("cowpea", "Northern Guinea Savanna", 75.0, 20.0, "cgspace:94bed861-c0a2-4073-a9cf-1909836782d4"),
            ("rice", "Sudan Savanna", 30.0, 15.0, "cgspace:10568/108804"),
            ("tomato", "Sudan Savanna", 60.0, 50.0, "tomato_sudan_2020"),
        ]
        cursor.executemany(
            "INSERT INTO spacing (crop, zone, row_cm, within_row_cm, source_id) VALUES (?, ?, ?, ?, ?)",
            spacings
        )
        print("Seeded spacing table.")

        # 7. Seed agrochemical table
        agrochemicals = [
            ("Glyphosate", None, 4.0, "l", 30, "nafdac_agrochemicals_2020"),
            ("Paraquat", "maize", 3.0, "l", 45, "nafdac_agrochemicals_2020"),
            ("Mancozeb", "tomato", 1.5, "kg", 7, "nafdac_agrochemicals_2020"),
            ("Atrazine", "maize", 4.0, "l", 60, "nafdac_agrochemicals_2020"),
            ("Cypermethrin", "cowpea", 1.0, "l", 14, "nafdac_agrochemicals_2020"),
            ("Glyphosate", "cassava", 4.0, "l", 30, "nafdac_agrochemicals_2020"),
            ("Mancozeb", "yam", 1.5, "kg", 7, "nafdac_agrochemicals_2020"),
            ("Butachlor", "rice", 4.0, "l", 21, "nafdac_agrochemicals_2020"),
        ]
        cursor.executemany(
            "INSERT INTO agrochemical (product_name, crop, rate_per_ha, rate_unit, pre_harvest_interval_days, source_id) VALUES (?, ?, ?, ?, ?, ?)",
            agrochemicals
        )
        print("Seeded agrochemical table.")

        conn.commit()
        print("Database successfully seeded.")
    finally:
        conn.close()

if __name__ == "__main__":
    seed_db()
