-- Database schema for the agri_calc module

CREATE TABLE IF NOT EXISTS product (
    product_name        TEXT PRIMARY KEY,     -- e.g. 'NPK 15-15-15', 'Urea'
    n_pct                REAL NOT NULL,        -- % nitrogen by weight
    p2o5_pct             REAL NOT NULL,        -- % phosphorus pentoxide by weight
    k2o_pct               REAL NOT NULL,        -- % potassium oxide by weight
    bag_weight_kg         REAL DEFAULT 50,      -- weight per bag in kg
    source_id             TEXT NOT NULL         -- citation source ID
);

CREATE TABLE IF NOT EXISTS planting_material (
    crop                  TEXT NOT NULL,
    material_type         TEXT NOT NULL,        -- 'seed' | 'cutting' | 'sett'
    unit_weight_g         REAL,                  -- 1000-seed weight in grams; NULL for cuttings/setts
    stands_per_unit        INTEGER DEFAULT 1,     -- e.g. cuttings per stand for cassava
    units_per_bundle       INTEGER,               -- standard bundle count if documented (e.g. 50)
    source_id              TEXT NOT NULL,        -- citation source ID
    PRIMARY KEY (crop, material_type)
);

CREATE TABLE IF NOT EXISTS fertilizer_rate (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    crop                  TEXT NOT NULL,
    zone                  TEXT NOT NULL,
    soil_class            TEXT,                 -- nullable soil class
    target_yield          REAL,                 -- nullable target yield in t/ha or similar
    n_rate_kg_ha          REAL NOT NULL,        -- nitrogen rate in kg/ha
    p2o5_rate_kg_ha       REAL NOT NULL,        -- phosphorus rate in kg/ha
    k2o_rate_kg_ha        REAL NOT NULL,        -- potassium rate in kg/ha
    source_id             TEXT NOT NULL         -- citation source ID
);

CREATE TABLE IF NOT EXISTS fertilizer_split (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    fertilizer_rate_id    INTEGER NOT NULL,
    split_number          INTEGER NOT NULL,
    timing                TEXT NOT NULL,        -- e.g., "basal (at planting)", "top-dress (6 WAP)"
    product_name          TEXT NOT NULL,        -- e.g., 'NPK 15-15-15', 'Urea'
    basis_nutrient        TEXT NOT NULL,        -- 'N' | 'P2O5' | 'K2O'
    split_fraction        REAL DEFAULT 1.0,     -- proportion of basis nutrient to apply
    source_id             TEXT NOT NULL,        -- citation source ID
    FOREIGN KEY (fertilizer_rate_id) REFERENCES fertilizer_rate(id),
    FOREIGN KEY (product_name) REFERENCES product(product_name)
);

CREATE TABLE IF NOT EXISTS spacing (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    crop                  TEXT NOT NULL,
    zone                  TEXT NOT NULL,
    row_cm                REAL NOT NULL,        -- row spacing in cm
    within_row_cm         REAL NOT NULL,        -- within-row spacing in cm
    source_id             TEXT NOT NULL,        -- citation source ID
    UNIQUE (crop, zone)
);

CREATE TABLE IF NOT EXISTS agrochemical (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name                  TEXT NOT NULL,
    crop                          TEXT,                 -- NULL if general recommendation
    rate_per_ha                   REAL NOT NULL,        -- recommended rate per hectare
    rate_unit                     TEXT NOT NULL,        -- 'ml', 'g', 'kg', 'l'
    pre_harvest_interval_days     INTEGER NOT NULL,     -- safety buffer period in days
    source_id                     TEXT NOT NULL,        -- citation source ID
    UNIQUE (product_name, crop)
);
