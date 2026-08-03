-- Database schema for the pest_lookup module.
--
-- Built by build_db.py from corpus/structured.db's auto-extracted `pest`
-- table, grouped by (canonical pest_name, crop) after canonicalization
-- (see normalize.py) -- so this table is small and exact-matchable, unlike
-- the raw structured.db table where the same real pest appears under many
-- noisy phrase variants.

CREATE TABLE IF NOT EXISTS pest (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    pest_name             TEXT NOT NULL,        -- canonical keyword, e.g. 'thrips', 'leaf spot'
    crop                  TEXT,                 -- NULL if not tied to a specific crop in the source text
    symptoms              TEXT,
    growth_stage          TEXT,
    cultural_control      TEXT,
    chemical_control      TEXT,
    confidence            REAL NOT NULL,        -- max confidence among the raw rows merged into this record
    needs_review          INTEGER NOT NULL DEFAULT 0,
    source_count          INTEGER NOT NULL,     -- how many raw structured.db rows were merged into this record
    source_ids            TEXT NOT NULL,        -- JSON array of contributing source_id citations
    UNIQUE (pest_name, crop)
);
