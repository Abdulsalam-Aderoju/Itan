-- Database schema for the fertilizer_rate module.
--
-- Unlike agri_calc's tables (hand-curated ground truth), this table is
-- built by build_db.py from corpus/structured.db's auto-extracted
-- fertilizer_rate rows -- so confidence/needs_review are carried through
-- rather than curated away, letting callers decide how much to trust a row.
-- Only rows with BOTH fertilizer_type and rate_kg_ha populated are copied
-- here (see build_db.py) -- a bare number with no identified nutrient is
-- not independently useful, mirroring pest_lookup's empty-content gate.

CREATE TABLE IF NOT EXISTS fertilizer_rate (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    crop                  TEXT,                 -- nullable: some rows are crop-agnostic
    fertilizer_type       TEXT NOT NULL,        -- e.g. 'nitrogen' -- corpus coverage today is nitrogen-only
    rate_kg_ha            TEXT NOT NULL,
    application_stage     TEXT,                 -- nullable; in practice often holds a unit label
                                                  -- from the source regex (e.g. 'kg_n_slash_ha'),
                                                  -- not an actual growth-stage descriptor -- see
                                                  -- build_db.py module docstring
    confidence            REAL NOT NULL,
    needs_review           INTEGER NOT NULL DEFAULT 0,
    source_id             TEXT NOT NULL         -- citation source ID
);
