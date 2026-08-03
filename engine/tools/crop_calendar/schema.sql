-- Database schema for the crop_calendar module.
--
-- Unlike agri_calc's tables (hand-curated ground truth), this table is
-- built by build_db.py directly from corpus/structured.db's auto-extracted
-- crop_calendar rows -- so confidence/needs_review are carried through
-- rather than curated away, letting callers decide how much to trust a row.

CREATE TABLE IF NOT EXISTS crop_calendar (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    crop                  TEXT NOT NULL,
    zone                  TEXT,                 -- agro-ecological zone; nullable, not always extractable
    state                 TEXT,                 -- Nigerian state; nullable
    activity              TEXT NOT NULL,        -- 'plant_in_month' | 'planting_window' | 'first_rains' | 'harvest_in_month'
    month_start           TEXT NOT NULL,
    month_end             TEXT,                 -- nullable, only set for window-style activities
    confidence            REAL NOT NULL,
    needs_review          INTEGER NOT NULL DEFAULT 0,
    source_id             TEXT NOT NULL         -- citation source ID
);
