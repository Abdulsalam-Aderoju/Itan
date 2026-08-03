"""SQLite database connection helper for the crop_calendar module.

Mirrors engine/tools/agri_calc/db.py: connections are not persisted
globally, and row outputs use standard dictionary-like mapping interfaces.
"""
import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "crop_calendar.db"

def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Return a new SQLite database connection.

    Row factory is set to sqlite3.Row for column-name accessibility.
    """
    if db_path is None:
        db_path = os.environ.get("CROP_CALENDAR_DB_PATH", str(DEFAULT_DB_PATH))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
