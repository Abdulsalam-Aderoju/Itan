"""Crop calendar lookup tool.

Matches the blueprint's Section 5.1 TOOLS contract exactly:

    'crop_calendar': (crop, agro_zone, year) -> {plant_window, harvest_window}

This is a Tier A (Section 3.3) tool: it returns one resolved "answer
card" -- a single plant_window and a single harvest_window -- not a raw
list of extracted rows, so the model only has to phrase the card, never
pick between candidates. `all_signals` is an Ìtàn-specific addition beyond
the blueprint's minimal signature: every raw row that fed the resolution,
kept for citation/debugging so the resolution logic itself is auditable.
`state` is likewise an addition -- the blueprint's table schema (4.3)
tracks it even though the tool signature in 5.1 doesn't mention it, and
much of this corpus's calendar data is state-level rather than
zone-level, so dropping it would throw away real signal.

Reads from the module-local crop_calendar.db built by build_db.py from
corpus/structured.db.

"No data for this crop/zone" is an expected, common outcome (corpus
coverage is sparse) -- returns plant_window/harvest_window as None rather
than raising. An unrecognized crop name IS an error, since that is a
caller mistake rather than a corpus gap.
"""
from dataclasses import dataclass, field
import json
from typing import Any

from engine.tools.crop_calendar.db import get_connection
from engine.tools.crop_calendar.errors import InvalidInputError
from corpus.common import CROPS, ZONES

PLANT_ACTIVITIES = ("planting_window", "plant_in_month")
HARVEST_ACTIVITIES = ("harvest_window", "harvest_in_month")


@dataclass(frozen=True)
class CropCalendarEvent:
    """A single raw planting/harvest-window data point, as extracted."""
    activity: str                # 'plant_in_month' | 'planting_window' | 'first_rains' | 'harvest_in_month'
    month_start: str
    month_end: str | None
    zone: str | None
    state: str | None
    confidence: float
    needs_review: bool
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity": self.activity,
            "month_start": self.month_start,
            "month_end": self.month_end,
            "zone": self.zone,
            "state": self.state,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class CalendarWindow:
    """A resolved plant_window or harvest_window answer card."""
    start_month: str
    end_month: str | None
    confidence: float
    needs_review: bool
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_month": self.start_month,
            "end_month": self.end_month,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class CropCalendarResult:
    """Result of a crop_calendar lookup."""
    crop: str
    agro_zone: str | None
    year: int | None
    state: str | None
    plant_window: CalendarWindow | None = None
    harvest_window: CalendarWindow | None = None
    all_signals: list[CropCalendarEvent] = field(default_factory=list)

    @property
    def data_available(self) -> bool:
        return self.plant_window is not None or self.harvest_window is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop": self.crop,
            "agro_zone": self.agro_zone,
            "year": self.year,
            "state": self.state,
            "data_available": self.data_available,
            "plant_window": self.plant_window.to_dict() if self.plant_window else None,
            "harvest_window": self.harvest_window.to_dict() if self.harvest_window else None,
            "all_signals": [e.to_dict() for e in self.all_signals],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _best_window(rows, activities: tuple[str, ...]) -> CalendarWindow | None:
    """Among `rows`, pick the highest-confidence row whose activity is in
    `activities` (checked in priority order), and build a CalendarWindow
    from it. Returns None if no row matches any of the activities."""
    for activity in activities:
        candidates = [r for r in rows if r["activity"] == activity]
        if candidates:
            best = max(candidates, key=lambda r: r["confidence"])
            return CalendarWindow(
                start_month=best["month_start"],
                end_month=best["month_end"],
                confidence=best["confidence"],
                needs_review=bool(best["needs_review"]),
                source_id=best["source_id"],
            )
    return None


def crop_calendar(
    crop: str,
    agro_zone: str | None = None,
    year: int | None = None,
    state: str | None = None,
) -> CropCalendarResult:
    """Resolve a crop's planting/harvest window for an agro-zone (+year).

    `year` is accepted for forward compatibility with the blueprint's
    signature, but no row in the current corpus carries a year, so it is
    NOT used to filter -- once year-versioned recommendations exist in
    the corpus, this will start narrowing by them. Until then, passing
    `year` has no effect on the result.

    If `agro_zone` or `state` is given, results are narrowed to rows
    matching it (rows with no recorded zone/state are excluded once a
    filter is applied, since they cannot be confirmed to match).

    Raises InvalidInputError if `crop` is not one of the 10 target crops,
    or if `agro_zone` is given but not one of the 6 recognized zones.
    Returns plant_window=None / harvest_window=None (data_available=False)
    if the crop is valid but the corpus has no calendar data for it -- as
    of 2026-08-17, 8 of the 10 crops have at least one row (missing:
    cowpea, pepper). 05_structure_extract.py does extract harvest_window
    (a dedicated "harvested between X and Y" pattern, mirroring the
    planting pattern), so it's populated wherever the corpus has that data
    -- e.g. crop_calendar("yam") resolves both plant_window and
    harvest_window today.
    """
    if not crop:
        raise InvalidInputError("Crop name cannot be empty.")
    clean_crop = crop.lower().strip()
    if clean_crop not in CROPS:
        raise InvalidInputError(
            f"Unrecognized crop '{crop}'. Must be one of: {', '.join(CROPS)}"
        )

    if agro_zone is not None and agro_zone not in ZONES:
        raise InvalidInputError(
            f"Unrecognized agro_zone '{agro_zone}'. Must be one of: {', '.join(ZONES)}"
        )

    conn = get_connection()
    try:
        query = "SELECT * FROM crop_calendar WHERE crop = ?"
        params: list[Any] = [clean_crop]
        if agro_zone is not None:
            query += " AND zone = ?"
            params.append(agro_zone)
        if state is not None:
            query += " AND state = ?"
            params.append(state.lower().strip())

        query += " ORDER BY id"
        rows = conn.execute(query, params).fetchall()

        plant_window = _best_window(rows, PLANT_ACTIVITIES)
        harvest_window = _best_window(rows, HARVEST_ACTIVITIES)

        all_signals = [
            CropCalendarEvent(
                activity=r["activity"],
                month_start=r["month_start"],
                month_end=r["month_end"],
                zone=r["zone"],
                state=r["state"],
                confidence=r["confidence"],
                needs_review=bool(r["needs_review"]),
                source_id=r["source_id"],
            )
            for r in rows
        ]

        return CropCalendarResult(
            crop=clean_crop,
            agro_zone=agro_zone,
            year=year,
            state=state,
            plant_window=plant_window,
            harvest_window=harvest_window,
            all_signals=all_signals,
        )
    finally:
        conn.close()
