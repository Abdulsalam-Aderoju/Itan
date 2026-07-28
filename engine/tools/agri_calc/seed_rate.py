"""Seed rate calculator module.

Calculates the required seed quantity (in kg) or planting material count (cuttings/setts/bundles)
for a given crop, area, spacing, and germination percentage by querying the database.
"""
from dataclasses import dataclass
import sqlite3
from typing import Any

from engine.tools.agri_calc.db import get_connection
from engine.tools.agri_calc.errors import InvalidInputError, MissingConstantError
from engine.tools.agri_calc.constants import round_seed_weight, round_cuttings

@dataclass(frozen=True)
class SeedRateResult:
    """Result of the seed rate calculation."""
    crop: str
    area_ha: float
    spacing_cm: tuple[float, float]
    germination_pct: float
    seeds_per_stand: int
    material_type: str                  # 'seed' | 'cutting' | 'sett'
    plant_population: float             # total stands/plants for the area
    source_ids: list[str]
    seed_kg: float | None = None        # only for 'seed' (rounded to 2 decimal places)
    cuttings_count: int | None = None   # only for 'cutting'/'sett' (rounded up)
    cuttings_exact: float | None = None # unrounded count, for audit
    bundles: int | None = None          # only if units_per_bundle is set (rounded up)
    bundles_exact: float | None = None  # unrounded bundle count, for audit

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "crop": self.crop,
            "area_ha": self.area_ha,
            "spacing_cm": list(self.spacing_cm),
            "germination_pct": self.germination_pct,
            "seeds_per_stand": self.seeds_per_stand,
            "material_type": self.material_type,
            "plant_population": self.plant_population,
            "seed_kg": self.seed_kg,
            "cuttings_count": self.cuttings_count,
            "cuttings_exact": self.cuttings_exact,
            "bundles": self.bundles,
            "bundles_exact": self.bundles_exact,
            "source_ids": self.source_ids,
        }

def seed_rate(
    crop: str,
    area_ha: float,
    spacing_cm: tuple[float, float],
    germination_pct: float,
    seeds_per_stand: int = 1,
) -> SeedRateResult:
    """Calculate seed weight or cutting count requirements.

    Formulas:
      stands_per_ha = 10,000 / (row_m * within_row_m)
      plant_population = stands_per_ha * area_ha

      For seeds:
        seed_kg = (plant_population * seeds_per_stand * (1000_seed_weight_g / 1000) / 1000) / (germination_pct / 100)

      For cuttings/setts:
        cuttings = plant_population * cuttings_per_stand (stands_per_unit)
        bundles = cuttings / units_per_bundle

    Reads from `planting_material` SQLite table.
    """
    # 1. Validation
    if not crop:
        raise InvalidInputError("Crop name cannot be empty.")
    if area_ha <= 0:
        raise InvalidInputError(f"Area must be greater than zero. Received: {area_ha}")
    if len(spacing_cm) != 2 or spacing_cm[0] <= 0 or spacing_cm[1] <= 0:
        raise InvalidInputError(f"Spacing must be positive (row_cm, within_row_cm). Received: {spacing_cm}")
    if germination_pct <= 0 or germination_pct > 100:
        raise InvalidInputError(f"Germination percentage must be in the range (0, 100]. Received: {germination_pct}")
    if seeds_per_stand <= 0:
        raise InvalidInputError(f"Seeds per stand must be at least 1. Received: {seeds_per_stand}")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM planting_material WHERE crop = ?",
            (crop,)
        )
        row = cursor.fetchone()
        if not row:
            raise MissingConstantError(f"No planting material record found for crop '{crop}'")

        material_type = row['material_type']
        source_id = row['source_id']

        # Spacing calculations
        row_cm, within_row_cm = spacing_cm
        row_m = row_cm / 100.0
        within_row_m = within_row_cm / 100.0

        # Stands per hectare
        stands_per_ha = 10000.0 / (row_m * within_row_m)
        plant_population = stands_per_ha * area_ha

        if material_type == 'seed':
            unit_weight_g = row['unit_weight_g']
            if unit_weight_g is None or unit_weight_g <= 0:
                raise MissingConstantError(
                    f"Seed weight (unit_weight_g) is missing or invalid for seed crop '{crop}'"
                )

            # seed weight arithmetic (avoiding 1000x scaling bugs)
            # 1. Target seed count (before germination adjustment)
            seed_count_base = plant_population * seeds_per_stand
            
            # 2. Adjust for germination rate (germination_pct is in percent, e.g. 85.0)
            seed_count_needed = seed_count_base / (germination_pct / 100.0)

            # 3. unit_weight_g is weight of 1000 seeds in grams
            # Weight of 1 seed in grams = unit_weight_g / 1000.0
            # Weight of 1 seed in kg = unit_weight_g / 1000.0 / 1000.0 = unit_weight_g / 1,000,000.0
            weight_per_seed_kg = unit_weight_g / 1000000.0

            # 4. Total seed weight in kg
            seed_kg_raw = seed_count_needed * weight_per_seed_kg
            seed_kg = round_seed_weight(seed_kg_raw)

            return SeedRateResult(
                crop=crop,
                area_ha=area_ha,
                spacing_cm=spacing_cm,
                germination_pct=germination_pct,
                seeds_per_stand=seeds_per_stand,
                material_type=material_type,
                plant_population=plant_population,
                seed_kg=seed_kg,
                source_ids=[source_id],
            )

        elif material_type in ('cutting', 'sett'):
            stands_per_unit = row['stands_per_unit'] or 1
            cuttings_exact = plant_population * stands_per_unit
            cuttings_count = round_cuttings(cuttings_exact)

            bundles_exact = None
            bundles = None
            units_per_bundle = row['units_per_bundle']
            if units_per_bundle is not None and units_per_bundle > 0:
                bundles_exact = cuttings_exact / units_per_bundle
                bundles = round_cuttings(bundles_exact)

            return SeedRateResult(
                crop=crop,
                area_ha=area_ha,
                spacing_cm=spacing_cm,
                germination_pct=germination_pct,
                seeds_per_stand=seeds_per_stand,
                material_type=material_type,
                plant_population=plant_population,
                cuttings_count=cuttings_count,
                cuttings_exact=cuttings_exact,
                bundles=bundles,
                bundles_exact=bundles_exact,
                source_ids=[source_id],
            )
        else:
            raise MissingConstantError(f"Unsupported material type '{material_type}' for crop '{crop}'")

    finally:
        conn.close()
