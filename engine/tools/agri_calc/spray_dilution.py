"""Agrochemical spray dilution calculator module.

Calculates the amount of agrochemical product (ml or g) to mix per spray tank
based on the application rate per hectare, tank size, and carrier volume.
Also retrieves the safety Pre-Harvest Interval (PHI) in days.
"""
from dataclasses import dataclass
import sqlite3
from typing import Any

from engine.tools.agri_calc.db import get_connection
from engine.tools.agri_calc.errors import InvalidInputError, MissingConstantError
from engine.tools.agri_calc.constants import round_dilution

@dataclass(frozen=True)
class SprayDilutionResult:
    """Result of the spray dilution calculation."""
    product_name: str
    crop: str | None
    amount_per_tank: float              # rounded to nearest whole unit (ml or g)
    unit: str                            # "ml" | "g"
    tank_litres: float
    spray_volume_l_per_ha: float
    pre_harvest_interval_days: int
    source_ids: list[str]
    conc_pct: float | None = None                    # active-ingredient % supplied by caller, if any
    active_ingredient_per_tank: float | None = None   # amount_per_tank's a.i. share, same unit; None if conc_pct not given

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "product_name": self.product_name,
            "crop": self.crop,
            "amount_per_tank": self.amount_per_tank,
            "unit": self.unit,
            "tank_litres": self.tank_litres,
            "spray_volume_l_per_ha": self.spray_volume_l_per_ha,
            "pre_harvest_interval_days": self.pre_harvest_interval_days,
            "source_ids": self.source_ids,
            "conc_pct": self.conc_pct,
            "active_ingredient_per_tank": self.active_ingredient_per_tank,
        }

def spray_dilution(
    product_name: str,
    tank_litres: float,
    rate_per_ha: float | None = None,
    spray_volume_l_per_ha: float = 200.0,
    crop: str | None = None,
    conc_pct: float | None = None,
) -> SprayDilutionResult:
    """Calculate agrochemical dilution per tank.

    Formula:
      amount_per_tank = rate_per_ha * (tank_litres / spray_volume_l_per_ha)

    If rate_per_ha unit is kg or L, converts to g or ml respectively.
    Reads from `agrochemical` SQLite table.

    conc_pct (blueprint Section 5.1's required parameter) is the product's
    declared active-ingredient concentration, e.g. a label reading "glyphosate
    41% SL" is conc_pct=41. rate_per_ha in this corpus is already a
    formulated-product rate (confirmed against every populated agrochemical
    row and the existing SD-001..004 gold cases, all in L/ha or kg/ha of
    product, never a.i.-per-ha), so conc_pct does not change amount_per_tank
    -- it is used only to additionally report how much of that amount is
    pure active ingredient (active_ingredient_per_tank), which is what a
    label's "% a.i." figure is actually for. When omitted, behaviour and
    output are unchanged from before this parameter existed.
    """
    # 1. Validation
    if not product_name:
        raise InvalidInputError("Product name cannot be empty.")
    if tank_litres <= 0:
        raise InvalidInputError(f"Tank volume must be greater than zero. Received: {tank_litres}")
    if spray_volume_l_per_ha <= 0:
        raise InvalidInputError(f"Spray volume per hectare must be greater than zero. Received: {spray_volume_l_per_ha}")
    if rate_per_ha is not None and rate_per_ha <= 0:
        raise InvalidInputError(f"User-overridden rate per hectare must be greater than zero. Received: {rate_per_ha}")
    if conc_pct is not None and not (0 < conc_pct <= 100):
        raise InvalidInputError(f"Concentration percent must be between 0 and 100. Received: {conc_pct}")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Query matching agrochemical products
        cursor.execute(
            "SELECT * FROM agrochemical WHERE product_name = ?",
            (product_name,)
        )
        rows = cursor.fetchall()
        if not rows:
            raise MissingConstantError(f"No agrochemical record found for product '{product_name}'")

        # Select best match (crop-specific > general/none)
        matched_row = None
        if crop:
            clean_crop = crop.lower().strip()
            for row in rows:
                if row['crop'] and row['crop'].lower().strip() == clean_crop:
                    matched_row = row
                    break
        
        if matched_row is None:
            # Fallback to general recommendation (crop is null or empty)
            for row in rows:
                if row['crop'] is None or row['crop'] == '':
                    matched_row = row
                    break

        if matched_row is None:
            # Fallback to first available row if no generic row exists
            matched_row = rows[0]

        db_rate = matched_row['rate_per_ha']
        rate_unit = matched_row['rate_unit'].lower().strip()
        phi = matched_row['pre_harvest_interval_days']
        source_id = matched_row['source_id']

        # Determine rate to use
        active_rate = rate_per_ha if rate_per_ha is not None else db_rate

        # Calculate raw amount per tank
        amount_raw = active_rate * (tank_litres / spray_volume_l_per_ha)

        # Normalize units and handle conversions (kg -> g, L -> ml)
        # Unit normalization:
        # 'kg' -> 'g' (multiply by 1000)
        # 'l' (litre) -> 'ml' (multiply by 1000)
        # 'g' -> 'g'
        # 'ml' -> 'ml'
        if rate_unit == 'kg':
            amount_raw *= 1000.0
            final_unit = 'g'
        elif rate_unit in ('l', 'litre', 'litres'):
            amount_raw *= 1000.0
            final_unit = 'ml'
        elif rate_unit == 'g':
            final_unit = 'g'
        elif rate_unit == 'ml':
            final_unit = 'ml'
        else:
            # Fallback to whatever unit is documented, but carry through
            final_unit = rate_unit

        # Apply dilution rounding policy
        amount_per_tank = float(round_dilution(amount_raw))

        # Optional: how much of amount_per_tank is active ingredient
        active_ingredient_per_tank = (
            float(round_dilution(amount_per_tank * (conc_pct / 100.0)))
            if conc_pct is not None else None
        )

        # Crop is recorded as the matched crop in the DB (or input crop)
        result_crop = matched_row['crop'] if matched_row['crop'] else crop

        return SprayDilutionResult(
            product_name=product_name,
            crop=result_crop,
            amount_per_tank=amount_per_tank,
            unit=final_unit,
            tank_litres=tank_litres,
            spray_volume_l_per_ha=spray_volume_l_per_ha,
            pre_harvest_interval_days=phi,
            source_ids=[source_id],
            conc_pct=conc_pct,
            active_ingredient_per_tank=active_ingredient_per_tank,
        )

    finally:
        conn.close()
