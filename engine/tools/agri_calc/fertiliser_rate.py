"""Fertiliser rate calculator module.

Calculates the required fertiliser products and application splits for a given
crop, area, zone, soil class, and target yield by querying the database.
"""
from dataclasses import dataclass
import sqlite3
from typing import Any

from engine.tools.agri_calc.db import get_connection
from engine.tools.agri_calc.errors import InvalidInputError, MissingConstantError
from engine.tools.agri_calc.constants import round_bags, round_seed_weight

@dataclass(frozen=True)
class Application:
    """Represents a single fertilizer split application."""
    timing_label: str          # e.g., "basal (at planting)" | "top-dress (6 WAP)"
    product_name: str
    product_kg: float          # rounded to 2 decimal places
    bags: int                   # rounded per policy (ceil)
    bags_exact: float           # unrounded, for audit

@dataclass(frozen=True)
class FertiliserRateResult:
    """Result of the fertiliser rate calculation."""
    crop: str
    area_ha: float
    zone: str
    matched_exactly: bool
    matched_soil_class: str | None
    matched_target_yield: float | None
    nutrient_totals_kg: dict[str, float]      # {"N": .., "P2O5": .., "K2O": ..} (actually supplied)
    unallocated_nutrients_kg: dict[str, float] # {"N": .., "P2O5": .., "K2O": ..} (recommended - supplied)
    applications: list[Application]            # one per split
    source_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "crop": self.crop,
            "area_ha": self.area_ha,
            "zone": self.zone,
            "matched_exactly": self.matched_exactly,
            "matched_soil_class": self.matched_soil_class,
            "matched_target_yield": self.matched_target_yield,
            "nutrient_totals_kg": self.nutrient_totals_kg,
            "unallocated_nutrients_kg": self.unallocated_nutrients_kg,
            "applications": [
                {
                    "timing_label": app.timing_label,
                    "product_name": app.product_name,
                    "product_kg": app.product_kg,
                    "bags": app.bags,
                    "bags_exact": app.bags_exact,
                }
                for app in self.applications
            ],
            "source_ids": self.source_ids,
        }

def fertiliser_rate(
    crop: str,
    area_ha: float,
    zone: str,
    soil_class: str | None = None,
    target_yield: float | None = None,
) -> FertiliserRateResult:
    """Calculate fertiliser application split schedules.

    Formula:
      product_kg_ha = (basis_nutrient_target * split_fraction) / (pct_concentration / 100)
      total_product_kg = product_kg_ha * area_ha
      bags = ceil(total_product_kg / product_bag_weight_kg)

    Reads from `fertilizer_rate`, `fertilizer_split`, and `product` SQLite tables.
    """
    # 1. Validation
    if not crop:
        raise InvalidInputError("Crop name cannot be empty.")
    if not zone:
        raise InvalidInputError("Zone name cannot be empty.")
    if area_ha <= 0:
        raise InvalidInputError(f"Area must be greater than zero. Received: {area_ha}")

    conn = get_connection()
    try:
        # 2. Lookup the crop first to see if it exists
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fertilizer_rate WHERE crop = ?", (crop,))
        if cursor.fetchone()[0] == 0:
            raise MissingConstantError(f"No fertilizer recommendation found for crop '{crop}'")

        # 3. Lookup the zone next within the crop
        cursor.execute("SELECT COUNT(*) FROM fertilizer_rate WHERE crop = ? AND zone = ?", (crop, zone))
        if cursor.fetchone()[0] == 0:
            raise MissingConstantError(f"No fertilizer recommendation found for crop '{crop}' in zone '{zone}'")

        # 4. Fetch all candidates for the crop and zone
        cursor.execute(
            "SELECT * FROM fertilizer_rate WHERE crop = ? AND zone = ?",
            (crop, zone)
        )
        candidates = cursor.fetchall()
        
        # Match-Ranking heuristic
        scored_candidates = []
        for candidate in candidates:
            # Soil class penalty
            if soil_class is not None:
                if candidate['soil_class'] == soil_class:
                    soil_penalty = 0
                elif candidate['soil_class'] is None:
                    soil_penalty = 1
                else:
                    soil_penalty = 2
            else:
                if candidate['soil_class'] is None:
                    soil_penalty = 0
                else:
                    soil_penalty = 1

            # Yield penalty
            if target_yield is not None:
                if candidate['target_yield'] is not None:
                    yield_penalty = abs(candidate['target_yield'] - target_yield)
                else:
                    yield_penalty = 1000000000.0  # Large penalty for missing yield config
            else:
                if candidate['target_yield'] is None:
                    yield_penalty = 0.0
                else:
                    yield_penalty = 1.0

            scored_candidates.append((soil_penalty, yield_penalty, candidate['id'], candidate))

        # Sort by soil penalty, then yield penalty, then ID (deterministic tie-breaker)
        scored_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        best_soil_penalty, best_yield_penalty, _, matched_row = scored_candidates[0]
        
        matched_exactly = (best_soil_penalty == 0 and best_yield_penalty == 0.0)

        # 5. Load Split Recipes
        cursor.execute(
            "SELECT * FROM fertilizer_split WHERE fertilizer_rate_id = ? ORDER BY split_number",
            (matched_row['id'],)
        )
        splits = cursor.fetchall()

        rec_N = matched_row['n_rate_kg_ha']
        rec_P = matched_row['p2o5_rate_kg_ha']
        rec_K = matched_row['k2o_rate_kg_ha']

        rem_N = rec_N
        rem_P = rec_P
        rem_K = rec_K

        supplied_N = 0.0
        supplied_P = 0.0
        supplied_K = 0.0

        applications = []
        source_ids_list = [matched_row['source_id']]

        # 6. Sequential Nutrient Netting
        for split in splits:
            source_ids_list.append(split['source_id'])
            
            # Lookup product concentration
            cursor.execute(
                "SELECT * FROM product WHERE product_name = ?",
                (split['product_name'],)
            )
            product_row = cursor.fetchone()
            if not product_row:
                raise MissingConstantError(f"Missing product constant for '{split['product_name']}'")
            
            source_ids_list.append(product_row['source_id'])

            basis = split['basis_nutrient']
            split_fraction = split['split_fraction']

            # Choose the remaining target based on basis nutrient
            if basis == 'N':
                target = rem_N
                pct = product_row['n_pct']
            elif basis == 'P2O5':
                target = rem_P
                pct = product_row['p2o5_pct']
            elif basis == 'K2O':
                target = rem_K
                pct = product_row['k2o_pct']
            else:
                raise InvalidInputError(f"Invalid basis nutrient: {basis}")

            if pct <= 0.0:
                raise MissingConstantError(
                    f"Product '{split['product_name']}' has 0% or negative concentration of basis nutrient '{basis}'"
                )

            # Clamp remaining target to 0 to prevent negative calculations (over-supply from previous splits)
            target = max(0.0, target)
            product_kg_ha = (target * split_fraction) / (pct / 100.0)
            total_product_kg = product_kg_ha * area_ha

            bag_weight = product_row['bag_weight_kg'] or 50.0
            bags_exact = total_product_kg / bag_weight
            bags = round_bags(bags_exact)

            # Update actual supplied totals and remaining needs
            n_supplied_ha = product_kg_ha * (product_row['n_pct'] / 100.0)
            p_supplied_ha = product_kg_ha * (product_row['p2o5_pct'] / 100.0)
            k_supplied_ha = product_kg_ha * (product_row['k2o_pct'] / 100.0)

            supplied_N += n_supplied_ha * area_ha
            supplied_P += p_supplied_ha * area_ha
            supplied_K += k_supplied_ha * area_ha

            rem_N -= n_supplied_ha
            rem_P -= p_supplied_ha
            rem_K -= k_supplied_ha

            applications.append(
                Application(
                    timing_label=split['timing'],
                    product_name=split['product_name'],
                    product_kg=round_seed_weight(total_product_kg),
                    bags=bags,
                    bags_exact=bags_exact,
                )
            )

        # 7. Formulate results and discrepancy outputs
        nutrient_totals = {
            "N": round_seed_weight(supplied_N),
            "P2O5": round_seed_weight(supplied_P),
            "K2O": round_seed_weight(supplied_K),
        }
        
        # Discrepancy: recommended rate total - actual supplied total
        unallocated = {
            "N": round_seed_weight((rec_N * area_ha) - supplied_N),
            "P2O5": round_seed_weight((rec_P * area_ha) - supplied_P),
            "K2O": round_seed_weight((rec_K * area_ha) - supplied_K),
        }

        # Keep original insertion order of sources, deduplicated
        source_ids = []
        for sid in source_ids_list:
            if sid not in source_ids:
                source_ids.append(sid)

        return FertiliserRateResult(
            crop=crop,
            area_ha=area_ha,
            zone=zone,
            matched_exactly=matched_exactly,
            matched_soil_class=matched_row['soil_class'],
            matched_target_yield=matched_row['target_yield'],
            nutrient_totals_kg=nutrient_totals,
            unallocated_nutrients_kg=unallocated,
            applications=applications,
            source_ids=source_ids,
        )

    finally:
        conn.close()
