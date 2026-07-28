"""Universal conversion constants and rounding policies for the agri_calc module.

No crop-specific or zone-specific agronomic constants should be hardcoded here.
All agronomic data must be queried from the database.
"""
import math
from decimal import Decimal, ROUND_HALF_UP

# Conversion factors
SQ_M_PER_HA = 10000.0

# Rounding configuration
ROUNDING_MODE = "default"

def round_bags(bags: float) -> int:
    """Round up to the next whole bag (ceil) per policy."""
    return math.ceil(round(bags, 9))

def round_seed_weight(weight: float) -> float:
    """Round seed weight to 2 decimal places per policy."""
    d = Decimal(str(round(weight, 9)))
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def round_cuttings(cuttings: float) -> int:
    """Round cutting/sett counts up to next whole unit per policy."""
    return math.ceil(round(cuttings, 9))

def round_dilution(amount: float) -> int:
    """Round dilution amount (ml or g) to the nearest whole unit per policy."""
    d = Decimal(str(round(amount, 9)))
    return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def round_currency(amount: float) -> int:
    """Round currency amount to nearest whole unit of currency per policy."""
    d = Decimal(str(round(amount, 9)))
    return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
