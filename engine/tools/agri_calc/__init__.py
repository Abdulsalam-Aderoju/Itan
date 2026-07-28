"""agri_calc public exports.

Exposes the four core calculator functions, required input structures,
and exception classes for error boundaries.
"""

from engine.tools.agri_calc.fertiliser_rate import fertiliser_rate
from engine.tools.agri_calc.seed_rate import seed_rate
from engine.tools.agri_calc.spray_dilution import spray_dilution
from engine.tools.agri_calc.gross_margin import gross_margin, CostItem
from engine.tools.agri_calc.errors import AgriCalcError, InvalidInputError, MissingConstantError

__all__ = [
    "fertiliser_rate",
    "seed_rate",
    "spray_dilution",
    "gross_margin",
    "CostItem",
    "AgriCalcError",
    "InvalidInputError",
    "MissingConstantError",
]
