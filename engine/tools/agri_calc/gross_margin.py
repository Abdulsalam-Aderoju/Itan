"""Gross margin calculator module.

Calculates the revenue, total costs, and net profit margin for crop sales,
maintaining itemized unrounded input cost lists to prevent rounding drift.
"""
from dataclasses import dataclass
from typing import Any

from engine.tools.agri_calc.errors import InvalidInputError
from engine.tools.agri_calc.constants import round_currency

@dataclass(frozen=True)
class CostItem:
    """Represents a single itemized cost entry (e.g. fertilizer, seeds, labor)."""
    label: str
    amount: float                       # exact unrounded input cost

@dataclass(frozen=True)
class GrossMarginResult:
    """Result of the gross margin calculation."""
    yield_kg: float
    price_per_kg: float
    revenue: float                      # rounded to nearest integer of currency
    total_cost: float                   # rounded to nearest integer of currency
    margin: float                       # rounded to nearest integer of currency
    input_costs: list[CostItem]          # identical list of inputs (unrounded)
    currency: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "yield_kg": self.yield_kg,
            "price_per_kg": self.price_per_kg,
            "revenue": self.revenue,
            "total_cost": self.total_cost,
            "margin": self.margin,
            "input_costs": [
                {"label": item.label, "amount": item.amount}
                for item in self.input_costs
            ],
            "currency": self.currency,
        }

def gross_margin(
    yield_kg: float,
    price_per_kg: float,
    input_costs: list[CostItem],
    currency: str = "NGN",
) -> GrossMarginResult:
    """Calculate gross margins and profits.

    Formulas:
      revenue = yield_kg * price_per_kg
      total_cost = sum(item.amount for item in input_costs)
      margin = revenue - total_cost

    Revenue, total_cost, and margin are rounded to nearest whole unit of currency.
    """
    # 1. Validation
    if yield_kg < 0:
        raise InvalidInputError(f"Yield cannot be negative. Received: {yield_kg}")
    if price_per_kg < 0:
        raise InvalidInputError(f"Price per kg cannot be negative. Received: {price_per_kg}")
    
    for item in input_costs:
        if item.amount < 0:
            raise InvalidInputError(f"Cost amount cannot be negative for '{item.label}'. Received: {item.amount}")

    # 2. Arithmetic
    revenue_raw = yield_kg * price_per_kg
    total_cost_raw = sum(item.amount for item in input_costs)
    margin_raw = revenue_raw - total_cost_raw

    # 3. Currency Rounding
    revenue = float(round_currency(revenue_raw))
    total_cost = float(round_currency(total_cost_raw))
    margin = float(round_currency(margin_raw))

    return GrossMarginResult(
        yield_kg=yield_kg,
        price_per_kg=price_per_kg,
        revenue=revenue,
        total_cost=total_cost,
        margin=margin,
        input_costs=input_costs,
        currency=currency,
    )
