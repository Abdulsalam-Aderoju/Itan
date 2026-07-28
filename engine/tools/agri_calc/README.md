# agri_calc Module

This module implements the four core offline arithmetic calculators used by the agricultural extension agent: `fertiliser_rate`, `seed_rate`, `spray_dilution`, and `gross_margin`. 

## Rounding Policy

Rounding rules are isolated in `constants.py` under the `ROUNDING_MODE = "default"` configuration. Calculations are protected from floating-point representation noise by performing intermediate rounding to 9 decimal places before final quantization.

| Quantity | Rule | Rationale |
| :--- | :--- | :--- |
| **Bag counts** (`fertiliser_rate`) | Round up to the next whole bag (`ceil`), and separately report the exact unrounded figure | Farmers cannot buy fractional bags; under-buying is unsafe. |
| **Seed weight, kg** (`seed_rate`) | Round to 2 decimal places | Seed is sold by weight; fine granularity is meaningful. |
| **Cuttings/setts** (`seed_rate`) | Round up to next whole unit (`ceil`) | Farmers cannot plant fractional cuttings. |
| **Dilution amount** (`spray_dilution`) | Round to nearest whole unit (round-half-up) | Matches practical measuring-cup precision. |
| **Currency amounts** (`gross_margin`) | Round to nearest whole currency unit (round-half-up) | No sub-unit precision needed for margin estimation. |

---

## Function Formulas

### 1. fertiliser_rate
Determines the recommendation matches in the database using a strict match-ranking penalty check on optional `soil_class` and `target_yield` variables (raising errors on crop/zone mismatches). It then nets out requirements sequentially across split applications:
$$\text{product\_kg\_ha} = \frac{\text{basis\_nutrient\_target} \times \text{split\_fraction}}{\text{nutrient\_concentration\_pct} / 100}$$
$$\text{total\_product\_kg} = \text{product\_kg\_ha} \times \text{area\_ha}$$
$$\text{bags} = \lceil \frac{\text{total\_product\_kg}}{\text{bag\_weight\_kg}} \rceil$$
Nutrients supplied by earlier splits are deducted from subsequent split targets to avoid double counting. Over- or under-allocations are reported in the `unallocated_nutrients_kg` result dictionary.

### 2. seed_rate
Calculates planting materials. It first looks up the crop's material type in the database. For true seed crops (`material_type = 'seed'`), it calculates plant population, adjusts it for germination rate, and multiplies by the 1000-seed weight:
$$\text{stands\_per\_ha} = \frac{10000.0}{\text{row\_m} \times \text{within\_row\_m}}$$
$$\text{seed\_kg} = \frac{\text{stands\_per\_ha} \times \text{seeds\_per\_stand} \times \text{area\_ha} \times (\text{unit\_weight\_g} / 1000) / 1000}{\text{germination\_pct} / 100}$$
For vegetative crops (`material_type` in `('cutting', 'sett')`), it multiplies plant population by the cuttings per stand, and converts it to bundles if a standard bundle size is documented:
$$\text{cuttings} = \text{stands\_per\_ha} \times \text{area\_ha} \times \text{stands\_per\_unit}$$
$$\text{bundles} = \lceil \frac{\text{cuttings}}{\text{units\_per\_bundle}} \rceil$$

### 3. spray_dilution
Calculates chemical dilution rates for knapsack sprayers. It reads recommendation rates and safety Pre-Harvest Intervals (PHI) from the database, then calculates the concentration needed per tank:
$$\text{amount\_per\_tank} = \text{rate\_per\_ha} \times \frac{\text{tank\_litres}}{\text{spray\_volume\_l\_per\_ha}}$$
If the database rate unit is `kg` or `L`, it multiplies the output by 1000 to convert to grams (`g`) or milliliters (`ml`) respectively to match measuring-cup precision.

### 4. gross_margin
Performs a financial analysis based on yield and price. To prevent rounding error accumulation, cost item details are kept as unrounded floats, and rounding is applied only to the final aggregates:
$$\text{revenue} = \text{yield\_kg} \times \text{price\_per\_kg}$$
$$\text{total\_cost} = \sum \text{cost\_item.amount}$$
$$\text{margin} = \text{revenue} - \text{total\_cost}$$
All three outputs are rounded to the nearest whole unit of the specified currency.
