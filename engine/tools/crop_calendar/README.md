# crop_calendar Module

Matches the Ìtàn blueprint's Section 5.1 `TOOLS` contract:

```
'crop_calendar': (crop, agro_zone, year) -> {plant_window, harvest_window}
```

This is a Tier A tool (Section 3.3 of the blueprint): it returns one
resolved answer card, not a list the model has to pick between. Built on
`corpus/structured.db`'s auto-extracted `crop_calendar` table (see
`corpus/05_structure_extract.py`), rather than hand-curated like
`agri_calc`'s tables -- so the underlying rows carry their original
`confidence` / `needs_review` flags, surfaced on the resolved window and
in `all_signals`.

## Resolution logic

Given (crop, agro_zone, state), the tool queries every matching row, then
resolves:
- **plant_window**: the highest-confidence row with activity
  `planting_window` (has both start and end month) if one exists;
  otherwise the highest-confidence `plant_in_month` row (start month
  only). A `first_rains` row is never used to construct plant_window on
  its own -- it's an ecological cue correlated with planting, not a
  planting instruction.
- **harvest_window**: same priority, but `05_structure_extract.py` has no
  harvest-window extraction pattern yet, so this is `None` for every crop
  today.
- **all_signals**: every raw matching row (Ìtàn-specific addition beyond
  the blueprint's minimal signature), so the resolution above is
  auditable and citable even for signals that didn't win (e.g. a
  `first_rains` row still shows up here).

## Data pipeline

`build_db.py` copies rows from `corpus/structured.db` into this module's
own `crop_calendar.db`, deduping exact repeats. Rebuild after re-running
the corpus pipeline:
```
python engine/tools/crop_calendar/build_db.py
```

## Known coverage gap

`corpus/structured.db`'s `crop_calendar` table has **4 rows total,
covering only 1 of the 10 target crops (yam)**. The other 9 crops have
zero calendar rows -- a sourcing gap, not a bug in this tool or the
extraction regex. Blueprint Section 4.3 flags Tier A tables as the
highest-value, most-scrutinized work; this table needs a targeted harvest
pass, not more code.

## Behavior on missing data

"No data for this crop/zone" is expected and common here -- `plant_window`
and `harvest_window` come back `None` rather than raising.
**Unrecognized crop name** (not one of the 10 target crops) or
**unrecognized agro_zone** IS a caller error and raises `InvalidInputError`.

`year` is accepted for forward compatibility with the blueprint's
signature but is currently a no-op: no row in the corpus carries a year,
so it does not filter anything -- it's simply echoed back in the result.

## Usage

```python
from engine.tools.crop_calendar import crop_calendar

result = crop_calendar(crop="yam")
print(result.to_json())
```

```json
{
  "crop": "yam",
  "agro_zone": null,
  "year": null,
  "state": null,
  "data_available": true,
  "plant_window": {
    "start_month": "December",
    "end_month": null,
    "confidence": 0.85,
    "needs_review": false,
    "source_id": "RAW_4BF42E261279_109c4015"
  },
  "harvest_window": null,
  "all_signals": [
    {
      "activity": "plant_in_month",
      "month_start": "December",
      "month_end": null,
      "zone": null,
      "state": null,
      "confidence": 0.85,
      "needs_review": false,
      "source_id": "RAW_4BF42E261279_109c4015"
    }
  ]
}
```
