# pest_lookup Module

Matches the Ìtàn blueprint's Section 5.1 `TOOLS` contract:

```
'pest_lookup' : (crop, symptom_terms[]) -> ranked [{pest, control, source}]
```

This is the tool Section 5.3's "guided pest triage" calls: once the agent
has narrowed a vague complaint ("my maize is dying") down to a crop plus a
short list of discriminating symptom terms via one structured follow-up
question, `pest_lookup(crop, symptom_terms)` deterministically ranks
candidate pests -- no model call, matching Pillar 2 ("tools, not
arithmetic"). Built on `corpus/structured.db`'s auto-extracted `pest`
table (see `corpus/05_structure_extract.py`).

## Ranking algorithm

Candidates are every record for `crop` **plus** every crop-generic record
(crop was `None` in the source text) -- a generic record can still be the
best symptom match even without crop-specific confirmation. For each
candidate, `match_score` = how many of the input `symptom_terms` appear
(case-insensitive substring) in `pest_name + symptoms + growth_stage`.
Chemical/cultural control text is deliberately excluded from matching --
it describes treatment, not symptoms, and would produce false positives
(e.g. "spray" matching almost every record). Only records with
`match_score >= 1` are returned, sorted by `(-match_score, -confidence)`.

## Why canonicalization is required upstream

The raw extractor's `PEST_NAME_RE` captures the noun phrase surrounding a
pest keyword, not a clean name -- the same real pest shows up as
`"usually attacked by thrips"`, `"Symptoms of thrips"`, `"season thrips"`
across different rows. `normalize.py` reduces any such phrase to the
single keyword it was matched against (from `PEST_KEYWORDS`: weevil,
borer, aphid, nematode, striga, armyworm, thrips, mealybug, mite, beetle,
caterpillar, worm, whitefly, moth, blight, rust, wilt, mosaic virus,
anthracnose, smut, rot, leaf spot), and `build_db.py` groups the ~12,200
raw rows by (canonical pest name, crop) into 183 clean records before
`pest_lookup` ever runs its symptom matching over them.

## Data pipeline

Rebuild after re-running the corpus pipeline:
```
python engine/tools/pest_lookup/build_db.py
```

## Coverage

All 10 target crops have pest data (maize 1798 raw rows, cassava 1124,
sorghum 850, groundnut 774, yam 626, tomato 482, cowpea 324, rice 274,
soybean 184, pepper 130, as of this writing).

## "Not found" behavior

- `crop` outside the 10 target crops is a caller mistake: raises
  `InvalidInputError`.
- `symptom_terms` empty, or containing only blank strings, is rejected the
  same way -- triage must supply at least one discriminating term.
- Zero matches for valid input is a legitimate outcome (wrong symptom
  vocabulary, or a genuine corpus gap): returns `matches: []` and
  `data_available: false`, not an error.

## Secondary capability: exact lookup by name

`pest_lookup_by_name(pest_name, crop=None)` is kept alongside the
blueprint's primary tool -- it isn't part of the registered `TOOLS` dict,
but it's the original interface built before the blueprint was available,
and it's still useful for direct programmatic use (e.g. once triage has
already identified "thrips" by name, skip re-ranking). See its docstring
in `pest_lookup.py` for the crop-specific-then-generic fallback behavior.

## Usage

```python
from engine.tools.pest_lookup import pest_lookup

result = pest_lookup(crop="groundnut", symptom_terms=["yellowing", "dwarfing"])
print(result.to_json())
```

```json
{
  "crop": "groundnut",
  "symptom_terms": ["yellowing", "dwarfing"],
  "data_available": true,
  "matches": [
    {
      "pest": "thrips",
      "match_score": 1,
      "matched_terms": ["dwarfing"],
      "symptoms": "of thrips damage include dwarfing and malformation of leaves ...",
      "growth_stage": "maturity",
      "cultural_control": "",
      "chemical_control": "",
      "confidence": 0.7,
      "needs_review": false,
      "source_count": 40,
      "source_ids": ["RAW_..."]
    }
  ]
}
```
