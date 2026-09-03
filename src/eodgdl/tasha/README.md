# eodgdl.tasha

The travel-demand model (TASHA / GTAModel style) consumes three tables —
`od_households.csv`, `od_people.csv`, `od_trips.csv` — with English column names
and single-letter category codes. This subpackage holds the definition of those
tables, the mapping from the Guadalajara EOD 2023 survey onto them, and a
checker for both.

```
model_schema.yaml   the contract   — what the tables must contain, which codes
                                     are legal, what each one means.
mappings.yaml       the mapping    — one entry per output column: the eodgdl
                                     source column, the {raw answer: code}
                                     lookup, and the caveats.
_schema.py          the glue       — loads both, hands you .map()-ready dicts,
                                     checks the mappings against the contract
                                     and against the survey's own schemas, and
                                     checks a produced table against it.
build.py            the builder    — turns the cleaned survey into the three
                                     tables, driven by mappings.yaml.
```

The split is the point. `model_schema.yaml` changes only when the *model's*
requirements change; `mappings.yaml` changes whenever a coding decision does, and
is meant to be read and edited by hand. Both YAML files ship inside the package,
so everything here works offline and from an installed wheel.

## Generating the files

```python
from eodgdl import load_eod, tasha

od = tasha.build(load_eod("data"))       # ODTables(households, people, trips)
assert tasha.validate_all(*od) == []
od.trips.to_csv("output/od_trips.csv", index=False)
```

or in one step:

```bash
eodgdl tasha build --data data/ --out output/    # builds, writes, and validates
```

The builder reads every lookup out of `mappings.yaml`, so editing a mapping
changes the output without touching `build.py`. It expects the tables as
`load_eod` returns them, trip chains already cleaned by
`eodgdl.eod.clean_trip_chains` — the 281 persons with an untimed trip
excluded, 84 mislabelled returns recoded, 468 returns home made from home
dropped, 120 trips after a return home made to start at home, 2,029
mistyped start hours repaired and flagged in `hora_inicio_ajuste` — and refuses
a trip table with untimed rows. What that cleaning does not repair, `tasha.chain_report(od.trips)`
counts; see "Validating" below.

Zone columns hold the survey's own AGEB CVEGEO or locality id as a **string**.
Read them back with `dtype=str` — `tasha.zone_columns(table)` lists them —
since most ids are all-digit and will otherwise parse as `int64`.

The trip column is spelled **`PurposeOrigin`**. The downstream model misspells it
`PuposeOrigin`; that typo belongs to the consumer, so alias it there or rename
the column after the build — never here.

## Reading the mapping

```python
from eodgdl import tasha

tasha.domain("PurposeDestination")   # {'H': 'Home', 'W': 'Work — first…', …}
tasha.mapping("Mode")                # source column, lookup, override, notes
tasha.build_map("Mode")              # {'A PIE': 'W', 'CAMIÓN O AUTOBÚS': 'B', …}
tasha.gaps()                         # what is assumed, constant, or unresolved
```

`build_map` returns only the literal lookup. Everything a lookup cannot express
— a fall-through default, a rank over a person's trips, a constant, an override
from a second column — is prose under `derivation`, `override` or `constant` in
the mapping entry. Read the entry before writing the transformation; the notes
carry the lossy collapses and the judgement calls.

## Building a column

```python
from eodgdl import load_eod, tasha

viv, hab, trips, legs = load_eod("data")

people["Occupation"] = (
    hab.giro_empresa.map(tasha.build_map("Occupation"))
       .fillna(tasha.mapping("Occupation")["default"])
       .mask(people.EmploymentStatus == "O", "O")   # per the entry's derivation
)
```

Columns whose name appears in more than one table (`HouseholdId`,
`PersonNumber`, `ExpansionFactor`) need `table=` to disambiguate.

## Editing a mapping

Edit `mappings.yaml` directly — it is the source, not build output. Each entry
takes:

| key          | meaning                                                    |
|--------------|------------------------------------------------------------|
| `source`     | raw eodgdl column(s) the value comes from                  |
| `values`     | `{raw answer: output code}` — a `.map()`-ready lookup      |
| `default`    | code for anything not listed in `values`                    |
| `constant`   | the whole column is this one value                          |
| `derivation` | prose, for what a lookup cannot express                     |
| `override`   | a second lookup applied on top of `values`, with its `when` |
| `note`       | caveats: lossy collapses, judgement calls, open questions   |
| `status`     | `ok` (default) / `assumed` / `not_surveyed` / `pending`     |

Then re-run the checker. It reads the mapping in both directions — against the
contract, and against the survey the mapping claims to read:

| direction | what it catches |
|---|---|
| vs. `model_schema.yaml` | a column with no mapping; a mapping for a column the contract does not have; a code the column's domain does not allow |
| vs. the pandera schemas | a `source` that is no longer a column of `viv`/`hab`/`trips`/`legs`; a `values` key that is not an answer that column can take |

The second direction matters because both failures are silent at build time: a
renamed source column and a typo'd Spanish answer both come out of `.map()` as
NaN, not as an error. `tasha.survey_columns()` is the table it checks against.

```
eodgdl tasha check
eodgdl tasha gaps
```

`tests/test_tasha.py` runs `check` too, so a mapping that drifts from either the
contract or the survey fails the test suite.

## Validating produced tables

```python
for problem in tasha.validate_all(households, people, trips):
    print(problem)
```

Or against a directory of written CSVs:

```
eodgdl tasha validate output/               # reads od_{households,people,trips}.csv
eodgdl tasha validate output/ --suffix _v2  # ...or the _v2 variants
```

`validate` checks required columns, unknown columns, key uniqueness, nulls, code
domains, dtypes, declared ranges, zone id shapes, cross-table joins, and the
invariants the contract states in prose rather than in a domain:

- a non-worker has no sector (`EmploymentStatus` and `Occupation` are `O` together);
- the return purposes `R`/`C` never appear as a trip *origin*;
- no trip goes from `H` to `H` — a return home made from home is not a trip;
- `HouseholdId` is dense and 0-based;
- every person's `TripNumber` runs 1..n with no gaps;
- `NumberOfPersons` is never below the person rows the household actually has —
  above is normal, since the EOD interviews ages 6+ and under-6s have no row.

It returns a list of strings, raises nothing and mutates nothing.

`validate` checks what a conforming table *must* satisfy. Two chain properties
are deliberately not in that list, because the survey violates them and a
builder cannot repair them without inventing data: that each trip starts in
the zone the previous one ended in, and that each trip starts after the
previous one arrived. `tasha.chain_report(trips)` counts those instead, together with tours
that do not begin or end at home, and the CLI prints it after every build and
validate without changing the exit code. On the shipped data:

```
32 trips (32 people) do not start in the zone the previous trip ended in
838 trips (765 people) start earlier than the trip before them
1667 trips (1517 people) start before the previous trip could have arrived
125 trips start at the same minute as the trip before them
899 people whose first trip does not start at home
412 people whose last trip does not end at home
```

## Open items

### Column gaps

`eodgdl tasha gaps` lists these; the full reasoning is in each mapping's `note`.

- **`DwellingType`** (pending) — no source exists. The dwellings file has no
  dwelling-type question and no address fields, so there is no interior-unit
  proxy either; `tenencia_vivienda` is tenure, not dwelling class. Constant `1`.
- **`IncomeClass`** (pending) — 10,432 of 17,901 dwellings (58%) refused or did
  not know and land in class 7. The band merge is provisional; an imputation
  model from AGEB census characteristics, vehicles, education and household size
  is the next step.
- **`FreeParking`** (pending) — constant `O`, but this survey does carry
  `estacionamiento_lugar` and `pago_estacionamiento` on every trip, so a real
  value is derivable.
- **`Formality`** (not surveyed) — no formality or social-security question in
  the EOD. Constant `O`, recorded rather than silently omitted.
- **`License`** (assumed) — age proxy, `edad >= 18`.
- **`TransitPass`** (not surveyed) — constant `N`.

### Structural, invisible to `tasha gaps`

No mapping `status` can carry these — they are about the shape of the output
rather than about one column's coding, so nothing surfaces them automatically.

- **Trip ordering — resolved 2026-09-01.** The row order is the chain and the
  contract now says so; the start times are the noisy field. `folio_viaje` is
  a generated number, but the zones and times pin the sequence and the row
  order matches it for 99.9% of clean multi-trip days; re-sequencing tours by
  time was tested and rejected (under 10% of inversions, misreads night
  shifts). The evidence and the five chain rules live with the loader,
  `eodgdl.eod.clean_trip_chains`; the mapping notes point there, and
  `reports/trip_chains.qmd` walks through every problem with examples. What
  remains is data quality the build reports rather than repairs
  (`tasha.chain_report`): 838 trips in 765 people still start before the
  trip before them, 883 start before the previous trip could have arrived by
  more than the 15-minute tolerance, and 32 trips do not start where the
  previous one ended. Two follow-ups were tried the
  same day: recoding the `Regresar a Casa` trips whose `tipo_lugar_destino`
  says the place was not a home is now a survey-level recode in `load_eod`,
  guarded by the zone (84 trips); imputing the untimed trips instead of
  excluding their person-days was prototyped and rejected — the numbers are
  in `eodgdl.eod`.

- **Zone system.** `build()` hardcodes the AGEB ids. `model_schema.yaml`'s
  `zones.alternatives` offers `ID_ZONAEOD` (71 zones) and `MZONA` (601) as the
  other choices, but there is no `zones=` selector, and the written tables record
  nothing about which system produced them — so the choice is invisible to
  whoever reads the CSVs. 1,701 AGEBs is also likely finer than a model with
  matching networks and skims wants.

  Separately, `validate()` checks a zone id's *shape* but never whether it joins.
  38 locality ids are absent from `RELACION_AGEBS-ZONA_con_datos_censales.parquet`
  and so carry no census attributes: 7,962 trip origins, 7,961 destinations,
  1,146 households, 1,275 `EmploymentZone`s and 449 `SchoolZone`s.

## Provenance

`model_schema.yaml` was reverse-engineered from an existing implementation of
these three tables, not from a published TMG file-format specification — hence
"TASHA / GTAModel *style*". Before relying on it as an interchange format, check
the code letters against TMG's own documentation; `Occupation` `G` is the one to
check first, since it is defined here as a fall-through ("everything else")
rather than as a named occupation class.
