# eodgdl

Loader, validation schemas, and traffic-analysis-zone (TAZ) tools for the **IMEPLAN
Guadalajara EOD 2023** household origin-destination survey.

```python
from eodgdl import load_eod

# Fetches the survey CSVs from the data mirror on first use (cached thereafter):
viv, hab, trips, legs = load_eod()

# Or read from a local copy (e.g. this repo's data/ dir, or $EODGDL_DATA_DIR):
viv, hab, trips, legs = load_eod("data")
```

`load_eod()` returns an `EODTables` named tuple with four linked, cleaned, schema-validated
tables:

| field   | grain                                              |
|---------|----------------------------------------------------|
| `viv`   | dwelling — `folio_vivienda`                         |
| `hab`   | person — `(folio_vivienda, folio_habitante)`        |
| `trips` | trip — `(…, folio_viaje)`                            |
| `legs`  | trip leg — `(…, folio_viaje, folio_traslado)`       |

## Zone system

```python
from eodgdl import load_eod, load_taz, load_zm_muns, reweight

taz = load_taz(drop_ap=True)                       # the centralidad polygons the survey was fielded on

# every census unit (urban AGEB, rural locality) in at most one zone: the polygons plus the
# survey's own coding; needs the `reweight` extra (census data through mxcensus)
tables = load_eod()
units = reweight.assign_units(tables.viv, tables.trips)   # CVEGEO -> zone, rule, population
zones = reweight.zone_shapes(units)                # the zones redrawn along AGEB edges
```

Every loader fetches from the mirror by default and accepts a local path override. An
urban AGEB is never split: a sampled one takes the zone the survey coded there, an
unsampled one takes the zone the survey coded for its trip ends, or else goes whole to the
polygon holding most of its population. `reports/reweight_inputs.qmd`
maps the result; `scripts/zone_system_map.py` writes the map and the redrawn zones.

## Travel-demand model schema

`eodgdl.tasha` targets the three tables the travel-demand model consumes
(`od_households.csv`, `od_people.csv`, `od_trips.csv`). The contract lives in
`model_schema.yaml`, the survey mapping in `mappings.yaml` — both shipped inside the
package and meant to be read by hand.

```python
from eodgdl import load_eod, tasha

od = tasha.build(load_eod("data"))   # ODTables(households, people, trips)
tasha.validate_all(*od)

tasha.build_map("Mode")              # {'A PIE': 'W', 'CAMIÓN O AUTOBÚS': 'B', …}
tasha.mapping("Mode")                # ...plus the override, notes and caveats
tasha.gaps()                         # what is assumed, constant, or unresolved
```

```bash
eodgdl tasha build --data data/ --out output/   # build, write and validate
eodgdl tasha check                              # mappings vs. contract
eodgdl tasha gaps                               # open items
eodgdl tasha validate output/                   # produced CSVs vs. contract
```

Zone columns carry the survey's own zone id as a string — a 13-character urban AGEB CVEGEO, a
9-character rural AGEB key (INEGI's rural AGEB, not a locality) or one of the 7 access points
(`99999000x`) — and the zone system places every one of them (`eodgdl.reweight.zoning`); read
them back with `dtype=str`.

See [`src/eodgdl/tasha/README.md`](src/eodgdl/tasha/README.md) for the full guide, the
mapping-entry format, and the open items.

## Trip-chain review

The chains `load_eod` leaves with a `problemas` code can be fixed by hand through a
review sheet: a CSV in the chain browser's table format (`notebooks/chain_browser.ipynb`),
one row per shipped trip, `shipped → cleaned` where the rules changed a value.

```bash
eodgdl review export --data data/ --out notebooks/chain_review.csv   # the pending chains, to edit by hand
eodgdl review verify notebooks/chain_review.csv --data data/         # recover the edits, apply, recompute problemas
```

Next to every column that may take a change stands an empty `new …` column: write the
value that should hold there (`dropped` under `new status` takes a row out, `restored`
brings back a row `load_eod` dropped) and say why under `note`. `verify` (and the
browser's *load sheet* box) reads the filled cells as the edits, applies them with
`<field>:revision` codes in `ajustes`, and reports per person which defects were cleared,
left or made. See `eodgdl.review` for the functions behind both.

## Installation

```bash
uv add eodgdl @ git+https://github.com/CentroFuturoCiudades/eodgdl
# or, for development:
git clone https://github.com/CentroFuturoCiudades/eodgdl && cd eodgdl && uv sync
```

## Data

The survey data lives in this repo's `data/` directory (committed, but **not** distributed
inside the installed package — the wheel ships code only). The package fetches the files it
needs on demand via [Pooch](https://www.fatiando.org/pooch/), caching them locally and
verifying checksums against `src/eodgdl/data/registry.txt`.

Override the defaults with environment variables:

| variable           | effect                                                            |
|--------------------|-------------------------------------------------------------------|
| `EODGDL_DATA_DIR`  | read data from this local directory instead of fetching           |
| `EODGDL_CACHE_DIR` | where Pooch caches downloaded files                               |
| `EODGDL_BASE_URL`  | mirror base URL (fork / different ref); keep the trailing `/`     |

CLI helpers: `eodgdl info` (show cache dir + mirror) and `eodgdl fetch [--dataset survey|zones|all]`.

### Data provenance

Source: **Encuesta Origen-Destino 2023**, IMEPLAN (Instituto Metropolitano de Planeación del
Área Metropolitana de Guadalajara). Public report and interactive visualizer:
<https://www.imeplan.mx/plataformas-de-informacion/visualizador-eod>

The full public report PDF (~148 MB) is **not** redistributed here — download it from the link
above. The 4 MB technical report (`Informe_Tecnico_Final_EOD_2023.pdf`) is included for
reference.

A small number of manual data-entry corrections are applied by the loaders (encoded as
documented constants in `eod.py`, `chains.py` and `reweight/zoning.py`): three trip-mode
fixes and four rural localities placed in the zone the survey coded for their rural AGEB.

IMEPLAN's AGEB-to-zone table (`RELACION_AGEBS-ZONA_con_datos_censales`) and micro-zones
(`AMG_MicroZONAS2023`) are kept under `data/` as delivered but no longer loaded: neither was
part of the survey design, and the table disagreed with the survey's own zone coding (it
left La Aurora, Juanacatlán, whose 221 sampled dwellings the survey coded `49F`, in no
zone). Zones are built from the census by `reweight.zoning` instead.

## Giro imputation model (`eodgdl[giro]`)

Most workers in the survey did not report the activity of their employer (`giro_empresa`). The optional
`eodgdl.giro` subpackage imputes it within the survey: a hybrid (with / without education) scikit-learn model
trained on the workers with an observed giro, using raw survey columns, the work-trip destination and mode, and
the DENUE establishment mix of the destination's urban or rural AGEB, the unit the survey codes and the zone system
(`reweight.zoning`) places (DENUE and the census are fetched through `mxcensus`).
It predicts the five native levels — Comercio, Servicio, Educación, Industria, Gobierno/sector público — and keeps
the full probability vector (`prob_giro_<slug>`); `giro_final` is the arg-max.

```bash
uv add "eodgdl[giro]"
```

```python
import eodgdl
from eodgdl import giro

workers = giro.impute(eodgdl.load_eod())   # fitted bundle fetched from the data mirror on first use
workers[giro.OUTPUT_COLUMNS].head()
```

The model runs on `eodgdl.impute`, the package's imputation engine: the task is defined in
`src/eodgdl/impute/tasks/giro.yaml` (classes, features, arms, candidate grid, evaluation settings), and the engine
scores, compares and retrains it:

```bash
uv run eodgdl impute score giro --data data             # the fitted bundle's probabilities -> output/impute/giro_scores.parquet
uv run eodgdl impute compare giro --spec candidates.yaml  # feature specifications at the published winner's hyperparameters, paired folds
uv run eodgdl impute retrain giro --data data           # grouped-CV selection, held-out evaluation, refit -> output/impute/giro/
quarto render reports/imputation_giro.qmd               # the retrain's evaluation (reads output/impute/giro/, trains nothing)
```

The fitted bundle (`data/od_giro_hybrid_model.joblib`, a scikit-learn pickle — see `metadata["sklearn_version"]`) is
checked against the task on load (scikit-learn version, category levels, features). A retrain writes a new one to
`output/impute/giro/`; to ship it, copy it to `data/` and update its sha256 in `src/eodgdl/data/registry.txt`. Feature
frames are cached under the eodgdl cache directory (`--refresh` rebuilds).
