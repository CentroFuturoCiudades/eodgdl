# eodgdl

Loader, validation schemas, and traffic-analysis-zone (TAZ) tools for the **IMEPLAN
Guadalajara EOD 2023** household origin-destination survey.

```python
from eodgdl import load_eod

# Reads the survey CSVs from the clone's data/ (or $EODGDL_DATA_DIR):
viv, hab, trips, legs = load_eod()
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

Every loader reads the clone's `data/` by default and accepts a path override. An
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

completed = tasha.load_completed("output/impute", draw=0)   # one completed dataset of the pipeline tasha
od = tasha.build(load_eod("data"), completed=completed)    # ODTables(households, people, trips)
tasha.validate_all(*od)

tasha.build_map("Mode")              # {'A PIE': 'W', 'CAMIÓN O AUTOBÚS': 'B', …}
tasha.mapping("Mode")                # ...plus the override, notes and caveats
tasha.gaps()                         # what is assumed, constant, or unresolved
```

```bash
eodgdl tasha build --data data/ --out output/   # build, write and validate (--impute output/impute --draw 0)
eodgdl tasha check                              # mappings vs. contract
eodgdl tasha gaps                               # open items
eodgdl tasha validate output/                   # produced CSVs vs. contract
```

Zone columns carry the survey's own zone id as a string — a 13-character urban AGEB CVEGEO, a
9-character rural AGEB key (INEGI's rural AGEB, not a locality) or one of the 7 access points
(`99999000x`) — and the zone system places every one of them (`eodgdl.reweight.zoning`); read
them back with `dtype=str`.

Some columns come from the imputation models (see Imputation models below), one completed
dataset at a time, of the pipeline `tasha` (persons and dwellings drawn jointly):
`IncomeClass` is the dwelling's AMAI socioeconomic level (its dwellings, the chain `nse`, as
the model's Monterrey inputs coded it, counting the members drawn as workers);
`EmploymentStatus` takes the drawn employment situation of the persons who did not answer,
and `P` is an informal worker (Monterrey's convention, not part-time), with `Formality`
mirroring it; a worker without a reported giro takes the drawn one for `Occupation`, and an
unanswered person drawn as a student is `S` (its persons, the chain `labour`). The build
writes `od_provenance.json` with the draw and the models; `--no-impute` leaves those columns
at their defaults.

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

eodgdl is a research pipeline, run from its repository: `uv sync` installs the package editable, and every loader
reads its files in place. Another project uses it as an editable path dependency
(`uv add --editable ../eodgdl`) and reads the same clone.

```bash
git clone https://github.com/CentroFuturoCiudades/eodgdl && cd eodgdl && uv sync
```

## Data

The survey and the other input data live in this repo's `data/` directory, the fitted
imputation models in `models/`, both committed and read in place (`eodgdl.data`: `data_dir()`,
`models_dir()`, `resolve(filename)`); nothing is downloaded, registered or released file by file.
The census, ENOE, ENIGH and DENUE come through `mxcensus`, which fetches and checks its own files.

Override the defaults with environment variables:

| variable            | effect                                                         |
|---------------------|----------------------------------------------------------------|
| `EODGDL_DATA_DIR`   | read the input data from this directory (default: `data/`)     |
| `EODGDL_MODELS_DIR` | read the model bundles from this directory (default: `models/`) |
| `EODGDL_CACHE_DIR`  | where eodgdl caches its feature frames (default: `.cache/`)   |

What a run of the processing chooses is `config/config.yaml`: the weight the survey's imputation tasks train on
(`design`, the survey's own, or `tmg`, TMG.SurveyReweight's household weight once it is in `data/`), the bootstrap refits
per task, the diagnostic chains' draws and the completed dataset the TASHA build reads (`docs/pipeline.md`).

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

## Imputation models (`eodgdl[giro]`)

`eodgdl.impute` is the package's categorical imputation engine; every imputation model on the survey runs on it. A
**task** is one model, a YAML under `src/eodgdl/impute/tasks/` (source, classes, features, arms, candidate grid,
evaluation settings). A **chain** strings tasks together, in parallel or in sequence, and passes each task's class
probabilities to the tasks downstream (YAMLs under `src/eodgdl/impute/chains/`). A **pipeline** draws the chains of
several row levels jointly (YAMLs under `src/eodgdl/impute/pipelines/`). There are four imputations, and the pipeline
the TASHA build reads:

| imputation | kind | trained on | scores |
|---|---|---|---|
| `giro` | task | EOD workers who reported their employer's activity | every EOD worker |
| `sector_informality` | chain: `giro` → `informality` (exact sum over the sector) | ENOE, 8 quarters of Jalisco's employed, harmonized to the survey | every EOD worker |
| `nse` | chain: `educacion_jefe` → `amai_banos` → `amai_dormitorios` → AMAI points, level, AGEB calibration (50 draws) | the EOD's heads (education); ENIGH 2022, households in cities of 100,000+ (bathrooms, bedrooms) | every EOD dwelling |
| `labour` | chain: `empleo` → `giro` → `informality`, all on the workers and the persons who did not answer (50 draws) | the EOD's persons aged 16+ who answered (employment); as above (giro, informality) | every EOD worker and every person aged 16+ without an employment answer |
| `tasha` | pipeline: `labour` on the persons → each dwelling's workers aged 14+ with the drawn ones → `nse` on the dwellings (50 draws, a bootstrap model per draw) | as `labour` and `nse` | as `labour` and `nse`: what the TASHA build reads |

The fitted bundle of every task is committed under `models/` (`models/od_<task>_model.joblib`,
giro's is `od_giro_hybrid_model.joblib`). It is a scikit-learn pickle, checked against its task when
it loads (scikit-learn version, category levels, features, scoring settings). `amai_trabajadores` is diagnostic only:
the survey counts the workers, so the chain `nse` does not use it, and only `evaluate nse` reads it.

Each task writes the same output columns, prefixed with its name `<t>`: `<t>_observado`, `<t>_imputado`, `<t>_final`
(the reported value, else the arg-max), `<t>_fue_imputado`, `<t>_model_used` (the arm), `<t>_prediction_confidence`,
`<t>_marginalized_features` (features whose level had no training support, averaged over) and the full probability
vector `prob_<t>_<class>`. Inside a chain, each task also gets `<t>_condicionado_en` (the upstream it was conditioned
on). `sector_informality` also writes `prob_informalidad_<class>_given_sector_<sector>`. `nse` also writes each
component's `<t>_puntaje_esperado` (expected AMAI points), `amai_puntos_media`, `prob_nse_<level>` and
`prob_nse_calibrado_<level>` (levels `e d d_mas c_menos c c_mas ab`). A chain's run also writes
`completions.parquet`, one row per completion (per row × sector for enumerate, per row × draw for draws) with its
weight: this is the table for any statistic beyond the marginals.

### Giro

Most workers in the survey did not report the activity of their employer (`giro_empresa`). The optional
`eodgdl.giro` subpackage imputes it within the survey: a hybrid (with / without education) scikit-learn model
trained on the workers with an observed giro, using raw survey columns, the work-trip destination and mode, and
the DENUE establishment mix of the destination's urban or rural AGEB, the unit the survey codes and the zone system
(`reweight.zoning`) places (DENUE and the census are fetched through `mxcensus`).
It predicts the five native levels — Comercio, Servicio, Educación, Industria, Gobierno/sector público — and keeps
the full probability vector (`prob_giro_<slug>`); `giro_final` is the arg-max.

```bash
uv sync --extra giro
```

```python
import eodgdl
from eodgdl import giro

workers = giro.impute(eodgdl.load_eod())   # the fitted bundle in models/
workers[giro.OUTPUT_COLUMNS].head()
```

The engine scores, compares and retrains the task (`src/eodgdl/impute/tasks/giro.yaml`):

```bash
uv run eodgdl impute score giro --data data             # the fitted bundle's probabilities -> output/impute/giro_scores.parquet
uv run eodgdl impute compare giro --spec candidates.yaml  # feature specifications at the published winner's hyperparameters, paired folds
uv run eodgdl impute retrain giro --data data           # grouped-CV selection, held-out evaluation, refit -> output/impute/giro/
quarto render reports/imputation_giro.qmd               # the retrain's evaluation (reads output/impute/giro/, trains nothing)
```

### Informality

Informality (trained on INEGI's ENOE through `mxcensus`, harmonized to the survey's workers) is imputed through the
giro model by the chain `sector_informality`:

```bash
uv run eodgdl impute score sector_informality --data data      # -> output/impute/sector_informality/
uv run eodgdl impute retrain informality --data data           # -> output/impute/informality/
uv run eodgdl impute evaluate sector_informality --data data   # benchmark, sensitivity, components, gap decomposition
                                                               # (reads giro's scenarios: needs `retrain giro` first)
quarto render reports/imputation_informality.qmd
```

### Socioeconomic level (NSE)

AMAI's socioeconomic level (NSE) of every dwelling is the chain `nse`. The head's education is imputed within the
survey (with the other members' highest education where one answered), and complete bathrooms and bedrooms from INEGI's ENIGH 2022 (households in cities of 100,000+). Internet, cars
and the members aged 14+ who worked come from the survey's own answers. AMAI's points and levels are carried through
50 multiple imputations, then rank-calibrated against AMAI's NSE by AGEB (`data/NSE_por_AGEB_AMAI.xlsx`):

```bash
uv run eodgdl impute score nse --data data                 # -> output/impute/nse/: prob_nse_<level>, prob_nse_calibrado_<level>, the draws
for task in educacion_jefe amai_banos amai_dormitorios amai_trabajadores; do uv run eodgdl impute retrain $task --data data; done
uv run eodgdl impute evaluate nse --data data              # propagation variants, Monte Carlo error, AMAI by AGEB
quarto render reports/imputation_nse.qmd
```

### Employment situation

4,370 persons aged 16+ did not answer whether they worked last week (`trabajo_semana_pasada`), with the occupation,
the giro and often the education blank too: the questionnaire's socio-economic block was skipped, mostly from
February on. The task `empleo` imputes their situation (works, student, home duties, retired, not working) from the
persons who answered: age, sex, relationship, the dwelling, the survey day's trips, the weekend block and the other
household members' answers, with an arm for the occupation and one for the education where given. The chain `labour`
draws it together with a giro and an informality for every worker (reported, or drawn as one). The pipeline `tasha`
runs it on the persons, then counts the members drawn as workers in their dwelling's AMAI level (the chain `nse` on the
dwellings, draw by draw); the TASHA build reads one of its 50 completed datasets:

```bash
uv run eodgdl impute retrain empleo --data data --bootstrap 50                              # -> output/impute/empleo/
uv run eodgdl impute score tasha --data data --retrained output/impute      # -> output/impute/tasha/ (persons/, dwellings/)
quarto render reports/imputation_empleo.qmd
```

### Retraining and shipping a bundle

A retrain writes the bundle, its scores and every evaluation table to `<out>/<task>/` (`--out` defaults to
`output/impute`; `retrain <chain>` does the same for each of its tasks). Every command that needs a bundle finds it
the same way: with `--retrained <dir>`, the one a retrain wrote under `<dir>/<task>/` when there is one, else the
shipped one. `score` and `compare` default to the shipped bundles; `evaluate` defaults to `--retrained output/impute`,
where it also reads an upstream's retrain `scenarios.parquet`. To ship a retrained bundle, copy it to `models/` and
commit it. Feature frames are cached under `.cache/impute/` (`$EODGDL_CACHE_DIR`; `--refresh` rebuilds).

### Figures

The figures of informal-jobs-model's informality and NSE notebooks (where these models were first built), redrawn from
the outputs above with their original names and Spanish labels, as PDF and 600-dpi PNG in `output/figures/`: ENOE and
the EOD's predictor distributions, ENOE's informality profiles, the informality models' calibration (with the isotonic
and the arms-combined held-out curves a retrain writes), ENOE against the EOD after the model, the head's education and
the NSE maps by AGEB (the AGEB polygons come from `mxcensus`, the `reweight` extra):

```bash
quarto render reports/imputation_figures.qmd   # reads output/impute/{informality,educacion_jefe,sector_informality,nse}/
```
