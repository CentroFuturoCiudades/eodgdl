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
from eodgdl import load_taz, load_mtaz, load_imeplan_agebs, load_zm_muns, zone_system_report

taz = load_taz(drop_ap=True)                       # traffic-analysis zones
mtaz = load_mtaz(taz, drop_ap=True)                # micro-zones, aligned to taz CRS
agebs = load_imeplan_agebs(taz, drop_ap=True)      # AGEB → zone with census population
zone_system_report(taz, mtaz)                       # diagnostics
```

Every loader fetches from the mirror by default and accepts a local path override.

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
documented constants in `eod.py` and `taz.py`): three trip-mode fixes, two micro-zone
population double-count adjustments, and three AGEB `MZONA` reassignments.

## Giro imputation model (`eodgdl[giro]`)

Most workers in the survey did not report the activity of their employer (`giro_empresa`). The optional
`eodgdl.giro` subpackage imputes it within the survey: a hybrid (with / without education) scikit-learn model
trained on the workers with an observed giro, using raw survey columns, the work-trip destination and mode, and
the destination's DENUE establishment mix (DENUE and the Marco Geoestadístico are fetched through `mxcensus`).
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

The fitted bundle (`data/od_giro_hybrid_model.joblib`, a scikit-learn pickle — see `metadata["sklearn_version"]`)
is trained and evaluated in `notebooks/giro_model.ipynb` (household-grouped cross-validation, one-standard-error
model selection, calibration, covariate-shift sensitivity); after retraining, update its sha256 in
`src/eodgdl/data/registry.txt`.
