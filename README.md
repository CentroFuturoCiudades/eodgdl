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

Zone columns carry the survey's own AGEB CVEGEO or locality id as a string, so the
output joins straight to the census tables; read them back with `dtype=str`.

See [`src/eodgdl/tasha/README.md`](src/eodgdl/tasha/README.md) for the full guide, the
mapping-entry format, and the open items.

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

The AGEB table also mixes locality rows with the AGEB rows that subdivide them, double-
counting those localities' population. `load_imeplan_agebs` keeps whichever side partitions
the locality more finely — the AGEBs where there is more than one, otherwise the locality —
which drops three rows and makes the AGEB and micro-zone population totals reconcile
exactly.
