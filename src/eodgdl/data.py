"""Where eodgdl reads its files: the clone's ``data/`` (the input data) and ``models/`` (the fitted model bundles).

eodgdl is a research pipeline run from its repository (``uv sync`` installs the package editable, so this module sits
at ``<clone>/src/eodgdl/data.py``): every file is read in place, nothing is downloaded or registered, and a new file
under ``data/`` or ``models/`` needs no more than its name in the code that reads it. ``$EODGDL_DATA_DIR`` and
``$EODGDL_MODELS_DIR`` point the reads elsewhere (the tests' temporary folders, another checkout's data);
``$EODGDL_CACHE_DIR`` moves the cache (eodgdl.impute's feature frames), by default ``<clone>/.cache/``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Survey master tables (raw IMEPLAN CSV, ISO-8859-1)
VIVIENDAS_CSV = "IMEPLAN_Base_Viviendas_Master.csv"
HABITANTES_CSV = "IMEPLAN_Base_Habitantes_Master.csv"
VIAJES_CSV = "IMEPLAN_Base_Viajes_Master.csv"

# Zone-system geometries (parquet)
ZONIFICACION_PARQUET = "AMG_Zonificacion_para_encuesta.parquet"

# CONAPO population projections for the 9 municipalities (municipality x sex x year x 5-year band)
CONAPO_CSV = "CONAPO_proyecciones_AMG.csv"

# INEGI registered motor vehicles (VMRC), annual, the 9 municipalities, 2015-2023
VMRC_CSV = "VMRC_AMG.csv"

# INEGI ENDUTIH share of dwellings with internet, 2020 and 2023 (scripts/endutih_internet.py)
ENDUTIH_CSV = "ENDUTIH_internet_AMG.csv"

SURVEY_FILES = [VIVIENDAS_CSV, HABITANTES_CSV, VIAJES_CSV]
ZONE_FILES = [ZONIFICACION_PARQUET]
CENSUS_FILES = [CONAPO_CSV, VMRC_CSV, ENDUTIH_CSV]

# The fitted eodgdl.impute bundles are models/<file>.joblib, each named by its task (impute/tasks/<task>.yaml,
# `bundle:`) and written by `eodgdl impute retrain <task>`; models/od_empleo_design_model.joblib is the empleo task
# trained on the survey's design weight, pinned for the reweight records (reweight/spec.yaml, `imputed`).
MODEL_SUFFIX = ".joblib"


def _directory(variable: str, default: Path) -> Path:
    if value := os.environ.get(variable):
        return Path(value).expanduser().resolve()
    return default


def data_dir() -> Path:
    """The input data: ``$EODGDL_DATA_DIR``, else the clone's ``data/``."""
    return _directory("EODGDL_DATA_DIR", REPO / "data")


def models_dir() -> Path:
    """The fitted model bundles: ``$EODGDL_MODELS_DIR``, else the clone's ``models/``."""
    return _directory("EODGDL_MODELS_DIR", REPO / "models")


def cache_dir() -> Path:
    """What eodgdl caches (eodgdl.impute's feature frames): ``$EODGDL_CACHE_DIR``, else the clone's ``.cache/``."""
    return _directory("EODGDL_CACHE_DIR", REPO / ".cache")


def resolve(filename: str) -> Path:
    """The path of ``filename``: a model bundle (``*.joblib``) in :func:`models_dir`, any other file in
    :func:`data_dir`. Fails, naming where it looked, when the file is not there."""
    folder = models_dir() if filename.endswith(MODEL_SUFFIX) else data_dir()
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} is not in {folder} (eodgdl reads the clone's data/ and models/, "
                                "or $EODGDL_DATA_DIR / $EODGDL_MODELS_DIR)")
    return path
