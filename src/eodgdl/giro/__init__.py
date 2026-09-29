"""Imputation of the economic activity (``giro_empresa``) of EOD workers who did not report it.

Optional extra ``eodgdl[giro]`` (scikit-learn, joblib, pyyaml, mxcensus). The model is trained within the survey
on the workers with an observed giro (``eodgdl impute retrain giro``, evaluated in ``reports/imputation_giro.qmd``) and predicts the survey's five native levels
(Comercio, Servicio, Educación, Industria, Gobierno/sector público) from raw survey columns, the work-trip
destination and mode, and the destination's DENUE establishment mix (fetched through ``mxcensus``).

Typical use::

    import eodgdl
    from eodgdl import giro

    workers = giro.impute(eodgdl.load_eod())          # fitted bundle fetched from the data mirror
    workers[["giro_final"] + giro.PROBABILITY_COLUMNS]

giro is the engine's task ``giro`` (``impute/tasks/giro.yaml``; :mod:`eodgdl.impute` scores, retrains and compares it)
and the first step of the chain ``sector_informality``. This package keeps giro's stable entry points on top of the
engine: :func:`impute`, :func:`load_model` (the fitted v2 bundle, :mod:`eodgdl.impute.bundle`), the column names and
giro's feature builders (:func:`build_worker_features`, the DENUE destination mix). The notebook-era bundle layout and
its entry points (``impute_giro``, the old keys of ``load_model``) were retired on 2026-09-29.
"""

from ._config import (
    DENUE_RELEASE, DENUE_STATE_CODE, DESTINATION_AMBITO_LEVELS, DESTINATION_FEATURES, EMPLOYED_CATEGORIES, GIRO_CLASSES,
    GIRO_LABELS, GIRO_SLUGS, KEYS, MOBILITY_FEATURES, NO_ESPECIFICADO, NUMERIC_FEATURES, ROBUST_SECTOR_FEATURES,
    SECTOR_FEATURES, SHIFT_PROFILE_FEATURES, TASK, build_category_levels, load_config,
)
from .features import add_destination_features, build_worker_features, compute_work_trip_destination, zone_units

__all__ = [
    # the imputation
    "impute", "load_model", "MODEL_FILE", "OUTPUT_COLUMNS", "PROBABILITY_COLUMNS",
    # the feature frame
    "build_worker_features", "add_destination_features", "compute_work_trip_destination", "zone_units",
    # the task's constants
    "TASK", "load_config", "build_category_levels", "GIRO_CLASSES", "GIRO_LABELS", "GIRO_SLUGS", "KEYS", "NO_ESPECIFICADO",
    "EMPLOYED_CATEGORIES", "DESTINATION_FEATURES", "DESTINATION_AMBITO_LEVELS", "MOBILITY_FEATURES", "NUMERIC_FEATURES",
    "SECTOR_FEATURES", "ROBUST_SECTOR_FEATURES", "SHIFT_PROFILE_FEATURES", "DENUE_RELEASE", "DENUE_STATE_CODE",
]

MODEL_FILE = TASK.bundle_file
PROBABILITY_COLUMNS = TASK.probability_columns
OUTPUT_COLUMNS = KEYS + [
    "giro_observado", "giro_imputado", "giro_final", "giro_fue_imputado", "giro_model_used",
    "giro_prediction_confidence", "giro_marginalized_features",
] + PROBABILITY_COLUMNS


def load_model(path=None):
    """The fitted giro bundle, a v2 bundle (``arms``, ``auxiliary``, ``classes``, ``category_levels``, ``metadata``:
    :mod:`eodgdl.impute.bundle`), fetched from the data mirror unless ``path`` is given or ``$EODGDL_DATA_DIR`` holds a
    local copy. Scoring checks it against the task spec and the running scikit-learn (:func:`impute`)."""
    from eodgdl.impute.bundle import load_bundle

    return load_bundle(TASK, path)


def impute(tables=None, bundle=None, path=None):
    """Worker frame (:func:`build_worker_features`) scored with the fitted bundle (:func:`eodgdl.impute.run.score_task`):
    the bundle is checked against the task spec (scikit-learn version, category levels, features) first. With
    ``tables=None`` the survey is loaded and the features are read from the feature cache when present."""
    from eodgdl.impute.run import score_task

    return score_task(TASK, tables=tables, bundle=bundle, path=path)
