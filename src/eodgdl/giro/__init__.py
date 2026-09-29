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
engine: :func:`impute`, :func:`load_model` (the bundle with its old keys), :func:`impute_giro`, the column names
and giro's feature builders (:func:`build_worker_features`, the DENUE destination mix).
"""

import pandas as pd

from ._config import (
    DENUE_RELEASE, DENUE_STATE_CODE, DESTINATION_AMBITO_LEVELS, DESTINATION_FEATURES, EMPLOYED_CATEGORIES, GIRO_CLASSES,
    GIRO_LABELS, GIRO_SLUGS, KEYS, MOBILITY_FEATURES, NO_ESPECIFICADO, NUMERIC_FEATURES, ROBUST_SECTOR_FEATURES,
    SECTOR_FEATURES, SHIFT_PROFILE_FEATURES, TASK, build_category_levels, load_config,
)
from ._ml import prepare_model_features
from .features import add_destination_features, build_worker_features, compute_work_trip_destination, zone_units

MODEL_FILE = TASK.bundle_file
PROBABILITY_COLUMNS = TASK.probability_columns
OUTPUT_COLUMNS = KEYS + [
    "giro_observado", "giro_imputado", "giro_final", "giro_fue_imputado", "giro_model_used",
    "giro_prediction_confidence", "giro_marginalized_features",
] + PROBABILITY_COLUMNS


def load_model(path=None):
    """The fitted hybrid bundle as a dict with ``model_with_education``, ``model_without_education``, the feature
    lists, ``destination_models``, ``category_levels`` and ``metadata`` (:func:`eodgdl.impute.bundle.legacy_view`).
    Fetched from the data mirror unless ``path`` is given or ``$EODGDL_DATA_DIR`` holds a local copy. Requires the
    scikit-learn version recorded in ``metadata["sklearn_version"]``."""
    from eodgdl.impute.bundle import legacy_view, load_bundle

    return legacy_view(load_bundle(TASK, path))


def impute(tables=None, bundle=None, path=None):
    """Worker frame (:func:`build_worker_features`) scored with the fitted bundle (:func:`eodgdl.impute.run.score_task`):
    the bundle is checked against the task spec (scikit-learn version, category levels, features) first. With
    ``tables=None`` the survey is loaded and the features are read from the feature cache when present."""
    from eodgdl.impute.run import score_task

    return score_task(TASK, tables=tables, bundle=bundle, path=path)


def impute_giro(model_with_education, model_without_education, od, with_education_features=SECTOR_FEATURES, without_education_features=ROBUST_SECTOR_FEATURES, destination_models=None):
    """Score every worker of ``od`` (:func:`build_worker_features`) without an observed giro with the model their
    education supports, from the fitted models of :func:`load_model` (``destination_models``: P(destination | x)
    for the workers without a work trip). Adds ``giro_observado``, ``giro_imputado``, ``giro_final``,
    ``giro_fue_imputado``, ``giro_model_used``, ``giro_prediction_confidence``, ``giro_marginalized_features`` and
    ``prob_giro_<slug>`` (rows sum to one; observed rows get probability one on their giro)."""
    from eodgdl.impute.arms import Arm, assign_arms, impute_with_arms

    destination_models = destination_models or {}
    arms = [Arm(name, model, list(features), requires=list(TASK.arm(name).requires),
                auxiliary={"destino_trabajo": destination_models[name]} if name in destination_models else {})
            for name, model, features in (("with_education", model_with_education, with_education_features),
                                          ("without_education", model_without_education, without_education_features))]
    assigned = assign_arms(od, arms, TASK.is_missing)
    assigned[~od["giro_desconocido"].astype(bool)] = pd.NA

    return impute_with_arms(od, "giro", GIRO_CLASSES, arms, assigned, "giro", NUMERIC_FEATURES, build_category_levels(), missing_label=NO_ESPECIFICADO)
