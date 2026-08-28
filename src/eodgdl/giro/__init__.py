"""Imputation of the economic activity (``giro_empresa``) of EOD workers who did not report it.

Optional extra ``eodgdl[giro]`` (scikit-learn, joblib, pyyaml, mxcensus). The model is trained within the survey
on the workers with an observed giro (``notebooks/giro_model.ipynb``) and predicts the survey's five native levels
(Comercio, Servicio, Educación, Industria, Gobierno/sector público) from raw survey columns, the work-trip
destination and mode, and the destination's DENUE establishment mix (fetched through ``mxcensus``).

Typical use::

    import eodgdl
    from eodgdl import giro

    workers = giro.impute(eodgdl.load_eod())          # fitted bundle fetched from the data mirror
    workers[["giro_final"] + giro.PROBABILITY_COLUMNS]

``impute`` = :func:`build_worker_features` + :func:`load_model` + :func:`impute_giro`.
"""

from ._config import (
    DENUE_RELEASE, DENUE_STATE_CODE, DESTINATION_AMBITO_LEVELS, DESTINATION_FEATURES, EMPLOYED_CATEGORIES, GIRO_CLASSES,
    GIRO_LABELS, GIRO_SLUGS, KEYS, MOBILITY_FEATURES, NO_ESPECIFICADO, NUMERIC_FEATURES, ROBUST_SECTOR_FEATURES,
    SECTOR_FEATURES, SHIFT_PROFILE_FEATURES, build_category_levels, load_config,
)
from ._ml import (
    count_levels_without_training_support, fold_table, identify_missing_category, prepare_model_features, select_one_se,
)
from .features import add_destination_features, build_worker_features, compute_work_trip_destination
from .model import (
    PROBABILITY_COLUMNS, adjust_imputed_share, build_models, calculate_calibration, calculate_confidence_summary,
    calculate_distribution, calculate_feature_missingness, calculate_model_usage, calculate_probabilistic_distribution,
    compare_known_unknown_profiles, evaluate_model, fit_destination_models, get_best_model, impute_giro,
    impute_under_covariate_shift, missing_education, prepare_training_data, refit_model, split_known_data,
    test_metrics_with_uncertainty, tune_models, validate_probability_rows,
)

MODEL_FILE = "od_giro_hybrid_model.joblib"
OUTPUT_COLUMNS = KEYS + [
    "giro_observado", "giro_imputado", "giro_final", "giro_fue_imputado", "giro_model_used",
    "giro_prediction_confidence", "giro_marginalized_features",
] + PROBABILITY_COLUMNS


def load_model(path=None):
    """The fitted hybrid bundle (dict with ``model_with_education``, ``model_without_education``, the feature lists,
    ``destination_models``, ``category_levels`` and ``metadata``). Fetched from the data mirror unless ``path`` is
    given or ``$EODGDL_DATA_DIR`` holds a local copy. Requires the scikit-learn version recorded in
    ``metadata["sklearn_version"]``."""
    import joblib

    from eodgdl.data import resolve

    return joblib.load(path if path is not None else resolve(MODEL_FILE))


def impute(tables=None, bundle=None, path=None):
    """Worker frame (:func:`build_worker_features`) scored with the fitted bundle (:func:`impute_giro`)."""
    import eodgdl

    tables = tables if tables is not None else eodgdl.load_eod()
    bundle = bundle if bundle is not None else load_model(path)

    return impute_giro(
        bundle["model_with_education"], bundle["model_without_education"], build_worker_features(tables),
        with_education_features=bundle["features_with_education"], without_education_features=bundle["features_without_education"],
        destination_models=bundle.get("destination_models"),
    )
