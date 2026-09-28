"""Compatibility shim: the modelling helpers moved to :mod:`eodgdl.impute`; these names bind giro's numeric features,
missing label and category levels to them. Bundles pickled before the move reference
``eodgdl.giro._ml.prepare_model_features`` and keep loading through it."""

from eodgdl.impute import evaluate as _evaluate
from eodgdl.impute import levels as _levels
from eodgdl.impute import marginalize as _marginalize
from eodgdl.impute import models as _models
from eodgdl.impute.evaluate import bootstrap_by_group, calculate_calibration_table, calibration_metrics, marginal_log_loss
from eodgdl.impute.levels import normalize_predicted_probabilities, normalize_sample_weights
from eodgdl.impute.models import make_tree_preprocessor
from eodgdl.impute.select import FAMILY_COMPLEXITY, PARAMETER_COMPLEXITY_DIRECTION, complexity_key, fold_table, select_one_se

from ._config import NO_ESPECIFICADO, NUMERIC_FEATURES, build_category_levels

__all__ = [
    "FAMILY_COMPLEXITY", "PARAMETER_COMPLEXITY_DIRECTION", "assert_known_levels", "attach_training_level_shares", "bootstrap_by_group",
    "calculate_calibration_table", "calibration_metrics", "complexity_key", "compute_training_level_shares",
    "count_levels_without_training_support", "fit_level_model", "fold_table", "identify_missing_category", "make_tree_preprocessor",
    "marginal_log_loss", "normalize_predicted_probabilities", "normalize_sample_weights", "predict_level_shares",
    "predict_proba_marginalizing", "prepare_model_features", "reweight_to_target_profile", "select_one_se", "split_feature_types",
]


def split_feature_types(features):
    return _levels.split_feature_types(features, NUMERIC_FEATURES)


def identify_missing_category(series):
    return _levels.identify_missing(series, NO_ESPECIFICADO)


def prepare_model_features(dataframe, features):
    """Select the model features: numeric columns coerced, categorical columns as clean strings with a missing label."""
    return _levels.prepare_features(dataframe, features, NUMERIC_FEATURES, NO_ESPECIFICADO)


def assert_known_levels(X, category_levels=None):
    _levels.assert_known_levels(X, category_levels or build_category_levels())


def count_levels_without_training_support(X_train, X):
    return _levels.count_levels_without_training_support(X_train, X, NUMERIC_FEATURES)


def compute_training_level_shares(X, sample_weights):
    return _marginalize.compute_training_level_shares(X, sample_weights, NUMERIC_FEATURES)


def attach_training_level_shares(model, X, sample_weights):
    return _marginalize.attach_training_level_shares(model, X, sample_weights, NUMERIC_FEATURES)


def predict_proba_marginalizing(model, X, level_subsets=None, conditional_shares=None):
    return _marginalize.predict_proba_marginalizing(model, X, level_subsets=level_subsets, conditional_shares=conditional_shares, missing_label=NO_ESPECIFICADO)


def predict_level_shares(model, X, level_subsets=None):
    return _marginalize.predict_level_shares(model, X, level_subsets=level_subsets, missing_label=NO_ESPECIFICADO)


def fit_level_model(frame, target, features, sample_weights=None, random_state=42):
    return _models.fit_level_model(frame, target, features, NUMERIC_FEATURES, build_category_levels(), sample_weights=sample_weights, random_state=random_state, missing_label=NO_ESPECIFICADO)


def reweight_to_target_profile(source_rows, target_rows, features, source_weight_column, target_weight_column, random_state=42):
    return _evaluate.reweight_to_target_profile(source_rows, target_rows, features, source_weight_column, target_weight_column, NUMERIC_FEATURES, random_state=random_state, missing_label=NO_ESPECIFICADO)
