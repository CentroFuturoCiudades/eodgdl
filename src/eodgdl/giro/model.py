"""Probabilistic imputation of the economic activity (``giro_empresa``) of OD workers who did not report it.

Two specifications are tuned on the workers with an observed giro (household-grouped CV, one-standard-error
selection): with and without education. At scoring time each worker without a giro is sent to the specification
their data supports. The output is the full probability vector ``prob_giro_<slug>`` (rows sum to one); the
arg-max ``giro_final`` is a convenience.

The machinery is :mod:`eodgdl.impute`; these functions bind giro's features, levels and column names to it.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from eodgdl.impute import evaluate, select
from eodgdl.impute.arms import Arm, assign_arms, impute_with_arms
from eodgdl.impute.models import build_candidates

from ._config import (
    DESTINATION_FEATURES, GIRO_CLASSES, MISSING_EDUCATION_LEVELS, NO_ESPECIFICADO, NUMERIC_FEATURES, ROBUST_SECTOR_FEATURES, SECTOR_FEATURES,
    SHIFT_PROFILE_FEATURES, build_category_levels,
)
from ._ml import (
    attach_training_level_shares, fit_level_model, identify_missing_category, normalize_sample_weights, prepare_model_features,
    reweight_to_target_profile,
)

PROBABILITY_COLUMNS = [f"prob_giro_{giro}" for giro in GIRO_CLASSES]


def missing_education(od):
    """Rows whose education is unobserved (NA, blank, or a 'does not know' answer)."""
    return identify_missing_category(od["escolaridad"]) | od["escolaridad"].astype("string").isin(MISSING_EDUCATION_LEVELS).fillna(False)


# Diagnostics
def compare_known_unknown_profiles(od, columns, weight_column="ponderador"):
    """Weighted distribution of ``columns`` among workers with and without an observed giro."""
    return evaluate.compare_known_unknown_profiles(od, columns, od["giro_desconocido"], weight_column=weight_column)


def calculate_feature_missingness(od, features=SECTOR_FEATURES, weight_column="ponderador"):
    return evaluate.calculate_feature_missingness(od, features, od["giro_desconocido"], weight_column=weight_column, missing_label=NO_ESPECIFICADO)


# Train-test split and feature preparation
def split_known_data(od, n_splits=5, test_fold=0, random_state=42):
    """Household-grouped, stratified split of the workers with an observed giro into training and test rows."""
    known = od[~od["giro_desconocido"]].reset_index(drop=True)
    cross_validation = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_index, test_index = list(cross_validation.split(known, known["giro"], groups=known["folio_vivienda"].astype("string")))[test_fold]

    return known.iloc[train_index].copy(), known.iloc[test_index].copy()


def prepare_training_data(od, features=SECTOR_FEATURES, weight_column="ponderador"):
    training = od[~od["giro_desconocido"]].reset_index(drop=True)
    X = prepare_model_features(training, features)
    y = training["giro"].astype(str).reset_index(drop=True)
    sample_weights = normalize_sample_weights(training[weight_column]).reset_index(drop=True)
    groups = training["folio_vivienda"].astype("string").reset_index(drop=True)

    return X, y, sample_weights, groups, training


# Models
def build_models(features=SECTOR_FEATURES, random_state=42, native_categoricals=True):
    """Candidate families and their grids (:func:`eodgdl.impute.models.build_candidates`). Each pipeline is
    self-contained: the first step selects and cleans the features from the raw worker frame, so a pickled bundle
    applies directly to ``build_worker_features`` output."""
    from eodgdl.impute.models import DEFAULT_FAMILIES

    families = {name: dict(config) for name, config in DEFAULT_FAMILIES.items()}
    families["GradientBoosting"]["native_categoricals"] = native_categoricals

    return build_candidates(features, NUMERIC_FEATURES, build_category_levels(), families=families, random_state=random_state, missing_label=NO_ESPECIFICADO)


def tune_models(X, y, sample_weights, groups, features=SECTOR_FEATURES, cv_splits=5, random_state=42, n_jobs=-1):
    """Grid search of every family under a household-grouped stratified CV with weighted metrics, one-SE selection
    within and across families (:func:`eodgdl.impute.select.tune`). Returns ``(summary, best_models)``;
    ``summary.attrs["grid_results"]`` holds every configuration."""
    return select.tune(build_models(features=features, random_state=random_state), X, y, sample_weights, groups, cv_splits=cv_splits, random_state=random_state, n_jobs=n_jobs)


get_best_model = select.get_best_model


# Test evaluation
def evaluate_model(model, test_data, features=SECTOR_FEATURES, weight_column="ponderador"):
    """Held-out metrics, per-class metrics, weighted confusion matrix and observed vs predicted class shares."""
    return evaluate.evaluate_classifier(model, test_data, features, "giro", GIRO_CLASSES, NUMERIC_FEATURES, weight_column=weight_column, class_column="giro", missing_label=NO_ESPECIFICADO)


def calculate_calibration(model, test_data, features=SECTOR_FEATURES, n_bins=10, weight_column="ponderador"):
    """One-vs-rest calibration of the giro probabilities on held-out data: ``(in_the_large, reliability)``."""
    return evaluate.calibration_by_class(model, test_data, features, "giro", NUMERIC_FEATURES, n_bins=n_bins, weight_column=weight_column, class_column="giro", missing_label=NO_ESPECIFICADO)


def test_metrics_with_uncertainty(model, test_data, features=SECTOR_FEATURES, n_bootstrap=500, random_state=42, weight_column="ponderador"):
    """Held-out log loss / accuracy / macro-F1 with household-bootstrap intervals, plus the weighted-marginal baseline."""
    return evaluate.test_metrics_with_uncertainty(model, test_data, features, "giro", NUMERIC_FEATURES, n_bootstrap=n_bootstrap, random_state=random_state, weight_column=weight_column, group_column="folio_vivienda", missing_label=NO_ESPECIFICADO)


# Final models and imputation
def refit_model(model, od, features=SECTOR_FEATURES, weight_column="ponderador"):
    """Refit a selected pipeline on every worker with an observed giro and attach the training level shares."""
    X, y, sample_weights, _, training = prepare_training_data(od, features=features, weight_column=weight_column)
    final_model = clone(model).fit(X, y, classifier__sample_weight=sample_weights)
    attach_training_level_shares(final_model, X, sample_weights)

    return final_model, training


def fit_destination_models(od, with_education_features=SECTOR_FEATURES, without_education_features=ROBUST_SECTOR_FEATURES, random_state=42, weight_column="ponderador"):
    """Auxiliary models P(destino_trabajo | x) on workers with a work trip (one per specification, without the
    destination and mode features), used to marginalize workers without a work trip with their own distribution."""
    models = {}
    for key, features in (("with_education", with_education_features), ("without_education", without_education_features)):
        predictors = [column for column in features if column not in DESTINATION_FEATURES and column != "modo_trabajo"]
        models[key] = fit_level_model(od, "destino_trabajo", predictors, sample_weights=od[weight_column], random_state=random_state)

    return models


def impute_giro(model_with_education, model_without_education, od, with_education_features=SECTOR_FEATURES, without_education_features=ROBUST_SECTOR_FEATURES, destination_models=None):
    """Score every worker without an observed giro with the specification their education supports.

    Adds ``giro_observado``, ``giro_imputado``, ``giro_final``, ``giro_fue_imputado``, ``giro_model_used``,
    ``giro_prediction_confidence``, ``giro_marginalized_features`` and ``prob_giro_<slug>`` (rows sum to one;
    observed rows get probability one on their giro).
    """
    destination_models = destination_models or {}
    arms = [
        Arm("with_education", model_with_education, list(with_education_features), requires=["escolaridad"],
            auxiliary={"destino_trabajo": destination_models["with_education"]} if "with_education" in destination_models else {}),
        Arm("without_education", model_without_education, list(without_education_features),
            auxiliary={"destino_trabajo": destination_models["without_education"]} if "without_education" in destination_models else {}),
    ]
    assigned = assign_arms(od, arms, lambda frame, feature: missing_education(frame))
    assigned[~od["giro_desconocido"].astype(bool)] = pd.NA

    return impute_with_arms(od, "giro", GIRO_CLASSES, arms, assigned, "giro", NUMERIC_FEATURES, build_category_levels(), missing_label=NO_ESPECIFICADO)


def validate_probability_rows(od, tolerance=1e-8):
    return evaluate.validate_probability_rows(od, PROBABILITY_COLUMNS, tolerance=tolerance)


# Summaries
def calculate_model_usage(od, weight_column="ponderador"):
    imputed = od[od["giro_fue_imputado"]]
    usage = imputed.groupby("giro_model_used").agg(sample_workers=("giro_model_used", "size"), weighted_population=(weight_column, "sum")).reset_index()
    usage["sample_share"] = usage["sample_workers"] / usage["sample_workers"].sum()
    usage["weighted_share"] = usage["weighted_population"] / usage["weighted_population"].sum()

    return usage


def calculate_confidence_summary(od):
    return od[od["giro_fue_imputado"]].groupby("giro_model_used")["giro_prediction_confidence"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()


def calculate_distribution(od, giro_column="giro_final", weight_column="ponderador"):
    return evaluate.weighted_distribution(od, giro_column, weight_column=weight_column, class_column="giro")


def calculate_probabilistic_distribution(od, weight_column="ponderador"):
    return evaluate.probabilistic_distribution(od, "giro", GIRO_CLASSES, weight_column=weight_column, class_column="giro")


# Sensitivity of the imputation to the covariate shift between known- and unknown-giro workers
def impute_under_covariate_shift(model_with_education, model_without_education, od, with_education_features=SECTOR_FEATURES, without_education_features=ROBUST_SECTOR_FEATURES, profile_features=SHIFT_PROFILE_FEATURES, weight_column="ponderador"):
    """Re-impute after refitting the selected models on known-giro rows reweighted to the unknown-giro profile.
    Returns ``(od_imputed_under_shift, diagnostics)``."""
    known, unknown = od[~od["giro_desconocido"]], od[od["giro_desconocido"]]
    reweighted_known, diagnostics = reweight_to_target_profile(known, unknown, profile_features, weight_column, weight_column)
    shifted = pd.concat([reweighted_known, unknown]).sort_index()
    shift_with, _ = refit_model(model_with_education, shifted, features=with_education_features, weight_column=weight_column)
    shift_without, _ = refit_model(model_without_education, shifted, features=without_education_features, weight_column=weight_column)

    return impute_giro(shift_with, shift_without, od, with_education_features=with_education_features, without_education_features=without_education_features), diagnostics


def adjust_imputed_share(od_imputed, giro, target_share=None, weight_column="ponderador"):
    """Delta adjustment: scale the imputed probability of ``giro`` so the weighted imputed share equals
    ``target_share`` (default: the observed share among known-giro workers), renormalizing the other classes."""
    return evaluate.adjust_imputed_share(od_imputed, "giro", GIRO_CLASSES, giro, target_share=target_share, weight_column=weight_column)
