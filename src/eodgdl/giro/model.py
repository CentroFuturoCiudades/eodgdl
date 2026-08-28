"""Probabilistic imputation of the economic activity (``giro_empresa``) of OD workers who did not report it.

Two specifications are tuned on the workers with an observed giro (household-grouped CV, one-standard-error
selection): with and without education. At scoring time each worker without a giro is sent to the specification
their data supports. The output is the full probability vector ``prob_giro_<slug>`` (rows sum to one); the
arg-max ``giro_final`` is a convenience.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ._config import DESTINATION_FEATURES, GIRO_CLASSES, MISSING_EDUCATION_LEVELS, NO_ESPECIFICADO, ROBUST_SECTOR_FEATURES, SECTOR_FEATURES, SHIFT_PROFILE_FEATURES, build_category_levels
from ._ml import (
    assert_known_levels, attach_training_level_shares, bootstrap_by_group, calculate_calibration_table, calibration_metrics,
    fit_level_model, identify_missing_category, make_tree_preprocessor, marginal_log_loss, normalize_predicted_probabilities,
    normalize_sample_weights, predict_level_shares, predict_proba_marginalizing, prepare_model_features, reweight_to_target_profile,
    select_one_se, split_feature_types,
)

PROBABILITY_COLUMNS = [f"prob_giro_{giro}" for giro in GIRO_CLASSES]


def missing_education(od):
    """Rows whose education is unobserved (NA, blank, or a 'does not know' answer)."""
    return identify_missing_category(od["escolaridad"]) | od["escolaridad"].astype("string").isin(MISSING_EDUCATION_LEVELS).fillna(False)


# Diagnostics
def compare_known_unknown_profiles(od, columns, weight_column="ponderador"):
    """Weighted distribution of ``columns`` among workers with and without an observed giro."""
    known, unknown = od[~od["giro_desconocido"]], od[od["giro_desconocido"]]
    comparisons = []
    for column in columns:
        known_distribution = known.groupby(column, dropna=False)[weight_column].sum().reset_index(name="known_population")
        unknown_distribution = unknown.groupby(column, dropna=False)[weight_column].sum().reset_index(name="unknown_population")
        known_distribution["known_share"] = known_distribution["known_population"] / known_distribution["known_population"].sum()
        unknown_distribution["unknown_share"] = unknown_distribution["unknown_population"] / unknown_distribution["unknown_population"].sum()
        comparison = known_distribution.merge(unknown_distribution, on=column, how="outer")
        comparison["variable"] = column
        comparison["category"] = comparison[column].astype("string")
        comparison[["known_share", "unknown_share"]] = comparison[["known_share", "unknown_share"]].fillna(0)
        comparison["difference_pp"] = (comparison["unknown_share"] - comparison["known_share"]) * 100
        comparisons.append(comparison[["variable", "category", "known_population", "unknown_population", "known_share", "unknown_share", "difference_pp"]])

    return pd.concat(comparisons, ignore_index=True)


def calculate_feature_missingness(od, features=SECTOR_FEATURES, weight_column="ponderador"):
    known, unknown = od[~od["giro_desconocido"]], od[od["giro_desconocido"]]
    rows = []
    for column in features:
        rows.append({
            "variable": column,
            "known_missing_share": known.loc[identify_missing_category(known[column]), weight_column].sum() / known[weight_column].sum(),
            "unknown_missing_share": unknown.loc[identify_missing_category(unknown[column]), weight_column].sum() / unknown[weight_column].sum(),
        })

    return pd.DataFrame(rows)


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
    """Candidate families and their grids. Each pipeline is self-contained: the first step selects and cleans the
    features from the raw worker frame, so a pickled bundle applies directly to ``build_worker_features`` output."""
    numerical_features, categorical_features = split_feature_types(features)
    category_levels = build_category_levels()
    categories = [category_levels[column] for column in categorical_features]

    linear_numerical = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    linear_categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=NO_ESPECIFICADO)), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error"))])
    tree_numerical = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    tree_categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=NO_ESPECIFICADO)), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error", sparse_output=False))])
    prepare = FunctionTransformer(prepare_model_features, kw_args={"features": list(features)})
    linear_preprocessor = ColumnTransformer([("numerical", linear_numerical, numerical_features), ("categorical", linear_categorical, categorical_features)])
    tree_preprocessor = ColumnTransformer([("numerical", tree_numerical, numerical_features), ("categorical", tree_categorical, categorical_features)])
    boosting_preprocessor, boosting_categorical = make_tree_preprocessor(numerical_features, categorical_features, categories, native_categoricals=native_categoricals)

    return {
        "LogisticRegression": {
            "model": Pipeline([("prepare", prepare), ("preprocessor", linear_preprocessor), ("classifier", LogisticRegression(max_iter=2000, random_state=random_state))]),
            "params": {"classifier__C": [0.1, 1.0, 10.0]},
        },
        "RandomForest": {
            "model": Pipeline([("prepare", prepare), ("preprocessor", tree_preprocessor), ("classifier", RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1))]),
            "params": {"classifier__max_leaf_nodes": [25, 50, 100], "classifier__max_features": ["sqrt", 0.7], "classifier__min_samples_leaf": [1, 5, 10]},
        },
        "GradientBoosting": {
            # early stopping would use a row-level split that ignores households; max_iter is tuned in the grouped CV instead
            "model": Pipeline([("prepare", prepare), ("preprocessor", boosting_preprocessor), ("classifier", HistGradientBoostingClassifier(early_stopping=False, random_state=random_state, categorical_features=boosting_categorical if native_categoricals else None))]),
            "params": {"classifier__max_iter": [50, 100, 200, 400], "classifier__learning_rate": [0.05, 0.1], "classifier__max_leaf_nodes": [15, 31], "classifier__l2_regularization": [0.0, 1.0]},
        },
    }


def tune_models(X, y, sample_weights, groups, features=SECTOR_FEATURES, cv_splits=5, random_state=42):
    """Grid search of every family under a household-grouped stratified CV with weighted metrics; within each family
    and then across families the simplest configuration within one standard error of the best is selected.
    Returns ``(summary, best_models)``; ``summary.attrs["grid_results"]`` holds every configuration."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    sample_weights, groups = sample_weights.reset_index(drop=True), groups.reset_index(drop=True)
    models = build_models(features=features, random_state=random_state)
    splits = list(StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state).split(X, y, groups=groups))

    model_results, grid_results, best_models = [], [], {}
    for model_name, model_config in models.items():
        family_results = []
        for params in ParameterGrid(model_config["params"]):
            folds = {"log_loss": [], "balanced_accuracy": [], "accuracy": [], "f1_macro": []}
            for train_index, validation_index in splits:
                model = clone(model_config["model"]).set_params(**params)
                model.fit(X.iloc[train_index], y.iloc[train_index], classifier__sample_weight=sample_weights.iloc[train_index])
                X_validation, y_validation, w_validation = X.iloc[validation_index], y.iloc[validation_index], sample_weights.iloc[validation_index]
                predictions = model.predict(X_validation)
                probabilities = normalize_predicted_probabilities(model.predict_proba(X_validation))
                classes = model.named_steps["classifier"].classes_
                folds["log_loss"].append(log_loss(y_validation, probabilities, labels=classes, sample_weight=w_validation))
                folds["balanced_accuracy"].append(balanced_accuracy_score(y_validation, predictions, sample_weight=w_validation))
                folds["accuracy"].append(accuracy_score(y_validation, predictions, sample_weight=w_validation))
                folds["f1_macro"].append(f1_score(y_validation, predictions, average="macro", sample_weight=w_validation, zero_division=0))
            family_results.append({
                "model": model_name,
                "weighted_log_loss": np.mean(folds["log_loss"]), "weighted_log_loss_std": np.std(folds["log_loss"]),
                "weighted_balanced_accuracy": np.mean(folds["balanced_accuracy"]), "weighted_accuracy": np.mean(folds["accuracy"]),
                "weighted_f1_macro": np.mean(folds["f1_macro"]), "best_params": params,
                "fold_log_losses": [float(value) for value in folds["log_loss"]],
            })
        family_table = select_one_se(pd.DataFrame(family_results))
        best_result = family_results[int(family_table.index[family_table["selected"]][0])]
        best_model = clone(model_config["model"]).set_params(**best_result["best_params"])
        best_model.fit(X, y, classifier__sample_weight=sample_weights)
        grid_results.extend(family_results)
        model_results.append(best_result)
        best_models[model_name] = best_model

    summary = select_one_se(pd.DataFrame(model_results)).sort_values("weighted_log_loss").reset_index(drop=True)
    summary.attrs["grid_results"] = pd.DataFrame(grid_results)

    return summary, best_models


def get_best_model(summary, best_models):
    name = summary.loc[summary["selected"], "model"].iloc[0]

    return name, best_models[name]


# Test evaluation
def evaluate_model(model, test_data, features=SECTOR_FEATURES, weight_column="ponderador"):
    """Held-out metrics, per-class metrics, weighted confusion matrix and observed vs predicted class shares."""
    X = prepare_model_features(test_data, features)
    y_true = test_data["giro"].astype(str)
    weights = test_data[weight_column].astype(float)
    predictions = model.predict(X)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = list(model.named_steps["classifier"].classes_)

    metrics = pd.DataFrame({
        "accuracy": [accuracy_score(y_true, predictions)],
        "balanced_accuracy": [balanced_accuracy_score(y_true, predictions)],
        "f1_macro": [f1_score(y_true, predictions, average="macro", zero_division=0)],
        "weighted_accuracy": [accuracy_score(y_true, predictions, sample_weight=weights)],
        "weighted_balanced_accuracy": [balanced_accuracy_score(y_true, predictions, sample_weight=weights)],
        "weighted_f1_macro": [f1_score(y_true, predictions, average="macro", sample_weight=weights, zero_division=0)],
        "weighted_log_loss": [log_loss(y_true, probabilities, labels=classes, sample_weight=weights)],
    })
    precision, recall, f1, support = precision_recall_fscore_support(y_true, predictions, labels=GIRO_CLASSES, sample_weight=weights, zero_division=0)
    class_metrics = pd.DataFrame({"giro": GIRO_CLASSES, "precision": precision, "recall": recall, "f1": f1, "weighted_support": support})
    confusion = pd.DataFrame(confusion_matrix(y_true, predictions, labels=GIRO_CLASSES, sample_weight=weights), index=GIRO_CLASSES, columns=GIRO_CLASSES)

    observed = pd.Series(weights.to_numpy(), index=y_true.to_numpy()).groupby(level=0).sum()
    hard = pd.Series(weights.to_numpy(), index=predictions).groupby(level=0).sum()
    probabilistic = pd.Series((weights.to_numpy()[:, None] * probabilities).sum(axis=0), index=classes)
    distribution = pd.DataFrame({"giro": GIRO_CLASSES})
    for name, series in (("observed", observed), ("hard_predicted", hard), ("probabilistic_predicted", probabilistic)):
        distribution[f"{name}_population"] = distribution["giro"].map(series).fillna(0.0)
        distribution[f"{name}_share"] = distribution[f"{name}_population"] / distribution[f"{name}_population"].sum()
    distribution["probabilistic_difference_pp"] = (distribution["probabilistic_predicted_share"] - distribution["observed_share"]) * 100

    return metrics, class_metrics, confusion, distribution


def calculate_calibration(model, test_data, features=SECTOR_FEATURES, n_bins=10, weight_column="ponderador"):
    """One-vs-rest calibration of the giro probabilities on held-out data: ``(in_the_large, reliability)``."""
    X = prepare_model_features(test_data, features)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = list(model.named_steps["classifier"].classes_)
    weights = test_data[weight_column].astype(float).to_numpy()
    rows, tables = [], []
    for index, giro in enumerate(classes):
        outcome = (test_data["giro"].astype(str).to_numpy() == giro).astype(float)
        metrics = calibration_metrics(outcome, probabilities[:, index], weights, n_bins=n_bins)
        rows.append({"giro": giro, "observed_share": np.average(outcome, weights=weights), "predicted_share": np.average(probabilities[:, index], weights=weights), "gap_pp": metrics["calibration_gap_pp"], "ece": metrics["ece"], "calibration_slope": metrics["calibration_slope"]})
        tables.append(calculate_calibration_table(outcome, probabilities[:, index], weights, n_bins=n_bins).assign(giro=giro))

    return pd.DataFrame(rows), pd.concat(tables, ignore_index=True)


def test_metrics_with_uncertainty(model, test_data, features=SECTOR_FEATURES, n_bootstrap=500, random_state=42, weight_column="ponderador"):
    """Held-out log loss / accuracy / macro-F1 with household-bootstrap intervals, plus the weighted-marginal baseline."""
    X = prepare_model_features(test_data, features)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = list(model.named_steps["classifier"].classes_)
    columns = [f"p_{c}" for c in classes]
    frame = pd.DataFrame(probabilities, columns=columns)
    frame["y"] = test_data["giro"].astype(str).to_numpy()
    frame["w"] = test_data[weight_column].astype(float).to_numpy()
    frame["household"] = test_data["folio_vivienda"].astype(str).to_numpy()

    def metrics(data):
        p = data[columns].to_numpy(); predicted = np.array(classes)[p.argmax(axis=1)]
        return {
            "weighted_log_loss": log_loss(data["y"], p, labels=classes, sample_weight=data["w"]),
            "marginal_log_loss": marginal_log_loss(data["y"], data["w"], classes),
            "weighted_accuracy": accuracy_score(data["y"], predicted, sample_weight=data["w"]),
            "weighted_f1_macro": f1_score(data["y"], predicted, average="macro", sample_weight=data["w"], zero_division=0),
        }

    summary = bootstrap_by_group(frame, "household", metrics, n_bootstrap=n_bootstrap, random_state=random_state)
    summary.loc["relative_improvement_over_marginal", "estimate"] = 1 - summary.loc["weighted_log_loss", "estimate"] / summary.loc["marginal_log_loss", "estimate"]

    return summary


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
    od = od.copy()
    for model in (model_with_education, model_without_education):
        assert set(model.named_steps["classifier"].classes_) == set(GIRO_CLASSES), f"Model classes {list(model.named_steps['classifier'].classes_)} differ from GIRO_CLASSES"
    unknown = od["giro_desconocido"].astype(bool)
    no_education = missing_education(od)
    arms = {"with_education": (unknown & ~no_education, model_with_education, with_education_features), "without_education": (unknown & no_education, model_without_education, without_education_features)}

    od["giro_observado"] = od["giro"].where(~unknown, pd.NA).astype("string")
    od["giro_imputado"] = pd.Series(pd.NA, index=od.index, dtype="string")
    od["giro_final"] = od["giro_observado"].copy()
    od["giro_fue_imputado"] = unknown
    od["giro_model_used"] = pd.Series("observed", index=od.index, dtype="string")
    od["giro_prediction_confidence"] = np.nan
    od["giro_marginalized_features"] = pd.Series("", index=od.index, dtype="string")
    for giro in GIRO_CLASSES:
        od[f"prob_giro_{giro}"] = (~unknown & od["giro"].eq(giro)).astype(float)

    for key, (mask, model, features) in arms.items():
        if not mask.any():
            continue
        X = prepare_model_features(od.loc[mask], features)
        assert_known_levels(X)
        conditional = None
        if destination_models is not None:
            predictors = [column for column in features if column not in DESTINATION_FEATURES]
            conditional = {"destino_trabajo": predict_level_shares(destination_models[key], X[predictors])}
        probabilities, marginalized = predict_proba_marginalizing(model, X, conditional_shares=conditional)
        classes = model.named_steps["classifier"].classes_
        predictions = classes[probabilities.argmax(axis=1)]
        od.loc[mask, "giro_imputado"] = predictions
        od.loc[mask, "giro_final"] = predictions
        od.loc[mask, "giro_model_used"] = key
        od.loc[mask, "giro_prediction_confidence"] = probabilities.max(axis=1)
        od.loc[mask, "giro_marginalized_features"] = marginalized.to_numpy()
        for index, giro in enumerate(classes):
            od.loc[mask, f"prob_giro_{giro}"] = probabilities[:, index]

    return od


def validate_probability_rows(od, tolerance=1e-8):
    maximum_error = float(np.max(np.abs(od[PROBABILITY_COLUMNS].sum(axis=1) - 1.0)))
    if maximum_error > tolerance:
        raise ValueError(f"Giro probabilities do not sum to one. Maximum error: {maximum_error:.3e}")

    return maximum_error


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
    distribution = od.groupby(giro_column, dropna=False)[weight_column].sum().reset_index(name="weighted_population").rename(columns={giro_column: "giro"})
    distribution["weighted_share"] = distribution["weighted_population"] / distribution["weighted_population"].sum()

    return distribution


def calculate_probabilistic_distribution(od, weight_column="ponderador"):
    distribution = pd.DataFrame({"giro": GIRO_CLASSES, "weighted_population": [(od[weight_column] * od[f"prob_giro_{giro}"]).sum() for giro in GIRO_CLASSES]})
    distribution["weighted_share"] = distribution["weighted_population"] / distribution["weighted_population"].sum()

    return distribution


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
    od_imputed = od_imputed.copy()
    imputed = od_imputed["giro_fue_imputado"].to_numpy()
    weights = od_imputed[weight_column].astype(float)
    column = f"prob_giro_{giro}"
    if target_share is None:
        target_share = (weights[~imputed] * (od_imputed.loc[~imputed, "giro_final"] == giro).to_numpy(dtype=float)).sum() / weights[~imputed].sum()
    current_share = (weights[imputed] * od_imputed.loc[imputed, column]).sum() / weights[imputed].sum()
    factor = target_share / current_share
    other_columns = [f"prob_giro_{other}" for other in GIRO_CLASSES if other != giro]
    adjusted = (od_imputed.loc[imputed, column] * factor).clip(upper=1.0)
    remaining = 1.0 - adjusted
    other_total = od_imputed.loc[imputed, other_columns].sum(axis=1).replace(0, np.nan)
    for other in other_columns:
        od_imputed.loc[imputed, other] = (od_imputed.loc[imputed, other] / other_total * remaining).fillna(0.0)
    od_imputed.loc[imputed, column] = adjusted
    od_imputed.loc[imputed, "giro_final"] = np.array(GIRO_CLASSES)[od_imputed.loc[imputed, PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)]
    od_imputed.loc[imputed, "giro_imputado"] = od_imputed.loc[imputed, "giro_final"]
    validate_probability_rows(od_imputed)

    return od_imputed, float(factor)
