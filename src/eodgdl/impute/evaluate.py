"""Evaluation of a fitted classifier and of an imputation: held-out metrics with household-bootstrap intervals, the
weighted-marginal baseline, one-vs-rest reliability, known vs unknown profiles, density-ratio reweighting for
covariate-shift sensitivity, and the delta adjustment of an imputed share."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support

from .levels import MISSING_LABEL, identify_missing, normalize_predicted_probabilities, prepare_features, split_feature_types


# Calibration
def calculate_calibration_table(y_true, probabilities, sample_weights, n_bins=10):
    """Weighted reliability table: observed rate vs mean predicted probability per probability bin."""
    calibration = pd.DataFrame({
        "outcome": np.asarray(y_true, dtype=float),
        "probability": np.asarray(probabilities, dtype=float),
        "weight": np.asarray(sample_weights, dtype=float),
    })
    calibration["bin"] = pd.cut(calibration["probability"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    calibration["weighted_outcome"] = calibration["weight"] * calibration["outcome"]
    calibration["weighted_probability"] = calibration["weight"] * calibration["probability"]
    table = calibration.groupby("bin", observed=True).agg(sample_workers=("outcome", "size"), weighted_population=("weight", "sum"), weighted_positive=("weighted_outcome", "sum"), weighted_probability=("weighted_probability", "sum")).reset_index()
    table["observed_rate"] = table["weighted_positive"] / table["weighted_population"]
    table["predicted_probability"] = table["weighted_probability"] / table["weighted_population"]

    return table


def calibration_metrics(y_true, probabilities, sample_weights, n_bins=10):
    """Population-weighted calibration summary for a binary probability: ``ece``, ``calibration_gap_pp``
    (predicted − observed aggregate rate, pp), ``calibration_slope`` / ``calibration_intercept`` (weighted logistic
    regression of the outcome on logit(p̂); slope 1, intercept 0 = perfect)."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    w = np.asarray(sample_weights, dtype=float)
    table = calculate_calibration_table(y, p, w, n_bins=n_bins)
    ece = float((table["weighted_population"] / table["weighted_population"].sum() * (table["observed_rate"] - table["predicted_probability"]).abs()).sum())
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y.astype(int), sample_weight=w)

    return {
        "ece": ece,
        "calibration_gap_pp": float((np.average(p, weights=w) - np.average(y, weights=w)) * 100),
        "calibration_slope": float(fit.coef_[0, 0]),
        "calibration_intercept": float(fit.intercept_[0]),
    }


def calibration_by_class(model, test_data, features, target, numeric, n_bins=10, weight_column="ponderador", class_column="class", missing_label=MISSING_LABEL):
    """One-vs-rest calibration of every class on held-out data: ``(in_the_large, reliability)``."""
    X = prepare_features(test_data, features, numeric, missing_label)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = list(model.named_steps["classifier"].classes_)
    weights = test_data[weight_column].astype(float).to_numpy()
    rows, tables = [], []
    for index, label in enumerate(classes):
        outcome = (test_data[target].astype(str).to_numpy() == label).astype(float)
        metrics = calibration_metrics(outcome, probabilities[:, index], weights, n_bins=n_bins)
        rows.append({class_column: label, "observed_share": np.average(outcome, weights=weights), "predicted_share": np.average(probabilities[:, index], weights=weights), "gap_pp": metrics["calibration_gap_pp"], "ece": metrics["ece"], "calibration_slope": metrics["calibration_slope"]})
        tables.append(calculate_calibration_table(outcome, probabilities[:, index], weights, n_bins=n_bins).assign(**{class_column: label}))

    return pd.DataFrame(rows), pd.concat(tables, ignore_index=True)


# Density-ratio reweighting of one population to another's covariate profile
def reweight_to_target_profile(source_rows, target_rows, features, source_weight_column, target_weight_column, numeric, random_state=42, missing_label=MISSING_LABEL):
    """Reweight ``source_rows`` so their covariate distribution matches ``target_rows`` (weighted logistic density
    ratio on ``features``; weights rescaled to the original total). Returns ``(reweighted_rows, diagnostics)`` with
    the effective sample size and the weight share of the top decile of odds ratios."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X_source = prepare_features(source_rows, features, numeric, missing_label)
    X_target = prepare_features(target_rows, features, numeric, missing_label)
    source_weight = source_rows[source_weight_column].astype(float).to_numpy()
    target_weight = target_rows[target_weight_column].astype(float).to_numpy()
    stacked = pd.concat([X_source, X_target], ignore_index=True)
    label = np.r_[np.zeros(len(X_source)), np.ones(len(X_target))]
    # equal total mass per group, mean weight 1 on the source side (keeps the penalty term from dominating)
    weight = np.r_[source_weight / source_weight.mean(), target_weight / target_weight.mean() * (len(source_weight) / len(target_weight))]
    numerical_features, categorical_features = split_feature_types(features, numeric)
    preprocessor = ColumnTransformer([("numerical", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical_features), ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)])
    classifier = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=random_state))])
    classifier.fit(stacked, label, classifier__sample_weight=weight)
    probability = classifier.predict_proba(X_source)[:, 1].clip(1e-4, 1 - 1e-4)
    odds = probability / (1 - probability)

    reweighted = source_rows.copy()
    new_weight = source_weight * odds
    reweighted[source_weight_column] = new_weight * source_weight.sum() / new_weight.sum()
    top_decile = odds >= np.quantile(odds, 0.9)
    diagnostics = pd.Series({
        "rows": len(reweighted),
        "effective_sample_size": float(new_weight.sum() ** 2 / (new_weight ** 2).sum()),
        "top_decile_weight_share": float(new_weight[top_decile].sum() / new_weight.sum()),
        "odds_ratio_median": float(np.median(odds)),
        "odds_ratio_p90": float(np.quantile(odds, 0.9)),
    })

    return reweighted, diagnostics


# Baselines and uncertainty
def marginal_log_loss(y_true, sample_weights, classes=None):
    """Weighted log loss of the constant predictor that outputs the weighted class shares (the honest reference for a
    probabilistic classifier)."""
    y = pd.Series(np.asarray(y_true)); w = np.asarray(sample_weights, dtype=float)
    classes = list(classes) if classes is not None else sorted(y.unique())
    shares = np.array([w[(y == c).to_numpy()].sum() for c in classes]) / w.sum()
    probabilities = np.tile(shares, (len(y), 1))

    return float(log_loss(y, probabilities, labels=classes, sample_weight=w))


def bootstrap_by_group(frame, group_column, metric_function, n_bootstrap=500, random_state=42, alpha=0.05):
    """Cluster bootstrap: resample the groups (households) of ``frame`` with replacement and recompute
    ``metric_function(frame) -> dict``. Returns point estimate and percentile interval per metric."""
    rng = np.random.default_rng(random_state)
    groups = frame[group_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    members = pd.Series(np.arange(len(frame))).groupby(groups).apply(lambda s: s.to_numpy()).to_dict()
    point = metric_function(frame)
    samples = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([members[g] for g in drawn])
        samples.append(metric_function(frame.iloc[index]))
    samples = pd.DataFrame(samples)

    return pd.DataFrame({"estimate": pd.Series(point), "ci_low": samples.quantile(alpha / 2), "ci_high": samples.quantile(1 - alpha / 2), "bootstrap_sd": samples.std(ddof=1)})


# Held-out evaluation
def binary_metrics(y_true, probabilities, classes, positive, sample_weights, n_bins=10):
    """Weighted ROC AUC, Brier score, F1 of the positive class, observed and predicted rates and calibration of a
    two-class probability matrix (``positive``: the class the rates and the calibration are about)."""
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(y_true) == positive).astype(int)
    w = np.asarray(sample_weights, dtype=float)
    p = np.asarray(probabilities, dtype=float)[:, list(classes).index(positive)]
    observed, predicted = np.average(y, weights=w), np.average(p, weights=w)
    metrics = {"weighted_roc_auc": roc_auc_score(y, p, sample_weight=w), "weighted_brier_score": float(np.average((p - y) ** 2, weights=w)),
               "weighted_f1": f1_score(y, (p >= 0.5).astype(int), sample_weight=w, zero_division=0),
               "observed_rate": observed, "predicted_rate": predicted, "calibration_gap_pp": (predicted - observed) * 100}
    metrics.update({key: value for key, value in calibration_metrics(y, p, w, n_bins=n_bins).items() if key != "calibration_gap_pp"})
    return metrics


def evaluate_classifier(model, test_data, features, target, classes, numeric, weight_column="ponderador", class_column="class", missing_label=MISSING_LABEL, positive=None, n_bins=10):
    """Held-out metrics, per-class metrics, weighted confusion matrix and observed vs predicted class shares:
    ``(metrics, class_metrics, confusion, distribution)``. With ``positive`` (a two-class target) the metrics add
    :func:`binary_metrics`."""
    X = prepare_features(test_data, features, numeric, missing_label)
    y_true = test_data[target].astype(str)
    weights = test_data[weight_column].astype(float)
    predictions = model.predict(X)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    model_classes = list(model.named_steps["classifier"].classes_)

    metrics = pd.DataFrame({
        "accuracy": [accuracy_score(y_true, predictions)],
        "balanced_accuracy": [balanced_accuracy_score(y_true, predictions)],
        "f1_macro": [f1_score(y_true, predictions, average="macro", zero_division=0)],
        "weighted_accuracy": [accuracy_score(y_true, predictions, sample_weight=weights)],
        "weighted_balanced_accuracy": [balanced_accuracy_score(y_true, predictions, sample_weight=weights)],
        "weighted_f1_macro": [f1_score(y_true, predictions, average="macro", sample_weight=weights, zero_division=0)],
        "weighted_log_loss": [log_loss(y_true, probabilities, labels=model_classes, sample_weight=weights)],
    })
    if positive is not None:
        for name, value in binary_metrics(y_true, probabilities, model_classes, positive, weights, n_bins=n_bins).items():
            metrics[name] = [value]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, predictions, labels=classes, sample_weight=weights, zero_division=0)
    class_metrics = pd.DataFrame({class_column: classes, "precision": precision, "recall": recall, "f1": f1, "weighted_support": support})
    confusion = pd.DataFrame(confusion_matrix(y_true, predictions, labels=classes, sample_weight=weights), index=classes, columns=classes)

    observed = pd.Series(weights.to_numpy(), index=y_true.to_numpy()).groupby(level=0).sum()
    hard = pd.Series(weights.to_numpy(), index=predictions).groupby(level=0).sum()
    probabilistic = pd.Series((weights.to_numpy()[:, None] * probabilities).sum(axis=0), index=model_classes)
    distribution = pd.DataFrame({class_column: classes})
    for name, series in (("observed", observed), ("hard_predicted", hard), ("probabilistic_predicted", probabilistic)):
        distribution[f"{name}_population"] = distribution[class_column].map(series).fillna(0.0)
        distribution[f"{name}_share"] = distribution[f"{name}_population"] / distribution[f"{name}_population"].sum()
    distribution["probabilistic_difference_pp"] = (distribution["probabilistic_predicted_share"] - distribution["observed_share"]) * 100

    return metrics, class_metrics, confusion, distribution


def test_metrics_with_uncertainty(model, test_data, features, target, numeric, n_bootstrap=500, random_state=42, weight_column="ponderador", group_column="folio_vivienda", missing_label=MISSING_LABEL, positive=None):
    """Held-out log loss / accuracy / macro-F1 with group-bootstrap intervals, plus the weighted-marginal baseline and
    the relative improvement over it."""
    X = prepare_features(test_data, features, numeric, missing_label)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = list(model.named_steps["classifier"].classes_)
    columns = [f"p_{c}" for c in classes]
    frame = pd.DataFrame(probabilities, columns=columns)
    frame["y"] = test_data[target].astype(str).to_numpy()
    frame["w"] = test_data[weight_column].astype(float).to_numpy()
    frame["household"] = test_data[group_column].astype(str).to_numpy()

    def metrics(data):
        p = data[columns].to_numpy(); predicted = np.array(classes)[p.argmax(axis=1)]
        return {
            "weighted_log_loss": log_loss(data["y"], p, labels=classes, sample_weight=data["w"]),
            "marginal_log_loss": marginal_log_loss(data["y"], data["w"], classes),
            "weighted_accuracy": accuracy_score(data["y"], predicted, sample_weight=data["w"]),
            "weighted_f1_macro": f1_score(data["y"], predicted, average="macro", sample_weight=data["w"], zero_division=0),
            **({key: value for key, value in binary_metrics(data["y"], p, classes, positive, data["w"]).items() if key in ("weighted_roc_auc", "weighted_brier_score", "calibration_gap_pp")}
               if positive is not None else {}),
        }

    summary = bootstrap_by_group(frame, "household", metrics, n_bootstrap=n_bootstrap, random_state=random_state)
    summary.loc["relative_improvement_over_marginal", "estimate"] = 1 - summary.loc["weighted_log_loss", "estimate"] / summary.loc["marginal_log_loss", "estimate"]

    return summary


# Known vs unknown rows
def compare_known_unknown_profiles(frame, columns, unknown, weight_column="ponderador"):
    """Weighted distribution of ``columns`` among rows with an observed target and rows without (``unknown``)."""
    unknown = np.asarray(unknown, dtype=bool)
    known_rows, unknown_rows = frame[~unknown], frame[unknown]
    comparisons = []
    for column in columns:
        known_distribution = known_rows.groupby(column, dropna=False)[weight_column].sum().reset_index(name="known_population")
        unknown_distribution = unknown_rows.groupby(column, dropna=False)[weight_column].sum().reset_index(name="unknown_population")
        known_distribution["known_share"] = known_distribution["known_population"] / known_distribution["known_population"].sum()
        unknown_distribution["unknown_share"] = unknown_distribution["unknown_population"] / unknown_distribution["unknown_population"].sum()
        comparison = known_distribution.merge(unknown_distribution, on=column, how="outer")
        comparison["variable"] = column
        comparison["category"] = comparison[column].astype("string")
        comparison[["known_share", "unknown_share"]] = comparison[["known_share", "unknown_share"]].fillna(0)
        comparison["difference_pp"] = (comparison["unknown_share"] - comparison["known_share"]) * 100
        comparisons.append(comparison[["variable", "category", "known_population", "unknown_population", "known_share", "unknown_share", "difference_pp"]])

    return pd.concat(comparisons, ignore_index=True)


def calculate_feature_missingness(frame, features, unknown, weight_column="ponderador", missing_label=MISSING_LABEL):
    """Weighted share of rows with each feature missing, among known- and unknown-target rows."""
    unknown = np.asarray(unknown, dtype=bool)
    known_rows, unknown_rows = frame[~unknown], frame[unknown]
    rows = []
    for column in features:
        rows.append({
            "variable": column,
            "known_missing_share": known_rows.loc[identify_missing(known_rows[column], missing_label), weight_column].sum() / known_rows[weight_column].sum(),
            "unknown_missing_share": unknown_rows.loc[identify_missing(unknown_rows[column], missing_label), weight_column].sum() / unknown_rows[weight_column].sum(),
        })

    return pd.DataFrame(rows)


# Outputs of an imputation
def validate_probability_rows(frame, columns, tolerance=1e-8):
    """Raise unless every row of ``frame[columns]`` sums to one; returns the largest error."""
    maximum_error = float(np.max(np.abs(frame[columns].sum(axis=1) - 1.0)))
    if maximum_error > tolerance:
        raise ValueError(f"Probabilities in {columns} do not sum to one. Maximum error: {maximum_error:.3e}")

    return maximum_error


def adjust_imputed_share(imputed, prefix, classes, label, target_share=None, weight_column="ponderador"):
    """Delta adjustment: scale the imputed probability of ``label`` so the weighted imputed share equals
    ``target_share`` (default: its observed share among the rows with an observed target), renormalizing the other
    classes. Returns ``(adjusted, factor)``. The scaled probability is clipped at one, so the target is not reached
    where the clip binds."""
    imputed = imputed.copy()
    columns = [f"prob_{prefix}_{c}" for c in classes]
    rows = imputed[f"{prefix}_fue_imputado"].to_numpy()
    weights = imputed[weight_column].astype(float)
    column = f"prob_{prefix}_{label}"
    if target_share is None:
        target_share = (weights[~rows] * (imputed.loc[~rows, f"{prefix}_final"] == label).to_numpy(dtype=float)).sum() / weights[~rows].sum()
    current_share = (weights[rows] * imputed.loc[rows, column]).sum() / weights[rows].sum()
    factor = target_share / current_share
    other_columns = [c for c in columns if c != column]
    adjusted = (imputed.loc[rows, column] * factor).clip(upper=1.0)
    remaining = 1.0 - adjusted
    other_total = imputed.loc[rows, other_columns].sum(axis=1).replace(0, np.nan)
    for other in other_columns:
        imputed.loc[rows, other] = (imputed.loc[rows, other] / other_total * remaining).fillna(0.0)
    imputed.loc[rows, column] = adjusted
    imputed.loc[rows, f"{prefix}_final"] = np.array(classes)[imputed.loc[rows, columns].to_numpy().argmax(axis=1)]
    imputed.loc[rows, f"{prefix}_imputado"] = imputed.loc[rows, f"{prefix}_final"]
    validate_probability_rows(imputed, columns)

    return imputed, float(factor)


def weighted_distribution(frame, column, weight_column="ponderador", class_column="class"):
    """Weighted population and share of each value of ``column`` (hard labels: a summary, never a model input)."""
    distribution = frame.groupby(column, dropna=False)[weight_column].sum().reset_index(name="weighted_population").rename(columns={column: class_column})
    distribution["weighted_share"] = distribution["weighted_population"] / distribution["weighted_population"].sum()

    return distribution


def probabilistic_distribution(frame, prefix, classes, weight_column="ponderador", class_column="class"):
    """Weighted population and share of each class, summing the probabilities."""
    distribution = pd.DataFrame({class_column: classes, "weighted_population": [(frame[weight_column] * frame[f"prob_{prefix}_{c}"]).sum() for c in classes]})
    distribution["weighted_share"] = distribution["weighted_population"] / distribution["weighted_population"].sum()

    return distribution


# Isotonic recalibration (a diagnostic; shipped only when the task spec asks for it)
class IsotonicCalibrated:
    """A fitted pipeline whose class probabilities pass through one isotonic map per class (one-vs-rest) and are
    renormalized. Exposes what scoring relies on (``named_steps``, ``predict_proba``, ``predict``,
    ``training_level_shares_``), so marginalization calibrates every combination before averaging."""

    def __init__(self, pipeline, calibrators):
        self.pipeline = pipeline
        self.calibrators = calibrators          # class -> fitted IsotonicRegression, in the pipeline's class order
        if hasattr(pipeline, "training_level_shares_"):
            self.training_level_shares_ = pipeline.training_level_shares_

    @property
    def named_steps(self):
        return self.pipeline.named_steps

    @property
    def classes_(self):
        return self.pipeline.named_steps["classifier"].classes_

    def calibrate(self, raw):
        calibrated = np.column_stack([self.calibrators[label].predict(raw[:, index]) for index, label in enumerate(self.classes_)])
        calibrated = np.clip(calibrated, 0.0, 1.0)
        totals = calibrated.sum(axis=1, keepdims=True)
        return np.where(totals > 0, calibrated / np.where(totals > 0, totals, 1.0), raw)

    def predict_proba(self, X):
        return self.calibrate(self.pipeline.predict_proba(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def out_of_fold_probabilities(model, X, y, sample_weights, splits):
    """Each row's probabilities from a clone of ``model`` fitted on the other folds (households never shared)."""
    from sklearn.base import clone

    X = X.reset_index(drop=True); y = pd.Series(np.asarray(y)); w = pd.Series(np.asarray(sample_weights, dtype=float))
    probabilities, classes = None, None
    for train_index, validation_index in splits:
        fold_model = clone(model).fit(X.iloc[train_index], y.iloc[train_index], classifier__sample_weight=w.iloc[train_index])
        fold_classes = fold_model.named_steps["classifier"].classes_
        if probabilities is None:
            classes = fold_classes
            probabilities = np.full((len(X), len(classes)), np.nan)
        assert list(fold_classes) == list(classes), "a fold is missing a class"
        probabilities[validation_index] = fold_model.predict_proba(X.iloc[validation_index])
    assert not np.isnan(probabilities).any()
    return probabilities, classes


def fit_isotonic(fitted, unfitted, X, y, sample_weights, splits):
    """:class:`IsotonicCalibrated` around ``fitted``, its maps learned from the out-of-fold probabilities of
    ``unfitted`` (the same configuration) on ``X`` under ``splits``, so the maps are not optimistic."""
    from sklearn.isotonic import IsotonicRegression

    probabilities, classes = out_of_fold_probabilities(unfitted, X, y, sample_weights, splits)
    y = np.asarray(y); w = np.asarray(sample_weights, dtype=float)
    calibrators = {label: IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(probabilities[:, index], (y == label).astype(float), sample_weight=w)
                   for index, label in enumerate(classes)}
    return IsotonicCalibrated(fitted, calibrators)


def multiclass_calibration_summary(y_true, probabilities, classes, sample_weights, n_bins=10):
    """Weighted log loss and the population-weighted mean one-vs-rest ECE of a probability matrix."""
    y = np.asarray(y_true); w = np.asarray(sample_weights, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    eces = [calibration_metrics((y == label).astype(float), probabilities[:, index], w, n_bins=n_bins)["ece"] for index, label in enumerate(classes)]
    shares = np.array([w[y == label].sum() for label in classes]) / w.sum()
    return {"weighted_log_loss": float(log_loss(y, np.clip(probabilities, 1e-15, 1), labels=list(classes), sample_weight=w)),
            "mean_ece": float(np.mean(eces)), "share_weighted_ece": float(np.dot(shares, eces))}
