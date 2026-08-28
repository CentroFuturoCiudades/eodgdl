"""Generic modelling helpers used by the giro model: feature preparation, marginalization of categorical levels
without training support, one-standard-error model selection, calibration, density-ratio reweighting, cluster
bootstrap and the auxiliary level model. Self-contained (no dependency outside sklearn/pandas)."""

import numpy as np
import pandas as pd

from ._config import NO_ESPECIFICADO, NUMERIC_FEATURES, build_category_levels


def split_feature_types(features):
    numerical = [column for column in features if column in NUMERIC_FEATURES]
    categorical = [column for column in features if column not in numerical]

    return numerical, categorical


def identify_missing_category(series):
    text = series.astype("string").str.strip().str.lower()
    missing = series.isna() | text.eq("").fillna(False) | text.eq(NO_ESPECIFICADO).fillna(False)

    return missing.fillna(False).astype(bool)


def prepare_model_features(dataframe, features):
    """Select the model features: numeric columns coerced, categorical columns as clean strings with a missing label."""
    X = dataframe[features].copy()
    numerical_features, categorical_features = split_feature_types(features)
    for column in numerical_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical_features:
        X[column] = X[column].astype("string").str.strip().replace("", NO_ESPECIFICADO).fillna(NO_ESPECIFICADO).astype(object)

    return X


def assert_known_levels(X, category_levels=None):
    """Raise if a categorical feature holds a value outside the declared level list."""
    category_levels = category_levels or build_category_levels()
    unknown = {}
    for column in X.columns:
        if column in category_levels:
            values = set(X[column].dropna().unique()) - set(category_levels[column])
            if values:
                unknown[column] = sorted(map(str, values))
    if unknown:
        raise ValueError(f"Values outside the declared category levels: {unknown}")


def count_levels_without_training_support(X_train, X):
    """Rows of ``X`` whose categorical value never appears in ``X_train`` (no training support), per feature."""
    _, categorical_features = split_feature_types(list(X.columns))
    rows = []
    for column in categorical_features:
        seen = set(X_train[column].dropna().unique())
        unsupported = ~X[column].isin(seen)
        if unsupported.any():
            rows.append({"feature": column, "rows": int(unsupported.sum()), "levels": sorted(map(str, X.loc[unsupported, column].unique()))})

    return pd.DataFrame(rows, columns=["feature", "rows", "levels"])


def normalize_sample_weights(sample_weights):
    if sample_weights.isna().any():
        raise ValueError("Sample weights contain missing values.")
    sample_weights = sample_weights.astype(float)

    return sample_weights / sample_weights.mean()


def normalize_predicted_probabilities(probabilities, tolerance=1e-8):
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values.")
    probability_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(probability_sums <= 0):
        raise ValueError("At least one predicted probability row has a non-positive sum.")
    maximum_error = np.max(np.abs(probability_sums.ravel() - 1.0))
    if maximum_error > tolerance:
        raise ValueError(f"Predicted probabilities do not sum to one. Maximum error: {maximum_error:.3e}")

    return probabilities / probability_sums


# Marginalization over categorical levels without training support
def compute_training_level_shares(X, sample_weights):
    """Weighted share of each level of every categorical feature in the training data."""
    _, categorical_features = split_feature_types(list(X.columns))
    weights = pd.Series(np.asarray(sample_weights, dtype=float), index=X.index)
    shares = {}
    for column in categorical_features:
        totals = weights.groupby(X[column].astype(str)).sum()
        shares[column] = totals / totals.sum()

    return shares


def attach_training_level_shares(model, X, sample_weights):
    """Store the training level shares on the fitted pipeline so the pickled bundle is self-contained."""
    model.training_level_shares_ = compute_training_level_shares(X, sample_weights)

    return model


def predict_proba_marginalizing(model, X, level_subsets=None, conditional_shares=None):
    """``model.predict_proba`` where a categorical value with no training support is marginalized out.

    For every row whose value of feature *f* is unsupported (or is the missing label), predict once per supported
    level of *f* and average with the training share of each level; rows with several unsupported features are
    expanded over all combinations. ``level_subsets`` (feature -> levels) restricts the levels a feature is averaged
    over; ``conditional_shares`` (feature -> DataFrame, one row per row of ``X``, one column per level) replaces the
    global training shares with row-specific P(level | x) from an auxiliary model (:func:`fit_level_model`).

    Returns ``(probabilities, marginalized_features)``; the second element is a per-row string listing the features
    that were marginalized ("" if none).
    """
    shares = getattr(model, "training_level_shares_", None)
    if not shares:
        raise ValueError("The model has no training_level_shares_; refit it with attach_training_level_shares.")
    if level_subsets:
        shares = dict(shares)
        for feature, levels in level_subsets.items():
            subset = shares[feature].reindex(levels).dropna()
            if subset.sum() <= 0:
                raise ValueError(f"No training support for the requested {feature} levels {list(levels)}")
            shares[feature] = pd.Series(0.0, index=shares[feature].index).add(subset / subset.sum(), fill_value=0.0)
    X = X.reset_index(drop=True)
    conditional = {feature: table.reset_index(drop=True) for feature, table in (conditional_shares or {}).items()}
    for feature, table in conditional.items():
        assert len(table) == len(X), f"conditional shares for {feature} must have one row per row of X"
    features = [column for column in X.columns if column in shares]
    n_classes = len(model.named_steps["classifier"].classes_)
    result = np.zeros((len(X), n_classes))
    marginalized_features = [[] for _ in range(len(X))]

    def fill(rows, weights, remaining):
        if len(rows) == 0:
            return
        if not remaining:
            result[rows.index] += weights[:, None] * model.predict_proba(rows)
            return
        feature, rest = remaining[0], remaining[1:]
        # The missing label is never a "supported" level: a few training rows with no_especificado must not turn an
        # unobserved value into a category of its own.
        supported = shares[feature][(shares[feature] > 0) & (shares[feature].index != NO_ESPECIFICADO)]
        supported = supported / supported.sum()
        unsupported = ~rows[feature].astype(str).isin(supported.index)
        fill(rows[~unsupported], weights[~unsupported.to_numpy()], rest)
        if unsupported.any():
            for row in rows.index[unsupported]:
                if feature not in marginalized_features[row]:
                    marginalized_features[row].append(feature)
            if feature in conditional:
                table = conditional[feature].loc[rows.index[unsupported]]
                row_shares = table.reindex(columns=supported.index, fill_value=0.0).to_numpy(dtype=float)
                row_shares = row_shares / np.where(row_shares.sum(axis=1, keepdims=True) > 0, row_shares.sum(axis=1, keepdims=True), 1.0)
                for column_index, level in enumerate(supported.index):
                    fill(rows[unsupported].assign(**{feature: level}), weights[unsupported.to_numpy()] * row_shares[:, column_index], rest)
            else:
                for level, share in supported.items():
                    fill(rows[unsupported].assign(**{feature: level}), weights[unsupported.to_numpy()] * share, rest)

    fill(X, np.ones(len(X)), features)
    marginalized = pd.Series(["+".join(names) for names in marginalized_features], index=X.index, dtype=object)

    return normalize_predicted_probabilities(result), marginalized


# Model selection with a one-standard-error rule on paired folds
FAMILY_COMPLEXITY = {"LogisticRegression": 0, "RandomForest": 1, "GradientBoosting": 2}
# +1: larger value = more complex; -1: larger value = simpler (more regularization)
PARAMETER_COMPLEXITY_DIRECTION = {
    "classifier__max_iter": 1, "classifier__max_leaf_nodes": 1, "classifier__learning_rate": 1, "classifier__max_features": 1,
    "classifier__C": 1, "classifier__l2_regularization": -1, "classifier__min_samples_leaf": -1,
}


def complexity_key(model_name, params):
    """Sort key: simpler families first, then simpler hyperparameters (lexicographic over sorted parameter names)."""
    values = []
    for name in sorted(params):
        value = params[name]
        if value == "sqrt":
            value = 0.3
        values.append(PARAMETER_COMPLEXITY_DIRECTION.get(name, 1) * float(value))

    return (FAMILY_COMPLEXITY.get(model_name, 99), tuple(values))


def select_one_se(results):
    """Mark the configuration to keep under a one-standard-error rule: every candidate whose paired fold-difference
    to the best mean log loss is within one standard error of zero is eligible, and the simplest eligible candidate
    (:func:`complexity_key`) is selected."""
    table = results.reset_index(drop=True).copy()
    losses = np.array([np.asarray(row, dtype=float) for row in table["fold_log_losses"]])
    table["weighted_log_loss"] = losses.mean(axis=1)
    best = int(table["weighted_log_loss"].idxmin())
    differences = losses - losses[best]
    table["mean_diff_vs_best"] = differences.mean(axis=1)
    table["se_diff_vs_best"] = differences.std(axis=1, ddof=1) / np.sqrt(losses.shape[1])
    table["within_one_se"] = table["mean_diff_vs_best"] <= table["se_diff_vs_best"]
    keys = [complexity_key(model, params) for model, params in zip(table["model"], table["best_params"])]
    table["complexity_rank"] = pd.Series(keys).rank(method="first").astype(int)
    eligible = table.index[table["within_one_se"]]
    selected = min(eligible, key=lambda index: keys[index])
    table["selected"] = False
    table.loc[selected, "selected"] = True

    return table


def fold_table(model_summary):
    """Per-fold log losses of the selected configuration of each family (from the tuning summary)."""
    rows = {}
    for _, row in model_summary.iterrows():
        rows[row["model"]] = pd.Series(row["fold_log_losses"], index=[f"fold_{k}" for k in range(len(row["fold_log_losses"]))])
    table = pd.DataFrame(rows).T
    table["mean"] = table.mean(axis=1)
    table["sd_across_folds"] = table.iloc[:, :-1].std(axis=1, ddof=1)

    return table


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


# Density-ratio reweighting of one population to another's covariate profile
def reweight_to_target_profile(source_rows, target_rows, features, source_weight_column, target_weight_column, random_state=42):
    """Reweight ``source_rows`` so their covariate distribution matches ``target_rows`` (weighted logistic density
    ratio on ``features``; weights rescaled to the original total). Returns ``(reweighted_rows, diagnostics)``."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X_source = prepare_model_features(source_rows, features)
    X_target = prepare_model_features(target_rows, features)
    source_weight = source_rows[source_weight_column].astype(float).to_numpy()
    target_weight = target_rows[target_weight_column].astype(float).to_numpy()
    stacked = pd.concat([X_source, X_target], ignore_index=True)
    label = np.r_[np.zeros(len(X_source)), np.ones(len(X_target))]
    weight = np.r_[source_weight / source_weight.mean(), target_weight / target_weight.mean() * (len(source_weight) / len(target_weight))]
    numerical_features, categorical_features = split_feature_types(features)
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
    """Weighted log loss of the constant predictor that outputs the weighted class shares."""
    from sklearn.metrics import log_loss

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


# Auxiliary model for an unobserved categorical feature
def fit_level_model(frame, target, features, sample_weights=None, random_state=42):
    """Fit P(target level | features) with a boosted-tree pipeline (same preprocessing contract as the main models);
    rows whose target is the missing label are excluded. Returns the fitted pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

    observed = ~identify_missing_category(frame[target])
    data = frame[observed]
    weights = None if sample_weights is None else normalize_sample_weights(pd.Series(np.asarray(sample_weights))[observed.to_numpy()])
    numerical, categorical = split_feature_types(features)
    levels = build_category_levels()
    preprocessor = ColumnTransformer([
        ("numerical", SimpleImputer(strategy="median"), numerical),
        ("categorical", OneHotEncoder(categories=[levels[column] for column in categorical], handle_unknown="error", sparse_output=False), categorical),
    ])
    model = Pipeline([
        ("prepare", FunctionTransformer(prepare_model_features, kw_args={"features": list(features)})),
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=random_state)),
    ])
    model.fit(data, data[target].astype(str), classifier__sample_weight=None if weights is None else weights.to_numpy())

    return model


def predict_level_shares(model, X):
    """Row-wise P(level | x) as a DataFrame (columns = the model's classes) for :func:`predict_proba_marginalizing`."""
    probabilities = model.predict_proba(X)

    return pd.DataFrame(probabilities, columns=list(model.named_steps["classifier"].classes_), index=X.index)


def make_tree_preprocessor(numerical_features, categorical_features, categories, native_categoricals=False):
    """Preprocessor for tree models: one-hot (dense) by default, or ordinal-encoded with the declared categories so
    ``HistGradientBoostingClassifier`` can use native categorical splits (returns the categorical positions too)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    numerical = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    if native_categoricals:
        categorical = OrdinalEncoder(categories=categories, handle_unknown="error")
    else:
        categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=NO_ESPECIFICADO)), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error", sparse_output=False))])
    preprocessor = ColumnTransformer([("numerical", numerical, numerical_features), ("categorical", categorical, categorical_features)])
    categorical_positions = list(range(len(numerical_features), len(numerical_features) + len(categorical_features)))

    return preprocessor, categorical_positions
