"""The category-level contract every model is built with: the declared levels of each categorical feature plus the
missing label. A value outside the declared levels raises (:func:`assert_known_levels`); a declared level with no
training rows is marginalized at scoring time (:mod:`eodgdl.impute.marginalize`); a missing value becomes the missing
label (:func:`prepare_features`)."""

import numpy as np
import pandas as pd

MISSING_LABEL = "no_especificado"


def split_feature_types(features, numeric):
    """``(numerical, categorical)``: the features in ``numeric`` and the rest, both in the order of ``features``."""
    numerical = [column for column in features if column in numeric]
    categorical = [column for column in features if column not in numerical]

    return numerical, categorical


def identify_missing(series, missing_label=MISSING_LABEL):
    """Rows whose value is NA, blank or the missing label (case-insensitive)."""
    text = series.astype("string").str.strip().str.lower()
    missing = series.isna() | text.eq("").fillna(False) | text.eq(missing_label).fillna(False)

    return missing.fillna(False).astype(bool)


def prepare_features(dataframe, features, numeric, missing_label=MISSING_LABEL):
    """Select the model features: numeric columns coerced, categorical columns as clean strings with the missing
    label for NA and blanks. The first step of every pipeline (picklable through ``FunctionTransformer`` kw_args)."""
    X = dataframe[features].copy()
    numerical_features, categorical_features = split_feature_types(features, numeric)
    for column in numerical_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical_features:
        X[column] = X[column].astype("string").str.strip().replace("", missing_label).fillna(missing_label).astype(object)

    return X


def assert_known_levels(X, category_levels):
    """Raise if a categorical feature holds a value outside its declared level list."""
    unknown = {}
    for column in X.columns:
        if column in category_levels:
            values = set(X[column].dropna().unique()) - set(category_levels[column])
            if values:
                unknown[column] = sorted(map(str, values))
    if unknown:
        raise ValueError(f"Values outside the declared category levels: {unknown}")


def count_levels_without_training_support(X_train, X, numeric):
    """Rows of ``X`` whose categorical value never appears in ``X_train`` (no training support), per feature."""
    _, categorical_features = split_feature_types(list(X.columns), numeric)
    rows = []
    for column in categorical_features:
        seen = set(X_train[column].dropna().unique())
        unsupported = ~X[column].isin(seen)
        if unsupported.any():
            rows.append({"feature": column, "rows": int(unsupported.sum()), "levels": sorted(map(str, X.loc[unsupported, column].unique()))})

    return pd.DataFrame(rows, columns=["feature", "rows", "levels"])


def normalize_sample_weights(sample_weights):
    """Weights rescaled to mean one (every model is trained on survey weights normalized this way)."""
    if sample_weights.isna().any():
        raise ValueError("Sample weights contain missing values.")
    sample_weights = sample_weights.astype(float)

    return sample_weights / sample_weights.mean()


def normalize_predicted_probabilities(probabilities, tolerance=1e-8):
    """Probability rows checked to be finite and to sum to one (within ``tolerance``), then renormalized exactly."""
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
