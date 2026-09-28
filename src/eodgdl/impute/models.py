"""Candidate pipelines: prepare (select and clean the features from the raw frame) → preprocess (impute, encode with
the declared levels, ``handle_unknown="error"``) → classifier. Every pipeline is self-contained, so a pickled model
applies directly to the frame its source builds."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .levels import MISSING_LABEL, identify_missing, normalize_sample_weights, prepare_features, split_feature_types
from .marginalize import attach_training_level_shares

# The default grid (giro's): family -> fixed classifier arguments and the ``params`` grid (classifier arguments).
DEFAULT_FAMILIES = {
    "LogisticRegression": {"max_iter": 2000, "params": {"C": [0.1, 1.0, 10.0]}},
    "RandomForest": {"n_estimators": 500, "n_jobs": -1, "params": {"max_leaf_nodes": [25, 50, 100], "max_features": ["sqrt", 0.7], "min_samples_leaf": [1, 5, 10]}},
    # early stopping would use a row-level split that ignores households; max_iter is tuned in the grouped CV instead
    "GradientBoosting": {"native_categoricals": True, "early_stopping": False,
                         "params": {"max_iter": [50, 100, 200, 400], "learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]}},
}
CLASSIFIERS = {"LogisticRegression": LogisticRegression, "RandomForest": RandomForestClassifier, "GradientBoosting": HistGradientBoostingClassifier}


def prepare_step(features, numeric, missing_label=MISSING_LABEL):
    """The first pipeline step: :func:`~eodgdl.impute.levels.prepare_features` on ``features``."""
    return FunctionTransformer(prepare_features, kw_args={"features": list(features), "numeric": list(numeric), "missing_label": missing_label})


def make_tree_preprocessor(numerical_features, categorical_features, categories, native_categoricals=False, missing_label=MISSING_LABEL):
    """Preprocessor for tree models: one-hot (dense) by default, or ordinal-encoded with the declared categories so
    ``HistGradientBoostingClassifier`` can use native categorical splits (returns the categorical positions too)."""
    numerical = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    if native_categoricals:
        categorical = OrdinalEncoder(categories=categories, handle_unknown="error")
    else:
        categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=missing_label)), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error", sparse_output=False))])
    preprocessor = ColumnTransformer([("numerical", numerical, numerical_features), ("categorical", categorical, categorical_features)])
    categorical_positions = list(range(len(numerical_features), len(numerical_features) + len(categorical_features)))

    return preprocessor, categorical_positions


def build_candidates(features, numeric, category_levels, families=None, random_state=42, missing_label=MISSING_LABEL):
    """``{family: {"model": pipeline, "params": grid}}`` for :func:`eodgdl.impute.select.tune`, one entry per family in
    ``families`` (default :data:`DEFAULT_FAMILIES`), grid keys prefixed ``classifier__``."""
    families = DEFAULT_FAMILIES if families is None else families
    numerical_features, categorical_features = split_feature_types(features, numeric)
    categories = [category_levels[column] for column in categorical_features]
    step = prepare_step(features, numeric, missing_label=missing_label)

    candidates = {}
    for name, config in families.items():
        fixed = {key: value for key, value in config.items() if key not in ("params", "native_categoricals")}
        grid = {f"classifier__{key}": list(values) for key, values in config["params"].items()}
        if name == "LogisticRegression":
            linear_numerical = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            linear_categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=missing_label)), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error"))])
            preprocessor = ColumnTransformer([("numerical", linear_numerical, numerical_features), ("categorical", linear_categorical, categorical_features)])
            classifier = LogisticRegression(**fixed, random_state=random_state)
        elif name == "RandomForest":
            preprocessor, _ = make_tree_preprocessor(numerical_features, categorical_features, categories, missing_label=missing_label)
            classifier = RandomForestClassifier(**fixed, random_state=random_state)
        elif name == "GradientBoosting":
            native = config.get("native_categoricals", False)
            preprocessor, positions = make_tree_preprocessor(numerical_features, categorical_features, categories, native_categoricals=native, missing_label=missing_label)
            classifier = HistGradientBoostingClassifier(**fixed, random_state=random_state, categorical_features=positions if native else None)
        else:
            raise ValueError(f"Unknown model family {name!r}; known: {sorted(CLASSIFIERS)}")
        candidates[name] = {"model": Pipeline([("prepare", step), ("preprocessor", preprocessor), ("classifier", classifier)]), "params": grid}

    return candidates


def fit_level_model(frame, target, features, numeric, category_levels, sample_weights=None, random_state=42, missing_label=MISSING_LABEL):
    """Auxiliary model P(target level | features) (one-hot boosted trees with the declared levels); rows whose target
    is missing are excluded. It carries ``training_level_shares_`` so its own unsupported predictor levels are
    marginalized at scoring time (:func:`eodgdl.impute.marginalize.predict_level_shares`)."""
    observed = ~identify_missing(frame[target], missing_label)
    data = frame[observed]
    weights = None if sample_weights is None else normalize_sample_weights(pd.Series(np.asarray(sample_weights))[observed.to_numpy()])
    numerical, categorical = split_feature_types(features, numeric)
    preprocessor = ColumnTransformer([
        ("numerical", SimpleImputer(strategy="median"), numerical),
        ("categorical", OneHotEncoder(categories=[category_levels[column] for column in categorical], handle_unknown="error", sparse_output=False), categorical),
    ])
    model = Pipeline([
        ("prepare", prepare_step(features, numeric, missing_label=missing_label)),
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=random_state)),
    ])
    model.fit(data, data[target].astype(str), classifier__sample_weight=None if weights is None else weights.to_numpy())
    X = prepare_features(data, list(features), numeric, missing_label)
    attach_training_level_shares(model, X, np.ones(len(data)) if weights is None else weights.to_numpy(), numeric)

    return model
