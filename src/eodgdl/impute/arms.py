"""Hybrid scoring: a task has several feature arms (e.g. with and without education), each a fitted model; every row
to impute is scored by the first arm whose required covariates it has observed, with the unsupported levels of its
features marginalized (:mod:`eodgdl.impute.marginalize`)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .levels import MISSING_LABEL, assert_known_levels, prepare_features
from .marginalize import predict_level_shares, predict_proba_marginalizing


@dataclass
class Arm:
    """One feature arm: its fitted ``model`` on ``features``, the covariates a row must have observed to use it
    (``requires``), and auxiliary level models (``{feature: model}``) giving row-specific P(level | x) for the rows
    whose value of that feature is unsupported."""

    name: str
    model: object
    features: list
    requires: list = field(default_factory=list)
    auxiliary: dict = field(default_factory=dict)


def assign_arms(frame, arms, is_missing):
    """Name of the first arm of ``arms`` whose ``requires`` are all observed, per row (``is_missing(frame, feature)``
    -> boolean Series); NA where no arm applies."""
    assigned = pd.Series(pd.NA, index=frame.index, dtype="string")
    for arm in arms:
        eligible = assigned.isna()
        for feature in arm.requires:
            eligible &= ~is_missing(frame, feature)
        assigned[eligible] = arm.name

    return assigned


def score_arm(arm, rows, numeric, category_levels, level_subsets=None, missing_label=MISSING_LABEL):
    """Probabilities of ``rows`` under one arm (columns in the model's class order) and the per-row marginalized
    features: ``(probabilities, marginalized, classes)``."""
    X = prepare_features(rows, arm.features, numeric, missing_label)
    assert_known_levels(X, category_levels)
    conditional = {feature: predict_level_shares(model, X, level_subsets=level_subsets, missing_label=missing_label) for feature, model in arm.auxiliary.items()} or None
    probabilities, marginalized = predict_proba_marginalizing(arm.model, X, level_subsets=level_subsets, conditional_shares=conditional, missing_label=missing_label)

    return probabilities, marginalized, arm.model.named_steps["classifier"].classes_


def predict_arms(frame, arms, assigned, classes, numeric, category_levels, level_subsets=None, missing_label=MISSING_LABEL):
    """Every row of ``frame`` with an arm in ``assigned`` scored by that arm (:func:`score_arm`): ``(probabilities,
    marginalized)``, a rows x ``classes`` matrix in the order of ``classes`` (NaN where no arm is assigned) and the
    marginalized features per row ("" where none or unassigned)."""
    probabilities = np.full((len(frame), len(classes)), np.nan)
    marginalized = np.full(len(frame), "", dtype=object)
    for arm in arms:
        assert set(arm.model.named_steps["classifier"].classes_) == set(classes), f"Model classes {list(arm.model.named_steps['classifier'].classes_)} differ from {list(classes)}"
    for arm in arms:
        mask = (assigned == arm.name).fillna(False).to_numpy()
        if not mask.any():
            continue
        predicted, names, model_classes = score_arm(arm, frame.loc[mask], numeric, category_levels, level_subsets=level_subsets, missing_label=missing_label)
        probabilities[mask] = predicted[:, [list(model_classes).index(label) for label in classes]]
        marginalized[mask] = names.to_numpy()

    return probabilities, marginalized


def set_class_probabilities(frame, prefix, classes, probabilities, rows):
    """Write ``probabilities`` (one row per selected row, columns in the order of ``classes``) to ``prob_<prefix>_<class>``
    of the ``rows`` (a boolean mask) of ``frame``, in place, with their arg-max as ``<prefix>_imputado`` and
    ``<prefix>_final`` and their maximum as ``<prefix>_prediction_confidence``."""
    rows = np.asarray(rows, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    for index, label in enumerate(classes):
        frame.loc[rows, f"prob_{prefix}_{label}"] = probabilities[:, index]
    predictions = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
    frame.loc[rows, f"{prefix}_imputado"] = predictions
    frame.loc[rows, f"{prefix}_final"] = predictions
    frame.loc[rows, f"{prefix}_prediction_confidence"] = probabilities.max(axis=1)

    return frame


def impute_with_arms(frame, target, classes, arms, assigned, prefix, numeric, category_levels, level_subsets=None, missing_label=MISSING_LABEL):
    """Score every row with an arm in ``assigned`` and write the task's outputs: ``<prefix>_observado``,
    ``<prefix>_imputado``, ``<prefix>_final`` (arg-max, a convenience), ``<prefix>_fue_imputado``,
    ``<prefix>_model_used``, ``<prefix>_prediction_confidence``, ``<prefix>_marginalized_features`` and
    ``prob_<prefix>_<class>`` (rows sum to one; observed rows are one-hot). ``target`` holds the observed class, NA
    where unknown; rows to impute are those with an assigned arm."""
    frame = frame.copy()
    unknown = assigned.notna().to_numpy()

    frame[f"{prefix}_observado"] = frame[target].where(~unknown, pd.NA).astype("string")
    frame[f"{prefix}_imputado"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame[f"{prefix}_final"] = frame[f"{prefix}_observado"].copy()
    frame[f"{prefix}_fue_imputado"] = unknown
    frame[f"{prefix}_model_used"] = pd.Series("observed", index=frame.index, dtype="string")
    frame[f"{prefix}_prediction_confidence"] = np.nan
    frame[f"{prefix}_marginalized_features"] = pd.Series("", index=frame.index, dtype="string")
    for label in classes:
        frame[f"prob_{prefix}_{label}"] = (~unknown & frame[target].eq(label)).astype(float)

    if unknown.any():
        probabilities, marginalized = predict_arms(frame, arms, assigned, classes, numeric, category_levels, level_subsets=level_subsets, missing_label=missing_label)
        set_class_probabilities(frame, prefix, classes, probabilities[unknown], unknown)
        frame.loc[unknown, f"{prefix}_model_used"] = assigned[unknown].to_numpy()
        frame.loc[unknown, f"{prefix}_marginalized_features"] = marginalized[unknown]

    return frame
