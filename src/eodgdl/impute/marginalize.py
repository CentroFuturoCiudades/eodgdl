"""Marginalization of categorical values without training support.

A model has seen only the levels its training rows hold. For a scored row whose value of feature *f* is another
declared level, or the missing label, the prediction is averaged over the supported levels of *f*: with the training
share of each level, or with a row-specific P(level | x) from an auxiliary model. Rows with several unsupported
features are expanded over every combination.

The expansion is built for all rows at once and the model is called **once** (:func:`expand_unsupported`,
:func:`predict_proba_marginalizing`); the probabilities are the same, to the last bit, as a recursion that calls
``predict_proba`` feature by feature and level by level, because each row's contributions are added in the same order
(earlier features major, levels in the order of the shares).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .levels import MISSING_LABEL, normalize_predicted_probabilities, split_feature_types


def compute_training_level_shares(X, sample_weights, numeric):
    """Weighted share of each level of every categorical feature of ``X`` (the training rows)."""
    _, categorical_features = split_feature_types(list(X.columns), numeric)
    weights = pd.Series(np.asarray(sample_weights, dtype=float), index=X.index)
    shares = {}
    for column in categorical_features:
        totals = weights.groupby(X[column].astype(str)).sum()
        shares[column] = totals / totals.sum()

    return shares


def attach_training_level_shares(model, X, sample_weights, numeric):
    """Store the training level shares on the fitted pipeline (``training_level_shares_``), so a pickled model
    carries what its marginalization needs."""
    model.training_level_shares_ = compute_training_level_shares(X, sample_weights, numeric)

    return model


@dataclass
class Expansion:
    """Every scored row expanded over the supported levels of its unsupported features.

    ``frame`` holds one row per combination (the input's other columns unchanged), ``row`` the input position each
    comes from, ``weight`` its weight (the product of the level shares; one where nothing was marginalized), and
    ``marginalized`` the features marginalized per input row, in column order. Rows of the same input row are
    consecutive, in the order their contributions must be added."""

    frame: pd.DataFrame
    row: np.ndarray
    weight: np.ndarray
    marginalized: list

    def combine(self, probabilities, n_rows):
        """Weighted sum of the combinations' ``probabilities`` per input row (added in expansion order)."""
        probabilities = np.asarray(probabilities, dtype=float)
        result = np.zeros((n_rows, probabilities.shape[1]))
        np.add.at(result, self.row, self.weight[:, None] * probabilities)

        return result


def _supported_levels(shares, missing_label):
    # The missing label is never a supported level: a few training rows with the missing label must not turn an
    # unobserved value into a category of its own (it would be routed like the residual training level).
    supported = shares[(shares > 0) & (shares.index != missing_label)]

    return supported / supported.sum()


def _row_sums(table):
    """Row sums added left to right, column by column. ``table.sum(axis=1)`` rounds differently with the array's
    memory layout (pairwise along a contiguous row, sequential across columns otherwise), so a row's normalized shares
    would depend on which rows are scored with it; this order is the same for any batch (and is what numpy gives a
    column-major table of several rows)."""
    totals = table[:, :1].copy()
    for column in range(1, table.shape[1]):
        totals += table[:, column:column + 1]

    return totals


def _averaging_shares(shares, level_subsets):
    """The shares unsupported rows are averaged over: the training shares, or the renormalized subset of levels
    ``level_subsets`` requests for a feature (e.g. an unsampled municipality averaged over the sampled ones only)."""
    averaging = dict(shares)
    for feature, levels in (level_subsets or {}).items():
        if feature not in shares:          # a feature this model does not use (another arm's)
            continue
        subset = shares[feature].reindex(levels).dropna()
        if subset.sum() <= 0:
            raise ValueError(f"No training support for the requested {feature} levels {list(levels)}")
        averaging[feature] = pd.Series(0.0, index=shares[feature].index).add(subset / subset.sum(), fill_value=0.0)

    return averaging


def expand_unsupported(X, shares, level_subsets=None, conditional_shares=None, missing_label=MISSING_LABEL):
    """Expand the rows of ``X`` over the supported levels of every categorical feature (a key of ``shares``, in the
    column order of ``X``) whose value has no training support or is the missing label.

    ``shares`` (feature -> training level shares) decides which observed levels are supported. ``level_subsets``
    (feature -> levels) only narrows what unsupported rows are averaged over: a row whose observed level has training
    support is kept as is even when that level lies outside the subset. ``conditional_shares`` (feature -> DataFrame,
    one row per row of ``X`` in order, one column per level) replaces the averaging shares with row-specific
    P(level | x), renormalized over the averaging levels."""
    X = X.reset_index(drop=True)
    averaging_shares = _averaging_shares(shares, level_subsets)
    conditional = {feature: table.reset_index(drop=True) for feature, table in (conditional_shares or {}).items()}
    for feature, table in conditional.items():
        assert len(table) == len(X), f"conditional shares for {feature} must have one row per row of X"

    row = np.arange(len(X))
    weight = np.ones(len(X))
    values = {}  # feature -> its value per combination, once the feature has been expanded
    marginalized = [[] for _ in range(len(X))]
    for feature in [column for column in X.columns if column in shares]:
        supported = _supported_levels(shares[feature], missing_label)
        averaging = _supported_levels(averaging_shares[feature], missing_label)
        current = X[feature].to_numpy()[row]
        unsupported = ~pd.Series(current).astype(str).isin(supported.index).to_numpy()
        if not unsupported.any():
            continue
        for position in np.unique(row[unsupported]):
            marginalized[position].append(feature)

        levels = averaging.index.to_numpy(dtype=object)
        if feature in conditional:
            table = conditional[feature].reindex(columns=averaging.index, fill_value=0.0).to_numpy(dtype=float)
            totals = _row_sums(table)
            table = table / np.where(totals > 0, totals, 1.0)
            level_weights = table[row[unsupported]]                       # one row of shares per expanded combination
        else:
            level_weights = np.tile(averaging.to_numpy(dtype=float), (int(unsupported.sum()), 1))

        # Each unsupported combination becomes len(levels) consecutive combinations, one per level, in place.
        repeats = np.where(unsupported, len(levels), 1)
        take = np.repeat(np.arange(len(row)), repeats)
        expanded = np.repeat(unsupported, repeats)
        new_weight = weight[take].copy()
        new_weight[expanded] = new_weight[expanded] * level_weights.ravel()
        for name in values:
            values[name] = values[name][take]
        current = current[take].astype(object)
        current[expanded] = np.tile(levels, int(unsupported.sum()))
        values[feature] = current
        row, weight = row[take], new_weight

    frame = X.iloc[row].reset_index(drop=True)
    for feature, column in values.items():
        frame[feature] = column

    return Expansion(frame=frame, row=row, weight=weight, marginalized=marginalized)


def predict_proba_marginalizing(model, X, level_subsets=None, conditional_shares=None, missing_label=MISSING_LABEL):
    """``model.predict_proba`` where a categorical value with no training support (or the missing label) is
    marginalized out over the supported levels (:func:`expand_unsupported`), with one ``predict_proba`` call.

    Returns ``(probabilities, marginalized_features)``: rows sum to one; the second element is a per-row string with
    the marginalized features joined by "+" ("" if none)."""
    shares = getattr(model, "training_level_shares_", None)
    if not shares:
        raise ValueError("The model has no training_level_shares_; refit it with attach_training_level_shares.")
    expansion = expand_unsupported(X, shares, level_subsets=level_subsets, conditional_shares=conditional_shares, missing_label=missing_label)
    probabilities = expansion.combine(model.predict_proba(expansion.frame), len(X))
    marginalized = pd.Series(["+".join(names) for names in expansion.marginalized], index=pd.RangeIndex(len(X)), dtype=object)

    return normalize_predicted_probabilities(probabilities), marginalized


def predict_level_shares(model, X, level_subsets=None, missing_label=MISSING_LABEL):
    """Row-wise P(level | x) of an auxiliary level model as a DataFrame (columns = its classes), for the
    ``conditional_shares`` of :func:`predict_proba_marginalizing`. Rows with a predictor level without training
    support are marginalized like the main model's (honouring ``level_subsets``); a model fitted without
    ``training_level_shares_`` falls back to ``predict_proba``."""
    shares = getattr(model, "training_level_shares_", None)
    if shares:
        subsets = {feature: levels for feature, levels in (level_subsets or {}).items() if feature in shares}
        probabilities, _ = predict_proba_marginalizing(model, X, level_subsets=subsets, missing_label=missing_label)
    else:
        probabilities = model.predict_proba(X)

    return pd.DataFrame(probabilities, columns=list(model.named_steps["classifier"].classes_), index=X.index)
