"""Model selection: a grid search of every candidate family under household-grouped, stratified CV with weighted
metrics, and a one-standard-error rule on paired fold differences, within each family and then across families.

Every (configuration, fold) fit runs in its own single-threaded worker (``n_jobs``; ``n_jobs=1`` runs them in this
process one after another), and a boosting grid over ``max_iter`` is fitted once per fold, to its largest value, and
scored at every stage in the grid: without early stopping the first n iterations of the chain are the model fitted
with ``max_iter=n``."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from .levels import normalize_predicted_probabilities

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
    (:func:`complexity_key`) is selected. ``results`` has one row per candidate with ``model``, ``best_params`` and
    ``fold_log_losses`` (the same folds for every row)."""
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


def grouped_splits(X, y, groups, cv_splits=5, random_state=42):
    """The folds every selection uses: stratified by the target, grouped (households never straddle two folds)."""
    return list(StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state).split(X, y, groups=groups))


def fold_metrics(y, weights, predictions, probabilities, classes):
    """Weighted log loss, balanced accuracy, accuracy and macro F1 of one validation fold."""
    probabilities = normalize_predicted_probabilities(probabilities)
    return {
        "log_loss": log_loss(y, probabilities, labels=classes, sample_weight=weights),
        "balanced_accuracy": balanced_accuracy_score(y, predictions, sample_weight=weights),
        "accuracy": accuracy_score(y, predictions, sample_weight=weights),
        "f1_macro": f1_score(y, predictions, average="macro", sample_weight=weights, zero_division=0),
    }


def fit_fold(model, params, stages, X, y, sample_weights, train_index, validation_index):
    """Fit one configuration on one fold, single-threaded, and score it on the fold's validation rows. With ``stages``
    (a boosting chain fitted to the largest ``max_iter``), score it at every ``max_iter`` in ``stages`` instead."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        model = clone(model).set_params(**params)
        if isinstance(model.named_steps["classifier"], RandomForestClassifier):
            model.set_params(classifier__n_jobs=1)
        model.fit(X.iloc[train_index], y.iloc[train_index], classifier__sample_weight=sample_weights.iloc[train_index])
        X_validation, y_validation, w_validation = X.iloc[validation_index], y.iloc[validation_index], sample_weights.iloc[validation_index]
        classifier = model.named_steps["classifier"]
        if not stages:
            return [fold_metrics(y_validation, w_validation, model.predict(X_validation), model.predict_proba(X_validation), classifier.classes_)]
        transformed = model[:-1].transform(X_validation)
        staged = zip(classifier.staged_predict(transformed), classifier.staged_predict_proba(transformed))
        return [fold_metrics(y_validation, w_validation, predictions, probabilities, classifier.classes_)
                for n, (predictions, probabilities) in enumerate(staged, start=1) if n in stages]


def cv_tasks(candidates):
    """The fits of a grid search, per fold: one per configuration, except that a boosting grid over ``max_iter``
    becomes one chain per setting of the other parameters, scored at every ``max_iter``. Yields
    ``(family, fit_params, scored_params)``."""
    for name, config in candidates.items():
        grid = list(ParameterGrid(config["params"]))
        stages = config["params"].get("classifier__max_iter")
        if stages is None or not isinstance(config["model"].named_steps["classifier"], HistGradientBoostingClassifier):
            for params in grid:
                yield name, params, [params]
            continue
        chains = {}
        for params in grid:
            rest = tuple(sorted((key, value) for key, value in params.items() if key != "classifier__max_iter"))
            chains.setdefault(rest, []).append(params)
        for members in chains.values():
            members = sorted(members, key=lambda params: params["classifier__max_iter"])
            yield name, members[-1], members


def cross_validate_grid(candidates, X, y, sample_weights, splits, n_jobs=-1, progress=True):
    """Every configuration of every family scored on every fold. Returns ``{(family, repr(sorted params)): {fold:
    metrics}}``."""
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    tasks = [(name, fit, scored, fold) for name, fit, scored in cv_tasks(candidates) for fold in range(len(splits))]
    jobs = (delayed(fit_fold)(candidates[name]["model"], fit, [params["classifier__max_iter"] for params in scored] if len(scored) > 1 else None,
                              X, y, sample_weights, *splits[fold]) for name, fit, scored, fold in tasks)
    scores = {}
    results = Parallel(n_jobs=n_jobs, return_as="generator")(jobs)
    for (name, _, scored, fold), metrics in tqdm(zip(tasks, results), total=len(tasks), desc="grouped CV fits", disable=not progress):
        for params, metrics_ in zip(scored, metrics, strict=True):
            scores.setdefault((name, repr(sorted(params.items()))), {})[fold] = metrics_

    return scores


def tune(candidates, X, y, sample_weights, groups, cv_splits=5, random_state=42, n_jobs=-1, progress=True):
    """Grid search of every family in ``candidates`` (``{family: {"model": pipeline, "params": grid}}``) under a
    household-grouped stratified CV with weighted metrics; within each family and then across families the simplest
    configuration within one standard error of the best is selected, and each family's selection is refitted on all
    rows. Returns ``(summary, best_models)``; ``summary.attrs["grid_results"]`` holds every configuration."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    sample_weights, groups = sample_weights.reset_index(drop=True), groups.reset_index(drop=True)
    splits = grouped_splits(X, y, groups, cv_splits=cv_splits, random_state=random_state)
    scores = cross_validate_grid(candidates, X, y, sample_weights, splits, n_jobs=n_jobs, progress=progress)

    model_results, grid_results, best_models = [], [], {}
    for model_name, model_config in candidates.items():
        family_results = []
        for params in ParameterGrid(model_config["params"]):
            by_fold = scores[(model_name, repr(sorted(params.items())))]
            folds = {metric: [by_fold[fold][metric] for fold in range(len(splits))] for metric in by_fold[0]}
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
    """``(family, fitted pipeline)`` of the configuration the summary selected."""
    name = summary.loc[summary["selected"], "model"].iloc[0]

    return name, best_models[name]


def split_known(frame, target, group, unknown_column, n_splits=5, test_fold=0, random_state=42):
    """Household-grouped, stratified split of the rows with an observed target into training and held-out rows
    (fold ``test_fold`` of ``n_splits``)."""
    known = frame[~frame[unknown_column].astype(bool)].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_index, test_index = list(splitter.split(known, known[target], groups=known[group].astype("string")))[test_fold]

    return known.iloc[train_index].copy(), known.iloc[test_index].copy()
