"""Running a task in its three modes.

- **score** (:func:`score_task`): load the task's bundle, check it, score the source's rows (seconds).
- **retrain** (:func:`retrain`): grouped-CV selection per arm on the training rows, evaluation on the held-out
  households, refit on every observed row, auxiliary models, scoring, the covariate-shift and delta scenarios; writes
  the bundle and every evaluation table (:func:`write_retrain`) for a report to read.
- **compare** (:func:`compare`): feature specifications evaluated cheaply: the published winner's family and
  hyperparameters fixed, the same grouped folds on the training rows (the held-out fold never touched), each
  candidate's fold log losses paired against the baseline's.

A mode that needs a fitted bundle takes it from :func:`eodgdl.impute.bundle.load_bundle`: ``retrained=<dir>`` reads
the one a retrain wrote under ``<dir>/<task>/`` when there is one, else the installed data file.
"""

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import evaluate, select
from .arms import assign_arms, impute_with_arms, predict_arms
from .bundle import as_v2, bundle_arms, check_bundle, load_bundle, make_bundle, save_bundle
from .features import build_frame, builder_config
from .levels import count_levels_without_training_support, normalize_sample_weights, prepare_features
from .marginalize import attach_training_level_shares
from .models import build_candidates, fit_level_model
from .sources import Context, get_source
from .spec import load_task, parse_task

OUTPUT_SUFFIXES = ("observado", "imputado", "final", "fue_imputado", "model_used", "prediction_confidence", "marginalized_features")


def task_levels(spec):
    """The task's category-level contract: its declared levels, else its source's."""
    return spec.category_levels(get_source(spec.source).schema_levels)


def expected_score_column(spec):
    return f"{spec.prefix}_puntaje_esperado"


def output_columns(spec, keys):
    """Row keys plus the task's output columns (and its expected score when the target is ordinal)."""
    columns = list(keys) + [f"{spec.prefix}_{suffix}" for suffix in OUTPUT_SUFFIXES] + spec.probability_columns
    return columns + ([expected_score_column(spec)] if spec.scores else [])


def output_frame(spec, scored, weight=False):
    """The keys of the task's scoring source and the task's output columns of ``scored`` (plus the source's weight)."""
    source = get_source(spec.score_source)
    return scored[output_columns(spec, source.keys) + ([source.weight] if weight else [])]


def select_rows(rows, keep=None, drop=None):
    """``rows`` whose columns take one of the values listed in ``keep`` and none of those in ``drop`` (each
    ``{column: [values]}``, compared as text; empty: every row), index reset."""
    mask = np.ones(len(rows), bool)
    for column, values in (keep or {}).items():
        mask &= rows[column].astype(str).isin([str(value) for value in values]).to_numpy()
    for column, values in (drop or {}).items():
        mask &= ~rows[column].astype(str).isin([str(value) for value in values]).to_numpy()
    return rows[mask].reset_index(drop=True)


def _spec(task):
    return load_task(task) if isinstance(task, str) else task


# Score
def score_frame(spec, frame, bundle, level_subsets=None):
    """Score ``frame`` (a task frame: :func:`~eodgdl.impute.features.build_frame`) with a checked ``bundle``: every
    row whose target is unobserved goes to the first arm whose required covariates it has; returns the frame with the
    task's output columns (:func:`~eodgdl.impute.arms.impute_with_arms`). ``level_subsets`` defaults to the task's
    ``scoring.level_subsets``."""
    levels = task_levels(spec)
    bundle = check_bundle(as_v2(bundle, spec), spec, levels)
    arms = bundle_arms(bundle)
    assigned = assign_arms(frame, arms, spec.is_missing)
    assigned[~frame[spec.unknown_column].astype(bool).to_numpy()] = pd.NA
    scored = impute_with_arms(frame, spec.target, spec.class_slugs, arms, assigned, spec.prefix, spec.numeric, levels,
                              level_subsets=(spec.level_subsets or None) if level_subsets is None else level_subsets, missing_label=spec.missing_label)
    evaluate.validate_probability_rows(scored, spec.probability_columns)
    if spec.scores:
        scored[expected_score_column(spec)] = scored[spec.probability_columns].to_numpy() @ np.array([spec.scores[slug] for slug in spec.class_slugs])

    return scored


def predict_rows(spec, bundle, rows, auxiliary=True):
    """P(class | x) of every row of ``rows``, whether its target is observed or not, under the bundle's arms (the
    first whose covariates the row has): one column per class slug. Without ``auxiliary`` an unsupported level is
    averaged over the training shares instead of P(level | x)."""
    arms = bundle_arms(bundle, auxiliary=auxiliary)
    probabilities, _ = predict_arms(rows, arms, assign_arms(rows, arms, spec.is_missing), spec.class_slugs, spec.numeric, task_levels(spec),
                                    level_subsets=spec.level_subsets or None, missing_label=spec.missing_label)
    return pd.DataFrame(probabilities, columns=spec.class_slugs, index=rows.index)


def score_task(task, tables=None, bundle=None, path=None, retrained=None, context=None):
    """Load (or take) the task's bundle and score its scoring source's rows. ``tables`` are the survey tables
    (loaded, and the features cached, when omitted); ``bundle`` a fitted bundle (v2 or a legacy giro dict), else read
    by :func:`~eodgdl.impute.bundle.load_bundle` (``path``, else ``<retrained>/<task>/``, else the data file)."""
    spec = _spec(task)
    context = context or Context(tables=tables)
    bundle = bundle if bundle is not None else load_bundle(spec, path, retrained)

    return score_frame(spec, build_frame(spec, context, role="score").frame, bundle)


# Retrain
def training_data(spec, rows, features, weight, group):
    """``(X, y, sample_weights, groups, rows)`` of the rows with an observed target; weights normalized to mean one."""
    rows = rows[~rows[spec.unknown_column].astype(bool)].reset_index(drop=True)
    X = prepare_features(rows, features, spec.numeric, spec.missing_label)
    y = rows[spec.target].astype(str).reset_index(drop=True)
    sample_weights = normalize_sample_weights(rows[weight]).reset_index(drop=True)
    groups = rows[group].astype("string").reset_index(drop=True)

    return X, y, sample_weights, groups, rows


def refit(model, spec, rows, features, weight, group):
    """A clone of ``model`` fitted on every row of ``rows`` with an observed target, with its training level shares."""
    X, y, sample_weights, _, _ = training_data(spec, rows, features, weight, group)
    final = clone(model).fit(X, y, classifier__sample_weight=sample_weights)
    return attach_training_level_shares(final, X, sample_weights, spec.numeric)


def heldout(model, spec, rows, weight):
    """``(y, probabilities, classes, weights)`` of ``rows`` under a fitted pipeline (whose first step selects its
    features), for the evaluators of :mod:`eodgdl.impute.evaluate`."""
    return rows[spec.target].astype(str), model.predict_proba(rows), list(model.named_steps["classifier"].classes_), rows[weight].astype(float)


def _metrics(model, spec, rows, weight, positive, n_bins):
    """The one-row metrics table of :func:`eodgdl.impute.evaluate.evaluate_classifier` on ``rows``."""
    return evaluate.evaluate_classifier(*heldout(model, spec, rows, weight), labels=spec.class_slugs, positive=positive, n_bins=n_bins)[0]


@dataclass
class RetrainResult:
    """What :func:`retrain` produced: the ``bundle``, the scored frame, the scenario frames and the evaluation
    ``tables`` ({name: DataFrame}) and ``summary`` (JSON-serializable)."""

    spec: object
    bundle: dict
    scored: pd.DataFrame
    scenarios: dict
    tables: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def component_spec(spec, name, config=None):
    """The task with one of its ``evaluation.components`` as the target: the component's column (its ``classes``, else
    the harmonized levels of that name, e.g. si / no), the same features and arms, prefix ``<name>``."""
    import dataclasses

    from .harmonize import load_harmonization

    config = config or spec.evaluation.get("components", {}).get(name, {})
    levels = config.get("classes") or load_harmonization("common")["levels"][name]
    classes = {str(level): str(level) for level in levels}
    return dataclasses.replace(spec, target_column=name, classes=classes, prefix=name, scores={})


def _add_components(spec, frame):
    """The task's ``evaluation.components`` as specs, their targets set in ``frame``: unobserved wherever the task's own
    target is (a component of an unknown label is unknown too) or the component holds the missing label."""
    components = {name: component_spec(spec, name, config) for name, config in spec.evaluation.get("components", {}).items()}
    for name, component in components.items():
        labels = frame[name].astype("string")
        frame[component.target] = labels.mask(frame[spec.unknown_column].astype(bool) | labels.eq(spec.missing_label).fillna(False))
        frame[component.unknown_column] = frame[component.target].isna()
    return components


def within_support(spec, target_rows, source_rows):
    """``target_rows`` with the values of the task's ``level_subsets`` features that the source never holds (e.g. a
    municipality the ENOE sample does not cover) set to the missing label, so a density ratio is not fitted on a level
    one side lacks."""
    target_rows = target_rows.copy()
    for feature in spec.level_subsets:
        outside = ~target_rows[feature].isin(set(source_rows[feature].dropna()))
        target_rows.loc[outside, feature] = spec.missing_label
    return target_rows


def _selected_row(summary):
    row = summary.loc[summary["selected"]].iloc[0]
    return {"model": row["model"], "best_params": dict(row["best_params"]), "weighted_log_loss": float(row["weighted_log_loss"]),
            "fold_log_losses": [float(value) for value in row["fold_log_losses"]]}


def _heldout_tables(spec, model, X, y, w, g, test, weight, group, ev, seed, n_splits):
    """The held-out evaluation of one arm's selected model: metrics, bootstrap intervals, per-class tables,
    calibration and the isotonic recalibration (maps from out-of-fold probabilities on the training rows), with its
    own calibration tables. Returns ``(tables, uncertainty, probabilities)``: the held-out probabilities, columns in
    sorted class order (the evaluators' convention: the model's order, as ``predict_proba`` gives it)."""
    n_bins, positive = ev.get("calibration_bins", 10), ev.get("positive_class")
    y_test, probabilities, classes, w_test = heldout(model, spec, test, weight)
    metrics, class_metrics, confusion, distribution = evaluate.evaluate_classifier(y_test, probabilities, classes, w_test, labels=spec.class_slugs, positive=positive, n_bins=n_bins)
    uncertainty = evaluate.test_metrics_with_uncertainty(y_test, probabilities, classes, w_test, test[group].astype(str), n_bootstrap=ev.get("bootstrap", 500),
                                                        random_state=seed, positive=positive)
    in_the_large, reliability = evaluate.calibration_by_class(y_test, probabilities, classes, w_test, n_bins=n_bins)
    calibrated = evaluate.fit_isotonic(model, X, y, w, select.grouped_splits(X, y, g, cv_splits=n_splits, random_state=seed))
    recalibrated = calibrated.predict_proba(test)
    isotonic = pd.DataFrame([
        {"probabilities": "selected", **evaluate.multiclass_calibration_summary(y_test.to_numpy(), probabilities, classes, w_test.to_numpy(), n_bins)},
        {"probabilities": "isotonic", **evaluate.multiclass_calibration_summary(y_test.to_numpy(), recalibrated, calibrated.classes_, w_test.to_numpy(), n_bins)},
    ])
    isotonic_in_the_large, isotonic_reliability = evaluate.calibration_by_class(y_test, recalibrated, calibrated.classes_, w_test, n_bins=n_bins)
    in_class_order = pd.DataFrame(probabilities, columns=classes).reindex(columns=sorted(spec.class_slugs), fill_value=0.0).to_numpy()
    return {"test_metrics": metrics, "test_uncertainty": uncertainty.rename_axis("metric").reset_index(), "class_metrics": class_metrics,
            "confusion": confusion.rename_axis("observed").reset_index(), "test_distribution": distribution, "calibration": in_the_large,
            "reliability": reliability, "isotonic": isotonic, "isotonic_calibration": isotonic_in_the_large,
            "isotonic_reliability": isotonic_reliability}, uncertainty, in_class_order


def _hybrid_tables(spec, test, probabilities, weight, positive, n_bins):
    """The held-out evaluation of the arms as scoring combines them: each held-out row predicted by the arm that
    would score it (the first whose required features it has; ``probabilities``: {arm: held-out probabilities, columns
    in sorted class order}). Tables ``<name>__hybrid`` (informal-jobs-model's hybrid calibration)."""
    classes = sorted(spec.class_slugs)
    assigned = assign_arms(test, spec.arms, spec.is_missing)
    combined = np.zeros((len(test), len(classes)))
    for arm in spec.arms:
        rows = assigned.eq(arm.name).fillna(False).to_numpy(bool)
        combined[rows] = probabilities[arm.name][rows]
    y_test, w_test = test[spec.target].astype(str), test[weight].astype(float)
    metrics = evaluate.evaluate_classifier(y_test, combined, classes, w_test, labels=spec.class_slugs, positive=positive, n_bins=n_bins)[0]
    in_the_large, reliability = evaluate.calibration_by_class(y_test, combined, classes, w_test, n_bins=n_bins)
    rows = assigned.value_counts().reindex([arm.name for arm in spec.arms], fill_value=0)
    return {"test_metrics__hybrid": metrics.assign(**{f"rows_{name}": int(count) for name, count in rows.items()}),
            "calibration__hybrid": in_the_large, "reliability__hybrid": reliability}


def _compare_populations(spec, ev, train, test, selected_models, weight, group, positive, n_bins):
    """Each arm's selected configuration refitted on the training fold restricted to each ``evaluation.populations``
    entry, all evaluated on the same held-out rows of each ``evaluation.population_tests`` entry: which training
    population predicts best where the model is applied."""
    rows = []
    for test_name, test_filter in ev.get("population_tests", {"held-out": {}}).items():
        held_out = select_rows(test, keep=test_filter)
        for population, conditions in ev["populations"].items():
            training = select_rows(train, keep=conditions)
            for arm in spec.arms:
                X, y, w, _, _ = training_data(spec, training, list(arm.features), weight, group)
                model = clone(selected_models[arm.name]).fit(X, y, classifier__sample_weight=w)
                metrics = _metrics(model, spec, held_out, weight, positive, n_bins).iloc[0]
                rows.append({"test": test_name, "population": population, "arm": arm.name, "training_rows": len(X), "test_rows": len(held_out),
                             "weighted_log_loss": metrics["weighted_log_loss"], "weighted_accuracy": metrics["weighted_accuracy"]})
    return pd.DataFrame(rows)


def _imputation_tables(spec, frame, known, rows, unknown, scored, weight, rows_weight, ev):
    """What the imputation looks like: the training population (rows with an observed target) against the imputed one,
    each with its own weight (profiles, missingness), the arms used, their confidence, the marginalized features and
    each arm's levels without training support."""
    tables = {}
    imputed = scored[scored[f"{spec.prefix}_fue_imputado"]]
    shared = [column for column in dict.fromkeys(ev.get("profiles", []) + spec.features) if column in frame.columns and column in rows.columns]
    population = pd.concat([frame.loc[known, shared].assign(__weight=frame.loc[known, weight].to_numpy()),
                            rows.loc[unknown, shared].assign(__weight=rows.loc[unknown, rows_weight].to_numpy())], ignore_index=True)
    is_imputed = np.r_[np.zeros(int(known.sum()), bool), np.ones(int(unknown.sum()), bool)]
    tables["profiles"] = evaluate.compare_known_unknown_profiles(population, ev["profiles"], is_imputed, weight_column="__weight") if ev.get("profiles") else pd.DataFrame()
    if unknown.any():
        tables["missingness"] = evaluate.calculate_feature_missingness(population, [f for f in spec.features if f in shared], is_imputed, spec.is_missing, weight_column="__weight")
    usage = imputed.groupby(f"{spec.prefix}_model_used").agg(sample_rows=(rows_weight, "size"), weighted_population=(rows_weight, "sum")).reset_index()
    usage["weighted_share"] = usage["weighted_population"] / usage["weighted_population"].sum()
    tables["model_usage"] = usage
    tables["confidence"] = imputed.groupby(f"{spec.prefix}_model_used")[f"{spec.prefix}_prediction_confidence"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    tables["marginalized"] = imputed[f"{spec.prefix}_marginalized_features"].value_counts().rename_axis("features").reset_index()
    for arm in spec.arms:
        features = list(arm.features)
        tables[f"unsupported_levels__{arm.name}"] = count_levels_without_training_support(
            prepare_features(frame[known], features, spec.numeric, spec.missing_label), prepare_features(rows[unknown], features, spec.numeric, spec.missing_label), spec.numeric)
    return tables


def retrain(task, context=None, n_jobs=-1, progress=True):
    """Select, evaluate, refit and score ``task``; returns a :class:`RetrainResult` (write it with
    :func:`write_retrain`). Every model is trained with the training source's weights and CV groups; the rows scored
    are the task's scoring source (the training source unless the task names another, e.g. ENOE -> the EOD)."""
    spec = _spec(task)
    context = context or Context()
    timings, started = {}, time.perf_counter()
    source = build_frame(spec, context)
    target = source if spec.score_source == spec.source else build_frame(spec, context, role="score")
    frame, weight, group = source.frame, source.weight, source.group
    rows, rows_weight = target.frame, target.weight
    levels = task_levels(spec)
    cv, ev = spec.selection["cv"], spec.evaluation
    seed, n_splits = cv["seed"], cv["splits"]
    n_bins, positive = ev.get("calibration_bins", 10), ev.get("positive_class")
    ship_isotonic = ev.get("isotonic", {}).get("ship", False)
    components = _add_components(spec, frame)
    full_frame = frame
    if spec.selection.get("population"):                                  # the training population (e.g. ENIGH's large cities)
        frame = select_rows(full_frame, keep=spec.selection["population"])
    known = ~frame[spec.unknown_column].astype(bool).to_numpy()          # training rows with an observed target
    unknown = rows[spec.unknown_column].astype(bool).to_numpy()          # scored rows to impute
    train, test = select.split_known(frame, spec.target, group, spec.unknown_column, n_splits=n_splits, test_fold=cv["test_fold"], random_state=seed)
    test = select_rows(test, drop=cv.get("test_exclude"))
    timings["features"] = time.perf_counter() - started

    tables, summary = {}, {"task": spec.name, "source": spec.source, "score_source": spec.score_source, "rows": len(rows),
                           "training_population": spec.selection.get("population") or "all", "source_rows": len(full_frame),
                           "known": int(known.sum()), "unknown": int(unknown.sum()), "train_rows": len(train), "test_rows": len(test),
                           "train_groups": int(train[group].nunique()), "test_groups": int(test[group].nunique()), "arms": {}}
    arms, selected_models, auxiliary, selected, test_metrics, heldout_probabilities = {}, {}, {}, {}, {}, {}
    for arm in spec.arms:
        features = list(arm.features)
        started = time.perf_counter()
        X, y, w, g, _ = training_data(spec, train, features, weight, group)
        candidates = build_candidates(features, spec.numeric, levels, spec.selection["families"], random_state=seed, missing_label=spec.missing_label)
        tuning, model = select.tune(candidates, X, y, w, g, cv_splits=n_splits, random_state=seed, n_jobs=n_jobs, progress=progress)
        timings[f"tune_{arm.name}"] = time.perf_counter() - started
        selected[arm.name] = _selected_row(tuning)
        selected_models[arm.name] = model

        started = time.perf_counter()
        arm_tables, uncertainty, heldout_probabilities[arm.name] = _heldout_tables(spec, model, X, y, w, g, test, weight, group, ev, seed, n_splits)
        test_metrics[arm.name] = arm_tables["test_metrics"].iloc[0].to_dict()
        timings[f"evaluate_{arm.name}"] = time.perf_counter() - started
        arm_tables = {"selection": tuning, "grid": tuning.attrs["grid_results"], **arm_tables, "fold_losses": select.fold_table(tuning).rename_axis("model").reset_index()}
        tables.update({f"{name}__{arm.name}": table for name, table in arm_tables.items()})

        if ev.get("robustness_groups"):
            # each selected configuration cross-validated with another group (e.g. ENOE's sampling unit)
            alternative = train[~train[spec.unknown_column].astype(bool)][ev["robustness_groups"]].astype("string").reset_index(drop=True)
            robust = [fit for fold in select.grouped_splits(X, y, alternative, cv_splits=n_splits, random_state=seed)
                      for fit in select.fit_fold(model, {}, None, X, y, w, *fold)]
            tables[f"robustness_cv__{arm.name}"] = pd.DataFrame({"fold": range(n_splits), "household_grouped": selected[arm.name]["fold_log_losses"],
                                                                 f"{ev['robustness_groups']}_grouped": [fit["log_loss"] for fit in robust]})
        for name, component in components.items():
            # each component of the label with the arm's selected configuration, judged on the same held-out rows
            X_component, y_component, w_component, _, _ = training_data(component, train, features, weight, group)
            component_model = clone(model).fit(X_component, y_component, classifier__sample_weight=w_component)
            component_metrics = _metrics(component_model, component, test, weight, ev["components"][name]["positive"], n_bins)
            tables.setdefault("components_heldout", []).append(component_metrics.assign(component=name, arm=arm.name))

        started = time.perf_counter()
        final = refit(model, spec, frame, features, weight, group)
        if ship_isotonic:
            X_all, y_all, w_all, g_all, _ = training_data(spec, frame, features, weight, group)
            final = evaluate.fit_isotonic(final, X_all, y_all, w_all, select.grouped_splits(X_all, y_all, g_all, cv_splits=n_splits, random_state=seed))
        arms[arm.name] = final
        # the auxiliary models: P(feature | x) fitted on every training row, whatever its target
        auxiliary[arm.name] = {feature: fit_level_model(frame, feature, spec.auxiliary_predictors(feature, arm.name), spec.numeric, levels,
                                                        sample_weights=frame[weight], random_state=seed, missing_label=spec.missing_label)
                               for feature in spec.auxiliary}
        timings[f"refit_{arm.name}"] = time.perf_counter() - started
        summary["arms"][arm.name] = {"selected": selected[arm.name], "test_metrics": test_metrics[arm.name],
                                     "test_uncertainty": uncertainty["estimate"].to_dict(), "isotonic": arm_tables["isotonic"].set_index("probabilities").to_dict("index")}

    if len(spec.arms) > 1:
        tables.update(_hybrid_tables(spec, test, heldout_probabilities, weight, positive, n_bins))
    if "components_heldout" in tables:
        tables["components_heldout"] = pd.concat(tables["components_heldout"], ignore_index=True)
    component_models = {name: {"classes": dict(component.classes), "positive": ev["components"][name]["positive"],
                               "arms": {arm.name: refit(selected_models[arm.name], component, frame, list(arm.features), weight, group) for arm in spec.arms}}
                        for name, component in components.items()}
    if ev.get("populations"):
        # every candidate population is drawn from the unfiltered source, with its own grouped held-out fold
        full_train, full_test = (train, test) if full_frame is frame else select.split_known(
            full_frame, spec.target, group, spec.unknown_column, n_splits=n_splits, test_fold=cv["test_fold"], random_state=seed)
        tables["populations"] = _compare_populations(spec, ev, full_train, full_test, selected_models, weight, group, positive, n_bins)
    if ev.get("target_profile"):
        # the held-out rows reweighted to the profile of the rows an arm scores (e.g. those without education)
        profile = ev["target_profile"]
        target_rows = rows[unknown & spec.is_missing(rows, profile["missing"]).to_numpy()] if profile.get("missing") else rows[unknown]
        reweighted_test, diagnostics = evaluate.reweight_to_target_profile(test, within_support(spec, target_rows, frame[known]), profile["features"], weight, rows_weight,
                                                                           spec.numeric, random_state=seed, missing_label=spec.missing_label)
        tables["target_profile"] = pd.concat([_metrics(selected_models[arm.name], spec, data, weight, positive, n_bins).assign(weighting=weighting, arm=arm.name)
                                              for weighting, data in (("training weights", test), ("reweighted to the scored rows' profile", reweighted_test))
                                              for arm in spec.arms], ignore_index=True)
        tables["target_profile_diagnostics"] = diagnostics.rename_axis("metric").reset_index(name="value")

    metadata = {
        "selected": selected, "test_metrics": test_metrics, "random_state": seed, "cv": cv, "isotonic_shipped": bool(ship_isotonic),
        "data_versions": {"training": source.versions, "scoring": target.versions}, "builders": {name: builder_config(spec, name) for name in spec.all_builders},
    }
    bundle = make_bundle(spec, arms, auxiliary, levels, metadata)
    if component_models:
        bundle["components"] = component_models

    started = time.perf_counter()
    scored = score_frame(spec, rows, bundle)
    timings["score"] = time.perf_counter() - started
    tables.update(_imputation_tables(spec, frame, known, rows, unknown, scored, weight, rows_weight, ev))

    scenarios = {}
    observed = evaluate.weighted_distribution(frame[known], spec.target, weight_column=weight)
    if "shift" in ev:
        # the training rows reweighted to the imputed rows' profile, every arm refitted (the auxiliary models kept)
        started = time.perf_counter()
        reweighted, diagnostics = evaluate.reweight_to_target_profile(frame[known], within_support(spec, rows[unknown], frame[known]), ev["shift"]["profile"], weight, rows_weight,
                                                                      spec.numeric, random_state=seed, missing_label=spec.missing_label)
        shift_arms = {arm.name: refit(selected_models[arm.name], spec, reweighted, list(arm.features), weight, group) for arm in spec.arms}
        scenarios["shift_weighted"] = score_frame(spec, rows, make_bundle(spec, shift_arms, auxiliary, levels, {"scenario": "shift_weighted"}))
        tables["shift_diagnostics"] = diagnostics.rename_axis("metric").reset_index(name="value")
        timings["shift"] = time.perf_counter() - started
    if "delta" in ev:
        # the imputed share of one class scaled to its observed share
        label = ev["delta"]["class"]
        share = float(observed.set_index("class")["weighted_share"].get(label, 0.0))
        scenarios["delta_adjusted"], summary["delta_factor"] = evaluate.adjust_imputed_share(scored, spec.prefix, spec.class_slugs, label, target_share=share, weight_column=rows_weight)

    shares = {"observed_known": observed.set_index("class")["weighted_share"].reindex(spec.class_slugs)}
    for name, scenario in {"imputed": scored, **scenarios}.items():
        chosen = scenario[scenario[f"{spec.prefix}_fue_imputado"]]
        shares[name] = evaluate.probabilistic_distribution(chosen, spec.prefix, spec.class_slugs, weight_column=rows_weight).set_index("class")["weighted_share"]
    imputed = scored[scored[f"{spec.prefix}_fue_imputado"]]
    shares["imputed_hard"] = evaluate.weighted_distribution(imputed, f"{spec.prefix}_final", weight_column=rows_weight).set_index("class")["weighted_share"].reindex(spec.class_slugs).fillna(0.0)
    shares["final"] = evaluate.probabilistic_distribution(scored, spec.prefix, spec.class_slugs, weight_column=rows_weight).set_index("class")["weighted_share"]
    tables["shares"] = pd.DataFrame(shares).rename_axis("class").reset_index()
    summary["timings_s"] = {name: round(value, 1) for name, value in timings.items()}
    summary["versions"] = {**target.versions, "training": source.versions, "sklearn": bundle["metadata"]["sklearn_version"], "eodgdl": bundle["metadata"]["eodgdl_version"]}

    return RetrainResult(spec, bundle, scored, scenarios, tables, summary)


def _serializable(frame):
    frame = frame.copy()
    frame.attrs = {}
    for column in frame.columns[frame.dtypes.eq("category") | frame.dtypes.map(lambda dtype: isinstance(dtype, pd.IntervalDtype))]:
        frame[column] = frame[column].astype(str)   # e.g. the reliability bins (intervals)
    for column in frame.columns:
        if frame[column].dtype == object and frame[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            frame[column] = frame[column].map(lambda value: json.dumps(value, default=str, sort_keys=True))
    frame.columns = [str(column) for column in frame.columns]
    return frame


def write_retrain(result, out):
    """Write a retrain to the task's directory ``out`` (``<root>/<task>``, where ``retrained=<root>`` finds it):
    ``<bundle file>`` (the v2 bundle), ``scores.parquet`` (keys, outputs and weight), ``scenarios.parquet`` (the same
    per scenario), ``evaluation/<table>.parquet`` and ``summary.json``, after removing the tables and scenarios an
    earlier retrain left there (read later as this one's). Returns the bundle's path and sha256."""
    spec, out = result.spec, Path(out)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    for stale in [*(out / "evaluation").glob("*.parquet"), out / "scenarios.parquet"]:
        stale.unlink(missing_ok=True)
    output_frame(spec, result.scored, weight=True).to_parquet(out / "scores.parquet", index=False)
    if result.scenarios:
        pd.concat([output_frame(spec, frame, weight=True).assign(scenario=name) for name, frame in result.scenarios.items()], ignore_index=True).to_parquet(out / "scenarios.parquet", index=False)
    for name, table in result.tables.items():
        _serializable(table).to_parquet(out / "evaluation" / f"{name}.parquet", index=False)
    bundle_path = out / spec.bundle_name
    digest = save_bundle(result.bundle, bundle_path)
    summary = {**result.summary, "bundle": {"file": bundle_path.name, "sha256": digest}, "spec_hash": spec.hash}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    return bundle_path, digest


# Compare
def _merge(base, fragment):
    merged = copy.deepcopy(base)
    for key, value in fragment.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and key != "arms":
            merged[key] = _merge(merged[key], value)
        elif key == "arms":
            merged["arms"] = {**merged["arms"], **copy.deepcopy(value)}
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def compare(task, candidates, seeds=(42,), bundle=None, retrained=None, context=None, n_jobs=-1, progress=True):
    """Fold log losses of candidate specifications against the task's own, per arm and seed.

    ``candidates`` maps a name to a spec fragment merged into the task's YAML (e.g. ``{"arms": {"with_education":
    {"features": [...], "requires": [...]}}}``, new ``features.groups`` or ``builders``; an arm entry replaces the
    whole arm). Each arm keeps the family and hyperparameters of the bundle's winner (``bundle``, else
    :func:`~eodgdl.impute.bundle.load_bundle` with ``retrained``); the folds are the task's grouped folds, reseeded
    per seed, over the training rows only. Returns ``(folds, paired)``: every fold's loss, and per candidate, arm and
    seed the mean loss, the mean paired difference to the baseline and its standard error, plus the difference
    averaged over seeds."""
    spec = _spec(task)
    context = context or Context()
    bundle = as_v2(bundle, spec) if bundle is not None else load_bundle(spec, retrained=retrained)
    winners = bundle["metadata"]["selected"]
    cv = spec.selection["cv"]
    specs = {"baseline": spec, **{name: parse_task(_merge(spec.raw, fragment)) for name, fragment in candidates.items()}}

    rows = []
    for name, candidate in specs.items():
        source = build_frame(candidate, context)
        frame = select_rows(source.frame, keep=candidate.selection.get("population"))
        train, _ = select.split_known(frame, candidate.target, source.group, candidate.unknown_column, n_splits=cv["splits"], test_fold=cv["test_fold"], random_state=cv["seed"])
        levels = task_levels(candidate)
        for arm in candidate.arms:
            winner = winners[arm.name]
            family = dict(candidate.selection["families"][winner["model"]])
            family["params"] = {key.removeprefix("classifier__"): [value] for key, value in winner["best_params"].items()}
            models = build_candidates(list(arm.features), candidate.numeric, levels, {winner["model"]: family}, random_state=cv["seed"], missing_label=candidate.missing_label)
            X, y, w, g, _ = training_data(candidate, train, list(arm.features), source.weight, source.group)
            for seed in seeds:
                splits = select.grouped_splits(X, y, g, cv_splits=cv["splits"], random_state=seed)
                [(_, by_fold)] = select.cross_validate_grid(models, X, y, w, splits, n_jobs=n_jobs, progress=progress).items()
                rows += [{"candidate": name, "arm": arm.name, "seed": seed, "fold": fold, "log_loss": by_fold[fold]["log_loss"], "features": len(arm.features)}
                         for fold in range(len(splits))]
    folds = pd.DataFrame(rows)
    baseline = folds[folds["candidate"] == "baseline"].set_index(["arm", "seed", "fold"])["log_loss"]
    folds["difference"] = folds["log_loss"].to_numpy() - baseline.reindex(pd.MultiIndex.from_frame(folds[["arm", "seed", "fold"]])).to_numpy()
    paired = folds.groupby(["candidate", "arm", "seed"]).agg(mean_log_loss=("log_loss", "mean"), mean_difference=("difference", "mean"),
                                                             se_difference=("difference", lambda d: d.std(ddof=1) / np.sqrt(len(d))), folds=("fold", "size")).reset_index()
    paired["mean_difference_over_seeds"] = paired.groupby(["candidate", "arm"])["mean_difference"].transform("mean")

    return folds, paired
