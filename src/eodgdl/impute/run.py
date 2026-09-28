"""Running a task in its three modes.

- **score** (:func:`score_task`): load the task's bundle, check it, score the source's rows (seconds).
- **retrain** (:func:`retrain`): grouped-CV selection per arm on the training rows, evaluation on the held-out
  households, refit on every observed row, auxiliary models, scoring, the covariate-shift and delta scenarios; writes
  the bundle and every evaluation table (:func:`write_retrain`) for a report to read.
- **compare** (:func:`compare`): feature specifications evaluated cheaply: the published winner's family and
  hyperparameters fixed, the same grouped folds on the training rows (the held-out fold never touched), each
  candidate's fold log losses paired against the baseline's.
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
from .arms import Arm, assign_arms, impute_with_arms
from .bundle import as_v2, check_bundle, load_bundle, make_bundle, save_bundle
from .features import build_frame, builder_config
from .levels import count_levels_without_training_support, normalize_sample_weights, prepare_features
from .marginalize import attach_training_level_shares
from .models import build_candidates, fit_level_model
from .sources import Context, get_source
from .spec import load_task, parse_task


def task_levels(spec):
    """The task's category-level contract: its declared levels, else its source's."""
    return spec.category_levels(get_source(spec.source).schema_levels)


def expected_score_column(spec):
    return f"{spec.prefix}_puntaje_esperado"


def output_columns(spec, keys):
    """Row keys plus the task's output columns (and its expected score when the target is ordinal)."""
    columns = list(keys) + [f"{spec.prefix}_{suffix}" for suffix in ("observado", "imputado", "final", "fue_imputado", "model_used", "prediction_confidence", "marginalized_features")] + spec.probability_columns
    return columns + ([expected_score_column(spec)] if spec.scores else [])


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
    arms = [Arm(name, entry["model"], entry["features"], requires=entry["requires"], auxiliary=bundle["auxiliary"].get(name, {}))
            for name, entry in bundle["arms"].items()]
    assigned = assign_arms(frame, arms, spec.is_missing)
    assigned[~frame[spec.unknown_column].astype(bool).to_numpy()] = pd.NA
    scored = impute_with_arms(frame, spec.target, spec.class_slugs, arms, assigned, spec.prefix, spec.numeric, levels,
                              level_subsets=(spec.level_subsets or None) if level_subsets is None else level_subsets, missing_label=spec.missing_label)
    evaluate.validate_probability_rows(scored, spec.probability_columns)
    if spec.scores:
        scored[expected_score_column(spec)] = scored[spec.probability_columns].to_numpy() @ np.array([spec.scores[slug] for slug in spec.class_slugs])

    return scored


def score_task(task, tables=None, bundle=None, path=None, context=None):
    """Load (or take) the task's bundle and score its source's rows. ``tables`` are the survey tables (loaded, and
    the features cached, when omitted); ``bundle`` a fitted bundle (v2 or a legacy giro dict), else read from
    ``path`` or the task's data file."""
    spec = _spec(task)
    context = context or Context(tables=tables)
    bundle = bundle if bundle is not None else load_bundle(spec, path)
    frame = build_frame(spec, context, role="score").frame

    return score_frame(spec, frame, bundle)


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
    keys: list = field(default_factory=list)


def component_spec(spec, name, config=None):
    """The task with one of its ``evaluation.components`` as the target: the component's column (its ``classes``, else
    the harmonized levels of that name, e.g. si / no), the same features and arms, prefix ``<name>``."""
    import dataclasses

    from .harmonize import load_harmonization

    config = config or spec.evaluation.get("components", {}).get(name, {})
    levels = config.get("classes") or load_harmonization("common")["levels"][name]
    classes = {str(level): str(level) for level in levels}
    return dataclasses.replace(spec, target_column=name, classes=classes, prefix=name, scores={})


def _within_support(spec, target_rows, source_rows):
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


def retrain(task, context=None, n_jobs=-1, progress=True):
    """Select, evaluate, refit and score ``task``; returns a :class:`RetrainResult` (write it with
    :func:`write_retrain`). Every model is trained with the training source's weights and CV groups; the rows scored
    are the task's scoring source (the training source unless the task names another, e.g. ENOE -> the EOD)."""
    spec = _spec(task)
    context = context or Context()
    timings = {}
    started = time.perf_counter()
    source = build_frame(spec, context)
    target = source if spec.score_source == spec.source else build_frame(spec, context, role="score")
    frame, weight, group = source.frame, source.weight, source.group
    rows, rows_weight = target.frame, target.weight
    levels = task_levels(spec)
    cv, ev = spec.selection["cv"], spec.evaluation
    seed, n_splits = cv["seed"], cv["splits"]
    n_bins = ev.get("calibration_bins", 10)
    positive = ev.get("positive_class")
    ship_isotonic = ev.get("isotonic", {}).get("ship", False)
    components = {name: component_spec(spec, name, config) for name, config in ev.get("components", {}).items()}
    for name, component in components.items():
        frame[component.target] = frame[name].astype("string")
        frame[component.unknown_column] = frame[component.target].isna()
    known = ~frame[spec.unknown_column].astype(bool).to_numpy()          # training rows with an observed target
    unknown = rows[spec.unknown_column].astype(bool).to_numpy()          # scored rows to impute
    train, test = select.split_known(frame, spec.target, group, spec.unknown_column, n_splits=n_splits, test_fold=cv["test_fold"], random_state=seed)
    for column, values in cv.get("test_exclude", {}).items():
        test = test[~test[column].isin(values)].reset_index(drop=True)
    timings["features"] = time.perf_counter() - started

    tables, summary = {}, {"task": spec.name, "source": spec.source, "score_source": spec.score_source, "rows": len(rows),
                           "known": int(known.sum()), "unknown": int(unknown.sum()), "train_rows": len(train), "test_rows": len(test),
                           "train_groups": int(train[group].nunique()), "test_groups": int(test[group].nunique()), "arms": {}}
    arms, selected_models, auxiliary, selected, test_metrics = {}, {}, {}, {}, {}
    for arm in spec.arms:
        features = list(arm.features)
        started = time.perf_counter()
        X, y, w, g, _ = training_data(spec, train, features, weight, group)
        candidates = build_candidates(features, spec.numeric, levels, spec.selection["families"], random_state=seed, missing_label=spec.missing_label)
        tuning, best = select.tune(candidates, X, y, w, g, cv_splits=n_splits, random_state=seed, n_jobs=n_jobs, progress=progress)
        family, model = select.get_best_model(tuning, best)
        timings[f"tune_{arm.name}"] = time.perf_counter() - started
        selected[arm.name] = _selected_row(tuning)
        selected_models[arm.name] = model

        started = time.perf_counter()
        metrics, class_metrics, confusion, distribution = evaluate.evaluate_classifier(model, test, features, spec.target, spec.class_slugs, spec.numeric, weight_column=weight,
                                                                                      missing_label=spec.missing_label, positive=positive, n_bins=n_bins)
        uncertainty = evaluate.test_metrics_with_uncertainty(model, test, features, spec.target, spec.numeric, n_bootstrap=ev.get("bootstrap", 500), random_state=seed,
                                                            weight_column=weight, group_column=group, missing_label=spec.missing_label, positive=positive)
        in_the_large, reliability = evaluate.calibration_by_class(model, test, features, spec.target, spec.numeric, n_bins=n_bins, weight_column=weight, missing_label=spec.missing_label)
        # isotonic: maps from out-of-fold probabilities on the training rows, applied to the selected model on the held-out rows
        splits = select.grouped_splits(X, y, g, cv_splits=n_splits, random_state=seed)
        calibrated = evaluate.fit_isotonic(model, candidates[family]["model"].set_params(**selected[arm.name]["best_params"]), X, y, w, splits)
        X_test = prepare_features(test, features, spec.numeric, spec.missing_label)
        test_w, test_y = test[weight].astype(float).to_numpy(), test[spec.target].astype(str).to_numpy()
        isotonic = pd.DataFrame([
            {"probabilities": "selected", **evaluate.multiclass_calibration_summary(test_y, model.predict_proba(X_test), model.named_steps["classifier"].classes_, test_w, n_bins)},
            {"probabilities": "isotonic", **evaluate.multiclass_calibration_summary(test_y, calibrated.predict_proba(X_test), calibrated.classes_, test_w, n_bins)},
        ])
        test_metrics[arm.name] = metrics.iloc[0].to_dict()
        timings[f"evaluate_{arm.name}"] = time.perf_counter() - started
        for name, table in (("selection", tuning), ("grid", tuning.attrs["grid_results"]), ("test_metrics", metrics), ("test_uncertainty", uncertainty.rename_axis("metric").reset_index()),
                            ("class_metrics", class_metrics), ("confusion", confusion.rename_axis("observed").reset_index()), ("test_distribution", distribution),
                            ("calibration", in_the_large), ("reliability", reliability), ("isotonic", isotonic),
                            ("fold_losses", select.fold_table(tuning).rename_axis("model").reset_index())):
            tables[f"{name}__{arm.name}"] = table

        if ev.get("robustness_groups"):
            alternative = train[~train[spec.unknown_column].astype(bool)][ev["robustness_groups"]].astype("string").reset_index(drop=True)
            robust = [fit for fold in select.grouped_splits(X, y, alternative, cv_splits=n_splits, random_state=seed)
                      for fit in select.fit_fold(model, {}, None, X, y, w, *fold)]
            tables[f"robustness_cv__{arm.name}"] = pd.DataFrame({"fold": range(n_splits), "household_grouped": selected[arm.name]["fold_log_losses"],
                                                                 f"{ev['robustness_groups']}_grouped": [fit["log_loss"] for fit in robust]})
        for name, component in components.items():
            X_component, y_component, w_component, _, _ = training_data(component, train, features, weight, group)
            component_model = clone(model).fit(X_component, y_component, classifier__sample_weight=w_component)
            component_metrics = evaluate.evaluate_classifier(component_model, test, features, component.target, component.class_slugs, spec.numeric, weight_column=weight,
                                                             missing_label=spec.missing_label, positive=ev["components"][name]["positive"], n_bins=n_bins)[0]
            tables.setdefault("components_heldout", []).append(component_metrics.assign(component=name, arm=arm.name))

        started = time.perf_counter()
        final = refit(model, spec, frame, features, weight, group)
        if ship_isotonic:
            X_all, y_all, w_all, g_all, _ = training_data(spec, frame, features, weight, group)
            final = evaluate.fit_isotonic(final, clone(model), X_all, y_all, w_all, select.grouped_splits(X_all, y_all, g_all, cv_splits=n_splits, random_state=seed))
        arms[arm.name] = final
        auxiliary[arm.name] = {feature: fit_level_model(frame, feature, spec.auxiliary_predictors(feature, arm.name), spec.numeric, levels,
                                                        sample_weights=frame[weight], random_state=seed, missing_label=spec.missing_label)
                               for feature in spec.auxiliary}
        timings[f"refit_{arm.name}"] = time.perf_counter() - started
        summary["arms"][arm.name] = {"selected": selected[arm.name], "test_metrics": test_metrics[arm.name],
                                     "test_uncertainty": uncertainty["estimate"].to_dict(), "isotonic": isotonic.set_index("probabilities").to_dict("index")}

    if "components_heldout" in tables:
        tables["components_heldout"] = pd.concat(tables["components_heldout"], ignore_index=True)
    component_models = {name: {"classes": dict(component.classes), "positive": ev["components"][name]["positive"],
                               "arms": {arm.name: refit(selected_models[arm.name], component, frame, list(arm.features), weight, group) for arm in spec.arms}}
                        for name, component in components.items()}
    if ev.get("target_profile"):
        profile = ev["target_profile"]
        target_rows = rows[unknown & spec.is_missing(rows, profile["missing"]).to_numpy()] if profile.get("missing") else rows[unknown]
        reweighted_test, diagnostics = evaluate.reweight_to_target_profile(test, _within_support(spec, target_rows, frame[known]), profile["features"], weight, rows_weight,
                                                                           spec.numeric, random_state=seed, missing_label=spec.missing_label)
        evaluations = []
        for weighting, data in (("training weights", test), ("reweighted to the scored rows' profile", reweighted_test)):
            for arm in spec.arms:
                metrics = evaluate.evaluate_classifier(selected_models[arm.name], data, list(arm.features), spec.target, spec.class_slugs, spec.numeric, weight_column=weight,
                                                       missing_label=spec.missing_label, positive=positive, n_bins=n_bins)[0]
                evaluations.append(metrics.assign(weighting=weighting, arm=arm.name))
        tables["target_profile"] = pd.concat(evaluations, ignore_index=True)
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
    imputed = scored[scored[f"{spec.prefix}_fue_imputado"]]
    observed = evaluate.weighted_distribution(frame[known], spec.target, weight_column=weight)
    # the training population (rows with an observed target) against the imputed one, each with its own weight
    shared = [column for column in dict.fromkeys(ev.get("profiles", []) + spec.features) if column in frame.columns and column in rows.columns]
    population = pd.concat([frame.loc[known, shared].assign(__weight=frame.loc[known, weight].to_numpy()),
                            rows.loc[unknown, shared].assign(__weight=rows.loc[unknown, rows_weight].to_numpy())], ignore_index=True)
    is_imputed = np.r_[np.zeros(int(known.sum()), bool), np.ones(int(unknown.sum()), bool)]
    tables["profiles"] = evaluate.compare_known_unknown_profiles(population, ev["profiles"], is_imputed, weight_column="__weight") if ev.get("profiles") else pd.DataFrame()
    tables["missingness"] = evaluate.calculate_feature_missingness(population, [f for f in spec.features if f in shared], is_imputed, weight_column="__weight", missing_label=spec.missing_label)
    usage = imputed.groupby(f"{spec.prefix}_model_used").agg(sample_rows=(rows_weight, "size"), weighted_population=(rows_weight, "sum")).reset_index()
    usage["weighted_share"] = usage["weighted_population"] / usage["weighted_population"].sum()
    tables["model_usage"] = usage
    tables["confidence"] = imputed.groupby(f"{spec.prefix}_model_used")[f"{spec.prefix}_prediction_confidence"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    tables["marginalized"] = imputed[f"{spec.prefix}_marginalized_features"].value_counts().rename_axis("features").reset_index()
    for arm in spec.arms:
        features = list(arm.features)
        tables[f"unsupported_levels__{arm.name}"] = count_levels_without_training_support(
            prepare_features(frame[known], features, spec.numeric, spec.missing_label), prepare_features(rows[unknown], features, spec.numeric, spec.missing_label), spec.numeric)

    scenarios = {}
    if "shift" in ev:
        started = time.perf_counter()
        reweighted, diagnostics = evaluate.reweight_to_target_profile(frame[known], _within_support(spec, rows[unknown], frame[known]), ev["shift"]["profile"], weight, rows_weight, spec.numeric,
                                                                      random_state=seed, missing_label=spec.missing_label)
        shift_arms = {arm.name: refit(selected_models[arm.name], spec, reweighted, list(arm.features), weight, group) for arm in spec.arms}
        # the auxiliary models are the final ones: P(feature | x) is fitted on every training row, whatever its target
        shift_bundle = make_bundle(spec, shift_arms, auxiliary, levels, {"scenario": "shift_weighted"})
        scenarios["shift_weighted"] = score_frame(spec, rows, shift_bundle)
        tables["shift_diagnostics"] = diagnostics.rename_axis("metric").reset_index(name="value")
        timings["shift"] = time.perf_counter() - started
    if "delta" in ev:
        label = ev["delta"]["class"]
        share = float(observed.set_index("class")["weighted_share"].get(label, 0.0))
        scenarios["delta_adjusted"], factor = evaluate.adjust_imputed_share(scored, spec.prefix, spec.class_slugs, label, target_share=share, weight_column=rows_weight)
        summary["delta_factor"] = factor

    shares = {"observed_known": observed.set_index("class")["weighted_share"].reindex(spec.class_slugs)}
    for name, scenario in {"imputed": scored, **scenarios}.items():
        chosen = scenario[scenario[f"{spec.prefix}_fue_imputado"]]
        shares[name] = evaluate.probabilistic_distribution(chosen, spec.prefix, spec.class_slugs, weight_column=rows_weight).set_index("class")["weighted_share"]
    shares["imputed_hard"] = evaluate.weighted_distribution(imputed, f"{spec.prefix}_final", weight_column=rows_weight).set_index("class")["weighted_share"].reindex(spec.class_slugs).fillna(0.0)
    shares["final"] = evaluate.probabilistic_distribution(scored, spec.prefix, spec.class_slugs, weight_column=rows_weight).set_index("class")["weighted_share"]
    tables["shares"] = pd.DataFrame(shares).rename_axis("class").reset_index()
    summary["timings_s"] = {name: round(value, 1) for name, value in timings.items()}
    summary["versions"] = {**target.versions, "training": source.versions, "sklearn": bundle["metadata"]["sklearn_version"], "eodgdl": bundle["metadata"]["eodgdl_version"]}

    return RetrainResult(spec, bundle, scored, scenarios, tables, summary, list(target.keys))


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
    """Write a retrain: ``<bundle file>`` (the v2 bundle), ``scores.parquet`` (keys and outputs),
    ``scenarios.parquet`` (the same per scenario), ``evaluation/<table>.parquet`` and ``summary.json``. Returns the
    bundle's path and sha256."""
    spec, out = result.spec, Path(out)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    columns = output_columns(spec, result.keys)
    weight = get_source(spec.score_source).config["weight"]
    result.scored[columns + [weight]].to_parquet(out / "scores.parquet", index=False)
    if result.scenarios:
        pd.concat([frame[columns + [weight]].assign(scenario=name) for name, frame in result.scenarios.items()], ignore_index=True).to_parquet(out / "scenarios.parquet", index=False)
    for name, table in result.tables.items():
        _serializable(table).to_parquet(out / "evaluation" / f"{name}.parquet", index=False)
    bundle_path = out / (spec.bundle_file or f"{spec.name}.joblib")
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


def compare(task, candidates, seeds=(42,), bundle=None, context=None, n_jobs=-1, progress=True):
    """Fold log losses of candidate specifications against the task's own, per arm and seed.

    ``candidates`` maps a name to a spec fragment merged into the task's YAML (e.g. ``{"arms": {"with_education":
    {"features": [...], "requires": [...]}}}``, new ``features.groups`` or ``builders``; an arm entry replaces the
    whole arm). Each arm keeps the family and hyperparameters of the published bundle's winner; the folds are the
    task's grouped folds, reseeded per seed, over the training rows only. Returns ``(folds, paired)``: every fold's
    loss, and per candidate, arm and seed the mean loss, the mean paired difference to the baseline and its standard
    error, plus the difference averaged over seeds."""
    spec = _spec(task)
    context = context or Context()
    bundle = as_v2(bundle if bundle is not None else load_bundle(spec), spec)
    winners = bundle["metadata"]["selected"]
    cv = spec.selection["cv"]
    specs = {"baseline": spec, **{name: parse_task(_merge(spec.raw, fragment)) for name, fragment in candidates.items()}}

    rows = []
    for name, candidate in specs.items():
        source = build_frame(candidate, context)
        train, _ = select.split_known(source.frame, candidate.target, source.group, candidate.unknown_column, n_splits=cv["splits"], test_fold=cv["test_fold"], random_state=cv["seed"])
        levels = task_levels(candidate)
        for arm in candidate.arms:
            winner = winners[arm.name]
            family = dict(candidate.selection["families"][winner["model"]])
            family["params"] = {key.removeprefix("classifier__"): [value] for key, value in winner["best_params"].items()}
            models = build_candidates(list(arm.features), candidate.numeric, levels, {winner["model"]: family}, random_state=cv["seed"], missing_label=candidate.missing_label)
            X, y, w, g, _ = training_data(candidate, train, list(arm.features), source.weight, source.group)
            for seed in seeds:
                splits = select.grouped_splits(X, y, g, cv_splits=cv["splits"], random_state=seed)
                scores = select.cross_validate_grid(models, X, y, w, splits, n_jobs=n_jobs, progress=progress)
                [(_, by_fold)] = scores.items()
                rows += [{"candidate": name, "arm": arm.name, "seed": seed, "fold": fold, "log_loss": by_fold[fold]["log_loss"], "features": len(arm.features)}
                         for fold in range(len(splits))]
    folds = pd.DataFrame(rows)
    baseline = folds[folds["candidate"] == "baseline"].set_index(["arm", "seed", "fold"])["log_loss"]
    folds["difference"] = folds["log_loss"].to_numpy() - baseline.reindex(pd.MultiIndex.from_frame(folds[["arm", "seed", "fold"]])).to_numpy()
    paired = folds.groupby(["candidate", "arm", "seed"]).agg(mean_log_loss=("log_loss", "mean"), mean_difference=("difference", "mean"),
                                                             se_difference=("difference", lambda d: d.std(ddof=1) / np.sqrt(len(d))), folds=("fold", "size")).reset_index()
    paired["mean_difference_over_seeds"] = paired.groupby(["candidate", "arm"])["mean_difference"].transform("mean")

    return folds, paired
