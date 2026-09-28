"""Diagnostics of a chain's imputation against its training source, driven by the chain's ``evaluation:`` section
(e.g. ``chains/sector_informality.yaml``). Ported from informal-jobs-model's notebook 05.

- ``benchmark``: the training source's observed rate where the chain is applied (``exclude`` rows, by ``by``).
- ``by_upstream``: the expected rate within each level of the upstream feature (Σ w P(s) P(y | x, s) / Σ w P(s)).
- ``upstream_scenarios``: the chain rerun with an upstream task's retrain scenarios in place of its scores.
- ``bounds``: the unsupported values of a feature averaged over the listed levels only (without auxiliary models).
- ``without_auxiliary``: the unsupported values averaged over the training shares instead of P(level | x).
- ``components``: the chain rerun with each component model of the task (``evaluation.components`` of its retrain).
- ``gap_decomposition``: direct standardization of the benchmark to the scored rows' profile: observed → predicted
  by the model → both reweighted to the scored profile → the chain's expected rate.
- ``raking``: the scored rows reweighted to the benchmark's profile.

:func:`evaluate_chain` returns the tables; :func:`write_evaluation` writes them for a report.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate
from .arms import Arm, assign_arms
from .bundle import as_v2, load_bundle, make_bundle
from .chain import TaskStep, load_chain, run_chain
from .features import build_frame
from .levels import prepare_features
from .marginalize import predict_proba_marginalizing
from .run import _within_support, component_spec, task_levels
from .sources import Context
from .spec import load_task


def _bundle(task, retrained):
    """The task's bundle from its retrain directory when there is one, else its data file."""
    spec = load_task(task)
    path = Path(retrained) / task / (spec.bundle_file or f"{task}.joblib")
    return as_v2(load_bundle(spec, path if path.exists() else None), spec)


def _rate(frame, column, weight):
    return float((frame[weight] * frame[column]).sum() / frame[weight].sum())


def _upstream_probabilities(chain, result, step):
    """P(level | x) of the feature a one-upstream step takes, from the upstream's probabilities and the transform."""
    [use] = step.uses
    upstream = load_task(use.task)
    transform = dict(use.transform) if use.transform else {slug: slug for slug in upstream.class_slugs}
    levels = sorted(set(transform.values()))
    return use.feature, pd.DataFrame({level: sum(result.frame[f"prob_{upstream.prefix}_{slug}"] for slug, target in transform.items() if target == level) for level in levels})


def predict_rows(spec, bundle, rows):
    """P(class | x) of ``rows`` under the bundle's arms (first arm whose covariates are observed), unsupported levels
    marginalized over the training shares; every row is predicted, whether its target is observed or not."""
    arms = [Arm(name, entry["model"], entry["features"], requires=entry["requires"]) for name, entry in bundle["arms"].items()]
    assigned = assign_arms(rows, arms, spec.is_missing)
    probabilities = np.full((len(rows), len(spec.class_slugs)), np.nan)
    for arm in arms:
        mask = (assigned == arm.name).fillna(False).to_numpy()
        if mask.any():
            X = prepare_features(rows[mask], arm.features, spec.numeric, spec.missing_label)
            predicted, _ = predict_proba_marginalizing(arm.model, X, missing_label=spec.missing_label)
            order = [list(arm.model.named_steps["classifier"].classes_).index(slug) for slug in spec.class_slugs]
            probabilities[mask] = predicted[:, order]
    return pd.DataFrame(probabilities, columns=spec.class_slugs, index=rows.index)


def evaluate_chain(chain, retrained="output/impute", context=None):
    """Every diagnostic of ``chain``'s ``evaluation:`` section, with the bundles of the retrains under ``retrained``
    (``<retrained>/<task>/``; a task without one uses its data file). Returns ``({name: DataFrame}, summary)``."""
    chain = load_chain(chain) if isinstance(chain, str) else chain
    ev = chain.raw.get("evaluation") or {}
    context = context or Context()
    task = ev["task"]
    spec = load_task(task)
    step = next(s for s in chain.steps if isinstance(s, TaskStep) and s.task == task)
    positive = spec.evaluation.get("positive_class", spec.class_slugs[0])
    bundles = {t: _bundle(t, retrained) for t in chain.tasks}
    scored_rows = build_frame(spec, context, role="score")
    weight, keys = scored_rows.weight, list(scored_rows.keys)
    target_frame = scored_rows.frame
    training = build_frame(spec, context)
    benchmark = training.frame[~training.frame[spec.unknown_column].astype(bool)]
    for column, values in ev.get("benchmark", {}).get("exclude", {}).items():
        benchmark = benchmark[~benchmark[column].isin(values)]
    benchmark = benchmark.reset_index(drop=True)
    train_weight = training.weight
    probability = f"prob_{spec.prefix}_{positive}"

    def run(**kwargs):
        result = run_chain(chain, context=context, bundles={**bundles, **kwargs.pop("bundles", {})}, **kwargs)
        result.frame = result.frame.merge(target_frame[keys + [weight]], on=keys, how="left", validate="one_to_one")
        return result

    tables, summary = {}, {"chain": chain.name, "task": task, "positive": positive}
    shipped = run()
    variants = {"shipped": shipped}
    for upstream, names in ev.get("upstream_scenarios", {}).items():
        scenarios = pd.read_parquet(Path(retrained) / upstream / "scenarios.parquet")
        for name in names:
            given = scenarios[scenarios["scenario"] == name].drop(columns="scenario")
            variants[f"{upstream}: {name}"] = run(upstream_outputs={upstream: given})
    for feature, levels in ev.get("bounds", {}).items():
        variants[f"{feature} averaged over {', '.join(levels)} only"] = run(options={task: {"level_subsets": {feature: levels}, "auxiliary": False}})
    if ev.get("without_auxiliary"):
        variants["training shares instead of the auxiliary models"] = run(options={task: {"auxiliary": False}})

    marginalized = {feature: shipped.frame[f"{spec.prefix}_marginalized_features"].str.contains(feature, regex=False).fillna(False).to_numpy()
                    for feature in ev.get("bounds", {})}
    rows = []
    for name, result in variants.items():
        row = {"variant": name, "expected_rate": _rate(result.frame, probability, weight)}
        for feature, mask in marginalized.items():
            row[f"mean_probability_{feature}_marginalized"] = float(result.frame.loc[mask, probability].mean())
        if ev.get("by_upstream") and len(step.uses) == 1:
            feature, upstream_probabilities = _upstream_probabilities(chain, result, step)
            for level in upstream_probabilities.columns:
                share = upstream_probabilities[level].to_numpy() * result.frame[weight].to_numpy()
                row[f"within_{level}"] = float((share * result.frame[f"prob_{spec.prefix}_{positive}_given_{feature}_{level}"]).sum() / share.sum())
        rows.append(row)
    headline = pd.DataFrame(rows)
    headline["difference_pp"] = (headline["expected_rate"] - headline["expected_rate"].iloc[0]) * 100
    tables["headline"] = headline
    summary["expected_rate"] = float(headline["expected_rate"].iloc[0])
    summary["rows"] = len(shipped.frame)
    summary["marginalized_rows"] = {feature: int(mask.sum()) for feature, mask in marginalized.items()}

    if ev.get("by_upstream") and len(step.uses) == 1:
        feature, upstream_probabilities = _upstream_probabilities(chain, shipped, step)
        by = []
        for level in upstream_probabilities.columns:
            share = upstream_probabilities[level].to_numpy() * shipped.frame[weight].to_numpy()
            informal = share * shipped.frame[f"prob_{spec.prefix}_{positive}_given_{feature}_{level}"].to_numpy()
            by.append({feature: level, "weighted_population": share.sum(), f"expected_{positive}_population": informal.sum(), "expected_rate": informal.sum() / share.sum()})
        tables["by_upstream"] = pd.DataFrame(by)

    # benchmark: the training source's observed rate where the chain is applied
    observed = (benchmark[spec.target] == positive).astype(float)
    bench = [{"group": "all", "rows": len(benchmark), "observed_rate": float(np.average(observed, weights=benchmark[train_weight]))}]
    by = ev.get("benchmark", {}).get("by")
    if by:
        for value, part in benchmark.groupby(by):
            bench.append({"group": f"{by} = {value}", "rows": len(part), "observed_rate": float(np.average((part[spec.target] == positive).astype(float), weights=part[train_weight]))})
    tables["benchmark"] = pd.DataFrame(bench)
    summary["benchmark_rate"] = bench[0]["observed_rate"]

    # components: each component's models through the same chain
    components = []
    for name in ev.get("components", []):
        stored = bundles[task].get("components", {}).get(name)
        if stored is None:
            continue
        component = component_spec(spec, name)
        component_bundle = make_bundle(component, stored["arms"], bundles[task]["auxiliary"], task_levels(spec), {"component_of": task})
        result = run(specs={task: component}, bundles={task: component_bundle})
        components.append({"component": name, "expected_rate": _rate(result.frame, f"prob_{name}_{stored['positive']}", weight),
                           "benchmark_rate": float(np.average((benchmark[name] == stored["positive"]).astype(float), weights=benchmark[train_weight]))})
    if components:
        table = pd.DataFrame(components)
        table.loc[len(table)] = {"component": "sum of the components", "expected_rate": table["expected_rate"].sum(), "benchmark_rate": table["benchmark_rate"].sum()}
        table.loc[len(table)] = {"component": "the task itself", "expected_rate": summary["expected_rate"], "benchmark_rate": summary["benchmark_rate"]}
        tables["components"] = table

    # direct standardization of the benchmark to the scored rows' profile
    scored = target_frame.merge(shipped.frame[keys + [probability]], on=keys, how="left", validate="one_to_one")
    if ev.get("gap_decomposition"):
        profile = list(ev["gap_decomposition"])
        target = scored.copy()
        if len(step.uses) == 1:
            feature, upstream_probabilities = _upstream_probabilities(chain, shipped, step)
            target[feature] = np.asarray(upstream_probabilities.columns, dtype=object)[upstream_probabilities.to_numpy().argmax(axis=1)]
        target = _within_support(spec, target, benchmark)
        reweighted, diagnostics = evaluate.reweight_to_target_profile(benchmark, target, profile, train_weight, weight, spec.numeric, missing_label=spec.missing_label)
        predicted = predict_rows(spec, bundles[task], benchmark)[positive].to_numpy()
        steps = [("training source observed (benchmark)", np.average(observed, weights=benchmark[train_weight]), "—"),
                 ("training source predicted by the model", np.average(predicted, weights=benchmark[train_weight]), "model bias on the training source"),
                 ("observed, reweighted to the scored profile", np.average(observed, weights=reweighted[train_weight]), "composition (observed)"),
                 ("predicted, reweighted to the scored profile", np.average(predicted, weights=reweighted[train_weight]), "composition (predicted)"),
                 ("scored rows' expected rate", summary["expected_rate"], "residual: upstream imputation, unsupported levels, non-response")]
        table = pd.DataFrame(steps, columns=["step", "rate", "interpretation"])
        table["difference_pp"] = [np.nan, (steps[1][1] - steps[0][1]) * 100, (steps[2][1] - steps[0][1]) * 100, (steps[3][1] - steps[1][1]) * 100, (steps[4][1] - steps[3][1]) * 100]
        tables["gap_decomposition"] = table
        tables["gap_diagnostics"] = diagnostics.rename_axis("metric").reset_index(name="value")

    # the scored rows raked to the benchmark's profile
    if ev.get("raking"):
        raked, diagnostics = evaluate.reweight_to_target_profile(_within_support(spec, scored, benchmark), benchmark, list(ev["raking"]), weight, train_weight,
                                                                 spec.numeric, missing_label=spec.missing_label)
        tables["raking"] = pd.DataFrame({"rate": {"scored rows, as surveyed": summary["expected_rate"], "scored rows, raked to the benchmark profile": _rate(raked, probability, weight),
                                                  "benchmark observed": summary["benchmark_rate"]}}).rename_axis("population").reset_index()
        tables["raking_diagnostics"] = diagnostics.rename_axis("metric").reset_index(name="value")

    return tables, summary


def write_evaluation(tables, summary, out):
    """``<out>/evaluation/<table>.parquet`` and ``<out>/evaluation/summary.json``."""
    out = Path(out) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out / f"{name}.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out
