"""Tests for the imputation engine on synthetic data: batched marginalization against the recursive reference, level
subsets, staged vs unstaged tuning, serial vs parallel tuning, the one-SE rule, harmonization and arm dispatch."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from eodgdl.impute import harmonize, levels, marginalize, models, select
from eodgdl.impute.arms import Arm, assign_arms

MISSING = levels.MISSING_LABEL
FEATURES = ["a", "b", "c"]
NUMERIC = ["c"]
LEVELS = {"a": ["a0", "a1", "a2", "a3", MISSING], "b": ["b0", "b1", "b2", MISSING]}


def recursive_marginalizing(model, X, level_subsets=None, conditional_shares=None, missing_label=MISSING):
    """The recursion the batched marginalization replaced (informal-jobs-model's semantics, a9bf693): one
    ``predict_proba`` call per feature and level. The reference the batched version must equal bit for bit."""
    shares = model.training_level_shares_
    averaging_shares = dict(shares)
    for feature, subset_levels in (level_subsets or {}).items():
        subset = shares[feature].reindex(subset_levels).dropna()
        averaging_shares[feature] = pd.Series(0.0, index=shares[feature].index).add(subset / subset.sum(), fill_value=0.0)
    X = X.reset_index(drop=True)
    conditional = {feature: table.reset_index(drop=True) for feature, table in (conditional_shares or {}).items()}
    features = [column for column in X.columns if column in shares]
    result = np.zeros((len(X), len(model.named_steps["classifier"].classes_)))
    marginalized = [[] for _ in range(len(X))]

    def fill(rows, weights, remaining):
        if len(rows) == 0:
            return
        if not remaining:
            result[rows.index] += weights[:, None] * model.predict_proba(rows)
            return
        feature, rest = remaining[0], remaining[1:]
        supported = shares[feature][(shares[feature] > 0) & (shares[feature].index != missing_label)]
        supported = supported / supported.sum()
        averaging = averaging_shares[feature][(averaging_shares[feature] > 0) & (averaging_shares[feature].index != missing_label)]
        averaging = averaging / averaging.sum()
        unsupported = ~rows[feature].astype(str).isin(supported.index)
        fill(rows[~unsupported], weights[~unsupported.to_numpy()], rest)
        if unsupported.any():
            for row in rows.index[unsupported]:
                if feature not in marginalized[row]:
                    marginalized[row].append(feature)
            if feature in conditional:
                table = conditional[feature].loc[rows.index[unsupported]]
                row_shares = table.reindex(columns=averaging.index, fill_value=0.0).to_numpy(dtype=float)
                row_shares = row_shares / np.where(row_shares.sum(axis=1, keepdims=True) > 0, row_shares.sum(axis=1, keepdims=True), 1.0)
                for column_index, level in enumerate(averaging.index):
                    fill(rows[unsupported].assign(**{feature: level}), weights[unsupported.to_numpy()] * row_shares[:, column_index], rest)
            else:
                for level, share in averaging.items():
                    fill(rows[unsupported].assign(**{feature: level}), weights[unsupported.to_numpy()] * share, rest)

    fill(X, np.ones(len(X)), features)

    return levels.normalize_predicted_probabilities(result), pd.Series(["+".join(names) for names in marginalized], dtype=object)


def synthetic(n=600, seed=0):
    """Training rows never hold a3; b holds the missing label in a few rows."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "a": rng.choice(["a0", "a1", "a2"], n, p=[0.5, 0.3, 0.2]),
        "b": rng.choice(["b0", "b1", "b2", MISSING], n, p=[0.4, 0.3, 0.25, 0.05]),
        "c": rng.normal(size=n),
    })
    score = (frame["a"] == "a1") * 1.0 + (frame["b"] == "b2") * 1.5 + frame["c"]
    frame["y"] = np.where(score > 1.2, "z", np.where(score > 0.2, "y", "x"))
    frame["w"] = rng.uniform(0.5, 2.0, n)
    frame["g"] = np.arange(n) // 3

    return frame


@pytest.fixture(scope="module")
def fitted():
    frame = synthetic()
    candidate = models.build_candidates(FEATURES, NUMERIC, LEVELS, families={"GradientBoosting": {"native_categoricals": True, "early_stopping": False, "params": {"max_iter": [20]}}})["GradientBoosting"]
    model = candidate["model"].set_params(classifier__max_iter=20)
    X = levels.prepare_features(frame, FEATURES, NUMERIC)
    weights = levels.normalize_sample_weights(frame["w"])
    model.fit(X, frame["y"], classifier__sample_weight=weights)
    marginalize.attach_training_level_shares(model, X, weights, NUMERIC)

    return model


def scored_rows():
    rows = pd.DataFrame({
        "a": ["a0", "a3", "a0", "a3", MISSING, "a2", "a3", "a1"],
        "b": ["b1", "b0", MISSING, MISSING, "b2", "b2", "b1", None],
        "c": [0.1, -0.3, 1.2, 0.0, 0.5, np.nan, 2.0, -1.0],
    })
    return levels.prepare_features(rows, FEATURES, NUMERIC)


def test_batched_marginalization_equals_the_recursion(fitted):
    X = scored_rows()
    rng = np.random.default_rng(1)
    conditional = {"b": pd.DataFrame(rng.dirichlet(np.ones(2), len(X)), columns=["b0", "b2"])}  # b1 missing: weight 0
    for kwargs in ({}, {"level_subsets": {"a": ["a0", "a1"]}}, {"conditional_shares": conditional},
                   {"level_subsets": {"a": ["a1", "a2"]}, "conditional_shares": conditional}):
        batched, batched_names = marginalize.predict_proba_marginalizing(fitted, X, **kwargs)
        recursive, recursive_names = recursive_marginalizing(fitted, X, **kwargs)
        np.testing.assert_array_equal(batched, recursive)
        assert list(batched_names) == list(recursive_names)
    assert list(batched_names) == ["", "a", "b", "a+b", "a", "", "a", "b"]


def test_marginalization_does_not_depend_on_the_batch(fitted):
    X = scored_rows()
    rng = np.random.default_rng(2)
    conditional = pd.DataFrame(rng.dirichlet(np.ones(3), len(X)) * rng.uniform(0.3, 3.0, (len(X), 1)), columns=["b0", "b1", "b2"])
    together, _ = marginalize.predict_proba_marginalizing(fitted, X, conditional_shares={"b": conditional})
    for i in range(len(X)):
        alone, _ = marginalize.predict_proba_marginalizing(fitted, X.iloc[[i]], conditional_shares={"b": conditional.iloc[[i]]})
        np.testing.assert_array_equal(alone[0], together[i])


def test_level_subsets_only_narrow_the_averaging(fitted):
    X = scored_rows()
    plain, _ = marginalize.predict_proba_marginalizing(fitted, X)
    narrowed, names = marginalize.predict_proba_marginalizing(fitted, X, level_subsets={"a": ["a1"]})
    supported = ~names.str.contains("a").to_numpy()
    np.testing.assert_array_equal(plain[supported], narrowed[supported])     # a0 / a2 rows outside the subset: as is
    single = X[names.str.fullmatch("a").to_numpy()].assign(a="a1")
    np.testing.assert_allclose(narrowed[names.str.fullmatch("a").to_numpy()], fitted.predict_proba(single))


def test_marginalization_requires_training_shares(fitted):
    bare = models.build_candidates(FEATURES, NUMERIC, LEVELS, families={"GradientBoosting": {"params": {"max_iter": [5]}}})["GradientBoosting"]["model"]
    with pytest.raises(ValueError, match="training_level_shares_"):
        marginalize.predict_proba_marginalizing(bare, scored_rows())


def test_auxiliary_level_model_marginalizes_its_inputs():
    frame = synthetic()
    aux = models.fit_level_model(frame, "b", ["a", "c"], NUMERIC, LEVELS, sample_weights=frame["w"])
    assert set(aux.training_level_shares_) == {"a"}
    X = scored_rows()
    shares = marginalize.predict_level_shares(aux, X)
    unsupported = (X["a"] == "a3").to_numpy() | (X["a"] == MISSING).to_numpy()
    expected, _ = recursive_marginalizing(aux, levels.prepare_features(X, ["a", "c"], NUMERIC))
    np.testing.assert_array_equal(shares.to_numpy(), expected)
    np.testing.assert_array_equal(shares.to_numpy()[~unsupported], levels.normalize_predicted_probabilities(aux.predict_proba(X[~unsupported])))


def test_staged_boosting_equals_separate_fits():
    frame = synthetic()
    X, y, w = levels.prepare_features(frame, FEATURES, NUMERIC), frame["y"], levels.normalize_sample_weights(frame["w"])
    candidates = models.build_candidates(FEATURES, NUMERIC, LEVELS, families={"GradientBoosting": {"native_categoricals": True, "early_stopping": False, "params": {"max_iter": [3, 8], "learning_rate": [0.1]}}})
    splits = select.grouped_splits(X, y, frame["g"], cv_splits=3)
    staged = select.cross_validate_grid(candidates, X, y, w, splits, n_jobs=1, progress=False)
    for max_iter in (3, 8):
        params = {"classifier__learning_rate": 0.1, "classifier__max_iter": max_iter}
        for fold, (train, validation) in enumerate(splits):
            [separate] = select.fit_fold(candidates["GradientBoosting"]["model"], params, None, X, y, w, train, validation)
            for metric, value in separate.items():
                assert staged[("GradientBoosting", repr(sorted(params.items())))][fold][metric] == pytest.approx(value, abs=1e-12)


def test_parallel_tuning_equals_serial():
    frame = synthetic(n=300)
    X, y, w = levels.prepare_features(frame, FEATURES, NUMERIC), frame["y"], levels.normalize_sample_weights(frame["w"])
    families = {"LogisticRegression": {"max_iter": 500, "params": {"C": [0.1, 1.0]}},
                "GradientBoosting": {"native_categoricals": True, "early_stopping": False, "params": {"max_iter": [5, 10]}}}
    runs = [select.tune(models.build_candidates(FEATURES, NUMERIC, LEVELS, families=families), X, y, w, frame["g"], cv_splits=3, n_jobs=n_jobs, progress=False)[0]
            for n_jobs in (1, 2)]
    pd.testing.assert_frame_equal(runs[0].drop(columns=["best_params", "fold_log_losses"]), runs[1].drop(columns=["best_params", "fold_log_losses"]))
    assert list(runs[0]["fold_log_losses"]) == list(runs[1]["fold_log_losses"])


def test_fold_expansions_change_nothing_where_every_level_is_supported():
    frame = synthetic(n=300)
    frame["b"] = frame["b"].replace(MISSING, "b0")                 # every fold's training rows hold every level
    X, y, w = levels.prepare_features(frame, FEATURES, NUMERIC), frame["y"], levels.normalize_sample_weights(frame["w"])
    families = {"LogisticRegression": {"max_iter": 500, "params": {"C": [1.0]}},
                "GradientBoosting": {"native_categoricals": True, "early_stopping": False, "params": {"max_iter": [5, 10]}}}
    candidates = models.build_candidates(FEATURES, NUMERIC, LEVELS, families=families)
    splits = select.grouped_splits(X, y, frame["g"], cv_splits=3)
    expansions = [marginalize.expand_unsupported(X.iloc[validation].reset_index(drop=True), marginalize.compute_training_level_shares(X.iloc[train], w.iloc[train], NUMERIC))
                  for train, validation in splits]
    assert all(len(expansion.frame) == len(validation) for expansion, (_, validation) in zip(expansions, splits))
    assert select.cross_validate_grid(candidates, X, y, w, splits, n_jobs=1, progress=False, expansions=expansions) == \
        select.cross_validate_grid(candidates, X, y, w, splits, n_jobs=1, progress=False)


def test_one_se_picks_the_simplest_candidate_within_one_se():
    results = pd.DataFrame({
        "model": ["GradientBoosting", "LogisticRegression", "RandomForest"],
        "best_params": [{"classifier__max_iter": 100}, {"classifier__C": 1.0}, {"classifier__max_leaf_nodes": 50}],
        "fold_log_losses": [[0.90, 0.92, 0.91], [0.901, 0.925, 0.905], [1.0, 1.02, 1.01]],
    })
    table = select.select_one_se(results)
    assert table.loc[table["selected"], "model"].item() == "LogisticRegression"
    assert list(table["within_one_se"]) == [True, True, False]
    # simpler hyperparameters first within a family: smaller max_iter, larger l2
    assert select.complexity_key("GradientBoosting", {"classifier__max_iter": 50}) < select.complexity_key("GradientBoosting", {"classifier__max_iter": 100})
    assert select.complexity_key("GradientBoosting", {"classifier__l2_regularization": 1.0}) < select.complexity_key("GradientBoosting", {"classifier__l2_regularization": 0.0})


def test_harmonize_fails_on_unmapped_codes():
    mapping = {1: "hombre", 2: "mujer"}
    values = pd.Series([1, 2, None, 9], dtype="Int64")
    with pytest.raises(ValueError, match=r"Unmapped categories in sexo: \['9'\]"):
        harmonize.harmonize(values, mapping, name="sexo")
    mapped = harmonize.harmonize(values, mapping, allowed_unmapped=[9])
    assert mapped.tolist() == ["hombre", "mujer", pd.NA, MISSING]


def test_unknown_level_raises():
    with pytest.raises(ValueError, match="outside the declared category levels"):
        levels.assert_known_levels(pd.DataFrame({"a": ["a0", "a9"]}), LEVELS)


def test_arms_take_the_first_whose_covariates_are_observed():
    frame = pd.DataFrame({"a": ["a0", None, "a1"], "b": ["b0", "b1", MISSING]})
    arms = [Arm("full", None, ["a", "b"], requires=["a", "b"]), Arm("only_a", None, ["a"], requires=["a"]), Arm("none", None, [])]
    assigned = assign_arms(frame, arms, lambda data, feature: levels.identify_missing(data[feature]))
    assert assigned.tolist() == ["full", "none", "only_a"]


def test_feature_cache_hits_misses_and_keys(tmp_path, monkeypatch):
    from eodgdl.impute import features, sources, spec as spec_module

    calls = {"source": 0, "builder": 0}

    def build_source(context, config):
        calls["source"] += 1
        frame = synthetic(n=30)[["a", "b", "c", "y"]].assign(k=range(30), label=lambda f: f["y"].str.upper())
        return frame

    def build_columns(frame, context, config, spec):
        calls["builder"] += 1
        return pd.DataFrame({"d": frame["c"] * config["scale"], "e": pd.Categorical(frame["a"])}, index=frame.index)

    monkeypatch.setitem(sources._SOURCES, "test.rows", sources.Source("test.rows", build_source, {"keys": ["k"], "weight": "c", "group": "k"},
                                                                    lambda context, config: {"data": "v1"}, lambda column: LEVELS[column][:-1], __file__))
    monkeypatch.setitem(features._BUILDERS, "test.double", features.Builder("test.double", build_columns, {"scale": 2.0}, lambda context, config: {}, __file__))
    raw = {"task": "t", "source": "test.rows", "target": {"column": "label", "classes": {"X": "x", "Y": "y", "Z": "z"}},
           "features": {"builders": ["test.double"], "numeric": ["c", "d"]}, "arms": {"only": {"features": ["a", "b", "c", "d", "e"]}}}
    spec = spec_module.parse_task(raw)
    context = sources.Context(cache=True, cache_dir=tmp_path)
    first = features.build_frame(spec, context).frame
    again = features.build_frame(spec, context).frame
    assert calls == {"source": 1, "builder": 1}
    pd.testing.assert_frame_equal(first, again)
    assert first["e"].dtype == "string" and first["t"].tolist()[:3] == first["y"].tolist()[:3]
    assert not first["t_desconocido"].any()
    features.build_frame(spec, context, overrides={"test.double": {"scale": 3.0}})
    assert calls == {"source": 1, "builder": 2}                           # new configuration, new key
    features.build_frame(spec, sources.Context(cache=True, cache_dir=tmp_path, refresh=True))
    assert calls["source"] == 2 and calls["builder"] == 3                  # refresh rebuilds
    # a task with other classes shares the builder's columns, unless the builder reads the classes
    other = spec_module.parse_task({**raw, "task": "u", "target": {"column": "label", "classes": {"X": "x", "Y": "y", "Z": "z", "W": "w"}}})
    features.build_frame(other, context)
    assert calls == {"source": 2, "builder": 3}
    monkeypatch.setitem(features._BUILDERS, "test.double", features.Builder("test.double", build_columns, {"scale": 2.0}, lambda context, config: {}, __file__, reads_classes=True))
    features.build_frame(spec, context)
    features.build_frame(other, context)
    assert calls == {"source": 2, "builder": 5}
    # no shipped builder reads the classes since giro's destination shares went by SCIAN sector (v0.6.0)
    assert not features.get_builder("giro.destination").reads_classes and not features.get_builder("harmonize.enigh").reads_classes


def test_spec_validation():
    from eodgdl.impute.spec import parse_task

    base = {"task": "t", "source": "s", "target": {"column": "y", "classes": {"A": "a", "B": "b"}},
            "features": {"groups": {"g": ["u", "v"]}, "numeric": ["v"]}, "arms": {"one": {"features": ["g"], "requires": ["u"]}, "rest": {"features": ["v"]}}}
    spec = parse_task(base)
    assert spec.features == ["u", "v"] and spec.arm("one").requires == ("u",)
    with pytest.raises(ValueError, match="requires features it does not use"):
        parse_task({**base, "arms": {"one": {"features": ["g"], "exclude": ["u"], "requires": ["u"]}, "rest": {"features": ["v"]}}})
    with pytest.raises(ValueError, match="last arm must require nothing"):
        parse_task({**base, "arms": {"one": {"features": ["g"], "requires": ["u"]}}})
    with pytest.raises(ValueError, match="numeric features no arm uses"):
        parse_task({**base, "features": {"groups": {"g": ["u", "v"]}, "numeric": ["w"]}})
    with pytest.raises(ValueError, match="class slugs repeat"):
        parse_task({**base, "target": {"column": "y", "classes": {"A": "a", "B": "a"}}})
    assert parse_task(base).scoring_hash() == spec.scoring_hash()
    assert parse_task({**base, "missing_values": {"u": ["?"]}}).scoring_hash() != spec.scoring_hash()
    # levels a task adds to a schema column's (a builder's own level), before the missing label
    extra = parse_task({**base, "levels": {"extra": {"u": ["nuevo"]}}})
    assert extra.category_levels(lambda column: ["a", "b"])["u"] == ["a", "b", "nuevo", MISSING]
    with pytest.raises(ValueError, match="extra levels for 'v'"):
        parse_task({**base, "levels": {"extra": {"v": ["x"]}}})                      # v is numeric


SYNTHETIC_TASK = {
    "task": "synthetic", "source": "test.labelled", "target": {"column": "label", "classes": {"X": "x", "Y": "y", "Z": "z"}},
    "features": {"groups": {"core": ["a", "b", "c"]}, "numeric": ["c"]},
    "levels": {"declared": {"a": ["a0", "a1", "a2", "a3"], "b": ["b0", "b1", "b2", "?"]}},
    "missing_values": {"b": ["?"]},
    "arms": {"with_b": {"features": ["core"], "requires": ["b"]}, "without_b": {"features": ["core"], "exclude": ["b"]}},
    "auxiliary": {"a": {"exclude": ["a"]}},
    "selection": {"cv": {"splits": 3, "seed": 42, "test_fold": 0}, "families": {
        "LogisticRegression": {"max_iter": 500, "params": {"C": [1.0]}},
        "GradientBoosting": {"native_categoricals": True, "early_stopping": False, "params": {"max_iter": [5, 10]}}}},
    "evaluation": {"bootstrap": 10, "calibration_bins": 5, "profiles": ["a", "b"], "shift": {"profile": ["a", "c"]}, "delta": {"class": "z"}, "isotonic": {"ship": False}},
}


@pytest.fixture
def labelled_source(monkeypatch):
    from eodgdl.impute import sources

    def build(context, config):
        frame = synthetic(n=450, seed=3)
        frame["label"] = frame["y"].str.upper().where(np.arange(len(frame)) % 5 != 0)   # every fifth row unknown
        frame.loc[frame.index % 7 == 0, "b"] = "?"                                       # unobserved b: the second arm
        frame.loc[frame.index % 11 == 0, "a"] = MISSING                                  # unobserved a: marginalized with P(a | x)
        frame["k"] = np.arange(len(frame))
        return frame

    monkeypatch.setitem(sources._SOURCES, "test.labelled", sources.Source("test.labelled", build, {"keys": ["g", "k"], "weight": "w", "group": "g"},
                                                                        lambda context, config: {"data": "v1"}, lambda column: [], __file__))


def test_retrain_writes_a_bundle_that_scores_the_same(labelled_source, tmp_path, monkeypatch):
    from eodgdl.impute import bundle as bundles, run, sources
    from eodgdl.impute.spec import parse_task

    spec = parse_task(SYNTHETIC_TASK)
    context = sources.Context(cache_dir=tmp_path / "cache")
    result = run.retrain(spec, context=context, n_jobs=1, progress=False)
    assert set(result.bundle["arms"]) == {"with_b", "without_b"} and set(result.bundle["auxiliary"]["with_b"]) == {"a"}
    scored = result.scored
    assert scored["synthetic_fue_imputado"].sum() == 90
    assert set(scored.loc[scored["synthetic_fue_imputado"], "synthetic_model_used"]) == {"with_b", "without_b"}
    assert scored["synthetic_marginalized_features"].str.contains("a").any()
    assert set(result.scenarios) == {"shift_weighted", "delta_adjusted"}
    assert {"selection__with_b", "isotonic__without_b", "shares", "profiles", "unsupported_levels__with_b"} <= set(result.tables)
    assert {"isotonic_calibration__with_b", "isotonic_reliability__without_b", "test_metrics__hybrid", "calibration__hybrid", "reliability__hybrid"} <= set(result.tables)
    hybrid = result.tables["test_metrics__hybrid"].iloc[0]
    assert hybrid["rows_with_b"] + hybrid["rows_without_b"] == result.summary["test_rows"] and hybrid["rows_without_b"] > 0

    # the task's own definition of unobserved: b's "?" answers count as missing, as they do for the arms
    frame = result.scored
    rows = frame[frame["synthetic_fue_imputado"]]
    expected = rows.loc[spec.is_missing(rows, "b"), "w"].sum() / rows["w"].sum()
    assert result.tables["missingness"].set_index("variable").loc["b", "unknown_missing_share"] == pytest.approx(expected)
    assert (rows["b"] == "?").any() and expected > rows.loc[rows["b"] == MISSING, "w"].sum() / rows["w"].sum()
    # the delta-adjusted scenario keeps its outputs coherent: confidence and arg-max read from the adjusted probabilities
    delta = result.scenarios["delta_adjusted"]
    adjusted = delta.loc[delta["synthetic_fue_imputado"], spec.probability_columns]
    np.testing.assert_array_equal(delta.loc[delta["synthetic_fue_imputado"], "synthetic_prediction_confidence"], adjusted.max(axis=1))
    assert (delta.loc[delta["synthetic_fue_imputado"], "synthetic_final"].to_numpy() == adjusted.idxmax(axis=1).str.removeprefix("prob_synthetic_").to_numpy()).all()

    path, digest = run.write_retrain(result, tmp_path / "out" / spec.name)
    assert (tmp_path / "out" / spec.name / "scores.parquet").exists() and (tmp_path / "out" / spec.name / "evaluation" / "reliability__with_b.parquet").exists()
    reloaded = run.score_frame(spec, result.scored.drop(columns=run.output_columns(spec, [])), bundles.load_bundle(spec, path))
    pd.testing.assert_frame_equal(reloaded, scored)
    # a retrain written under <root>/<task>/ is what retrained=<root> reads
    assert bundles.bundle_path(spec, retrained=tmp_path / "out") == path
    assert bundles.load_bundle(spec, retrained=tmp_path / "out")["metadata"]["spec_hash"] == result.bundle["metadata"]["spec_hash"]

    scenario_bundles, score_frame = {}, run.score_frame

    def recording(spec, frame, bundle, **kwargs):
        scenario_bundles[bundle["metadata"].get("scenario", "scored")] = bundle
        return score_frame(spec, frame, bundle, **kwargs)

    monkeypatch.setattr(run, "score_frame", recording)
    shipped = run.retrain(parse_task({**SYNTHETIC_TASK, "evaluation": {**SYNTHETIC_TASK["evaluation"], "isotonic": {"ship": True}}}), context=context, n_jobs=1, progress=False)
    assert type(shipped.bundle["arms"]["with_b"]["model"]).__name__ == "IsotonicCalibrated"
    np.testing.assert_allclose(shipped.scored[spec.probability_columns].sum(axis=1), 1.0)
    # the shift scenario refits the arms as the bundle ships them
    assert {type(entry["model"]).__name__ for entry in scenario_bundles["shift_weighted"]["arms"].values()} == {"IsotonicCalibrated"}
    np.testing.assert_allclose(shipped.scenarios["shift_weighted"][spec.probability_columns].sum(axis=1), 1.0)


def test_bootstrap_bundles_refit_the_selected_configurations(labelled_source, tmp_path):
    """B refits of the bundle's selected configurations and auxiliary models on cluster bootstraps of the training rows:
    valid bundles, seeded, each its own model; written under <task>/bootstrap/ and read back in order."""
    from eodgdl.impute import bundle as bundles, run, sources
    from eodgdl.impute.spec import parse_task

    spec = parse_task(SYNTHETIC_TASK)
    context = sources.Context(cache_dir=tmp_path / "cache")
    fitted = run.retrain(spec, context=context, n_jobs=1, progress=False).bundle
    boot = run.bootstrap_bundles(spec, fitted, 3, context=context, n_jobs=1)
    task_levels = run.task_levels(spec)
    frame = run.build_frame(spec, context).frame
    scores = [run.score_frame(spec, frame, bundles.check_bundle(model, spec, task_levels))[spec.probability_columns].to_numpy() for model in [fitted, *boot]]
    assert [model["metadata"]["bootstrap"] for model in boot] == [0, 1, 2] and set(boot[0]["auxiliary"]["with_b"]) == {"a"}
    assert all(not np.allclose(scores[0], other) for other in scores[1:]) and not np.allclose(scores[1], scores[2])
    again = run.bootstrap_bundles(spec, fitted, 2, context=context, n_jobs=1)
    np.testing.assert_array_equal(run.score_frame(spec, frame, again[1])[spec.probability_columns].to_numpy(), scores[2])   # seeded per refit
    paths = run.write_bootstrap(spec, boot, tmp_path / "root" / spec.name)
    assert [path.name for path in paths] == ["synthetic_000.joblib", "synthetic_001.joblib", "synthetic_002.joblib"]
    read = bundles.load_bootstrap(spec, tmp_path / "root")
    np.testing.assert_array_equal(run.score_frame(spec, frame, read[2])[spec.probability_columns].to_numpy(), scores[3])
    assert bundles.load_bootstrap(spec, tmp_path / "elsewhere") == [] and bundles.load_bootstrap(spec, None) == []


def test_check_bundle_refuses_what_cannot_score(labelled_source, tmp_path):
    from eodgdl.impute import bundle as bundles, run, sources
    from eodgdl.impute.spec import parse_task

    spec = parse_task(SYNTHETIC_TASK)
    fitted = run.retrain(spec, context=sources.Context(cache_dir=tmp_path), n_jobs=1, progress=False).bundle
    task_levels = run.task_levels(spec)
    assert bundles.check_bundle(fitted, spec, task_levels) is fitted
    arms, auxiliary, metadata = fitted["arms"], fitted["auxiliary"], fitted["metadata"]
    cases = {
        "bundle of task 'other'": {**fitted, "task": "other"},
        "fitted with scikit-learn 0.1": {**fitted, "metadata": {**metadata, "sklearn_version": "0.1"}},
        r"category levels differ from the task's for \['a'\]": {**fitted, "category_levels": {**task_levels, "a": ["a0", MISSING]}},
        "classes .* differ from the task's": {**fitted, "classes": {"X": "x", "Y": "y"}},
        r"arms \['without_b', 'with_b'\] differ": {**fitted, "arms": dict(reversed(list(arms.items())))},
        "arm with_b: features differ from the task's": {**fitted, "arms": {**arms, "with_b": {**arms["with_b"], "features": ["a", "c"]}}},
        "arm with_b: the fitted pipeline selects other features": {**fitted, "arms": {**arms, "with_b": {**arms["with_b"], "model": arms["without_b"]["model"]}}},
        "arm with_b: auxiliary model for 'b', which the task does not declare": {**fitted, "auxiliary": {**auxiliary, "with_b": {"b": auxiliary["with_b"]["a"]}}},
        "arm with_b: the auxiliary a model uses other predictors": {**fitted, "auxiliary": {**auxiliary, "with_b": {"a": auxiliary["without_b"]["a"]}}},
        "scoring_hash differs from the task's": {**fitted, "metadata": {**metadata, "scoring_hash": "x"}},
        "level_contract_hash differs from the task's": {**fitted, "metadata": {**metadata, "level_contract_hash": "x"}},
    }
    for message, broken in cases.items():
        with pytest.raises(bundles.BundleMismatch, match=message):
            bundles.check_bundle(broken, spec, task_levels)


def test_bootstrap_samples_draw_whole_clusters():
    from eodgdl.impute import run

    frame = pd.DataFrame({"g": [1, 1, 2, 3, 3, 3], "x": range(6)})
    sample = run.bootstrap_sample(frame, "g", np.random.default_rng(0))
    sizes = frame["g"].value_counts()
    drawn = sample["g"].value_counts()
    assert all(drawn[g] % sizes[g] == 0 for g in drawn.index)                    # every drawn cluster brings all its rows
    assert sum(drawn[g] // sizes[g] for g in drawn.index) == frame["g"].nunique()  # as many clusters as the frame holds


def test_rubins_rules_and_cluster_shares():
    from eodgdl.impute import evaluate

    combined = evaluate.rubin([0.30, 0.32, 0.31, 0.29], [1e-4, 1.1e-4, 0.9e-4, 1e-4])
    between, within = np.var([0.30, 0.32, 0.31, 0.29], ddof=1), 1e-4
    total = within + 1.25 * between
    assert combined["estimate"] == pytest.approx(0.305) and combined["se_total"] == pytest.approx(np.sqrt(total))
    assert combined["fmi"] == pytest.approx(1.25 * between / total) and combined["df"] == pytest.approx(3 * (1 + within / (1.25 * between)) ** 2)
    assert combined["ci_low"] < 0.305 < combined["ci_high"]
    frame = pd.DataFrame({"d": [0] * 4 + [1] * 4, "y": [1, 0, 1, 1, 0, 0, 1, 0], "w": [1.0, 2.0, 1.0, 1.0] * 2, "c": [1, 1, 2, 3] * 2})
    table = evaluate.cluster_shares(frame, "y", "w", "c", ["d"])
    share = (1.0 + 1.0 + 1.0) / 5.0
    z = np.array([1.0 * (1 - share) + 2.0 * (0 - share), 1.0 * (1 - share), 1.0 * (1 - share)])   # completion 0's clusters
    assert table.loc[0, "share"] == pytest.approx(share) and table.loc[0, "variance"] == pytest.approx(3 / 2 * (z ** 2).sum() / 25)
    assert table.loc[0, "clusters"] == 3


def test_hybrid_heldout_predicts_each_row_with_its_arm():
    from eodgdl.impute import run
    from eodgdl.impute.spec import parse_task

    spec = parse_task({**SYNTHETIC_TASK, "target": {"column": "label", "classes": {"Z": "z", "X": "x", "Y": "y"}}})   # not sorted
    test = pd.DataFrame({"a": "a0", "b": ["b0", "?", "b1", MISSING], "c": 0.0, "w": [1.0, 2.0, 1.0, 1.0], spec.target: ["x", "y", "z", "x"]})
    probabilities = {"with_b": np.tile([0.8, 0.1, 0.1], (4, 1)), "without_b": np.tile([0.2, 0.6, 0.2], (4, 1))}   # columns x, y, z
    metrics = run._hybrid_tables(spec, test, probabilities, "w", None, 5)["test_metrics__hybrid"].iloc[0]
    assert (metrics["rows_with_b"], metrics["rows_without_b"]) == (2, 2)
    # b observed (rows 0, 2): the arm with b; "?" and the missing label (rows 1, 3): the arm without it
    assert metrics["weighted_log_loss"] == pytest.approx(-np.average(np.log([0.8, 0.6, 0.1, 0.2]), weights=[1, 2, 1, 1]))


def test_heldout_rows_are_predicted_as_scoring_predicts_them(fitted, labelled_source):
    from eodgdl.impute import run
    from eodgdl.impute.spec import parse_task

    spec = parse_task({"task": "h", "source": "test.labelled", "target": {"column": "label", "classes": {"X": "x", "Y": "y", "Z": "z"}},
                       "features": {"numeric": NUMERIC}, "levels": {"declared": {"a": LEVELS["a"][:-1], "b": LEVELS["b"][:-1]}},
                       "arms": {"only": {"features": FEATURES}}})
    rows = synthetic(n=80, seed=11).assign(h=lambda frame: frame["y"])
    rows.loc[::5, "b"] = None                                      # unobserved: the missing label
    rows.loc[::7, "a"] = "a3"                                      # a level without training support
    X = levels.prepare_features(rows, FEATURES, NUMERIC)
    averaged = ((X["b"] == MISSING) | (X["a"] == "a3")).to_numpy()
    y, probabilities, classes, _ = run.heldout(fitted, spec, rows, "w")
    np.testing.assert_array_equal(probabilities, marginalize.predict_proba_marginalizing(fitted, X)[0])
    direct = fitted.predict_proba(rows)                            # the missing label as the category the model learned
    np.testing.assert_allclose(probabilities[~averaged], direct[~averaged], rtol=0, atol=1e-12)
    assert not np.allclose(probabilities[averaged], direct[averaged])
    assert classes == list(fitted.named_steps["classifier"].classes_) and y.tolist() == rows["y"].tolist()
    # with an auxiliary model an unobserved b is averaged over P(b | x), as the bundle's arm scores it
    auxiliary = {"b": models.fit_level_model(synthetic(), "b", ["a", "c"], NUMERIC, LEVELS)}
    with_auxiliary = run.heldout(fitted, spec, rows, "w", auxiliary)[1]
    conditional = {"b": marginalize.predict_level_shares(auxiliary["b"], X)}
    np.testing.assert_array_equal(with_auxiliary, marginalize.predict_proba_marginalizing(fitted, X, conditional_shares=conditional)[0])
    assert not np.allclose(with_auxiliary[X["b"].eq(MISSING).to_numpy()], probabilities[X["b"].eq(MISSING).to_numpy()])


def test_selection_scores_each_fold_as_the_bundle_scores(labelled_source, tmp_path):
    """The grid's CV judges a configuration on a validation fold as a bundle fitted on the fold's training rows would
    score it: the fold model with its training shares, an unobserved a averaged over P(a | x) from an auxiliary model
    fitted on every row outside the validation fold."""
    from sklearn.base import clone

    from eodgdl.impute import run, sources
    from eodgdl.impute.arms import Arm, score_arm
    from eodgdl.impute.features import build_frame
    from eodgdl.impute.spec import parse_task

    spec = parse_task(SYNTHETIC_TASK)
    source = build_frame(spec, sources.Context(cache_dir=tmp_path / "cache"))
    frame, task_levels = source.frame, run.task_levels(spec)
    others = frame[frame[spec.unknown_column].astype(bool).to_numpy()]
    arm = spec.arm("with_b")
    X, y, w, g, rows = run.training_data(spec, frame, list(arm.features), source.weight, source.group)
    expand = run.fold_expander(spec, arm, rows, X, w, others, source.weight, task_levels, 42)
    candidates = models.build_candidates(list(arm.features), spec.numeric, task_levels, {"LogisticRegression": {"max_iter": 500, "params": {"C": [1.0]}}})
    splits = select.grouped_splits(X, y, g, cv_splits=3)
    [(_, as_scored)] = select.cross_validate_grid(candidates, X, y, w, splits, n_jobs=1, progress=False, expansions=[expand(*split) for split in splits]).items()
    [(_, as_is)] = select.cross_validate_grid(candidates, X, y, w, splits, n_jobs=1, progress=False).items()
    for fold, (train, validation) in enumerate(splits):
        model = clone(candidates["LogisticRegression"]["model"]).set_params(classifier__C=1.0)
        model.fit(X.iloc[train], y.iloc[train], classifier__sample_weight=w.iloc[train])
        marginalize.attach_training_level_shares(model, X.iloc[train], w.iloc[train], spec.numeric)
        side = pd.concat([rows.iloc[train], others], ignore_index=True)
        auxiliary = {"a": models.fit_level_model(side, "a", spec.auxiliary_predictors("a", arm.name), spec.numeric, task_levels, sample_weights=side[source.weight])}
        probabilities, marginalized, classes = score_arm(Arm(arm.name, model, list(arm.features), auxiliary=auxiliary), rows.iloc[validation], spec.numeric, task_levels)
        expected = select.fold_metrics(y.iloc[validation], w.iloc[validation], classes[probabilities.argmax(axis=1)], probabilities, classes)
        assert as_scored[fold] == pytest.approx(expected, abs=1e-12)
        assert marginalized.str.contains("a").any() and marginalized.str.contains("b").any()
        # read as it is, the missing label is the category the model learned: another predictor
        assert abs(as_scored[fold]["log_loss"] - as_is[fold]["log_loss"]) > 1e-6


def test_delta_adjustment_reaches_its_target_where_probabilities_reach_one():
    from eodgdl.impute import evaluate

    classes, columns = ["a", "b", "c"], ["prob_t_a", "prob_t_b", "prob_t_c"]
    imputed = pd.DataFrame({"t_fue_imputado": [True, True, True, False], "w": [1.0, 2.0, 1.0, 1.0], "t_final": [None, None, None, "a"],
                            "prob_t_a": [0.9, 0.5, 0.1, 1.0], "prob_t_b": [0.1, 0.3, 0.6, 0.0], "prob_t_c": [0.0, 0.2, 0.3, 0.0]})   # imputed share of a: 0.5

    def share(frame):
        rows = frame["t_fue_imputado"]
        return np.average(frame.loc[rows, "prob_t_a"], weights=frame.loc[rows, "w"])

    # scaling down: the ratio of the target to the current share
    adjusted, factor = evaluate.adjust_imputed_share(imputed, "t", classes, "a", target_share=0.25, weight_column="w")
    assert factor == pytest.approx(0.5) and share(adjusted) == pytest.approx(0.25)
    # scaling up past one (the ratio, 1.6, would leave the share at 0.69): the largest probabilities held at one
    adjusted, factor = evaluate.adjust_imputed_share(imputed, "t", classes, "a", target_share=0.8, weight_column="w")
    assert factor == pytest.approx(2.0) and share(adjusted) == pytest.approx(0.8)
    np.testing.assert_allclose(adjusted.loc[[0, 1, 2], "prob_t_a"], [1.0, 1.0, 0.2])
    np.testing.assert_allclose(adjusted[columns].sum(axis=1), 1.0)
    assert (adjusted.loc[[0, 1], ["prob_t_b", "prob_t_c"]] == 0).all().all() and adjusted.loc[3, columns].tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="No factor reaches"):      # a quarter of the imputed weight has a positive probability
        evaluate.adjust_imputed_share(imputed.assign(prob_t_a=[0.9, 0.0, 0.0, 1.0], prob_t_b=[0.1, 0.8, 0.7, 0.0]), "t", classes, "a", target_share=0.5, weight_column="w")


def test_compare_pairs_candidates_with_the_baseline(labelled_source, tmp_path):
    from eodgdl.impute import run, sources
    from eodgdl.impute.spec import parse_task

    spec = parse_task(SYNTHETIC_TASK)
    context = sources.Context(cache_dir=tmp_path / "cache")
    bundle = run.retrain(spec, context=context, n_jobs=1, progress=False).bundle
    folds, paired = run.compare(spec, {"same": {}, "no_c": {"arms": {"with_b": {"features": ["a", "b"], "requires": ["b"]}}}},
                                seeds=(1, 2), bundle=bundle, context=context, n_jobs=1, progress=False)
    assert len(folds) == 3 * 2 * 2 * 3                       # candidates x arms x seeds x folds
    same = paired[paired["candidate"] == "same"]
    assert (same["mean_difference"] == 0).all()
    dropped = paired[(paired["candidate"] == "no_c") & (paired["arm"] == "with_b")]
    assert (dropped["mean_difference"] != 0).all()
    assert (paired[(paired["candidate"] == "no_c") & (paired["arm"] == "without_b")]["mean_difference"] == 0).all()


def test_harmonization_variables():
    variables = {
        "rel": {"column": "par_c", "floordiv": 100, "map": {1: "jefe", 2: "conyuge"}},
        "mun": {"column": "mun", "map": {39: "gdl"}, "otherwise": {"level": "otro", "domain": [1, 125]}},
        "civil": {"column": "e_con", "map": {1: "union", 5: "casado"}, "allowed_unmapped": [9]},
        "size": {"column": "n", "number": {"map": {"1": 1, "2": 2, "10 y +": 10}, "cap": 7}},
        "size_cat": {"from": "size", "bins": "household_size"},
        "age": {"column": "eda", "number": {"unspecified": [98]}},
        "place": {"domains": {"p4b": [1, 2, 3], "scian": [6, 7, 8]}, "rules": [
            {"any": {"p4b": [2, 3]}, "level": "establecimiento"},
            {"any": {"p4b": [2]}, "all": {"scian": [6, 7]}, "level": "comercio"}]},
    }
    frame = pd.DataFrame({"par_c": pd.array([101, 203, None], "Int64"), "mun": pd.array([39, 120, None], "Int64"), "e_con": pd.array([1, 9, 5], "Int64"),
                          "n": ["1", "10 y +", None], "eda": pd.array([30, 98, None], "Int64"), "p4b": pd.array([2, 3, 1], "Int64"), "scian": pd.array([6, 8, 7], "Int64")})
    out = harmonize.apply_variables(frame, variables)
    assert out["rel"].tolist() == ["jefe", "conyuge", MISSING]
    assert out["mun"].tolist() == ["gdl", "otro", MISSING]
    assert out["civil"].tolist() == ["union", MISSING, "casado"]
    assert out["size"].tolist()[:2] == [1, 7] and pd.isna(out["size"].iloc[2])
    assert out["size_cat"].tolist() == ["1", "7_y_mas", MISSING]
    assert out["age"].tolist()[0] == 30 and out["age"].isna().tolist()[1:] == [True, True]
    assert out["place"].tolist() == ["comercio", "establecimiento", MISSING]
    with pytest.raises(ValueError, match=r"Unmapped categories in mun: \['200'\]"):
        harmonize.apply_variables(frame.assign(mun=pd.array([39, 200, None], "Int64")), {"mun": variables["mun"]})
    with pytest.raises(ValueError, match="outside its declared domain"):
        harmonize.apply_variables(frame.assign(p4b=pd.array([2, 4, 1], "Int64")), {"place": variables["place"]})


def test_eod_harmonization_covers_the_schema():
    # every label the eodgdl schemas allow in a harmonized EOD column is mapped (or deliberately left unmapped)
    from eodgdl.impute.sources.eod import schema_levels

    for name, definition in harmonize.load_harmonization("eod")["variables"].items():
        if "map" not in definition:
            continue
        column = {"destino_trabajo": "tipo_lugar_destino"}.get(definition["column"], definition["column"])
        unmapped = set(schema_levels(column)) - set(definition["map"]) - set(definition.get("allowed_unmapped", []))
        assert not unmapped, (name, sorted(unmapped))


def test_enoe_source_on_a_stand_in(monkeypatch):
    # The employed filter, the roster's dwelling size, the weight over pooled quarters and the cross-quarter household
    # key, on two quarters of a stand-in mxcensus (no network).
    mxcensus = pytest.importorskip("mxcensus")
    from eodgdl.impute.sources import Context, enoe

    def dwelling(v_sel, **extra):
        return {"tipo": "1", "mes_cal": "2", "cd_a": "14", "ent": "14", "con": "0100", "v_sel": v_sel, **extra}

    def persons(period, ent, canonical_filter, labels):
        base = {"n_hog": "1", "h_mud": "0", "mun": "39", "est_d_tri": "5", "upm": "7", "est": "2", "sex": "1", "pos_ocu": "1", "scian": "7",
                "cs_p13_1": "3", "emp_ppal": "1", "e_con": "5", "par_c": "101", "p4": "1", "p4b": "4", "p4e": "3", "p4f": "", "p4h": "1",
                "tue2": "5", "seg_soc": "2", "fac_tri": "300"}
        rows = [
            {**dwelling("01"), **base, "n_ren": "01", "eda": "30", "r_def": "00", "c_res": "1", "clase2": "1"},
            {**dwelling("01"), **base, "n_ren": "02", "eda": "11", "r_def": "00", "c_res": "1", "clase2": "1"},   # too young
            {**dwelling("02"), **base, "n_ren": "01", "eda": "45", "r_def": "00", "c_res": "2", "clase2": "1"},   # moved out
            {**dwelling("02"), **base, "n_ren": "02", "eda": "98", "r_def": "00", "c_res": "3", "clase2": "1"},   # age unspecified: kept
            {**dwelling("03"), **base, "n_ren": "01", "eda": "40", "r_def": "00", "c_res": "1", "clase2": "2"},   # not employed
        ]
        return pd.DataFrame(rows)

    def sdem(table, period, ent, labels):
        return pd.DataFrame([{**dwelling("01"), "c_res": "1"}] * 3 + [{**dwelling("02"), "c_res": "3"}, {**dwelling("02"), "c_res": "2"}])

    monkeypatch.setattr(mxcensus, "load_enoe_persons", persons)
    monkeypatch.setattr(mxcensus, "load_enoe", sdem)
    config = {**enoe.load_config()["enoe.workers"], "periods": ["2023t1", "2023t2"]}
    frame = enoe.workers(Context(cache=False), config)
    assert len(frame) == 4 and frame["period"].tolist() == ["2023t1", "2023t1", "2023t2", "2023t2"]
    assert frame["survey_weight"].tolist() == [150.0] * 4
    assert frame["dwelling_size"].tolist() == [3, 1, 3, 1]
    assert frame["hogar"].tolist()[:2] == ["14_14_100_1_1_0", "14_14_100_2_1_0"]          # no panel-visit fields
    from eodgdl.impute.features import get_builder

    harmonized = get_builder("harmonize.enoe").build(frame, None, {}, None)
    assert harmonized["edad_num"].isna().tolist() == [False, True, False, True]
    assert harmonized[["lugar_trabajo", "sector", "informalidad"]].iloc[0].tolist() == ["comercio_o_puesto", "comercio", "informal"]

    monkeypatch.setattr(mxcensus, "load_enoe_persons", lambda **kwargs: pd.concat([persons(**kwargs)] * 2))
    with pytest.raises(AssertionError, match="do not identify persons"):
        enoe.workers(Context(cache=False), config)


def test_retrain_diagnostics_robustness_profile_and_components(labelled_source, tmp_path):
    from eodgdl.impute import run, sources
    from eodgdl.impute.spec import parse_task

    raw = {**SYNTHETIC_TASK, "evaluation": {**SYNTHETIC_TASK["evaluation"], "positive_class": "z", "robustness_groups": "psu",
                                            "target_profile": {"features": ["a", "c"], "missing": "b"},
                                            "components": {"part": {"positive": "hi", "classes": ["hi", "lo"]}}}}
    spec = parse_task(raw)
    original = sources._SOURCES["test.labelled"]
    sources._SOURCES["test.labelled"] = sources.Source("test.labelled", lambda context, config: _with_parts(original, context, config),
                                                       original.config, original.versions, original.schema_levels, __file__)
    try:
        result = run.retrain(spec, context=sources.Context(cache_dir=tmp_path), n_jobs=1, progress=False)
    finally:
        sources._SOURCES["test.labelled"] = original
    robust = result.tables["robustness_cv__with_b"]
    assert list(robust.columns) == ["fold", "household_grouped", "psu_grouped"] and len(robust) == 3
    assert robust["household_grouped"].tolist() == result.bundle["metadata"]["selected"]["with_b"]["fold_log_losses"]
    profile = result.tables["target_profile"]
    assert set(profile["weighting"]) == {"training weights", "reweighted to the scored rows' profile"} and "weighted_roc_auc" in profile
    assert set(result.tables["components_heldout"]["component"]) == {"part"}
    assert set(result.bundle["components"]["part"]["arms"]) == {"with_b", "without_b"}
    component = result.bundle["components"]["part"]["arms"]["with_b"]
    assert set(component.named_steps["classifier"].classes_) == {"hi", "lo"}
    # the component is observed only where the label is (the source gives it on every row): trained on the same rows
    main = result.bundle["arms"]["with_b"]["model"]
    for feature, shares in main.training_level_shares_.items():
        pd.testing.assert_series_equal(component.training_level_shares_[feature], shares)


def _with_parts(original, context, config):
    frame = original.build(context, config)
    frame["psu"] = (np.arange(len(frame)) // 9).astype(str)
    frame["part"] = np.where(frame["c"] > 0, "hi", "lo")     # also where the label is unknown: the component must not train there
    return frame


def test_training_population_filters_the_training_rows(labelled_source, tmp_path):
    from eodgdl.impute import run, sources
    from eodgdl.impute.spec import parse_task

    raw = {**SYNTHETIC_TASK, "selection": {**SYNTHETIC_TASK["selection"], "population": {"a": ["a0", "a1"]}},
           "evaluation": {**SYNTHETIC_TASK["evaluation"], "populations": {"all": {}, "a0": {"a": ["a0"]}}, "population_tests": {"a1": {"a": ["a1"]}}}}
    result = run.retrain(parse_task(raw), context=sources.Context(cache_dir=tmp_path), n_jobs=1, progress=False)
    assert result.summary["source_rows"] == 450 and result.summary["known"] < 360
    assert result.summary["train_rows"] + result.summary["test_rows"] == result.summary["known"]
    populations = result.tables["populations"]
    assert set(populations["population"]) == {"all", "a0"} and (populations.groupby("population")["training_rows"].first()["all"] > populations.groupby("population")["training_rows"].first()["a0"])


def test_cross_source_task_scores_with_its_scoring_source_keys(labelled_source, monkeypatch, tmp_path):
    # A task trained on one source and scored on another (as informality: ENOE -> the EOD) writes the scoring source's
    # keys: `eodgdl impute score <task>` selects its outputs this way (it read the training source's keys before).
    from eodgdl.impute import run, sources
    from eodgdl.impute.spec import parse_task

    def scored_rows(context, config):
        frame = synthetic(n=60, seed=9)[["a", "b", "c", "w"]]
        frame["s"] = [f"row{i}" for i in range(len(frame))]
        return frame

    monkeypatch.setitem(sources._SOURCES, "test.scored", sources.Source("test.scored", scored_rows, {"keys": ["s"], "weight": "w", "group": "s"},
                                                                     lambda context, config: {"data": "v2"}, lambda column: [], __file__))
    spec = parse_task({**SYNTHETIC_TASK, "score_source": "test.scored", "evaluation": {"bootstrap": 5, "calibration_bins": 5}})
    context = sources.Context(cache_dir=tmp_path)
    result = run.retrain(spec, context=context, n_jobs=1, progress=False)
    scored = run.score_task(spec, bundle=result.bundle, context=context)
    outputs = run.output_frame(spec, scored, weight=True)
    assert list(outputs.columns[:1]) == ["s"] and outputs.columns[-1] == "w" and outputs["synthetic_fue_imputado"].all()
    # an earlier retrain's scenarios and tables are not left behind to be read as this one's (it has neither)
    (tmp_path / "out" / "evaluation").mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "out" / "scenarios.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "out" / "evaluation" / "stale_table.parquet")
    run.write_retrain(result, tmp_path / "out")
    assert list(pd.read_parquet(tmp_path / "out" / "scores.parquet").columns[:1]) == ["s"]
    assert not (tmp_path / "out" / "scenarios.parquet").exists() and not (tmp_path / "out" / "evaluation" / "stale_table.parquet").exists()
    assert (tmp_path / "out" / "evaluation" / "selection__with_b.parquet").exists()


def test_bundle_path_prefers_the_explicit_file_then_the_retrain(tmp_path):
    from eodgdl.impute import bundle as bundles
    from eodgdl.impute.spec import parse_task

    spec = parse_task({**SYNTHETIC_TASK, "bundle": "od_synthetic_model.joblib"})
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "synthetic" / "od_synthetic_model.joblib").write_bytes(b"")
    assert bundles.bundle_path(spec, path="elsewhere.joblib", retrained=tmp_path) == Path("elsewhere.joblib")
    assert bundles.bundle_path(spec, retrained=tmp_path) == tmp_path / "synthetic" / "od_synthetic_model.joblib"
    with pytest.raises(ValueError, match="names no bundle file"):
        bundles.bundle_path(parse_task(SYNTHETIC_TASK), retrained=tmp_path / "nothing")


def test_components_are_unobserved_where_the_label_is():
    from eodgdl.impute import run
    from eodgdl.impute.spec import parse_task

    spec = parse_task({**SYNTHETIC_TASK, "evaluation": {"components": {"part": {"positive": "hi", "classes": ["hi", "lo"]}}}})
    frame = pd.DataFrame({"part": ["hi", "lo", "hi", MISSING], "synthetic_desconocido": [False, True, False, False]})
    [component] = run._add_components(spec, frame).values()
    assert frame[component.unknown_column].tolist() == [False, True, False, True]
    assert frame["part"].tolist()[:1] == ["hi"] and frame["part"].isna().tolist() == [False, True, False, True]


def test_level_subsets_of_features_a_model_does_not_use(fitted):
    X = scored_rows()
    plain, _ = marginalize.predict_proba_marginalizing(fitted, X)
    other, _ = marginalize.predict_proba_marginalizing(fitted, X, level_subsets={"municipio": ["x"]})   # another arm's feature
    np.testing.assert_array_equal(plain, other)


def test_boosting_stages_only_without_early_stopping():
    families = {"GradientBoosting": {"native_categoricals": True, "params": {"max_iter": [3, 8]}}}   # early_stopping left to sklearn ("auto")
    candidates = models.build_candidates(FEATURES, NUMERIC, LEVELS, families=families)
    assert [len(scored) for _, _, scored in select.cv_tasks(candidates)] == [1, 1]
    candidates = models.build_candidates(FEATURES, NUMERIC, LEVELS, families={"GradientBoosting": {**families["GradientBoosting"], "early_stopping": False}})
    assert [len(scored) for _, _, scored in select.cv_tasks(candidates)] == [2]


def test_harmonizations_are_builders_keyed_by_the_harmonization_code():
    from eodgdl.impute import features

    for name in ("harmonize.eod", "harmonize.eod_viviendas", "harmonize.enoe", "harmonize.enigh"):
        builder = features.get_builder(name)
        assert builder.replaces and Path(builder.module_file).name == "harmonize.py"
        assert set(builder.versions(None, {})) == {"harmonization", "common"}


def test_a_work_trip_level_for_the_workers_without_one(stages):
    """eod.work_trip leaves the destination and mode of a worker without a work trip missing, unless the task names a
    level for that state (giro: sin_viaje)."""
    from eodgdl.impute.sources import Context
    from eodgdl.impute.sources.eod import load_config, work_trip, workers

    context = Context(tables=stages.revised)
    config = load_config()
    frame = workers(context, config["eod.workers"])
    missing = work_trip(frame, context, config["eod.work_trip"], None)
    named = work_trip(frame, context, {**config["eod.work_trip"], "no_trip": "sin_viaje"}, None)
    without = missing["destino_trabajo"].isna()
    assert without.equals(missing["modo_trabajo"].isna()) and 0.1 < without.mean() < 0.2
    assert (named.loc[without, ["destino_trabajo", "modo_trabajo"]] == "sin_viaje").all().all()
    pd.testing.assert_frame_equal(named[~without].astype("string"), missing[~without].astype("string"))
    assert named.loc[without, "destino_cvegeo"].isna().all()                             # no destination to place


def test_amai_points_levels_and_calibration(monkeypatch):
    """AMAI's points (the tasks' class scores plus the dwelling's own answers), the cut into levels, and the rank
    calibration: within an AGEB with AMAI's distribution the dwellings ordered by points take the levels of its
    cumulative distribution at their weighted mid-rank, ties in a random order per completion; elsewhere the level stays."""
    from eodgdl.impute.derive import amai

    monkeypatch.setattr(amai, "_task_points", lambda values, task: values.astype(float).to_numpy())       # values are points already
    base = pd.DataFrame({"internet": ["Sí", "No", "Sí", "No", "Sí", "Sí"], "ageb": ["A", "A", "A", "A", "B", "C"], "w": 1.0})
    completions = pd.DataFrame({"row": [0, 1, 2, 3, 4, 5], "completion": 0, "weight": 1.0, "educacion": [59.0, 11.0, 27.0, 27.0, 85.0, 0.0]})
    points = amai.amai_points(completions, base, {"tasks": ["educacion"], "observed": {"internet": {"Sí": 32, "No": 0}}})["amai_puntos"]
    assert points.tolist() == [91.0, 11.0, 59.0, 27.0, 117.0, 32.0]
    completions["amai_puntos"] = points.to_numpy()
    levels = amai.nse_level(completions, base, {"from": "amai_puntos", "edges": [-np.inf, 48, 95, 116, np.inf], "labels": ["e", "d", "d_mas", "c_menos"]})["nse"]
    assert levels.tolist() == ["d", "e", "d", "e", "c_menos", "e"]
    completions["nse"] = levels.to_numpy()
    shares = pd.DataFrame({"e": [0.5, 0.0], "d": [0.0, 1.0], "c_mas": [0.5, 0.0]}, index=["A", "B"])        # AGEB C has no AMAI distribution
    monkeypatch.setattr(amai, "nse_ageb_shares", lambda config: shares)
    config = {"from": "amai_puntos", "level": "nse", "ageb": "ageb", "weight": "w", "levels": {"nse_calibrado": ["e", "d", "c_mas"]}, "seed": np.random.SeedSequence(1)}
    calibrated = amai.nse_ageb_calibrated(completions, base, config)["nse_calibrado"].tolist()
    # AGEB A by points: 11 (row 1), 27 (row 3), 59 (row 2), 91 (row 0): the lower half E, the upper half C+
    assert calibrated == ["c_mas", "e", "c_mas", "e", "d", "e"]
    with pytest.raises(ValueError, match="complete datasets"):
        amai.nse_ageb_calibrated(completions, base, {**config, "propagation": "enumerate"})
    # equal points share the level split at random, anew in each completion
    tied = pd.DataFrame({"row": np.tile([0, 1, 2, 3], 200), "completion": np.repeat(np.arange(200), 4), "weight": 1 / 200, "amai_puntos": 50.0, "nse": "d"})
    tied_base = base.iloc[:4]
    split = amai.nse_ageb_calibrated(tied, tied_base, {**config, "seed": np.random.SeedSequence(2)})["nse_calibrado"].to_numpy().reshape(200, 4)
    assert ((split == "e").sum(axis=1) == 2).all()                                       # AMAI's half and half in every completion
    assert 0.35 < (split[:, 0] == "e").mean() < 0.65                                      # whichever dwelling comes first


def test_eod_sources_on_the_survey(stages):
    """The EOD's sources on the cleaned survey: the workers with their interview month, one dwelling per row with its
    head and its members aged 14+ who worked."""
    from eodgdl.impute.sources import Context
    from eodgdl.impute.sources.eod import dwellings, load_config, workers

    context = Context(tables=stages.revised)
    config = load_config()
    frame = workers(context, config["eod.workers"])
    hab = stages.revised.hab
    employed = hab["trabajo_semana_pasada"].isin(config["eod.workers"]["employed_categories"])
    assert len(frame) == int(employed.sum()) == 26_913 and not frame.duplicated(["folio_vivienda", "folio_habitante"]).any()
    assert set(frame["mes_entrevista"]) == {"1", "2", "3", "4"} and frame["centralidad"].notna().all()
    homes = dwellings(context, config["eod.dwellings"])
    assert len(homes) == len(stages.revised.viv) == 17_901 and homes["folio_vivienda"].is_unique
    assert set(homes["jefe_fuente"]) == {"observado", "mayor_edad"} and homes["trabajadores_14_n"].ge(0).all()
    reported = homes["jefe_fuente"].eq("observado")                                   # else the oldest member stands in
    assert int(reported.sum()) == 16_873 and homes["sexo_jefe"].notna().all()
    assert homes["trabajadores_14_n"].sum() == int((hab["edad"].ge(14) & employed).sum()) == 26_797
    # a category the survey does not hold (a label split at its comma, as eod.yaml's flow list did) fails loudly
    with pytest.raises(ValueError, match=r"not levels of trabajo_semana_pasada: \['Tenía trabajo', 'pero no trabajó'\]"):
        dwellings(context, {**config["eod.dwellings"], "employed_categories": ["Tiempo completo", "Tenía trabajo", "pero no trabajó"]})
