"""Tests for the imputation engine on synthetic data: batched marginalization against the recursive reference, level
subsets, staged vs unstaged tuning, serial vs parallel tuning, the one-SE rule, harmonization and arm dispatch."""
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
        return sources.SourceFrame(frame, ["k"], "c", "k", {"data": "v1"}, lambda column: LEVELS[column][:-1])

    def build_columns(frame, context, config, spec):
        calls["builder"] += 1
        return pd.DataFrame({"d": frame["c"] * config["scale"], "e": pd.Categorical(frame["a"])}, index=frame.index)

    monkeypatch.setitem(sources._SOURCES, "test.rows", sources.Source("test.rows", build_source, {"keys": ["k"], "weight": "c", "group": "k"},
                                                                    lambda context, config: {"data": "v1"}, lambda column: LEVELS[column][:-1], __file__))
    monkeypatch.setitem(features._BUILDERS, "test.double", features.Builder("test.double", build_columns, {"scale": 2.0}, lambda context, config: {}, __file__))
    spec = spec_module.parse_task({
        "task": "t", "source": "test.rows", "target": {"column": "label", "classes": {"X": "x", "Y": "y", "Z": "z"}},
        "features": {"builders": ["test.double"], "numeric": ["c", "d"]}, "arms": {"only": {"features": ["a", "b", "c", "d", "e"]}},
    })
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


def test_spec_validation():
    from eodgdl.impute.spec import parse_task

    base = {"task": "t", "source": "s", "target": {"column": "y", "classes": {"A": "a", "B": "b"}},
            "features": {"groups": {"g": ["u", "v"]}, "numeric": ["v"]}, "arms": {"one": {"features": ["g"], "requires": ["u"]}}}
    spec = parse_task(base)
    assert spec.features == ["u", "v"] and spec.arm("one").requires == ("u",)
    with pytest.raises(ValueError, match="requires features it does not use"):
        parse_task({**base, "arms": {"one": {"features": ["g"], "exclude": ["u"], "requires": ["u"]}}})
    with pytest.raises(ValueError, match="numeric features no arm uses"):
        parse_task({**base, "features": {"groups": {"g": ["u"]}, "numeric": ["v"]}})
    with pytest.raises(ValueError, match="class slugs repeat"):
        parse_task({**base, "target": {"column": "y", "classes": {"A": "a", "B": "a"}}})
    assert parse_task(base).scoring_hash() == spec.scoring_hash()
    assert parse_task({**base, "missing_values": {"u": ["?"]}}).scoring_hash() != spec.scoring_hash()


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
        return sources.SourceFrame(frame, ["g", "k"], "w", "g", {"data": "v1"}, None)

    def build_with_key(context, config):
        result = build(context, config)
        result.frame["k"] = np.arange(len(result.frame))
        return result

    monkeypatch.setitem(sources._SOURCES, "test.labelled", sources.Source("test.labelled", build_with_key, {"keys": ["g", "k"], "weight": "w", "group": "g"},
                                                                        lambda context, config: {"data": "v1"}, lambda column: [], __file__))


def test_retrain_writes_a_bundle_that_scores_the_same(labelled_source, tmp_path):
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

    path, digest = run.write_retrain(result, tmp_path / "out")
    assert (tmp_path / "out" / "scores.parquet").exists() and (tmp_path / "out" / "evaluation" / "reliability__with_b.parquet").exists()
    reloaded = run.score_frame(spec, result.scored.drop(columns=run.output_columns(spec, [])), bundles.load_bundle(spec, path))
    pd.testing.assert_frame_equal(reloaded, scored)

    shipped = run.retrain(parse_task({**SYNTHETIC_TASK, "evaluation": {**SYNTHETIC_TASK["evaluation"], "isotonic": {"ship": True}}}), context=context, n_jobs=1, progress=False)
    assert type(shipped.bundle["arms"]["with_b"]["model"]).__name__ == "IsotonicCalibrated"
    np.testing.assert_allclose(shipped.scored[spec.probability_columns].sum(axis=1), 1.0)


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
