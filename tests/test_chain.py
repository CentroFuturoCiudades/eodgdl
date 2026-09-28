"""Chains of imputation tasks on synthetic data: enumerate is exact, draws converge to it, observed upstream values
pass through, parallel chains score each task alone, expected propagation plugs in the upstream score, and derive
steps are carried per completion."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from eodgdl.impute import levels, marginalize, models, sources
from eodgdl.impute.bundle import make_bundle
from eodgdl.impute.chain import parse_chain, run_chain
from eodgdl.impute.run import score_frame
from eodgdl.impute.spec import parse_task

FAMILY = {"GradientBoosting": {"native_categoricals": True, "early_stopping": False, "params": {"max_iter": [15]}}}
U1 = {"a": "ua", "b": "ub", "c": "ub"}    # y1's three classes collapse to two levels of the feature u1


def task(name, label, classes, features, numeric=(), declared=None, scores=None):
    target = {"column": label, "classes": classes}
    if scores:
        target["scores"] = scores
    return parse_task({"task": name, "source": "test.chain", "target": target,
                       "features": {"numeric": ["c", *numeric]}, "levels": {"declared": {k: v for k, v in (declared or {}).items() if k in features}},
                       "arms": {"only": {"features": features}}})


LEVELS = {"x1": ["a0", "a1", "a2"], "x2": ["b0", "b1"], "u1": ["ua", "ub"], "u2": ["p", "q"]}
T1 = task("t1", "lab1", {"A": "a", "B": "b", "C": "c"}, ["x1", "x2", "c"], declared=LEVELS, scores={"a": 1, "b": 2, "c": 3})
T2 = task("t2", "lab2", {"P": "p", "Q": "q"}, ["x1", "c", "u1"], declared=LEVELS)
T3 = task("t3", "lab3", {"M": "m", "N": "n"}, ["x2", "c", "u1", "u2"], declared=LEVELS)
T2E = task("t2e", "lab2", {"P": "p", "Q": "q"}, ["x1", "c", "v1"], numeric=["v1"], declared=LEVELS)
SPECS = {spec.name: spec for spec in (T1, T2, T3, T2E)}


def synthetic_frame(n=900, seed=5):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"k": np.arange(n), "x1": rng.choice(LEVELS["x1"], n), "x2": rng.choice(LEVELS["x2"], n), "c": rng.normal(size=n), "w": rng.uniform(0.5, 2, n)})
    logits = np.column_stack([(frame["x1"] == "a0") * 1.5 + frame["c"], (frame["x1"] == "a1") * 1.5, (frame["x2"] == "b1") * 1.0 - frame["c"]])
    y1 = np.array(["a", "b", "c"])[(logits + rng.gumbel(size=logits.shape)).argmax(axis=1)]
    frame["u1"] = pd.Series(y1).map(U1)
    frame["v1"] = pd.Series(y1).map({"a": 1.0, "b": 2.0, "c": 3.0})
    y2 = np.where((frame["u1"] == "ua") * 2.0 + frame["c"] + rng.logistic(size=n) > 1.0, "p", "q")
    frame["u2"] = y2
    y3 = np.where((frame["u1"] == "ub") * 1.5 + (frame["u2"] == "p") * 1.5 - frame["c"] + rng.logistic(size=n) > 1.5, "m", "n")
    hide = lambda rate: rng.random(n) < rate
    frame["lab1"] = pd.Series(y1).str.upper().where(~hide(0.3))
    frame["lab2"] = pd.Series(y2).str.upper().where(~hide(0.3))
    frame["lab3"] = pd.Series(y3).str.upper().where(~hide(0.3))
    return frame


@pytest.fixture(scope="module")
def chain_source(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _chain_source(monkeypatch, tmp_path_factory.mktemp("chain"))


def _chain_source(monkeypatch, tmp_path):
    frame = synthetic_frame()
    build = lambda context, config: sources.SourceFrame(frame.copy(), ["k"], "w", "k", {"data": "chain"}, lambda column: [])
    monkeypatch.setitem(sources._SOURCES, "test.chain", sources.Source("test.chain", build, {"keys": ["k"], "weight": "w", "group": "k"},
                                                                     lambda context, config: {"data": "chain"}, lambda column: [], __file__))
    from eodgdl.impute.features import build_frame

    context = sources.Context(cache_dir=tmp_path)
    bundles = {}
    for spec in SPECS.values():
        rows = build_frame(spec, context).frame
        known = rows[~rows[spec.unknown_column]]
        features = list(spec.arms[0].features)
        X = levels.prepare_features(known, features, spec.numeric)
        weights = levels.normalize_sample_weights(known["w"].reset_index(drop=True))
        model = models.build_candidates(features, spec.numeric, spec.category_levels(lambda column: []), FAMILY)["GradientBoosting"]["model"]
        model.fit(X, known[spec.target].astype(str), classifier__sample_weight=weights)
        marginalize.attach_training_level_shares(model, X, weights, spec.numeric)
        bundles[spec.name] = make_bundle(spec, {"only": model}, {}, spec.category_levels(lambda column: []), {})
    return context, bundles


def chain(steps, **settings):
    return parse_chain({"chain": "test", "steps": steps, **settings}, load=SPECS.__getitem__)


def scored_with(spec, frame, bundle, **fixed):
    return score_frame(spec, frame.assign(**fixed), bundle)[spec.probability_columns].to_numpy()


def test_enumerate_is_the_exact_scenario_sum(chain_source):
    context, bundles = chain_source
    spec = chain(["t1", {"t2": {"uses": {"t1": {"as": "u1", "transform": U1}}}}], mode="sequential", propagation="enumerate")
    result = run_chain(spec, context=context, bundles=bundles, specs=SPECS)
    from eodgdl.impute.features import build_frame

    frame1, frame2 = build_frame(T1, context).frame, build_frame(T2, context).frame
    p1 = score_frame(T1, frame1, bundles["t1"])[T1.probability_columns].to_numpy()
    expected = sum(p1[:, [i]] * scored_with(T2, frame2, bundles["t2"], u1=U1[slug]) for i, slug in enumerate(T1.class_slugs))
    np.testing.assert_allclose(result.frame[T2.probability_columns].to_numpy(), expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.frame[T2.probability_columns].sum(axis=1), 1.0)
    # conditionals per level of u1, and the observed t2 labels stay one-hot
    for level in ("ua", "ub"):
        conditional = result.frame[[f"prob_t2_{slug}_given_u1_{level}" for slug in T2.class_slugs]].to_numpy()
        np.testing.assert_allclose(conditional, scored_with(T2, frame2, bundles["t2"], u1=level))    # every row, every level
    known = ~result.frame["t2_fue_imputado"].to_numpy()
    assert set(np.unique(result.frame.loc[known, T2.probability_columns].to_numpy())) <= {0.0, 1.0}
    assert (result.frame["t2_condicionado_en"] == "t1:enumerate").all()


def test_observed_upstream_passes_through(chain_source):
    context, bundles = chain_source
    spec = chain(["t1", {"t2": {"uses": {"t1": {"as": "u1", "transform": U1}}}}])
    result = run_chain(spec, context=context, bundles=bundles, specs=SPECS)
    from eodgdl.impute.features import build_frame

    frame2 = build_frame(T2, context).frame
    observed = ~build_frame(T1, context).frame["t1_desconocido"].to_numpy()        # t1's label reported: u1 is its level
    direct = score_frame(T2, frame2, bundles["t2"])[T2.probability_columns].to_numpy()  # frame2's u1 is the truth
    np.testing.assert_allclose(result.frame[T2.probability_columns].to_numpy()[observed], direct[observed], rtol=1e-12)


def test_draws_converge_to_enumerate_on_a_three_task_chain(chain_source):
    context, bundles = chain_source
    steps = ["t1", {"t2": {"uses": {"t1": {"as": "u1", "transform": U1}}}},
             {"t3": {"uses": {"t1": {"as": "u1", "transform": U1}, "t2": {"as": "u2"}}}}]
    exact = run_chain(chain(steps, propagation="enumerate"), context=context, bundles=bundles, specs=SPECS).frame
    drawn = run_chain(chain(steps, propagation="draws", draws=400, seed=3), context=context, bundles=bundles, specs=SPECS)
    columns = T2.probability_columns + T3.probability_columns
    assert np.abs(drawn.frame[columns].to_numpy() - exact[columns].to_numpy()).max() < 0.06
    assert np.abs(drawn.frame[columns].to_numpy() - exact[columns].to_numpy()).mean() < 0.01
    np.testing.assert_allclose(drawn.frame[T1.probability_columns], exact[T1.probability_columns])   # the first step is not drawn
    assert len(drawn.completions) == 400 * len(exact) and {"t1", "t2", "weight", "completion"} <= set(drawn.completions.columns)
    again = run_chain(chain(steps, propagation="draws", draws=400, seed=3), context=context, bundles=bundles, specs=SPECS)
    pd.testing.assert_frame_equal(again.frame, drawn.frame)                                          # seeded


def test_parallel_chain_scores_each_task_alone_and_derives_per_completion(chain_source):
    context, bundles = chain_source
    from eodgdl.impute.features import build_frame

    combine = lambda completions, base, config: pd.DataFrame({"pair": completions["t1"].astype(str) + "-" + completions["t2"].astype(str)})
    result = run_chain(chain(["t1", "t2", {"derive": "pair"}], mode="parallel", propagation="enumerate"), context=context, bundles=bundles,
                       derive_functions={"pair": combine}, specs=SPECS)
    p1 = score_frame(T1, build_frame(T1, context).frame, bundles["t1"])[T1.probability_columns].to_numpy()
    p2 = score_frame(T2, build_frame(T2, context).frame, bundles["t2"])[T2.probability_columns].to_numpy()
    np.testing.assert_allclose(result.frame[T2.probability_columns].to_numpy(), p2)
    for i, a in enumerate(T1.class_slugs):
        for j, b in enumerate(T2.class_slugs):
            np.testing.assert_allclose(result.frame[f"prob_pair_{a}-{b}"], p1[:, i] * p2[:, j], atol=1e-12)
    assert (result.frame["t2_condicionado_en"] == "").all()


def test_expected_propagation_plugs_in_the_upstream_score(chain_source):
    context, bundles = chain_source
    from eodgdl.impute.features import build_frame

    result = run_chain(chain(["t1", {"t2e": {"uses": {"t1": {"as": "v1"}}}}], propagation="expected"), context=context, bundles=bundles, specs=SPECS)
    score = result.frame["t1_puntaje_esperado"].to_numpy()
    p1 = score_frame(T1, build_frame(T1, context).frame, bundles["t1"])[T1.probability_columns].to_numpy()
    np.testing.assert_allclose(score, p1 @ np.array([1.0, 2.0, 3.0]))
    direct = score_frame(T2E, build_frame(T2E, context).frame.assign(v1=score), bundles["t2e"])[T2E.probability_columns].to_numpy()
    np.testing.assert_allclose(result.frame[T2E.probability_columns].to_numpy(), direct, rtol=1e-12)
    assert (result.frame["t2e_condicionado_en"] == "t1:expected (plug-in)").all()


def test_chain_validation():
    with pytest.raises(ValueError, match="not an earlier step"):
        chain([{"t2": {"uses": {"t1": {"as": "u1", "transform": U1}}}}, "t1"])
    with pytest.raises(ValueError, match="parallel chain"):
        chain(["t1", {"t2": {"uses": {"t1": {"as": "u1", "transform": U1}}}}], mode="parallel")
    with pytest.raises(ValueError, match="must map every class"):
        chain(["t1", {"t2": {"uses": {"t1": {"as": "u1", "transform": {"a": "ua"}}}}}])
    with pytest.raises(ValueError, match="must be numeric"):
        chain(["t1", {"t2": {"uses": {"t1": {"as": "u1"}}}}], propagation="expected")
    with pytest.raises(ValueError, match="no feature"):
        chain(["t1", {"t2": {"uses": {"t1": {"as": "zz"}}}}])


def test_undeclared_levels_fail_and_runs_are_written(chain_source, tmp_path):
    from eodgdl.impute.chain import write_chain

    context, bundles = chain_source
    with pytest.raises(ValueError, match="does not declare"):
        run_chain(chain(["t1", {"t2": {"uses": {"t1": {"as": "u1", "transform": {**U1, "c": "uc"}}}}}]), context=context, bundles=bundles, specs=SPECS)
    result = run_chain(chain(["t1", {"t2": {"uses": {"t1": {"as": "u1", "transform": U1}}}}], propagation="draws", draws=5), context=context, bundles=bundles, specs=SPECS)
    out = write_chain(result, tmp_path / "run")
    assert pd.read_parquet(out / "completions.parquet")["completion"].max() == 4
    assert set(pd.read_parquet(out / "scores.parquet").columns) >= set(T1.probability_columns + T2.probability_columns)
