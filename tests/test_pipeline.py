"""Pipelines: the chains of several row levels drawn jointly (eodgdl.impute.pipeline), the workers aggregate between
the persons and the dwellings (eodgdl.impute.aggregate), and the pipeline tasha the TASHA build reads."""
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from eodgdl.impute import pipeline as pipeline_module
from eodgdl.impute.aggregate import get_aggregate, register_aggregate
from eodgdl.impute.chain import parse_chain
from eodgdl.impute.pipeline import load_pipeline, parse_pipeline, run_pipeline, write_pipeline

from test_chain import SPECS, U1, chain_source  # noqa: F401  (the fixture)

WORKERS = {"aggregate": "workers", "from": "persons", "key": "folio_vivienda", "base": "trabajadores_14_n", "task": "empleo",
           "value": "trabaja", "min_age": {"column": "edad", "value": 14}, "harmonize": {"table": "eod_viviendas", "variable": "trabajadores_14"}}


def test_the_workers_aggregate_counts_the_drawn_workers():
    # dwelling 1: an unanswered adult drawn working in draw 0 only; an unanswered 13-year-old drawn working and a
    # reported worker do not count (the reported one is in the source's count). Dwelling 2: three reported plus two
    # drawn, capped at "4 o más". Dwelling 3: nobody drawn.
    persons = pd.DataFrame({"folio_vivienda": [1, 1, 1, 2, 2], "folio_habitante": [1, 2, 3, 1, 2], "edad": [30, 13, 40, 50, 20],
                            "situacion_laboral_desconocido": [True, True, False, True, True]})
    drawn = {0: ["trabaja", "trabaja", "trabaja", "trabaja", "trabaja"], 1: ["hogar", "trabaja", "trabaja", "trabaja", "trabaja"]}
    completions = pd.concat([persons[["folio_vivienda", "folio_habitante"]].assign(completion=d, situacion_laboral=values) for d, values in drawn.items()])
    upstream = types.SimpleNamespace(base=persons, completions=completions.reset_index(drop=True), keys=["folio_vivienda", "folio_habitante"])
    dwellings = pd.DataFrame({"folio_vivienda": [1, 2, 3], "trabajadores_14_n": pd.array([0, 3, 2], dtype="Int64")})
    workers = get_aggregate("workers")
    out = workers(upstream, dwellings, WORKERS).set_index(["folio_vivienda", "completion"])
    assert out["trabajadores_14_n"].to_dict() == {("1", 0): 1, ("1", 1): 0, ("2", 0): 5, ("2", 1): 5, ("3", 0): 2, ("3", 1): 2}
    assert out["trabajadores_14"].to_dict() == {("1", 0): "1", ("1", 1): "0", ("2", 0): "4 o más", ("2", 1): "4 o más", ("3", 0): "2", ("3", 1): "2"}
    reported = workers(upstream, dwellings, WORKERS, counted=False)            # the parity switch: the source's count
    assert reported.groupby("folio_vivienda")["trabajadores_14_n"].unique().map(list).to_dict() == {"1": [0], "2": [3], "3": [2]}
    with pytest.raises(ValueError, match="not rows of the downstream source"):
        workers(upstream, dwellings[dwellings["folio_vivienda"] != 2], WORKERS)


def test_the_tasha_pipeline():
    from eodgdl.tasha._schema import completed_levels

    tasha = load_pipeline("tasha")
    assert [(level.name, level.chain) for level in tasha.levels] == [("persons", "labour"), ("dwellings", "nse")]
    assert tasha.column_level("nse_calibrado").name == "dwellings" and tasha.column_level("informalidad").name == "persons"
    assert tasha.chain("dwellings").uncertainty == "bootstrap" and tasha.chain("persons", draws=7).draws == 7
    assert completed_levels("tasha", "trabajadores_14") == {"0", "1", "2", "3", "4 o más"}
    assert completed_levels("tasha", "situacion_laboral") == {"trabaja", "estudiante", "hogar", "jubilado", "sin_trabajo"}
    raw = tasha.raw
    with pytest.raises(ValueError, match="which no task of the chain"):
        parse_pipeline({**raw, "levels": {**raw["levels"], "dwellings": {"chain": "nse", "given": {"edad": WORKERS}}}})
    with pytest.raises(ValueError, match="not an earlier level"):
        parse_pipeline({**raw, "levels": {"dwellings": raw["levels"]["dwellings"], "persons": raw["levels"]["persons"]}})
    with pytest.raises(ValueError, match="appear in two levels"):
        parse_pipeline({**raw, "levels": {"persons": {"chain": "labour"}, "workers": {"chain": "sector_informality"}}})


@register_aggregate("test_u1")
def _u1_of_the_upstream_draw(upstream, base, config, counted=True):
    """u1 of each row and completion: the upstream's drawn t1 through U1 (``counted=False``: always "ua")."""
    out = upstream.completions[["k", "completion"]].copy()
    out["u1"] = upstream.completions["t1"].map(U1) if counted else "ua"
    return out


@pytest.fixture
def synthetic_pipeline(chain_source, monkeypatch):
    chains = {"first": {"chain": "first", "propagation": "draws", "seed": 3, "steps": ["t1"]},
              "second": {"chain": "second", "propagation": "draws", "seed": 4, "steps": ["t2"]}}
    load = SPECS.__getitem__
    monkeypatch.setattr(pipeline_module, "load_chain", lambda name: parse_chain(chains[name], load=load))
    monkeypatch.setattr(pipeline_module, "parse_chain", lambda raw: parse_chain(raw, load=load))
    monkeypatch.setattr(pipeline_module, "load_task", load)
    spec = parse_pipeline({"pipeline": "test", "draws": 5, "levels": {"a": {"chain": "first"},
                                                                     "b": {"chain": "second", "given": {"u1": {"aggregate": "test_u1", "from": "a"}}}}})
    return spec, chain_source


def test_draw_d_of_a_level_reads_draw_d_of_the_level_it_aggregates(synthetic_pipeline, tmp_path):
    spec, (context, bundles) = synthetic_pipeline
    result = run_pipeline(spec, context=context, bundles=bundles, specs=SPECS)
    first, second = result.levels["a"].completions, result.levels["b"].completions
    assert (second[["k", "completion"]].to_numpy() == first[["k", "completion"]].to_numpy()).all()
    assert (second["u1"] == first["t1"].map(U1)).all()                          # completion d sees the upstream's draw d
    n = len(result.levels["a"].frame)
    out = write_pipeline(result, tmp_path / "test")
    for level in ("a", "b"):
        assert len(pd.read_parquet(out / level / "completions.parquet")) == n * 5
    assert set(pd.read_json(out / "provenance.json", typ="series")["levels"]) == {"a", "b"}
    # off: the second level reads the switched-off aggregate, the first does not move
    off = run_pipeline(spec, context=context, bundles=bundles, specs=SPECS, aggregate=False)
    assert off.levels["a"].completions.equals(first) and (off.levels["b"].completions["u1"] == "ua").all()
    assert result.provenance["aggregate"] and not off.provenance["aggregate"]


def test_bootstrap_draws_take_the_bundle_of_their_index(synthetic_pipeline):
    # with uncertainty bootstrap every level's draw d is scored with bootstrap bundle d mod B of its tasks, the second
    # level's given the aggregate of the first level's draw d: its marginal is the mean over draws of model d mod B's
    # conditional at that draw's u1
    from eodgdl.impute.features import build_frame

    from test_chain import T2, fit_bundle, scored_with

    spec, (context, bundles) = synthetic_pipeline
    frames = {name: build_frame(SPECS[name], context).frame for name in ("t1", "t2")}
    boot = {name: [fit_bundle(SPECS[name], frame[~frame[SPECS[name].unknown_column]].iloc[half::2]) for half in (0, 1)] for name, frame in frames.items()}
    result = run_pipeline(spec, context=context, bundles=bundles, specs=SPECS, uncertainty="bootstrap", bootstrap=boot)
    completions = result.levels["b"].completions
    rows, models = np.repeat(np.arange(len(frames["t2"])), 5), completions["completion"].to_numpy() % 2
    given = {(m, level): scored_with(T2, frames["t2"], boot["t2"][m], u1=level) for m in (0, 1) for level in ("ua", "ub")}
    per_draw = np.stack([given[(m, value)][row] for m, value, row in zip(models, completions["u1"], rows)])
    expected = per_draw.reshape(len(frames["t2"]), 5, -1).mean(axis=1)
    frame = result.levels["b"].frame
    unknown = frame["t2_fue_imputado"].to_numpy()
    np.testing.assert_allclose(frame[T2.probability_columns].to_numpy()[unknown], expected[unknown], rtol=1e-12)
    assert result.provenance["levels"]["a"]["bootstrap"] == {"t1": 2} and result.provenance["levels"]["b"]["bootstrap"] == {"t2": 2}
