"""The weight as a versioned input: a source's ``weight`` may name a column of a weight file under the data directory
(eodgdl.impute.sources.with_weight), whose sha256 joins the source's data versions, so the feature cache, a bundle's
training data version and tasha.load_completed's staleness check all move when the weight's values do."""
import json

import pandas as pd
import pytest

from eodgdl import tasha
from eodgdl.impute import sources
from eodgdl.impute.sources import Context, changed_versions, get_source, with_weight

WEIGHT = {"file": "pesos.csv", "column": "peso"}


def _weights(folder, values):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"folio_vivienda": [1, 2, 3], "peso": values}).to_csv(folder / "pesos.csv", index=False)
    return folder


@pytest.fixture
def weighted_source(monkeypatch):
    """A source ``test.weighted`` whose weight is the column ``peso`` of the file ``pesos.csv``."""
    config = {"keys": ["folio_vivienda", "folio_habitante"], "weight": WEIGHT, "group": "folio_vivienda"}
    rows = pd.DataFrame({"folio_vivienda": [2, 1, 2, 3], "folio_habitante": [1, 1, 2, 1]})
    source = sources.Source("test.weighted", lambda context, config: rows.copy(), config, lambda context, config: {"data": "v1", "eodgdl": "9.9"},
                            lambda column: [], __file__)
    monkeypatch.setitem(sources._SOURCES, "test.weighted", source)
    return source, rows


def test_a_weight_file_joins_on_the_key_and_versions_its_bytes(tmp_path, monkeypatch, weighted_source):
    source, rows = weighted_source
    monkeypatch.setenv("EODGDL_DATA_DIR", str(_weights(tmp_path / "design", [10.0, 20.0, 30.0])))
    assert source.weight == "peso"
    joined = with_weight(rows, source.config)
    assert joined[["folio_vivienda", "folio_habitante"]].equals(rows) and joined["peso"].tolist() == [20.0, 10.0, 20.0, 30.0]   # rows kept in order
    design = source.data_versions(Context())
    assert design["data"] == "v1" and len(design["weight"]) == 64

    # other values under the same name are another version, and a run recorded under the first sees it changed
    monkeypatch.setenv("EODGDL_DATA_DIR", str(_weights(tmp_path / "tmg", [11.0, 20.0, 30.0])))
    assert source.data_versions(Context())["weight"] != design["weight"]
    assert changed_versions("test.weighted", design) == ["weight"]
    assert changed_versions("test.weighted", source.data_versions(Context())) == []


@pytest.mark.parametrize(("values", "message"), [([10.0, 0.0, 30.0], "1 weights that are not positive"),
                                                 ([10.0, None, 30.0], "1 weights that are not positive")])
def test_a_weight_file_holds_positive_numbers(tmp_path, monkeypatch, weighted_source, values, message):
    source, rows = weighted_source
    monkeypatch.setenv("EODGDL_DATA_DIR", str(_weights(tmp_path, values)))
    with pytest.raises(ValueError, match=message):
        with_weight(rows, source.config)


def test_a_weight_file_covers_every_row_once(tmp_path, monkeypatch, weighted_source):
    source, rows = weighted_source
    monkeypatch.setenv("EODGDL_DATA_DIR", str(tmp_path))
    pd.DataFrame({"folio_vivienda": [1, 2], "peso": [1.0, 2.0]}).to_csv(tmp_path / "pesos.csv", index=False)
    with pytest.raises(ValueError, match=r"no weight for 1 rows \(e.g. folio_vivienda 3\)"):
        with_weight(rows, source.config)
    pd.DataFrame({"folio_vivienda": [1, 2, 2, 3], "peso": 1.0}).to_csv(tmp_path / "pesos.csv", index=False)
    with pytest.raises(ValueError, match="1 repeated folio_vivienda"):
        with_weight(rows, source.config)
    with pytest.raises(ValueError, match="already a column"):
        with_weight(rows.assign(peso=1.0), source.config)
    with pytest.raises(ValueError, match="a weight file is"):
        with_weight(rows, {**source.config, "weight": {"file": "pesos.csv"}})


def test_the_design_weight_keeps_the_source_s_versions():
    # `weight: ponderador` adds nothing: the feature cache and every bundle's training data version stay as they were
    for name in ("eod.workers", "eod.persons", "eod.labour", "eod.dwellings", "enoe.workers", "enigh.households"):
        source = get_source(name)
        assert isinstance(source.config["weight"], str) and sources.weight_versions(source.config) == {}


def test_load_completed_refuses_a_run_drawn_under_another_weight(tmp_path, monkeypatch, weighted_source):
    source, _ = weighted_source
    monkeypatch.setenv("EODGDL_DATA_DIR", str(_weights(tmp_path / "design", [10.0, 20.0, 30.0])))
    level = {"source": "test.weighted", "versions": {**source.data_versions(Context()), "features": {}}}
    folder = tmp_path / "root" / "tasha"
    persons = pd.DataFrame({"folio_vivienda": [1], "folio_habitante": [1], "completion": [0], "weight": 1.0, "situacion_laboral": ["trabaja"]})
    dwellings = pd.DataFrame({"folio_vivienda": [1], "completion": [0], "weight": 1.0, "nse_calibrado": ["c"]})
    for name, frame in (("persons", persons), ("dwellings", dwellings)):
        (folder / name).mkdir(parents=True)
        frame.to_parquet(folder / name / "completions.parquet", index=False)
    (folder / "provenance.json").write_text(json.dumps({"pipeline": "tasha", "draws": 1, "aggregate": True, "levels": {"persons": level, "dwellings": level}}))
    assert set(tasha.load_completed(tmp_path / "root")["tasha"]) == {"persons", "dwellings"}
    monkeypatch.setenv("EODGDL_DATA_DIR", str(_weights(tmp_path / "tmg", [11.0, 20.0, 30.0])))
    with pytest.raises(ValueError, match=r"persons: weight changed"):
        tasha.load_completed(tmp_path / "root")


def test_a_survey_source_reads_its_weight_from_a_file(stages, tmp_path, monkeypatch):
    # eod.persons with a household weight file: every person takes their dwelling's weight under the file's column
    from eodgdl.impute.features import build_frame
    from eodgdl.impute.spec import load_task

    from conftest import DATA_DIR

    viv = stages.revised.viv
    for path in DATA_DIR.iterdir():                     # the survey files, read locally (their versions are hashed)
        (tmp_path / path.name).symlink_to(path)
    pd.DataFrame({"folio_vivienda": viv.index, "peso": viv["ponderador"].astype(float) * 2}).to_csv(tmp_path / "pesos.csv", index=False)
    monkeypatch.setenv("EODGDL_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(get_source("eod.persons").config, "weight", WEIGHT)
    built = build_frame(load_task("empleo"), Context(tables=stages.revised))
    assert built.weight == "peso" and built.versions["weight"] == sources.file_digest(str(tmp_path / "pesos.csv"))
    expected = viv["ponderador"].astype(float).reindex(built.frame["folio_vivienda"]).to_numpy() * 2
    assert (built.frame["peso"].to_numpy() == expected).all()

