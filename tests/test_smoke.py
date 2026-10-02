"""Smoke tests — run against the in-repo data/ directory (no network)."""
from pathlib import Path

import pytest

import eodgdl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "IMEPLAN_Base_Viajes_Master.csv").exists()


def test_public_api():
    assert eodgdl.__all__
    assert callable(eodgdl.load_eod)
    assert callable(eodgdl.load_taz)


def test_rename_map_bundled():
    # Code config ships in the package and loads without any network access.
    m = eodgdl.imeplan_rename_map()
    # Exactly the three rename tables: categorical levels belong to schemas.py.
    assert set(m) == {"habitantes", "viviendas", "viajes"}
    assert all(isinstance(v, str) for table in m.values() for v in table.values())


def test_load_eod_local(stages):
    # the default load is the last of load_stages' three, table for table
    t = eodgdl.load_eod(DATA_DIR)
    assert all(len(df) > 0 for df in t)
    assert all(a.equals(b) for a, b in zip(t, stages.revised))
    assert [len(s.trips) for s in stages] == [154_662, 154_662 - 38, len(t.trips)]


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_taz_local():
    taz = eodgdl.load_taz(DATA_DIR / "AMG_Zonificacion_para_encuesta.parquet", drop_ap=True)
    assert len(taz) == 64 and taz.index.is_unique


def test_data_and_models_sit_in_the_clone():
    # every file the code names is read in place: the input data under data/, each task's model bundle under models/,
    # and no model bundle among the data
    from eodgdl import data
    from eodgdl.impute.spec import TASKS_DIR, load_task

    if not HAS_DATA:
        pytest.skip("in-repo data/ not present")
    assert data.data_dir() == DATA_DIR and data.models_dir() == DATA_DIR.parent / "models"
    for name in [*data.SURVEY_FILES, *data.ZONE_FILES, *data.CENSUS_FILES]:
        assert data.resolve(name) == DATA_DIR / name
    bundles = [load_task(path.stem).bundle_file for path in sorted(TASKS_DIR.glob("*.yaml"))]
    assert bundles and all(data.resolve(name).parent == data.models_dir() for name in bundles if name)
    assert not sorted(path.name for path in DATA_DIR.glob("*.joblib")), "model bundles belong in models/"


def test_resolve_reads_each_kind_from_its_directory(tmp_path, monkeypatch):
    # a model bundle (*.joblib) is read from $EODGDL_MODELS_DIR, any other file from $EODGDL_DATA_DIR; a file that is
    # not there fails, naming where it was looked for
    from eodgdl.data import VIAJES_CSV, resolve

    model = "od_giro_hybrid_model.joblib"
    for folder in ("data", "models"):
        (tmp_path / folder).mkdir()
    (tmp_path / "data" / VIAJES_CSV).write_text("x")
    (tmp_path / "models" / model).write_text("x")
    monkeypatch.setenv("EODGDL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EODGDL_MODELS_DIR", str(tmp_path / "models"))
    assert resolve(VIAJES_CSV) == (tmp_path / "data" / VIAJES_CSV).resolve()
    assert resolve(model) == (tmp_path / "models" / model).resolve()
    with pytest.raises(FileNotFoundError, match="not in"):
        resolve("od_missing_model.joblib")
