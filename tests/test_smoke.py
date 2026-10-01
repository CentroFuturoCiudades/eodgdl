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


def test_catalog_files_sit_where_the_mirror_serves_them():
    # the mirror serves the input data from data/ and the model bundles from models/: every catalog file is in its
    # directory with the registry's sha256, and no model bundle sits among the data
    import hashlib

    from eodgdl.data import FILES, MODEL_FILES, POOCH

    if not HAS_DATA:
        pytest.skip("in-repo data/ not present")
    root = DATA_DIR.parent
    for name in FILES:
        path = root / ("models" if name in MODEL_FILES else "data") / name
        assert path.exists(), path
        assert POOCH.registry[name] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), name
    assert not sorted(path.name for path in DATA_DIR.glob("*.joblib")), "model bundles belong in models/"


def test_resolve_reads_models_from_their_own_directory(tmp_path, monkeypatch):
    # a model bundle is read from $EODGDL_MODELS_DIR, else from the models/ directory beside $EODGDL_DATA_DIR (a
    # clone's layout); a data file from $EODGDL_DATA_DIR; the mirror serves each from its own directory
    import os

    from eodgdl.data import MODEL_FILES, POOCH, resolve
    from eodgdl.data._catalog import VIAJES_CSV
    from eodgdl.data._registry import REF

    model = MODEL_FILES[0]
    for folder in ("data", "models", "elsewhere"):
        (tmp_path / folder).mkdir()
    (tmp_path / "data" / VIAJES_CSV).write_text("x")
    (tmp_path / "models" / model).write_text("x")
    (tmp_path / "elsewhere" / model).write_text("x")
    monkeypatch.delenv("EODGDL_MODELS_DIR", raising=False)
    monkeypatch.setenv("EODGDL_DATA_DIR", str(tmp_path / "data"))
    assert resolve(model) == (tmp_path / "models" / model).resolve()
    assert resolve(VIAJES_CSV) == tmp_path / "data" / VIAJES_CSV
    monkeypatch.setenv("EODGDL_MODELS_DIR", str(tmp_path / "elsewhere"))
    assert resolve(model) == (tmp_path / "elsewhere" / model).resolve()
    if not {"EODGDL_BASE_URL", "EODGDL_MODELS_URL"} & set(os.environ):
        assert all(POOCH.get_url(name).endswith(f"/{REF}/models/{name}") for name in MODEL_FILES)
        assert POOCH.get_url(VIAJES_CSV).endswith(f"/{REF}/data/{VIAJES_CSV}")
