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
    assert {"habitantes", "viviendas", "viajes"} <= set(m)


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_eod_local():
    t = eodgdl.load_eod(DATA_DIR)
    assert t.viv.shape[0] > 0
    assert t.hab.shape[0] > 0
    assert t.trips.shape[0] > 0
    assert t.legs.shape[0] > 0


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_taz_local():
    taz = eodgdl.load_taz(DATA_DIR / "AMG_Zonificacion_para_encuesta.parquet", drop_ap=True)
    mtaz = eodgdl.load_mtaz(
        taz, DATA_DIR / "AMG_MicroZONAS2023.parquet", drop_ap=True
    )
    assert len(taz) > 0
    assert len(mtaz) > 0
