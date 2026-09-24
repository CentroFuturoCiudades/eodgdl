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


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_microzone_totals_drop_the_double_counted_localities():
    # IMEPLAN's micro-zone totals count two localities twice; _MTAZ_POBTOT_FIXES takes
    # them out once (6,637 people), leaving the population of the 601 micro-zones.
    taz = eodgdl.load_taz(DATA_DIR / "AMG_Zonificacion_para_encuesta.parquet")
    mtaz = eodgdl.load_mtaz(taz, DATA_DIR / "AMG_MicroZONAS2023.parquet")
    assert len(mtaz) == 601 and mtaz.POBTOT.sum() == 5_220_592
