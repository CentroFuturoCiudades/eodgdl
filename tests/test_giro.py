"""Tests for the giro extra, with no network: config, schema-derived levels, and the destination crosswalk on a
stand-in DENUE."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from eodgdl import giro
from eodgdl.giro._config import DENUE_SCIAN2


def test_config_and_levels():
    assert giro.GIRO_CLASSES == ["comercio", "servicio", "educacion", "industria", "gobierno"]
    assert set(giro.ROBUST_SECTOR_FEATURES) == set(giro.SECTOR_FEATURES) - {"escolaridad"}
    levels = giro.build_category_levels()
    for column in giro.SECTOR_FEATURES:
        if column not in giro.NUMERIC_FEATURES:
            assert levels[column][-1] == giro.NO_ESPECIFICADO, column
    assert giro.MODEL_FILE in __import__("eodgdl.data", fromlist=["FILES"]).FILES


def test_destination_is_its_zone_system_unit(monkeypatch):
    # An urban AGEB takes its own establishments, or its locality's when it has none; a rural AGEB takes every
    # establishment DENUE places in it, whatever its locality, and never a locality's mix.
    mxcensus = pytest.importorskip("mxcensus")
    sector = {giro_: code for code, giro_ in reversed(DENUE_SCIAN2.items())}
    rows = [  # (municipality, locality, AGEB, giro, large)
        ("039", "0001", "1293", "comercio", False),
        ("039", "0001", "1293", "industria", True),
        ("039", "0001", "1308", "servicio", False),
        ("097", "0194", "059A", "industria", True),  # in an unpopulated place: no census locality
        ("097", "0520", "059A", "comercio", False),
    ]
    denue = pd.DataFrame({
        "cve_ent": "14", "cve_mun": [r[0] for r in rows], "cve_loc": [r[1] for r in rows],
        "ageb": [r[2] for r in rows], "codigo_act": [sector[r[3]] + "1111" for r in rows],
        "per_ocu": ["51 a 100 personas" if r[4] else "0 a 5 personas" for r in rows],
    })
    monkeypatch.setattr(mxcensus, "load_denue", lambda state, release: denue)
    urban, rural = {"1403900011293", "1403900011308", "1403900011312"}, {"14097059A", "140701170"}
    od = pd.DataFrame({
        "destino_cvegeo": ["1403900011293", "1403900011312", "14097059A", "140701170", "999990001", None],
        "destino_zona": ["12F", "12F", "49", "31", "Acceso Norte", None],
    })
    out = giro.add_destination_features(od, urban, rural)
    assert out["destino_ambito"].tolist() == ["ageb_urbana", "ageb_urbana", "ageb_rural", "ageb_rural", "fuera_zm", "desconocido"]
    n = np.expm1(out["dest_establecimientos_log"]).round().tolist()
    assert n[:3] == [2, 3, 2] and np.isnan(n[3:]).all()   # 1312 has none: locality 1403900001's three
    assert out["dest_share_industria"].iloc[2] == 0.5 and out["dest_share_grandes"].iloc[2] == 0.5

    with pytest.raises(AssertionError, match="outside the zone system"):
        giro.add_destination_features(od.assign(destino_cvegeo=["140390001", *od["destino_cvegeo"][1:]]), urban, rural)
