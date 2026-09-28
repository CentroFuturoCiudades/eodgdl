"""Tests for the giro extra, with no network: config, schema-derived levels, and the destination crosswalk on a
stand-in DENUE."""
from pathlib import Path

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


FIXTURE = Path(__file__).parent / "data" / "giro_parity.parquet"
BUNDLE = Path(__file__).resolve().parent.parent / "data" / giro.MODEL_FILE


@pytest.mark.skipif(not BUNDLE.exists(), reason="in-repo giro bundle not present")
def test_scoring_reproduces_the_reference():
    # 384 workers of outputs/reference/od_giro_imputed.parquet (the 2026-09-28 bundle scored by the notebook), every
    # marginalization pattern among them: the engine's batched scoring gives the same outputs.
    fixture = pd.read_parquet(FIXTURE)
    expected = fixture.filter(like="expected__").rename(columns=lambda column: column.removeprefix("expected__"))
    bundle = giro.load_model(BUNDLE)
    scored = giro.impute_giro(
        bundle["model_with_education"], bundle["model_without_education"], fixture.drop(columns=expected.columns.map("expected__{}".format)),
        with_education_features=bundle["features_with_education"], without_education_features=bundle["features_without_education"],
        destination_models=bundle["destination_models"],
    )
    columns = [column for column in giro.OUTPUT_COLUMNS if column not in giro.KEYS]
    scored = scored[columns].reset_index(drop=True)
    # Exact on every row but one: the recursion that wrote the reference normalized a row's P(destino_trabajo | x)
    # with numpy's row sum, whose rounding depends on the array layout, and this row (the only one marginalized over
    # four features) was alone in its recursion call. The engine sums in one order for any batch; 2 ulp apart here.
    layout = (expected["giro_marginalized_features"] == "estado_civil+parentesco+destino_trabajo+modo_trabajo").to_numpy()
    assert layout.sum() == 1
    pd.testing.assert_frame_equal(scored[~layout], expected[columns][~layout], check_dtype=False, check_exact=True)
    floats = ["giro_prediction_confidence"] + giro.PROBABILITY_COLUMNS
    np.testing.assert_allclose(scored.loc[layout, floats].to_numpy(float), expected.loc[layout, floats].to_numpy(float), rtol=1e-14, atol=0)
    pd.testing.assert_frame_equal(scored[layout].drop(columns=floats), expected[columns][layout].drop(columns=floats), check_dtype=False, check_exact=True)
