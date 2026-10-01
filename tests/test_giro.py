"""Tests for the giro extra, with no network: config, schema-derived levels, and the destination crosswalk on a
stand-in DENUE."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from eodgdl import giro


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
    # establishment DENUE places in it, whatever its locality, and never a locality's mix. The shares are of staff.
    mxcensus = pytest.importorskip("mxcensus")
    rows = [  # (municipality, locality, AGEB, SCIAN sector, large)
        ("039", "0001", "1293", "46", False),
        ("039", "0001", "1293", "32", True),
        ("039", "0001", "1308", "54", False),
        ("097", "0194", "059A", "33", True),  # in an unpopulated place: no census locality
        ("097", "0520", "059A", "46", False),
    ]
    denue = pd.DataFrame({
        "cve_ent": "14", "cve_mun": [r[0] for r in rows], "cve_loc": [r[1] for r in rows],
        "ageb": [r[2] for r in rows], "codigo_act": [r[3] + "1111" for r in rows],
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
    staff = 75.5 / (75.5 + 3)   # 51 a 100 personas against 0 a 5
    assert out["dest_share_scian_31_33"].iloc[2] == staff and out["dest_share_grandes"].iloc[2] == staff
    assert out["dest_share_scian_46"].iloc[1] == 3 / (3 + 75.5 + 3) and out["dest_share_scian_54"].iloc[1] == 3 / (3 + 75.5 + 3)

    with pytest.raises(AssertionError, match="outside the zone system"):
        giro.add_destination_features(od.assign(destino_cvegeo=["140390001", *od["destino_cvegeo"][1:]]), urban, rural)


FIXTURE = Path(__file__).parent / "data" / "giro_parity.parquet"
BUNDLE = Path(__file__).resolve().parent.parent / "data" / giro.MODEL_FILE


@pytest.mark.skipif(not BUNDLE.exists(), reason="in-repo giro bundle not present")
def test_scoring_reproduces_the_reference():
    # 384 workers of outputs/reference/od_giro_imputed.parquet, every marginalization pattern among them, with the
    # installed bundle's scores (regenerated 2026-09-29 for the bundle that reads the interview month, which the
    # fixture gained, and again for the level sin_viaje its 131 workers without a work trip took; 2026-09-30 for v0.6.0's
    # staff-weighted SCIAN sector shares, which replaced the five giro shares): the engine scores them exactly so.
    from eodgdl.impute.run import score_frame

    fixture = pd.read_parquet(FIXTURE)
    expected = fixture.filter(like="expected__").rename(columns=lambda column: column.removeprefix("expected__"))
    scored = score_frame(giro.TASK, fixture.drop(columns=expected.columns.map("expected__{}".format)), giro.load_model(BUNDLE))
    columns = [column for column in giro.OUTPUT_COLUMNS if column not in giro.KEYS]
    pd.testing.assert_frame_equal(scored[columns].reset_index(drop=True), expected[columns], check_dtype=False, check_exact=True)


@pytest.mark.skipif(not BUNDLE.exists(), reason="in-repo giro bundle not present")
def test_bundle_checks_and_v2_round_trip(tmp_path):
    import sklearn

    from eodgdl.impute import bundle as bundles
    from eodgdl.impute.run import score_frame, task_levels

    spec, levels = giro.TASK, task_levels(giro.TASK)
    v2 = bundles.check_bundle(giro.load_model(BUNDLE), spec, levels)
    assert v2["format"] == bundles.FORMAT and list(v2["auxiliary"]) == ["with_education", "without_education"]
    for broken, message in (
        ({**v2, "metadata": {**v2["metadata"], "sklearn_version": "0.0"}}, "scikit-learn 0.0"),
        ({**v2, "category_levels": {**levels, "destino_ambito": ["localidad_rural"] + levels["destino_ambito"]}}, r"category levels differ .*destino_ambito"),
        ({**v2, "metadata": {**v2["metadata"], "scoring_hash": "x"}}, "scoring_hash"),
    ):
        with pytest.raises(bundles.BundleMismatch, match=message):
            bundles.check_bundle(broken, spec, levels)
    # the giro notebook's layout (model_with_education ... keys) is no longer converted
    with pytest.raises(bundles.BundleMismatch, match="format"):
        bundles.as_v2({"model_with_education": v2["arms"]["with_education"]["model"]}, spec)

    arms = {name: entry["model"] for name, entry in v2["arms"].items()}
    fresh = bundles.make_bundle(spec, arms, v2["auxiliary"], levels, {"selected": v2["metadata"]["selected"]})
    assert fresh["metadata"]["sklearn_version"] == sklearn.__version__
    path = tmp_path / "giro.joblib"
    bundles.save_bundle(fresh, path)
    frame = pd.read_parquet(FIXTURE).pipe(lambda fixture: fixture.drop(columns=[column for column in fixture if column.startswith("expected__")]))
    pd.testing.assert_frame_equal(score_frame(spec, frame, v2), score_frame(spec, frame, giro.load_model(path)))
