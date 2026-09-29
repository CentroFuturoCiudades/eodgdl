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
    # 384 workers of outputs/reference/od_giro_imputed.parquet, every marginalization pattern among them, with the
    # installed v2 bundle's scores (the engine's score_frame; regenerated 2026-09-29 for the bundle that reads the
    # interview month, which the fixture gained): giro's legacy entry points give the same outputs.
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
    pd.testing.assert_frame_equal(scored, expected[columns], check_dtype=False, check_exact=True)


@pytest.mark.skipif(not BUNDLE.exists(), reason="in-repo giro bundle not present")
def test_bundle_checks_and_v2_round_trip(tmp_path):
    import sklearn

    from eodgdl.impute import bundle as bundles
    from eodgdl.impute.run import score_frame, task_levels

    spec, levels = giro.TASK, task_levels(giro.TASK)
    legacy = giro.load_model(BUNDLE)                                         # the old keys (legacy_view of the v2 file)
    v2 = bundles.check_bundle(bundles.as_v2(legacy, spec), spec, levels)    # the legacy layout fits the task spec
    assert v2["metadata"]["converted_from"] == "legacy giro bundle"
    assert list(v2["auxiliary"]) == ["with_education", "without_education"]

    for broken, message in (
        ({**v2, "metadata": {**v2["metadata"], "sklearn_version": "0.0"}}, "scikit-learn 0.0"),
        ({**v2, "category_levels": {**levels, "destino_ambito": ["localidad_rural"] + levels["destino_ambito"]}}, r"category levels differ .*destino_ambito"),
        ({**v2, "metadata": {**v2["metadata"], "scoring_hash": "x"}}, "scoring_hash"),
    ):
        with pytest.raises(bundles.BundleMismatch, match=message):
            bundles.check_bundle(broken, spec, levels)

    arms = {name: entry["model"] for name, entry in v2["arms"].items()}
    fresh = bundles.make_bundle(spec, arms, v2["auxiliary"], levels, {"selected": legacy["metadata"]["selected"]})
    assert fresh["metadata"]["sklearn_version"] == sklearn.__version__
    path = tmp_path / "giro.joblib"
    bundles.save_bundle(fresh, path)
    view = giro.load_model(path)                                             # old keys, read from a v2 file
    assert set(view) >= {"model_with_education", "features_without_education", "destination_models", "category_levels", "metadata"}
    assert view["features_with_education"] == legacy["features_with_education"]

    fixture = pd.read_parquet(FIXTURE)
    frame = fixture.drop(columns=[column for column in fixture if column.startswith("expected__")])
    from_legacy = score_frame(spec, frame, legacy)
    from_v2 = score_frame(spec, frame, bundles.load_bundle(spec, path))
    pd.testing.assert_frame_equal(from_legacy, from_v2)
    expected = giro.impute_giro(legacy["model_with_education"], legacy["model_without_education"], frame,
                                with_education_features=legacy["features_with_education"], without_education_features=legacy["features_without_education"],
                                destination_models=legacy["destination_models"])
    pd.testing.assert_frame_equal(from_legacy[giro.OUTPUT_COLUMNS], expected[giro.OUTPUT_COLUMNS])
