"""Smoke tests for the giro extra (no network: config and schema-derived levels only)."""
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
