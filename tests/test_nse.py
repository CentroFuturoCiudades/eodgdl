"""The NSE chain's pieces: AMAI points, the cut, the AGEB rank calibration (synthetic), the EOD dwelling source (on
the survey) and ENIGH's harmonization (synthetic)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from eodgdl.impute.derive import amai
from eodgdl.impute.harmonize import apply_variables, load_harmonization

LEVELS = ["e", "d", "d_mas", "c_menos", "c", "c_mas", "ab"]


def test_points_cut_and_calibration(monkeypatch):
    base = pd.DataFrame({"tiene_internet": ["Sí", "No", "Sí", "No"], "n_autos_camionetas": ["1", "0", "4 o más", "0"],
                         "trabajadores_14": ["2", "1", "0", "4 o más"], "ageb": ["1403900010001"] * 4, "ponderador": [1.0, 1.0, 1.0, 1.0]})
    completions = pd.DataFrame({"row": [0, 1, 2, 3], "completion": [0, 0, 0, 0], "weight": [1.0] * 4,
                                "educacion_jefe": ["licenciatura", "primaria", "posgrado", "no_estudio"],
                                "amai_banos": ["1", "0", "2_o_mas", "0"], "amai_dormitorios": ["2", "1", "4_o_mas", "1"]})
    config = {"tasks": ["educacion_jefe", "amai_banos", "amai_dormitorios"],
              "observed": {"tiene_internet": {"Sí": 32, "No": 0}, "n_autos_camionetas": {"0": 0, "1": 22, "2": 43, "3": 43, "4 o más": 43},
                           "trabajadores_14": {"0": 0, "1": 15, "2": 31, "3": 46, "4 o más": 61}}}
    completions["amai_puntos"] = amai.amai_points(completions, base, config)["amai_puntos"].to_numpy()
    assert completions["amai_puntos"].tolist() == [59 + 24 + 16 + 32 + 22 + 31, 11 + 8 + 15, 85 + 47 + 32 + 32 + 43, 0 + 8 + 61]
    cut = {"from": "amai_puntos", "edges": [-np.inf, 48, 95, 116, 141, 168, 202, np.inf], "labels": LEVELS}
    completions["nse"] = amai.nse_level(completions, base, cut)["nse"].to_numpy()
    assert completions["nse"].tolist() == ["c_mas", "e", "ab", "d"]
    assert amai._task_points(pd.Series([12.5, 3.0]), "educacion_jefe").tolist() == [12.5, 3.0]

    # AMAI says the AGEB is half e, half ab: the two lowest-scoring dwellings are e, the two highest ab
    shares = pd.DataFrame([[0.5, 0, 0, 0, 0, 0, 0.5]], columns=LEVELS, index=["1403900010001"])
    monkeypatch.setattr(amai, "load_nse_ageb", lambda *args: shares)
    calibration = {"from": "amai_puntos", "level": "nse", "ageb": "ageb", "weight": "ponderador", "file": "x", "state": "Jalisco",
                   "municipalities": [], "file_columns": [], "counts": {level: level for level in LEVELS}, "levels": {"nse_calibrado": LEVELS}}
    calibrated = amai.nse_ageb_calibrated(completions, base, calibration)["nse_calibrado"]
    assert calibrated.tolist() == ["ab", "e", "ab", "e"]
    elsewhere = amai.nse_ageb_calibrated(completions, base.assign(ageb="1403900019999"), calibration)["nse_calibrado"]
    assert elsewhere.tolist() == completions["nse"].tolist()                      # no AMAI distribution: unchanged
    with pytest.raises(ValueError, match="complete datasets"):
        amai.nse_ageb_calibrated(completions, base, {**calibration, "propagation": "enumerate"})


def test_enigh_vehicles_sum_cars_vans_and_pickups():
    frame = pd.DataFrame({"num_auto": ["1", "0", "2"], "num_van": ["1", "0", "0"], "num_pickup": ["0", "0", "3"]})
    variables = {name: definition for name, definition in load_harmonization("enigh")["variables"].items() if name in ("autos_camionetas", "n_autos_camionetas")}
    assert apply_variables(frame, variables)["n_autos_camionetas"].tolist() == ["2", "0", "4 o más"]


def test_dwelling_source_heads_and_workers(stages):
    from eodgdl.impute.sources import Context, get_source

    source = get_source("eod.dwellings")
    frame = source.build(Context(tables=stages.revised), source.config).frame
    hab = stages.revised.hab.reset_index()
    assert len(frame) == len(stages.revised.viv) and frame["folio_vivienda"].is_unique
    reported = hab.loc[hab["parentesco"] == "Jefe del hogar", "folio_vivienda"].unique()
    assert set(frame.loc[frame["jefe_fuente"] == "observado", "folio_vivienda"]) == set(reported)
    assert frame["jefe_fuente"].notna().all()
    oldest = frame[frame["jefe_fuente"] == "mayor_edad"].set_index("folio_vivienda")["edad_jefe"]
    assert (oldest == hab[hab["folio_vivienda"].isin(oldest.index)].groupby("folio_vivienda")["edad"].max().reindex(oldest.index)).all()
    working = hab["edad"].ge(14) & hab["trabajo_semana_pasada"].isin(source.config["employed_categories"])
    assert frame.set_index("folio_vivienda")["trabajadores_14_n"].sum() == int(working.sum())
