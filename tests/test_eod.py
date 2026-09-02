"""load_eod: the chain cleaning the loader applies to the trip table."""
from pathlib import Path

import pandas as pd
import pytest

from eodgdl import clean_trip_chains, load_eod

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "IMEPLAN_Base_Viajes_Master.csv").exists()

AGEB = "1409700251418"          # the household's zone
ELSEWHERE = "1409700251419"
VIV = pd.DataFrame({"ageb": [AGEB]}, index=pd.Index([1], name="folio_vivienda"))
SHOP = "Comercio, mercado, tienda o centro comercial"


def _trips(rows):
    """A trips table in load_eod's shape from (person, hour, minute, motivo[, type, zone]) rows.

    Every trip leaves from home; a 'Regresar a Casa' arrives at home and at
    'Su casa', anything else arrives ELSEWHERE at a shop, unless the row says.
    """
    rows = [tuple(r) + (None,) * (6 - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=["person", "hora_inicio_h", "hora_inicio_m",
                                     "motivo_viaje", "tipo_lugar_destino", "destino"])
    home = df.motivo_viaje == "Regresar a Casa"
    df["tipo_lugar_destino"] = df.tipo_lugar_destino.fillna(home.map({True: "Su casa", False: SHOP}))
    df["destino"] = df.destino.fillna(home.map({True: AGEB, False: ELSEWHERE}))
    df["origen"] = AGEB
    df["folio_vivienda"] = 1
    df["folio_habitante"] = df.person
    df["folio_viaje"] = df.groupby("person").cumcount() + 1
    df["tipo_lugar_origen"] = "Su casa"
    df["hora_inicio_h"] = df.hora_inicio_h.astype("Int64")
    df["hora_inicio_m"] = df.hora_inicio_m.astype("Int64")
    return df.set_index(["folio_vivienda", "folio_habitante", "folio_viaje"]).drop(columns="person")


def test_incomplete_days_are_excluded_whole():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, None, None, None), (1, 19, 0, "Regresar a Casa"),
        (2, 8, 0, "Trabajar"), (2, 18, 0, "Regresar a Casa"),
    ])
    cleaned, dropped, counts = clean_trip_chains(trips, VIV)
    assert list(dropped) == [(1, 1)]
    assert counts["incomplete_persons"] == 1 and counts["incomplete_trips"] == 3
    assert cleaned.index.get_level_values("folio_habitante").unique().tolist() == [2]


def test_a_return_home_made_from_home_is_not_a_trip():
    trips = _trips([
        (1, 12, 0, "Regresar a Casa"),                       # first trip, from home: dropped
        (1, 13, 0, "Trabajar"), (1, 18, 0, "Regresar a Casa"),
        (1, 19, 0, "Regresar a Casa"), (1, 20, 0, "Regresar a Casa"),   # a run: both dropped
        (2, 8, 0, "Trabajar"), (2, 12, 0, "Regresar a Casa"),
        (2, 15, 0, "Compras (comida)"), (2, 16, 0, "Regresar a Casa"),  # intact
    ])
    cleaned, _, counts = clean_trip_chains(trips, VIV)
    assert counts["home_to_home"] == 3
    assert cleaned.loc[(1, 1)].motivo_viaje.tolist() == ["Trabajar", "Regresar a Casa"]
    assert len(cleaned.loc[(1, 2)]) == 4
    # folio_viaje is left as recorded, so the gap shows what was dropped.
    assert cleaned.loc[(1, 1)].index.tolist() == [2, 3]


def test_a_return_that_did_not_reach_home_takes_its_destination_type():
    other_shop = "1409700251420"
    trips = _trips([
        (1, 8, 0, "Trabajar"),
        (1, 12, 0, "Regresar a Casa", SHOP, other_shop),      # zone says not home: shopping
        (1, 13, 0, "Regresar a Casa"),                        # the real return, kept
        (2, 8, 0, "Trabajar"),
        (2, 17, 0, "Regresar a Casa", "Fábrica o taller"),    # zone says home: stays a return
        (2, 18, 0, "Regresar a Casa"),                        # ...so this one is home from home
    ])
    cleaned, _, counts = clean_trip_chains(trips, VIV)
    assert counts["recoded_returns"] == 1 and counts["home_to_home"] == 1
    assert cleaned.loc[(1, 1)].motivo_viaje.tolist() == [
        "Trabajar", "Compras (bienes, productos y servicios)", "Regresar a Casa"]
    assert cleaned.loc[(1, 2)].motivo_viaje.tolist() == ["Trabajar", "Regresar a Casa"]


def test_twelve_hour_clock_is_corrected_only_when_it_restores_order():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 5, 30, "Regresar a Casa"),            # 5:30 -> 17:30
        (2, 8, 0, "Trabajar"), (2, 5, 30, "Regresar a Casa"),
        (2, 6, 0, "Compras (comida)"), (2, 9, 0, "Regresar a Casa"),     # a run of PM entries
        (3, 19, 0, "Trabajar"), (3, 1, 0, "Regresar a Casa"),            # 13:00 < 19:00: left alone
        (4, 8, 0, "Trabajar"), (4, 7, 0, "Regresar a Casa"),
        (4, 13, 0, "Compras (comida)"), (4, 14, 0, "Regresar a Casa"),   # 19:00 > 13:00: left alone
    ])
    cleaned, _, counts = clean_trip_chains(trips, VIV)
    hours = cleaned.hora_inicio_h.groupby(level="folio_habitante").apply(list)
    assert hours[1] == [8, 17]
    # Strict by design: 5:30 -> 17:30 would overtake the 6:00 that follows, so
    # the run is left alone and tasha.chain_report counts it instead.
    assert hours[2] == [8, 5, 6, 9]
    assert hours[3] == [19, 1]
    assert hours[4] == [8, 7, 13, 14]
    assert counts["moved_12h"] == 1


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_eod_cleans_the_chains():
    viv, hab, trips, legs = load_eod(DATA_DIR)
    assert len(hab) == 58_061 - 281
    assert len(trips) == 153_052
    assert not trips.hora_inicio_h.isna().any()
    # hab and trips stay in step: the count is recounted, the legs follow the trips.
    counted = trips.groupby(level=["folio_vivienda", "folio_habitante"]).size()
    assert (hab.viajes_contados == counted.reindex(hab.index).fillna(0)).all()
    assert legs.index.droplevel("folio_traslado").isin(trips.index).all()
    # motivo_viaje keeps its categorical, with the recodes inside its levels.
    assert str(trips.motivo_viaje.dtype) == "category"


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_eod_can_return_the_survey_as_shipped():
    viv, hab, trips, legs = load_eod(DATA_DIR, clean_chains=False)
    assert (len(hab), len(trips), len(legs)) == (58_061, 154_662, 170_509)
    assert trips.ponderador.sum() == 11_796_386        # the published trip total
    assert trips.hora_inicio_h.isna().sum() == 325
