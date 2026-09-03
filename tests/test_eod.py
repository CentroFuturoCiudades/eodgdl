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
    """A trips table in load_eod's shape from (person, hour, minute, motivo[, type, zone, minutes]) rows.

    Every trip leaves from home; a 'Regresar a Casa' arrives at home and at
    'Su casa', anything else arrives ELSEWHERE at a shop, unless the row says.
    Travel minutes are 0 unless given.
    """
    rows = [tuple(r) + (None,) * (7 - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=["person", "hora_inicio_h", "hora_inicio_m",
                                     "motivo_viaje", "tipo_lugar_destino", "destino", "traslado1_min"])
    df["traslado1_min"] = df.traslado1_min.fillna(0).astype("Int64")
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
    # folio_viaje is left as shipped, so the gap shows what was dropped.
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


def test_a_trip_after_a_return_home_starts_at_home():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 17, 0, "Regresar a Casa"),
        (1, 18, 0, "Compras (comida)"), (1, 19, 0, "Regresar a Casa"),
        (2, 8, 0, "Trabajar", "Fábrica o taller", ELSEWHERE),   # a return that did not reach home...
        (2, 17, 0, "Regresar a Casa", "Su casa", ELSEWHERE),
        (2, 18, 0, "Compras (comida)"), (2, 19, 0, "Regresar a Casa"),
    ])
    trips.loc[(1, 1, 3), "origen"] = ELSEWHERE     # carried over from the morning stop
    cleaned, _, counts = clean_trip_chains(trips, VIV)
    assert counts["origins_repaired"] == 1
    assert cleaned.loc[(1, 1, 3), "origen"] == AGEB
    assert cleaned.loc[(1, 2, 3), "origen"] == AGEB   # ...leaves the next origin alone: either side could be wrong


def test_start_hours_are_repaired_by_the_fewest_typo_edits():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 5, 30, "Regresar a Casa"),            # 5:30 -> 17:30, a 12-hour entry
        (2, 8, 0, "Trabajar"), (2, 5, 30, "Regresar a Casa"),
        (2, 6, 0, "Compras (comida)"), (2, 9, 0, "Regresar a Casa"),     # a run of them: all three move
        (3, 19, 0, "Trabajar"), (3, 1, 0, "Regresar a Casa"),            # an overnight wrap is left alone
        (4, 8, 0, "Trabajar"), (4, 7, 0, "Regresar a Casa"),
        (4, 13, 0, "Compras (comida)"), (4, 14, 0, "Regresar a Casa"),   # no reading within the cost cap
        (5, 7, 24, "Compras (comida)"), (5, 7, 37, "Regresar a Casa"),
        (5, 19, 0, "Trabajar"), (5, 16, 30, "Regresar a Casa"),          # 19:00 -> 9:00, an extra leading 1
        (6, 14, 0, "Trabajar"), (6, 13, 50, "Regresar a Casa"),          # ten minutes back: minute noise
        (7, 11, 0, "Trabajar"), (7, 8, 0, "Regresar a Casa"),
        (7, 18, 13, "Llevar o recoger a alguien"), (7, 18, 22, "Regresar a Casa"),  # 8:00 -> 18:00: 20:00 would overtake
    ])
    cleaned, _, counts = clean_trip_chains(trips, VIV)
    hours = cleaned.hora_inicio_h.groupby(level="folio_habitante").apply(list)
    flags = cleaned.hora_inicio_ajuste.groupby(level="folio_habitante").apply(list)
    assert hours[1] == [8, 17] and flags[1] == ["", "+12h"]
    assert hours[2] == [8, 17, 18, 21]
    assert hours[3] == [19, 1] and flags[3] == ["", ""]
    assert hours[4] == [8, 7, 13, 14]
    assert hours[5] == [7, 7, 9, 16] and flags[5] == ["", "", "-10h", ""]
    assert hours[6] == [14, 13]
    assert hours[7] == [11, 18, 18, 18] and flags[7] == ["", "+10h", "", ""]
    assert counts["start_times_edited"] == 6 and counts["chains_repaired"] == 4


def test_a_trip_cannot_start_before_the_previous_one_arrived():
    # Household 8, person 3: the two-minute step at the end is a 32-minute
    # contradiction once the 30-minute drive is counted, and only one reading
    # within the menu resolves the whole chain.
    trips = _trips([
        (1, 7, 24, "Compras (comida)", None, None, 5), (1, 7, 37, "Regresar a Casa", None, None, 5),
        (1, 19, 0, "Trabajar", None, None, 30), (1, 16, 30, "Regresar a Casa", None, None, 30),
        (1, 18, 2, "Trabajar", None, None, 30), (1, 18, 0, "Regresar a Casa", None, None, 30),
    ])
    cleaned, _, counts = clean_trip_chains(trips, VIV)
    assert cleaned.hora_inicio_h.tolist() == [7, 7, 9, 16, 18, 20]
    assert cleaned.hora_inicio_ajuste.tolist() == ["", "", "-10h", "", "", "-10h+12h"]
    assert counts["start_times_edited"] == 2 and counts["chains_repaired"] == 1


def test_start_hour_search_declines_ambiguous_and_manufactured_readings():
    trips = _trips([
        (1, 7, 16, "Compras (comida)"), (1, 7, 31, "Regresar a Casa"),
        (1, 15, 30, "Estudiar"), (1, 19, 0, "Regresar a Casa"),
        (1, 18, 5, "Estudiar"), (1, 19, 36, "Regresar a Casa"),           # household 430: evening class, not a night shift
    ])
    cleaned, _, counts = clean_trip_chains(trips, VIV)
    assert cleaned.hora_inicio_h.tolist() == [7, 7, 15, 19, 20, 21]
    assert cleaned.hora_inicio_ajuste.tolist() == ["", "", "", "", "-10h+12h", "-10h+12h"]
    assert (cleaned.hora_inicio_h >= 5).all()


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


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_legs_unpivot_is_lossless():
    """Every filled traslado slot becomes a leg, in order, with a mode and minutes."""
    viv, hab, trips, legs = load_eod(DATA_DIR, clean_chains=False)
    trip_levels = ["folio_vivienda", "folio_habitante", "folio_viaje"]
    per_trip = legs.groupby(level=trip_levels).size()
    assert (per_trip.reindex(trips.index).fillna(0) == trips.n_traslados).all()
    # slots are filled in order: folio_traslado is 1..n within every trip
    position = legs.groupby(level=trip_levels).cumcount() + 1
    assert (position.to_numpy() == legs.index.get_level_values("folio_traslado").to_numpy()).all()
    assert not legs.traslado_medio.isna().any()
    assert (legs.traslado_min > 0).all()
    # the survey's main mode is always one of the trip's leg modes
    leg_modes = legs.traslado_medio.astype(str).groupby(level=trip_levels).agg(set).reindex(trips.index)
    assert all(m in modes for m, modes in zip(trips.modo_principal.astype(str), leg_modes))
    assert not any(c.startswith("traslado") for c in trips.columns)

