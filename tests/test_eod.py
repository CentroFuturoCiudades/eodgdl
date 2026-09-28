"""load_eod: the chain cleaning the loader applies to the trip table."""
from pathlib import Path

import pandas as pd
import pytest

from eodgdl import clean_trip_chains, flag_repeated_diaries, load_eod
from eodgdl.chains import (
    FIX_CODES,
    ISSUE_CODES,
    TIME_ORDER_ISSUES,
    _day_wraps,
    _donor_mask,
    _spread,
    _start_at_arrival,
    _trip_features,
    days_past_midnight,
    has_code,
    mark_issues,
    non_trips,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

AGEB = "1409700251418"          # the household's zone
ELSEWHERE = "1409700251419"
OTHER = "1409700251420"
VIV = pd.DataFrame({"ageb": [AGEB]}, index=pd.Index([1], name="folio_vivienda"))
SHOP = "Comercio, mercado, tienda o centro comercial"
CAR = "AUTOMÓVIL PARTICULAR"
PERSON = ["folio_vivienda", "folio_habitante"]


def _trips(rows):
    """A trips table in load_eod's shape from (person, hour, minute, motivo[, type, zone, minutes, mode]) rows.

    Every trip is on foot and starts where the previous one ended; a 'Regresar a
    Casa' arrives at home and at 'Su casa', anything else arrives ELSEWHERE at a
    shop, unless the row says.
    Travel minutes are 0 unless given. A row with no hour is an untimed trip and,
    like the survey's, has no motive or destination type either.
    """
    rows = [tuple(r) + (None,) * (8 - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=["person", "hora_inicio_h", "hora_inicio_m", "motivo_viaje",
                                     "tipo_lugar_destino", "destino", "traslado1_min", "modo_principal"])
    df["traslado1_min"] = df.traslado1_min.fillna(0).astype("Int64")
    df["modo_principal"] = df.modo_principal.fillna("A PIE")
    home = df.motivo_viaje == "Regresar a Casa"
    df["tipo_lugar_destino"] = df.tipo_lugar_destino.fillna(
        home.map({True: "Su casa", False: SHOP}).where(df.motivo_viaje.notna()))
    df["destino"] = df.destino.fillna(home.map({True: AGEB, False: ELSEWHERE}))
    # each trip starts where the person's previous one ended; the day starts at home
    df["origen"] = df.groupby("person").destino.shift(1).fillna(AGEB)
    df["folio_vivienda"] = 1
    df["folio_habitante"] = df.person
    df["folio_viaje"] = df.groupby("person").cumcount() + 1
    df["tipo_lugar_origen"] = "Su casa"
    df["hora_inicio_h"] = df.hora_inicio_h.astype("Int64")
    df["hora_inicio_m"] = df.hora_inicio_m.astype("Int64")
    return df.set_index(["folio_vivienda", "folio_habitante", "folio_viaje"]).drop(columns="person")


def viv_home(trips):
    """The household zone on every row of a fixture, as clean_trip_chains reads it from VIV."""
    return VIV.ageb.astype(str).reindex(trips.index.get_level_values("folio_vivienda")).to_numpy()


def _tour(person, out, back, minutes=5, mode="A PIE", motive="Compras (comida)", kind=None):
    """A person's out-and-back pair at the given (hour, minute) starts, as fixture rows."""
    return [(person, *out, motive, kind, None, minutes, mode), (person, *back, "Regresar a Casa", None, None, minutes, mode)]


def test_an_untimed_trip_is_imputed_from_its_nearest_timed_trips():
    donors = [r for p in range(2, 8) for r in _tour(p, (17, 30), (18, 0))]            # 25-minute errands on foot
    others = [r for p in range(8, 12) for r in _tour(p, (8, 0), (18, 0), 30, CAR, "Trabajar", "Oficina")]
    trips = _trips([
        (1, 8, 0, "Trabajar", "Oficina", OTHER, 30, CAR), (1, 12, 0, "Regresar a Casa", None, None, 30, CAR),
        (1, None, None, None, None, None, 5), (1, 18, 0, "Regresar a Casa", None, None, 5),   # the lost block and its return
        (12, 12, 0, None, None, None, 5), (12, 18, 0, "Regresar a Casa", None, None, 5),      # timed, no motive
        *donors, *others,
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert len(cleaned) == len(trips) and not cleaned.hora_inicio_h.isna().any() and not cleaned.motivo_viaje.isna().any()
    row = cleaned.loc[(1, 1, 3)]
    # 18:00 less 5 minutes of walking less the donors' 25-minute errand
    assert (row.hora_inicio_h, row.hora_inicio_m) == (17, 30)
    assert row.motivo_viaje == "Compras (comida)" and row.tipo_lugar_destino == SHOP
    assert row.ajustes == "hora:vecinos;motivo:vecinos" and row.problemas == ""
    timed = cleaned.loc[(1, 12, 1)]
    assert (timed.hora_inicio_h, timed.motivo_viaje, timed.ajustes) == (12, "Compras (comida)", "motivo:vecinos")
    assert cleaned.loc[(1, 1, 4), "ajustes"] == "" and cleaned.loc[(1, 1, 4), "problemas"] == ""
    assert counts["untimed_trips"] == 1 and counts["untimed_persons"] == 1
    assert counts["times_from_neighbours"] == 1 and counts["motives_from_neighbours"] == 2
    assert not non_trips(cleaned).any()


def test_donors_are_timed_activity_trips_whose_times_hold_up():
    trips = _trips([
        (1, 8, 0, "Trabajar", None, None, 30), (1, 17, 0, "Regresar a Casa", None, None, 30),     # a nine-hour day: donates
        (2, 19, 0, "Trabajar", None, None, 30), (2, 17, 0, "Regresar a Casa", None, None, 30),    # 19:00 for 9:00: a negative duration
        (3, 8, 0, "Trabajar", None, None, 30), (3, 8, 10, "Compras (comida)", None, None, 5),     # the errand starts inside the leg: the work
        (3, 12, 0, "Regresar a Casa", None, None, 5),                                              # trip's duration is negative, the errand too early
        (4, 8, 0, "Trabajar", None, None, 30), (4, 8, 20, "Compras (comida)", None, None, 5),     # ten minutes short: the errand is within the
        (4, 12, 0, "Regresar a Casa", None, None, 5),                                              # tolerance and donates; the work trip still not
        (5, None, None, None, None, None, 5), (5, 18, 0, "Regresar a Casa", None, None, 5),       # untimed: never a donor
    ])
    f = _trip_features(trips, viv_home(trips), None, None)
    donors = f.index[_donor_mask(f)].tolist()
    assert donors == [(1, 1, 1), (1, 4, 2)]


def test_an_untimed_return_takes_the_duplicate_that_follows_it():
    trips = _trips([
        (1, 8, 0, "Trabajar", "Oficina", None, 30, CAR),
        (1, None, None, None, None, AGEB, 30, CAR),              # the return, block lost
        (1, 18, 0, "Regresar a Casa", None, None, 5, CAR),       # home to home: the return's answers on a duplicate
    ])
    trips.loc[(1, 1, 2), "origen"] = ELSEWHERE
    cleaned, counts = clean_trip_chains(trips, VIV)
    ret = cleaned.loc[(1, 1, 2)]
    assert (ret.hora_inicio_h, ret.hora_inicio_m, ret.motivo_viaje, ret.tipo_lugar_destino) == (18, 0, "Regresar a Casa", "Su casa")
    assert ret.ajustes == "hora:duplicado;motivo:duplicado" and ret.problemas == ""
    assert (1, 1, 3) not in cleaned.index and len(cleaned) == 2     # the duplicate is dropped: its answers sit on the return
    assert non_trips(cleaned).tolist() == [False, False]
    assert counts["times_from_duplicate"] == 1 and counts["duplicate_returns"] == 1 and counts["home_to_home"] == 0


def test_a_return_home_made_from_home_is_not_a_trip():
    trips = _trips([
        (1, 12, 0, "Regresar a Casa"),                       # first trip, from home: marked
        (1, 13, 0, "Trabajar"), (1, 18, 0, "Regresar a Casa"),
        (1, 19, 0, "Regresar a Casa"), (1, 20, 0, "Regresar a Casa"),   # a run: both marked
        (2, 8, 0, "Trabajar"), (2, 12, 0, "Regresar a Casa"),
        (2, 15, 0, "Compras (comida)"), (2, 16, 0, "Regresar a Casa"),  # intact
        (3, 14, 0, "Regresar a Casa"), (3, 16, 0, "Compras (comida)"), (3, 17, 0, "Regresar a Casa"),
        (4, 14, 0, "Regresar a Casa"),
    ])
    # the day's answer is 'Su casa' but the first return leaves another zone: made away from home, a trip; the day
    # started there, at the place the person's own trips give that zone (a shop), whatever the answer said
    trips.loc[(1, 3, 1), "origen"] = ELSEWHERE
    # the first return leaves the home zone but the day started at another dwelling: a trip too
    trips.loc[(1, 4), "tipo_lugar_origen"] = "Otra vivienda"
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert counts["home_to_home"] == 3 and counts["first_return_answers"] == 1
    assert cleaned.loc[(1, 3, 1), "problemas"] == "inicio_fuera_de_casa"
    assert (cleaned.loc[(1, 3)].tipo_lugar_origen == SHOP).all() and (cleaned.loc[(1, 3)].ajustes == "tipo_origen:regreso").all()
    assert cleaned.loc[(1, 4, 1), "problemas"] == "inicio_fuera_de_casa"
    # nothing is dropped: the rows stay, marked, and the model build leaves them out
    assert len(cleaned) == len(trips)
    assert cleaned.loc[(1, 1)].problemas.tolist() == ["regreso_en_casa", "", "", "regreso_en_casa", "regreso_en_casa"]
    assert non_trips(cleaned).sum() == 3
    assert (cleaned.loc[(1, 2)].problemas == "").all()
    assert (cleaned.drop(index=3, level="folio_habitante").ajustes == "").all()
    assert (cleaned.loc[(1, 3)].problemas.tolist()[1:] == ["", ""])


def test_a_return_that_stops_short_of_home_does_not_put_the_person_at_home():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 12, 0, "Regresar a Casa", "Su casa", OTHER),   # 'home' in another zone
        (1, 19, 0, "Regresar a Casa"),                                              # ...so this is the way home: a trip
        (1, 20, 0, "Regresar a Casa"),                                              # and this one is made from home
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    # the return that stopped short went to a place nothing names (_short_returns): 'Otros', as the model read it
    assert cleaned.loc[(1, 1)].problemas.tolist() == ["", "", "", "regreso_en_casa"]
    assert cleaned.loc[(1, 1, 2), ["motivo_viaje", "tipo_lugar_destino", "ajustes"]].tolist() == [
        "Otros (especifique)", "Otros (especifique)", "motivo:otros"]
    assert counts["home_to_home"] == 1 and non_trips(cleaned).sum() == 1
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_a_return_left_away_from_home_goes_home_or_to_a_place_nothing_names():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 17, 0, "Regresar a Casa", "Su casa", OTHER),   # the day's last trip, to an AGEB
        (2, 8, 0, "Trabajar"), (2, 12, 0, "Regresar a Casa", "Su casa", OTHER),   # nothing else touches: home
        (2, 13, 0, "Compras (comida)"), (2, 14, 0, "Regresar a Casa", "Su casa", OTHER),   # the day comes back to it
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert cleaned.loc[(1, 1, 2), ["destino", "motivo_viaje", "ajustes", "problemas"]].tolist() == [
        AGEB, "Regresar a Casa", "destino:casa", ""]
    assert cleaned.loc[(1, 2)].motivo_viaje.tolist() == [
        "Trabajar", "Otros (especifique)", "Compras (comida)", "Otros (especifique)"]
    assert has_code(cleaned.ajustes, "motivo:otros").sum() == 2 and not has_code(cleaned.problemas, "regreso_sin_llegar").any()
    assert (counts["short_returns_home"], counts["short_returns_other"]) == (1, 2)
    # where a hand edit leaves a return away from home, the code says so
    assert has_code(mark_issues(trips, VIV), "regreso_sin_llegar").loc[(1, 1, 2)]


def test_a_return_home_after_midnight_from_an_afternoon_shift_is_overnight():
    trips = _trips([
        (1, 14, 0, "Trabajar", None, None, 30), (1, 2, 0, "Regresar a Casa"),       # second shift: overnight
        (2, 14, 0, "Trabajar", None, None, 30), (2, 2, 0, "Compras (comida)"),      # not a return: inverted
        (3, 19, 0, "Trabajar", None, None, 30), (3, 2, 0, "Compras (comida)"),      # an evening start: overnight
    ])
    cleaned, _ = clean_trip_chains(trips, VIV)
    assert (cleaned.ajustes == "").all()        # no typo reading fits any of them: the search leaves them
    second = cleaned.xs(2, level="folio_viaje").problemas
    assert [has_code(second, "hora_nocturna").loc[(1, p)] for p in (1, 2, 3)] == [True, False, True]
    assert [has_code(second, "hora_invertida").loc[(1, p)] for p in (1, 2, 3)] == [False, True, False]

def test_a_return_that_did_not_reach_home_takes_its_destination_type():
    trips = _trips([
        (1, 8, 0, "Trabajar"),
        (1, 12, 0, "Regresar a Casa", SHOP, OTHER),           # zone says not home: shopping
        (1, 13, 0, "Regresar a Casa"),                        # the real return, kept
        (2, 8, 0, "Trabajar"),
        (2, 17, 0, "Regresar a Casa", "Fábrica o taller"),    # zone says home: stays a return, type doubtful
        (2, 18, 0, "Regresar a Casa"),                        # ...so this one is home from home
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert counts["recoded_returns"] == 1 and counts["home_to_home"] == 1
    assert cleaned.loc[(1, 1)].motivo_viaje.tolist() == [
        "Trabajar", "Compras (bienes, productos y servicios)", "Regresar a Casa"]
    assert cleaned.loc[(1, 1)].ajustes.tolist() == ["", "motivo:tipo_destino", ""]
    assert cleaned.loc[(1, 2)].motivo_viaje.tolist() == ["Trabajar", "Regresar a Casa", "Regresar a Casa"]
    assert cleaned.loc[(1, 2)].problemas.tolist() == ["", "tipo_destino_dudoso", "regreso_en_casa"]


def test_every_trip_starts_where_the_previous_one_ended():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 17, 0, "Regresar a Casa"),
        (1, 18, 0, "Compras (comida)"), (1, 19, 0, "Regresar a Casa"),
        (2, 8, 0, "Trabajar", "Fábrica o taller", ELSEWHERE),   # a return recorded ending in its own origin's AGEB...
        (2, 17, 0, "Regresar a Casa", "Su casa", ELSEWHERE),
        (2, 18, 0, "Compras (comida)"), (2, 19, 0, "Regresar a Casa"),
        (3, 8, 0, "Trabajar", "Fábrica o taller", ELSEWHERE),   # work, then a trip recorded leaving home: the
        (3, 18, 0, "Compras (comida)", None, OTHER), (3, 19, 0, "Regresar a Casa"),   # return home was not recorded
    ])
    trips.loc[(1, 1, 3), "origen"] = ELSEWHERE     # carried over from the morning stop
    trips.loc[(1, 2, 3), "origen"] = AGEB          # ...followed by a trip that says it left from home
    trips.loc[(1, 3, 2), "origen"] = AGEB
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert counts["origins_repaired"] == 1 and counts["returns_copying_origin"] == 1 and counts["origins_continued"] == 1
    assert cleaned.loc[(1, 1, 3), "origen"] == AGEB and cleaned.loc[(1, 1, 3), "ajustes"] == "origen:casa"
    # ...went home: the next trip leaves from there, as recorded
    assert cleaned.loc[(1, 2, 2), ["destino", "ajustes"]].tolist() == [AGEB, "destino:copia"]
    assert (cleaned.loc[(1, 2)].problemas == "").all()
    # ...sets out from where the last trip left the person
    assert cleaned.loc[(1, 3, 2), ["origen", "ajustes"]].tolist() == [ELSEWHERE, "origen:anterior"]
    assert (cleaned.loc[(1, 3)].problemas == "").all() and counts["left_origen_discontinuo"] == 0
    # an origin break is still a code where a hand edit leaves one
    assert mark_issues(trips, VIV).loc[(1, 3, 2)] == "origen_discontinuo"


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
        (6, 14, 0, "Trabajar", None, None, 5), (6, 13, 50, "Regresar a Casa", None, None, 5),   # minute noise
        (7, 11, 0, "Trabajar"), (7, 8, 0, "Regresar a Casa"),
        (7, 18, 13, "Llevar o recoger a alguien"), (7, 18, 22, "Regresar a Casa"),  # 8:00 -> 18:00: 20:00 would overtake
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    hours = cleaned.hora_inicio_h.groupby(level="folio_habitante").apply(list)
    fixes = cleaned.ajustes.groupby(level="folio_habitante").apply(list)
    issues = cleaned.problemas.groupby(level="folio_habitante").apply(list)
    assert hours[1] == [8, 17] and fixes[1] == ["", "hora:+12h"]
    assert hours[2] == [8, 17, 18, 21]
    assert hours[3] == [19, 1] and fixes[3] == ["", ""] and issues[3] == ["", "hora_nocturna"]   # left, and said so
    assert hours[4] == [8, 7, 13, 14] and issues[4] == ["", "hora_invertida", "", ""]
    assert hours[5] == [7, 7, 9, 16] and fixes[5] == ["", "", "hora:-10h", ""]
    assert hours[6] == [14, 14] and fixes[6] == ["", "hora:llegada"] and issues[6] == ["", ""]   # ten minutes: to the arrival
    assert hours[7] == [11, 18, 18, 18] and fixes[7] == ["", "hora:+10h", "", ""]
    assert counts["start_times_edited"] == 6 and counts["chains_repaired"] == 4 and counts["starts_at_arrival"] == 1
    assert counts["left_hora_invertida"] == 1 and counts["left_hora_nocturna"] == 1 and counts["left_hora_anterior"] == 0


def test_a_trip_cannot_start_before_the_previous_one_arrived():
    # Household 8, person 3: the two-minute step at the end is a 32-minute
    # contradiction once the 30-minute drive is counted, and only one reading
    # within the menu resolves the whole chain.
    trips = _trips([
        (1, 7, 24, "Compras (comida)", None, None, 5), (1, 7, 37, "Regresar a Casa", None, None, 5),
        (1, 19, 0, "Trabajar", None, None, 30), (1, 16, 30, "Regresar a Casa", None, None, 30),
        (1, 18, 2, "Trabajar", None, None, 30), (1, 18, 0, "Regresar a Casa", None, None, 30),
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert cleaned.hora_inicio_h.tolist() == [7, 7, 9, 16, 18, 20]
    assert cleaned.ajustes.tolist() == ["", "", "hora:-10h", "", "", "hora:-10h+12h"]
    assert counts["start_times_edited"] == 2 and counts["chains_repaired"] == 1


def test_start_hour_search_declines_ambiguous_and_manufactured_readings():
    trips = _trips([
        (1, 7, 16, "Compras (comida)"), (1, 7, 31, "Regresar a Casa"),
        (1, 15, 30, "Estudiar"), (1, 19, 0, "Regresar a Casa"),
        (1, 18, 5, "Estudiar"), (1, 19, 36, "Regresar a Casa"),           # household 430: evening class, not a night shift
    ])
    cleaned, _ = clean_trip_chains(trips, VIV)
    assert cleaned.hora_inicio_h.tolist() == [7, 7, 15, 19, 20, 21]
    assert cleaned.ajustes.tolist() == ["", "", "", "", "hora:-10h+12h", "hora:-10h+12h"]
    assert (cleaned.hora_inicio_h >= 5).all()


def test_a_day_passes_midnight_where_a_start_is_read_as_the_next_day_s():
    trips = _trips([
        (1, 21, 0, "Trabajar", None, None, 30), (1, 2, 0, "Regresar a Casa", None, None, 30),   # a night shift...
        (1, 8, 0, "Compras (comida)"), (1, 8, 30, "Regresar a Casa"),                          # ...and the morning after
        (2, 14, 0, "Trabajar", None, None, 30), (2, 2, 0, "Regresar a Casa"),                  # a second shift's return
        (3, 14, 0, "Trabajar", None, None, 30), (3, 2, 0, "Compras (comida)"),                 # not a return: inverted
    ])
    cleaned, _ = clean_trip_chains(trips, VIV)
    days = days_past_midnight(cleaned)
    assert days.tolist() == [0, 1, 1, 1, 0, 1, 0, 0]
    # the trip where the day passes midnight is the one marked hora_nocturna
    assert (days.groupby(level="folio_habitante").diff().fillna(days) == 1).equals(has_code(cleaned.problemas, "hora_nocturna"))


def test_a_start_at_2301_as_reported_is_marked():
    trips = _trips([
        (1, 20, 0, "Trabajar"), (1, 23, 1, "Regresar a Casa"),                           # the survey's own 23:01
        (2, 20, 0, "Trabajar", None, None, 30), (2, 11, 1, "Regresar a Casa"),          # 11:01 read as 23:01
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert cleaned.loc[(1, 1, 2), "problemas"] == "hora_2301"
    assert cleaned.loc[(1, 2, 2), ["hora_inicio_h", "hora_inicio_m", "ajustes", "problemas"]].tolist() == [23, 1, "hora:+12h", ""]
    assert counts["left_hora_2301"] == 1 and mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_every_residual_defect_is_a_code():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 17, 0, "Regresar a Casa"),                                   # clean
        (2, 8, 0, "Trabajar"), (2, 17, 0, "Regresar a Casa"),                                   # the day starts at a shop
        (3, 8, 0, "Trabajar"), (3, 17, 0, "Regresar a Casa"),                                   # ...or at 'Su casa' in another zone
        (4, 8, 0, "Trabajar"), (4, 12, 0, "Regresar a Casa"), (4, 13, 0, "Compras (comida)"),   # ...or ends away from home
        (5, 8, 0, "Trabajar", "Su casa", AGEB), (5, 17, 0, "Compras (comida)"), (5, 18, 0, "Regresar a Casa"),   # work at home
        (6, 8, 0, "Guardería"), (6, 8, 0, "Regresar a Casa"),                                   # an escort, back the same minute
        (7, 8, 0, "Trabajar", None, None, 30), (7, 8, 0, "Regresar a Casa", None, None, 30),    # the same minute after a 30-minute leg...
        (7, 13, 0, "Compras (comida)"), (7, 14, 0, "Regresar a Casa"),                          # ...with no reading within the cost cap
        (8, 8, 0, "Trabajar", None, None, 30), (8, 8, 20, "Regresar a Casa", None, None, 30),   # back ten minutes before the leg ends
    ])
    trips.loc[trips.index.get_level_values("folio_habitante") == 2, "tipo_lugar_origen"] = SHOP
    trips.loc[(1, 3, 1), "origen"] = OTHER                     # not its destination's AGEB: a place of its own
    cleaned, counts = clean_trip_chains(trips, VIV)
    issues = cleaned.problemas.groupby(level="folio_habitante").apply(list)
    assert issues[1] == ["", ""]
    assert issues[2] == ["inicio_fuera_de_casa", ""]
    assert issues[3] == ["", ""]                                # the answer says home: the day starts there
    assert issues[4] == ["", "", "fin_fuera_de_casa"]
    assert issues[5] == ["actividad_en_casa", "", ""]
    assert issues[6] == ["motivo_guarderia", "hora_repetida"]
    assert issues[7] == ["", "hora_invertida", "", ""]          # one time code per row: the severe one
    assert issues[8] == ["", ""]                                # within the tolerance: moved to the arrival
    assert cleaned.ajustes[cleaned.ajustes != ""].to_dict() == {
        (1, 3, 1): "origen:respuesta", (1, 8, 2): "hora:llegada"} and not non_trips(cleaned).any()
    assert {k: v for k, v in counts.items() if k.startswith("left_") and v} == {
        "left_hora_invertida": 1, "left_hora_repetida": 1, "left_inicio_fuera_de_casa": 1,
        "left_fin_fuera_de_casa": 1, "left_actividad_en_casa": 1, "left_motivo_guarderia": 1}
    # an overlap the rules move to the arrival, and a first trip from 'Su casa' in another zone, are still codes
    # where a hand edit leaves one
    assert mark_issues(trips, VIV).loc[(1, 8, 2)] == "hora_traslapada"
    assert mark_issues(trips, VIV).loc[(1, 3, 1)] == "inicio_zona_ajena"
    before_start = _trips([(1, 10, 0, "Trabajar", None, None, 5), (1, 9, 55, "Regresar a Casa")])
    assert mark_issues(before_start, VIV).tolist() == ["", "hora_anterior"]


def test_a_first_trip_that_copies_home_away_starts_at_home():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 17, 0, "Regresar a Casa"),                  # leaves the AGEB it goes to: a copy
        (2, 8, 0, "Trabajar"), (2, 17, 0, "Regresar a Casa"),                  # leaves another dwelling: kept
        (3, 7, 0, "Regresar a Casa"), (3, 20, 0, "Trabajar"),                   # a return from elsewhere: kept
        (4, 8, 0, "Trabajar"), (4, 17, 0, "Regresar a Casa"),                  # the answer says home
        (5, 8, 0, "Trabajar"), (5, 17, 0, "Regresar a Casa"),                  # the answer copies the place type
    ])
    trips.loc[(1, 1, 1), "origen"] = ELSEWHERE           # the destination copied into the origin
    trips.loc[(1, 1), "tipo_lugar_origen"] = SHOP        # ...and its type into the day-start answer
    trips.loc[(1, 2, 1), "origen"] = OTHER
    trips.loc[(1, 2), "tipo_lugar_origen"] = "Otra vivienda"   # somebody spends the night there
    trips.loc[(1, 3, 1), "origen"] = ELSEWHERE
    trips.loc[(1, 3), "tipo_lugar_origen"] = "Fábrica o taller"   # the night shift's workplace
    trips.loc[(1, 4, 1), "origen"] = OTHER
    trips.loc[(1, 5, 1), "origen"] = OTHER
    trips.loc[(1, 5), "tipo_lugar_origen"] = SHOP        # the trip goes to a shop, and the day 'started at a shop'
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert cleaned.loc[(1, 1, 1), "origen"] == AGEB and (cleaned.loc[(1, 1)].tipo_lugar_origen == "Su casa").all()
    assert cleaned.loc[(1, 1)].ajustes.tolist() == ["origen:copia;tipo_origen:copia", "tipo_origen:copia"]
    assert cleaned.loc[(1, 1)].problemas.tolist() == ["", ""]
    assert cleaned.loc[(1, 2, 1), "origen"] == OTHER and cleaned.loc[(1, 2, 1), "problemas"] == "inicio_fuera_de_casa"
    assert cleaned.loc[(1, 3, 1), "origen"] == ELSEWHERE and (cleaned.loc[(1, 3)].ajustes == "").all()
    assert cleaned.loc[(1, 4, 1), ["origen", "ajustes", "problemas"]].tolist() == [AGEB, "origen:respuesta", ""]
    assert cleaned.loc[(1, 5, 1), ["origen", "ajustes", "problemas"]].tolist() == [
        AGEB, "origen:tipo_copia;tipo_origen:copia", ""]
    assert (cleaned.loc[(1, 5)].tipo_lugar_origen == "Su casa").all()
    assert (counts["first_origins_copied"], counts["first_origins_answered"], counts["first_origins_typed"]) == (1, 1, 1)
    assert counts["answers_set_home"] == 2
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_the_menu_reads_the_12_hour_clock_before_a_leading_1_and_no_half_day_at_an_errand():
    trips = _trips([
        (1, 19, 0, "Estudiar", "Escuela", None, 20), (1, 13, 0, "Regresar a Casa", None, None, 20),   # 19:00 is 07:00
        (2, 8, 0, "Compras (comida)", None, None, 80), (2, 22, 30, "Compras (comida)", None, None, 10),
        (2, 11, 0, "Regresar a Casa", None, None, 85),           # home at 23:00 would leave 13 hours at the first shops
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    starts = (cleaned.hora_inicio_h * 60 + cleaned.hora_inicio_m).groupby(level="folio_habitante").apply(list)
    fixes = cleaned.ajustes.groupby(level="folio_habitante").apply(list)
    # both readings fit; the 12-hour clock is the cheaper, and a sibling who went along says so (household 3627)
    assert starts[1] == [420, 780] and fixes[1] == ["hora:-12h", ""]
    # household 2585, person 1: the second errand is the morning's, and nobody stays half a day at the shops
    assert starts[2] == [480, 630, 660] and fixes[2] == ["", "hora:-12h", ""]
    assert counts["chains_repaired"] == 2 and mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_a_start_before_the_previous_arrival_by_the_tolerance_or_less_moves_to_it():
    trips = _trips([
        (1, 8, 0, "Trabajar", None, None, 30), (1, 8, 20, "Regresar a Casa", None, None, 30),     # ten minutes early
        (2, 8, 0, "Trabajar", None, None, 30), (2, 8, 25, "Compras (comida)", None, None, 30),    # five minutes early...
        (2, 8, 58, "Regresar a Casa", None, None, 10),                                             # ...and this one with it
        (3, 21, 0, "Trabajar", None, None, 30), (3, 21, 10, "Regresar a Casa", None, None, 30),   # 21:00 is 09:00
        (4, 22, 0, "Trabajar", None, None, 30), (4, 1, 0, "Regresar a Casa", None, None, 30),     # overnight: left
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    starts = (cleaned.hora_inicio_h * 60 + cleaned.hora_inicio_m).groupby(level="folio_habitante").apply(list)
    fixes = cleaned.ajustes.groupby(level="folio_habitante").apply(list)
    assert starts[1] == [480, 510] and fixes[1] == ["", "hora:llegada"]
    assert starts[2] == [480, 510, 540] and fixes[2] == ["", "hora:llegada", "hora:llegada"]
    assert starts[3] == [540, 1270] and fixes[3] == ["hora:-12h", ""]      # the -12h: a morning typed as evening
    assert starts[4] == [1320, 60] and cleaned.loc[(1, 4, 2), "problemas"] == "hora_nocturna"
    assert counts["starts_at_arrival"] == 3 and mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_the_search_reads_again_once_the_starts_move_to_the_arrival():
    # Household 834, person 1: the 20:10 return came 28 minutes before the 38-minute visit could end, which no reading
    # of the hours fixes, so the search read nothing, not even the 04:30 escort; moved to the arrival, it reads it
    trips = _trips([
        (1, 8, 0, "Trabajar", None, None, 5), (1, 4, 30, "Llevar o recoger a alguien", None, None, 5),
        (1, 18, 0, "Regresar a Casa", None, None, 5), (1, 20, 0, "Visitar a alguien", None, None, 38),
        (1, 20, 10, "Regresar a Casa", None, None, 38),
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert (cleaned.hora_inicio_h * 60 + cleaned.hora_inicio_m).tolist() == [480, 990, 1080, 1200, 1238]
    assert cleaned.ajustes.tolist() == ["", "hora:+12h", "", "", "hora:llegada"] and (cleaned.problemas == "").all()
    assert counts["chains_read_after_the_slide"] == 1 and counts["chains_repaired"] == 1


def test_a_night_is_no_night_shift_when_the_day_goes_to_work_again():
    trips = _trips([
        (1, 18, 0, "Trabajar", None, None, 30), (1, 8, 50, "Regresar a Casa", None, None, 20),   # household 2191,
        (1, 10, 13, "Trabajar", None, None, 30), (1, 7, 0, "Regresar a Casa", None, None, 20),   # person 2: a split shift
        (2, 21, 0, "Trabajar", None, None, 30), (2, 7, 0, "Regresar a Casa", None, None, 30),    # no reading: the hand's
        (2, 9, 0, "Trabajar", None, None, 30), (2, 17, 0, "Regresar a Casa", None, None, 30),
        (3, 19, 0, "Trabajar", None, None, 30), (3, 6, 0, "Regresar a Casa", None, None, 30),    # a night shift, and
        (3, 11, 0, "Compras (comida)", None, None, 5), (3, 12, 0, "Regresar a Casa", None, None, 5),   # a morning after
    ])
    cleaned, _ = clean_trip_chains(trips, VIV)
    hours = cleaned.hora_inicio_h.groupby(level="folio_habitante").apply(list)
    issues = cleaned.problemas.groupby(level="folio_habitante").apply(list)
    # to work at 18:00 and home at 08:50 is a night only if the day then ends: to work again at 10:13, it is 06:00
    assert hours[1] == [6, 8, 10, 19] and cleaned.loc[(1, 1)].ajustes.tolist() == ["hora:-12h", "", "", "hora:+12h"]
    assert issues[2] == ["", "hora_invertida", "", ""] and issues[3] == ["", "hora_nocturna", "", ""]
    assert days_past_midnight(cleaned).tolist() == [0] * 8 + [0, 1, 1, 1]
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)
    # a day is never read past 24 hours: work at 06:00 and 18:00, home at 07:00, is not a night shift after a day's work
    long_day = _trips([(1, 6, 0, "Trabajar", None, None, 30), (1, 18, 0, "Trabajar", None, None, 30),
                       (1, 7, 0, "Regresar a Casa", None, None, 30)])
    assert _day_wraps(long_day).tolist() == [False, False, False]
    assert _day_wraps(long_day.iloc[1:]).tolist() == [False, True]


def test_travel_minutes_a_review_corrected_come_before_the_rules():
    # household 7425, person 3: 120 minutes each way make the afternoon shift start before the ride home ends; the
    # search read its hours two later until 2026-09-27, when the ride home took the minutes up to the departure, and
    # the review's 60 minutes, read before any rule, let the survey's hours stand with no rule at all
    rows = [(1, 6, 5, "Trabajar", None, None, 120), (1, 12, 32, "Regresar a Casa", None, None, 120),
            (1, 14, 10, "Trabajar", None, None, 120), (1, 18, 8, "Regresar a Casa", None, None, 120)]
    trips = _trips(rows)
    as_reported, counts = clean_trip_chains(trips, VIV)
    assert as_reported.hora_inicio_h.tolist() == [6, 12, 14, 18] and counts["rides_home_to_departure"] == 1
    assert as_reported.ajustes.tolist() == ["", "minutos:regreso", "", ""]
    assert as_reported.traslado1_min.tolist() == [120, 98, 120, 120]
    minutes = pd.DataFrame({"household": 1, "person": 1, "trip": [1, 2, 3, 4], "before": 120, "after": 60})
    cleaned, counts = clean_trip_chains(trips, VIV, minutes=minutes)
    assert cleaned.hora_inicio_h.tolist() == [6, 12, 14, 18] and (cleaned.ajustes == "minutos:revision").all()
    assert cleaned.traslado1_min.tolist() == [60] * 4 and counts["leg_minutes_corrected"] == 4
    assert counts["rides_home_to_departure"] == 0
    assert (cleaned.problemas == "").all() and mark_issues(cleaned, VIV).equals(cleaned.problemas)
    # a correction is checked against the minutes it was made against
    with pytest.raises(ValueError, match="made against 90 minutes"):
        clean_trip_chains(trips, VIV, minutes=minutes.assign(before=90))
    # over several legs the minutes spread in proportion, a minute each at least (largest remainder)
    assert _spread([10, 25, 5], 20) == [5, 12, 3] and _spread([30], 20) == [20] and _spread([1, 1], 2) == [1, 1]


def test_a_ride_home_past_the_next_departure_took_the_minutes_up_to_it():
    trips = _trips([
        # household 15401, person 3: home from work by bus at 17:30, 60 minutes, then to the shop at 18:00 and back at
        # 18:20; until 2026-09-27 the search read both errand trips two hours later
        (1, 7, 0, "Trabajar", None, OTHER, 60), (1, 17, 30, "Regresar a Casa", None, None, 60),
        (1, 18, 0, "Compras (comida)", None, None, 8), (1, 18, 20, "Regresar a Casa", None, None, 8),
        # a departure two minutes after the ride's start is the capture's, not the ride's: it would keep 2 of 60 minutes
        (2, 7, 0, "Trabajar", None, OTHER, 60), (2, 17, 30, "Regresar a Casa", None, None, 60),
        (2, 17, 32, "Compras (comida)", None, None, 8), (2, 17, 50, "Regresar a Casa", None, None, 8),
        # a ride that goes elsewhere than home keeps its minutes
        (3, 7, 0, "Trabajar", None, OTHER, 60), (3, 17, 30, "Visitar a alguien", "Otra vivienda", None, 60),
        (3, 18, 0, "Regresar a Casa", None, None, 8),
        # so does a ride home that the next trip was recorded as leaving before
        (4, 7, 0, "Trabajar", None, OTHER, 60), (4, 17, 30, "Regresar a Casa", None, None, 60),
        (4, 17, 10, "Compras (comida)", None, None, 8), (4, 19, 0, "Regresar a Casa", None, None, 8),
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    minutes = cleaned.traslado1_min.groupby(level="folio_habitante").apply(list)
    fixes = cleaned.ajustes.groupby(level="folio_habitante").apply(list)
    assert minutes[1] == [60, 30, 8, 8] and fixes[1] == ["", "minutos:regreso", "", ""]
    assert (cleaned.loc[(1, 1)].hora_inicio_h * 60 + cleaned.loc[(1, 1)].hora_inicio_m).tolist() == [420, 1050, 1080, 1100]
    assert minutes[2][1] == minutes[3][1] == minutes[4][1] == 60 and counts["rides_home_to_departure"] == 1
    assert not has_code(cleaned.ajustes, "minutos:regreso").loc[(1, [2, 3, 4])].any()
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_two_members_who_report_a_trip_made_together_12_hours_apart_are_marked():
    school = ("Estudiar", "Escuela", ELSEWHERE, 5)
    trips = _trips([
        # household 4202: two children walk to school at 08:00, one comes home at 11:30 and the other at 23:30
        (1, 8, 0, *school), (1, 11, 30, "Regresar a Casa", None, None, 5),
        (2, 8, 0, *school), (2, 23, 30, "Regresar a Casa", None, None, 5),
        # household 1688: a straight shift beside a split one, both out at 08:00 and home at 21:00; the shared return
        # made the second trip to work of the one the companion of the morning trip of the other
        (3, 8, 0, "Trabajar", None, OTHER, 30), (3, 21, 0, "Regresar a Casa", None, None, 30),
        (4, 8, 0, "Trabajar", None, OTHER, 30), (4, 18, 32, "Regresar a Casa", None, None, 15),
        (4, 19, 50, "Trabajar", None, OTHER, 15), (4, 21, 0, "Regresar a Casa", None, None, 30),
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    marked = has_code(cleaned.problemas, "hora_acompanante_12h")
    assert marked[marked].index.tolist() == [(1, 1, 2), (1, 2, 2)] and counts["left_hora_acompanante_12h"] == 2
    assert (cleaned.ajustes == "").all() and mark_issues(cleaned, VIV).equals(cleaned.problemas)
    # once a hand decision reads one of the two, the pair is no longer marked
    settled = cleaned.copy()
    settled.loc[(1, 2, 2), ["hora_inicio_h", "ajustes"]] = [11, "hora:revision"]
    assert not has_code(mark_issues(settled, VIV), "hora_acompanante_12h").any()


def test_a_rotated_first_trip_went_from_home_to_the_place():
    trips = _trips([
        (1, 8, 0, "Trabajar", "Su casa", AGEB), (1, 17, 0, "Regresar a Casa"),           # to work, recorded from it
        (2, 8, 0, "Trabajar", "Su casa", AGEB), (2, 17, 0, "Compras (comida)"),          # ...and on from home
        (2, 18, 0, "Regresar a Casa"),
        (3, 8, 0, "Trabajar", "Su casa", AGEB), (3, 17, 0, "Compras (comida)", None, OTHER),   # a next trip that goes
        (3, 18, 0, "Regresar a Casa"),                                                          # on from home: kept
    ])
    for person in (1, 2, 3):
        trips.loc[(1, person, 1), "origen"] = ELSEWHERE
    trips.loc[(1, 1), "tipo_lugar_origen"] = "Oficina"      # the answer names the place
    trips.loc[(1, 3), "tipo_lugar_origen"] = "Oficina"
    cleaned, counts = clean_trip_chains(trips, VIV)
    one = cleaned.loc[(1, 1)]
    assert one[["origen", "destino"]].values.tolist() == [[AGEB, ELSEWHERE], [ELSEWHERE, AGEB]]
    assert one.tipo_lugar_destino.tolist() == ["Oficina", "Su casa"] and (one.tipo_lugar_origen == "Su casa").all()
    assert one.ajustes.tolist() == ["origen:rotado;destino:rotado;tipo_destino:rotado;tipo_origen:copia",
                                    "origen:anterior;tipo_origen:copia"]
    assert (one.problemas == "").all()
    # the answer said 'Su casa': rotated too, the place's type left as recorded
    two = cleaned.loc[(1, 2)]
    assert two[["origen", "destino"]].values.tolist()[:2] == [[AGEB, ELSEWHERE], [ELSEWHERE, ELSEWHERE]]
    assert two.problemas.tolist()[0] == "actividad_en_casa"
    # neither says home and the next trip is no return from home: left, as recorded
    three = cleaned.loc[(1, 3, 1)]
    assert [three.origen, three.destino, three.ajustes, three.problemas] == [
        ELSEWHERE, AGEB, "", "inicio_fuera_de_casa;actividad_en_casa"]
    assert counts["first_trips_rotated"] == 2 and mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_a_start_up_to_an_hour_before_the_arrival_moves_to_it_but_not_after_work():
    trips = _trips([
        (1, 10, 0, "Compras (comida)", None, None, 30), (1, 10, 10, "Regresar a Casa", None, None, 5),   # 20 min
        (2, 10, 0, "Trabajar", None, None, 30), (2, 10, 10, "Regresar a Casa", None, None, 5),           # after work
        (3, 10, 0, "Compras (comida)", None, None, 90), (3, 10, 10, "Regresar a Casa", None, None, 5),   # 80 min
    ])
    slid, moved = _start_at_arrival(trips, None)
    assert (slid.hora_inicio_h * 60 + slid.hora_inicio_m).tolist() == [600, 630, 600, 610, 600, 610]
    assert moved.tolist() == [False, True, False, False, False, False]


def test_a_night_shift_return_is_the_next_morning():
    trips = _trips([
        (1, 20, 0, "Trabajar", None, None, 30), (1, 8, 0, "Regresar a Casa", None, None, 30),     # a night shift
        (2, 7, 0, "Regresar a Casa", "Su casa", None, 30), (2, 20, 0, "Trabajar", None, None, 30),
        (2, 20, 30, "Regresar a Casa", None, None, 30),                                         # the next morning's
    ])
    trips.loc[(1, 2, 1), "origen"] = ELSEWHERE           # the day opens with the return from the job
    trips.loc[(1, 2), "tipo_lugar_origen"] = "Fábrica o taller"
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert (cleaned.loc[(1, 1)].ajustes == "").all()     # no reading of the hours: the typo search leaves it
    assert cleaned.loc[(1, 1)].problemas.tolist() == ["", "hora_nocturna"]
    assert days_past_midnight(cleaned.loc[[(1, 1, 1), (1, 1, 2)]]).tolist() == [0, 1]
    assert cleaned.loc[(1, 2)].problemas.tolist() == ["inicio_fuera_de_casa", "fin_fuera_de_casa", "regreso_dia_siguiente"]
    assert non_trips(cleaned).tolist() == [False, False, False, False, True] and counts["next_morning_returns"] == 1
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_the_motives_the_chain_implies():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 17, 0, "Compras (comida)", "Su casa", AGEB),     # arrives home: a return
        (2, 8, 0, "Trabajar", None, OTHER), (2, 13, 0, "Compras (comida)"),
        (2, 14, 0, "Regresar a Casa", "Su casa", OTHER), (2, 18, 0, "Regresar a Casa"),  # back to work's AGEB: work
        (3, 8, 0, "Guardería"), (3, 9, 0, "Regresar a Casa"),                        # an adult: the escort
        (4, 8, 0, "Guardería"), (4, 9, 0, "Regresar a Casa"),                        # a child: attends
    ])
    hab = pd.DataFrame({"edad": [40, 30, 35, 8], "sexo_nacimiento": "Mujeres", "ocupacion": "Empleado"},
                       index=pd.MultiIndex.from_tuples([(1, 1), (1, 2), (1, 3), (1, 4)], names=PERSON))
    cleaned, counts = clean_trip_chains(trips, VIV, hab=hab)
    assert cleaned.loc[(1, 1, 2), ["motivo_viaje", "ajustes", "problemas"]].tolist() == ["Regresar a Casa", "motivo:casa", ""]
    assert cleaned.loc[(1, 2, 3), ["motivo_viaje", "tipo_lugar_destino", "ajustes"]].tolist() == [
        "Trabajar", SHOP, "motivo:lugar"]
    assert (cleaned.loc[(1, 2)].problemas == "").all()
    assert cleaned.loc[(1, 3, 1), ["motivo_viaje", "ajustes", "problemas"]].tolist() == [
        "Llevar o recoger a alguien", "motivo:guarderia", ""]
    assert cleaned.loc[(1, 4, 1), ["motivo_viaje", "problemas"]].tolist() == ["Guardería", "motivo_guarderia"]
    assert (counts["arrivals_home"], counts["returns_elsewhere"], counts["daycare_escorts"]) == (1, 1, 1)
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_a_day_that_starts_in_the_small_hours_with_a_long_stay_is_read_by_the_typo_search():
    trips = _trips([
        (1, 1, 8, "Estudiar", None, None, 60), (1, 20, 7, "Regresar a Casa", None, None, 60),         # afternoon school
        (2, 4, 30, "Compras (comida)", None, None, 20), (2, 6, 0, "Regresar a Casa", None, None, 20),  # an early market trip
        (3, 3, 0, "Trabajar", None, None, 30), (3, 17, 0, "Regresar a Casa", None, None, 30),         # work at three
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    hours = cleaned.hora_inicio_h.groupby(level="folio_habitante").apply(list)
    assert hours[1] == [13, 20] and cleaned.loc[(1, 1)].ajustes.tolist() == ["hora:+12h", ""]
    assert hours[2] == [4, 6] and cleaned.loc[(1, 2)].problemas.tolist() == ["hora_madrugada", ""]   # left, and said so
    assert hours[3] == [3, 17] and (cleaned.loc[(1, 3)].problemas == "").all()
    assert counts["early_starts_read"] == 1 and counts["left_hora_madrugada"] == 1
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_a_return_home_half_a_day_after_a_short_errand_is_a_morning_one():
    trips = _trips([
        (1, 9, 0, "Al médico o atención de salud", None, None, 30), (1, 22, 0, "Regresar a Casa", None, None, 30),
        (2, 9, 0, "Visitar a alguien", None, None, 30), (2, 22, 0, "Regresar a Casa", None, None, 30),   # a visit: left
        (3, 9, 0, "Compras (comida)", None, None, 30), (3, 21, 0, "Regresar a Casa", None, None, 30),    # 11½ hours: left
        (4, 9, 0, "Compras (comida)", None, None, 30), (4, 22, 0, "Compras (comida)", None, None, 30),   # no return next
        (4, 22, 45, "Regresar a Casa", None, None, 30),
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    hours = cleaned.hora_inicio_h.groupby(level="folio_habitante").apply(list)
    # household 9541, person 1: to the doctor at 07:00 and home at 23:00, where a companion was home at 11:00
    assert hours[1] == [9, 10] and cleaned.loc[(1, 1)].ajustes.tolist() == ["", "hora:-12h"]
    assert hours[2] == [9, 22] and hours[3] == [9, 21] and hours[4] == [9, 22, 22]
    assert counts["long_errands_read"] == 1 and counts["early_starts_read"] == 0 and (cleaned.problemas == "").all()
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_a_return_the_capture_timed_is_marked():
    trips = _trips([
        (1, 17, 24, "Compras (comida)", None, None, 8), (1, 18, 25, "Regresar a Casa", None, None, 8),   # 1h01 after
        (2, 8, 0, "Trabajar", None, None, 30), (2, 13, 1, "Regresar a Casa", None, None, 30),          # 5h01 after
        (3, 8, 0, "Trabajar", None, None, 30), (3, 13, 0, "Regresar a Casa", None, None, 30),          # a time of its own
        (4, 8, 0, "Trabajar", None, None, 30), (4, 13, 1, "Regresar a Casa", None, None, 20),          # not a copy
        (5, 8, 0, "Trabajar", None, None, 30), (5, 1, 1, "Regresar a Casa", None, None, 30),           # 13:01 by a rule
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    issues = cleaned.problemas.groupby(level="folio_habitante").apply(list)
    assert issues[1] == ["", "hora_1h01"] and issues[2] == ["", "hora_5h01"]
    assert issues[3] == issues[4] == issues[5] == ["", ""]
    assert cleaned.loc[(1, 5, 2), ["hora_inicio_h", "ajustes"]].tolist() == [13, "hora:+12h"]
    assert counts["left_hora_1h01"] == counts["left_hora_5h01"] == 1
    assert mark_issues(cleaned, VIV).equals(cleaned.problemas)


def test_load_eod_cleans_the_chains_and_drops_only_the_duplicate_returns(stages):
    viv, hab, trips, legs = stages.rules   # load_eod(revise_chains=False): the rules alone
    assert (len(hab), len(trips), len(legs)) == (58_061, 154_662 - 38, 170_509 - 38)
    assert not trips.hora_inicio_h.isna().any() and not trips.motivo_viaje.isna().any()
    # every change and every defect left is on the row, in the documented vocabulary
    fixes = set(trips.ajustes.str.split(";").explode()) - {""}
    issues = set(trips.problemas.str.split(";").explode()) - {""}
    assert fixes <= set(FIX_CODES) and issues <= set(ISSUE_CODES)
    assert has_code(trips.ajustes, "hora:duplicado").sum() == 37 and has_code(trips.ajustes, "hora:vecinos").sum() == 288
    assert has_code(trips.ajustes, "motivo:vecinos").sum() == 294 and has_code(trips.ajustes, "motivo:duplicado").sum() == 38
    assert non_trips(trips).sum() == 556 + 75
    # every defect left is a code on the row, at these counts; a row carries at most one hora_* code
    assert {c: int(has_code(trips.problemas, c).sum()) for c in ISSUE_CODES} == {
        "regreso_en_casa": 556, "regreso_dia_siguiente": 75, "hora_invertida": 349, "hora_nocturna": 178,
        "hora_anterior": 0, "hora_repetida": 0, "hora_traslapada": 0, "origen_discontinuo": 0, "regreso_sin_llegar": 0,
        "tipo_destino_dudoso": 16, "inicio_fuera_de_casa": 381, "inicio_zona_ajena": 0,
        "fin_fuera_de_casa": 494, "actividad_en_casa": 536, "motivo_guarderia": 30, "hora_2301": 94, "hora_1h01": 728,
        "hora_5h01": 613, "hora_madrugada": 58, "hora_acompanante_12h": 10}
    # every change a rule makes, at these counts (the fifth review of the chains added five of them, 2026-09-26; the
    # review of the readings against household companions, the same day, read 236 extra leading 1's as the -12h; the
    # review of the leading-1 readings, 2026-09-27, let 240 rides home take the minutes up to the next departure, and
    # 345 fewer starts are read two hours on)
    assert {c: int(has_code(trips.ajustes, c).sum()) for c in FIX_CODES if not c.endswith(":revision")} == {
        "hora:duplicado": 37, "hora:vecinos": 288, "motivo:duplicado": 38, "motivo:vecinos": 294,
        "motivo:tipo_destino": 84, "destino:copia": 69, "motivo:casa": 231, "motivo:lugar": 67, "motivo:guarderia": 163,
        "destino:casa": 8, "motivo:otros": 21, "origen:casa": 147, "origen:anterior": 77, "origen:copia": 955,
        "origen:respuesta": 86, "origen:tipo_copia": 151, "origen:rotado": 47, "destino:rotado": 47,
        "tipo_destino:rotado": 20, "tipo_origen:copia": 1_602, "tipo_origen:regreso": 113, "minutos:regreso": 240,
        "hora:+12h": 706, "hora:-12h": 636 + 123, "hora:-10h": 142, "hora:+10h": 149, "hora:-10h+12h": 471,
        "hora:llegada": 648}
    hora = sum(has_code(trips.problemas, c) for c in TIME_ORDER_ISSUES)
    assert (hora <= 1).all() and (trips.problemas != "").sum() == 3_394 + 631
    # the travel minutes a review corrected are read before the rules (eodgdl/revisions/leg_minutes.csv.gz), and the
    # rides home that ran past the next departure take the minutes up to it: on those trips, and only there, the rules'
    # legs hold other minutes than the survey's
    minutes = legs.traslado_min.groupby(level=[0, 1, 2]).sum()
    shipped = stages.shipped.legs.traslado_min.groupby(level=[0, 1, 2]).sum().reindex(minutes.index)
    corrected, capped = has_code(trips.ajustes, "minutos:revision"), has_code(trips.ajustes, "minutos:regreso")
    assert set(minutes.index[minutes != shipped]) == set(trips.index[(corrected | capped).to_numpy()])
    assert corrected.sum() == 371 and capped.sum() == 240 and not (corrected & capped).any()
    # a day read past midnight holds together: no trip to work or school after the night, less than a day long
    kept = trips[~non_trips(trips)].sort_index()
    days = days_past_midnight(kept)
    start = (kept.hora_inicio_h * 60 + kept.hora_inicio_m + 1440 * days).groupby(level=PERSON)
    assert days.max() == 1 and ((start.max() - start.first()) < 1440).all()
    wrap = days.groupby(level=PERSON).diff().fillna(days) == 1                  # the trip where the day passes midnight
    assert not ((days == 1) & ~wrap & kept.motivo_viaje.isin(["Trabajar", "Estudiar"])).any()
    # the zone follows the AGEB on every trip end, the ones the rules moved home among them
    ends = pd.concat([pd.DataFrame({"ageb": trips[p].astype(str), "zone": trips[z].astype(str)})
                      for p, z in (("origen", "zona_origen"), ("destino", "zona_destino"))])
    assert (ends.groupby("ageb").zone.nunique() == 1).all()
    # hab and trips stay in step: viajes_contados follows the 38 dropped duplicates and the legs follow the trips
    counted = trips.groupby(level=PERSON).size()
    assert (hab.viajes_contados == counted.reindex(hab.index).fillna(0)).all()
    assert hab.viajes_contados.sum() == stages.shipped.hab.viajes_contados.sum() - 38
    assert legs.index.droplevel("folio_traslado").isin(trips.index).all()
    # the categoricals survive the imputation, with the imputed values inside their levels
    assert str(trips.motivo_viaje.dtype) == "category" and str(trips.tipo_lugar_destino.dtype) == "category"


def test_load_eod_can_return_the_survey_as_shipped(stages):
    viv, hab, trips, legs = stages.shipped
    assert (len(hab), len(trips), len(legs)) == (58_061, 154_662, 170_509)
    # load_eod's switch stops at the stage load_stages returns first
    assert all(a.equals(b) for a, b in zip(load_eod(DATA_DIR, clean_chains=False), stages.shipped))
    assert trips.ponderador.sum() == 11_796_386        # the published trip total
    assert trips.hora_inicio_h.isna().sum() == 325
    assert "ajustes" not in trips.columns and not non_trips(trips).any()


def test_legs_unpivot_is_lossless(stages):
    """Every filled traslado slot becomes a leg, in order, with a mode and minutes."""
    viv, hab, trips, legs = stages.shipped
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



# ------------------------------------------------------------- repeated diaries

HAB_DATE = pd.Timestamp("2023-01-28")


def _diary(hh, person, times, mode="CAMIÓN O AUTOBÚS", origin=AGEB, date=HAB_DATE):
    """One person's diary: a trip out and a trip home at the given (hour, minute) pairs, one leg each."""
    rows = []
    for k, (h, m) in enumerate(times):
        home = k % 2 == 1
        rows.append({"folio_vivienda": hh, "folio_habitante": person, "folio_viaje": k + 1,
                     "origen": origin if not home else ELSEWHERE, "destino": ELSEWHERE if not home else origin,
                     "motivo_viaje": "Regresar a Casa" if home else "Trabajar",
                     "tipo_lugar_destino": "Su casa" if home else "Oficina", "n_traslados": 1,
                     "traslado1_medio": mode, "traslado1_min": 30, "traslado1_pago": 9.5,
                     "hora_inicio_h": h, "hora_inicio_m": m})
    return rows, {"folio_vivienda": hh, "folio_habitante": person, "fecha": date}


def _tables(diaries, agebs):
    trips = pd.concat([pd.DataFrame(rows) for rows, _ in diaries]).set_index(["folio_vivienda", "folio_habitante", "folio_viaje"])
    for i in range(2, 6):                       # the empty leg slots the survey file carries
        for k in ("medio", "min", "pago"):
            trips[f"traslado{i}_{k}"] = float("nan")
    hab = pd.DataFrame([h for _, h in diaries]).set_index(["folio_vivienda", "folio_habitante"])
    viv = pd.DataFrame({"ageb": agebs}, index=pd.Index(list(agebs), name="folio_vivienda"))
    return trips, hab, viv


def test_a_diary_repeated_in_another_household_with_nudged_times_is_flagged():
    diaries = [
        _diary(1, 1, [(7, 0), (17, 0)]),                    # the pair: same diary, times 5 and 8 minutes apart
        _diary(2, 1, [(7, 5), (17, 8)]),
        _diary(3, 1, [(7, 11), (17, 20)]),                  # eleven and twelve minutes: outside the twin window
        _diary(4, 1, [(7, 0), (17, 0)], date=HAB_DATE + pd.Timedelta(days=1)),   # another interview date
        _diary(1, 2, [(8, 0), (12, 0)], mode="A PIE"),      # walks only: repeats by chance
        _diary(2, 2, [(8, 0), (12, 0)], mode="A PIE"),
        _diary(1, 3, [(9, 0)]),                             # a single trip
        _diary(5, 1, [(9, 0)]),
        _diary(6, 1, [(6, 0), (18, 0)]),                    # same diary in another AGEB
        _diary(7, 1, [(6, 2), (18, 3)]),
        _diary(8, 1, [(10, 0), (14, 0)]),                   # the same diary within one household
        _diary(8, 2, [(10, 3), (14, 3)]),
    ]
    agebs = {1: AGEB, 2: AGEB, 3: AGEB, 4: AGEB, 5: AGEB, 6: AGEB, 7: ELSEWHERE, 8: AGEB}
    trips, hab, viv = _tables(diaries, agebs)
    flag = flag_repeated_diaries(trips, hab, viv)
    assert flag.name == "diario_repetido" and flag.dtype == bool
    assert flag[flag].index.tolist() == [(1, 1), (2, 1)]


def test_load_eod_flags_repeated_diaries_and_drops_none(stages):
    viv, hab, trips, legs = stages.shipped
    assert hab.diario_repetido.dtype == bool
    assert hab.diario_repetido.sum() == 1_793
    assert len(hab) == 58_061 and len(trips) == 154_662
    # the twins that surfaced the pattern: 879/3 repeats 9530/3 whole; 879/7 adds two trips to 9560/4's diary
    assert hab.loc[(9530, 3), "diario_repetido"] and hab.loc[(879, 3), "diario_repetido"]
    assert not hab.loc[(9560, 4), "diario_repetido"] and not hab.loc[(879, 7), "diario_repetido"]
    # the same rule reads the legs table once the traslado columns are gone
    assert flag_repeated_diaries(trips, hab, viv, legs).equals(hab.diario_repetido)
    # the flag is set on the survey as shipped and survives the chain rules and the hand passes
    assert stages.revised.hab.diario_repetido.sum() == 1_793
