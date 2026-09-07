"""load_eod: the chain cleaning the loader applies to the trip table."""
from pathlib import Path

import pandas as pd
import pytest

from eodgdl import clean_trip_chains, flag_repeated_diaries, load_eod
from eodgdl.eod import FIX_CODES, ISSUE_CODES, _donor_mask, _trip_features, has_code, non_trips

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "IMEPLAN_Base_Viajes_Master.csv").exists()

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
    ])
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert counts["home_to_home"] == 3
    # nothing is dropped: the rows stay, marked, and the model build leaves them out
    assert len(cleaned) == len(trips)
    assert cleaned.loc[(1, 1)].problemas.tolist() == ["regreso_en_casa", "", "", "regreso_en_casa", "regreso_en_casa"]
    assert non_trips(cleaned).sum() == 3
    assert (cleaned.loc[(1, 2)].problemas == "").all() and (cleaned.ajustes == "").all()


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


def test_a_trip_after_a_return_home_starts_at_home():
    trips = _trips([
        (1, 8, 0, "Trabajar"), (1, 17, 0, "Regresar a Casa"),
        (1, 18, 0, "Compras (comida)"), (1, 19, 0, "Regresar a Casa"),
        (2, 8, 0, "Trabajar", "Fábrica o taller", ELSEWHERE),   # a return that did not reach home...
        (2, 17, 0, "Regresar a Casa", "Su casa", ELSEWHERE),
        (2, 18, 0, "Compras (comida)"), (2, 19, 0, "Regresar a Casa"),
    ])
    trips.loc[(1, 1, 3), "origen"] = ELSEWHERE     # carried over from the morning stop
    trips.loc[(1, 2, 3), "origen"] = AGEB          # ...followed by a trip that says it left from home
    cleaned, counts = clean_trip_chains(trips, VIV)
    assert counts["origins_repaired"] == 1
    assert cleaned.loc[(1, 1, 3), "origen"] == AGEB and cleaned.loc[(1, 1, 3), "ajustes"] == "origen:casa"
    # ...leaves the next origin alone, since either side could be wrong, and says so
    assert cleaned.loc[(1, 2, 3), "origen"] == AGEB
    assert cleaned.loc[(1, 2)].problemas.tolist() == ["", "regreso_sin_llegar", "origen_discontinuo", ""]
    assert counts["left_origen_discontinuo"] == 1 and counts["left_regreso_sin_llegar"] == 1


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
    cleaned, counts = clean_trip_chains(trips, VIV)
    hours = cleaned.hora_inicio_h.groupby(level="folio_habitante").apply(list)
    fixes = cleaned.ajustes.groupby(level="folio_habitante").apply(list)
    issues = cleaned.problemas.groupby(level="folio_habitante").apply(list)
    assert hours[1] == [8, 17] and fixes[1] == ["", "hora:+12h"]
    assert hours[2] == [8, 17, 18, 21]
    assert hours[3] == [19, 1] and fixes[3] == ["", ""] and issues[3] == ["", "hora_nocturna"]   # left, and said so
    assert hours[4] == [8, 7, 13, 14] and issues[4] == ["", "hora_invertida", "", ""]
    assert hours[5] == [7, 7, 9, 16] and fixes[5] == ["", "", "hora:-10h", ""]
    assert hours[6] == [14, 13] and issues[6] == ["", "hora_anterior"]
    assert hours[7] == [11, 18, 18, 18] and fixes[7] == ["", "hora:+10h", "", ""]
    assert counts["start_times_edited"] == 6 and counts["chains_repaired"] == 4
    assert counts["left_hora_invertida"] == 1 and counts["left_hora_nocturna"] == 1 and counts["left_hora_anterior"] == 1


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
    trips.loc[(1, 3, 1), "origen"] = ELSEWHERE
    cleaned, counts = clean_trip_chains(trips, VIV)
    issues = cleaned.problemas.groupby(level="folio_habitante").apply(list)
    assert issues[1] == ["", ""]
    assert issues[2] == ["inicio_fuera_de_casa", ""]
    assert issues[3] == ["inicio_zona_ajena", ""]
    assert issues[4] == ["", "", "fin_fuera_de_casa"]
    assert issues[5] == ["actividad_en_casa", "", ""]
    assert issues[6] == ["motivo_guarderia", "hora_repetida"]
    assert issues[7] == ["", "hora_invertida", "", ""]          # one time code per row: the severe one
    assert issues[8] == ["", "hora_traslapada"]                 # within the tolerance: rounding, not repaired
    assert (cleaned.ajustes == "").all() and not non_trips(cleaned).any()
    assert {k: v for k, v in counts.items() if k.startswith("left_") and v} == {
        "left_hora_invertida": 1, "left_hora_repetida": 1, "left_hora_traslapada": 1, "left_inicio_fuera_de_casa": 1,
        "left_inicio_zona_ajena": 1, "left_fin_fuera_de_casa": 1, "left_actividad_en_casa": 1, "left_motivo_guarderia": 1}


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_eod_cleans_the_chains_and_drops_only_the_duplicate_returns():
    viv, hab, trips, legs = load_eod(DATA_DIR)
    assert (len(hab), len(trips), len(legs)) == (58_061, 154_662 - 38, 170_509 - 38)
    assert not trips.hora_inicio_h.isna().any() and not trips.motivo_viaje.isna().any()
    # every change and every defect left is on the row, in the documented vocabulary
    fixes = set(trips.ajustes.str.split(";").explode()) - {""}
    issues = set(trips.problemas.str.split(";").explode()) - {""}
    assert fixes <= set(FIX_CODES) and issues <= set(ISSUE_CODES)
    assert has_code(trips.ajustes, "hora:duplicado").sum() == 37 and has_code(trips.ajustes, "hora:vecinos").sum() == 288
    assert has_code(trips.ajustes, "motivo:vecinos").sum() == 294 and has_code(trips.ajustes, "motivo:duplicado").sum() == 38
    assert non_trips(trips).sum() == 491
    # every defect left is a code on the row, at these counts; a row carries at most one hora_* code
    assert {c: int(has_code(trips.problemas, c).sum()) for c in ISSUE_CODES} == {
        "regreso_en_casa": 491, "hora_invertida": 885, "hora_nocturna": 93,
        "hora_anterior": 25, "hora_repetida": 101, "hora_traslapada": 569, "origen_discontinuo": 32, "regreso_sin_llegar": 163,
        "tipo_destino_dudoso": 16, "inicio_fuera_de_casa": 906, "inicio_zona_ajena": 639,
        "fin_fuera_de_casa": 514, "actividad_en_casa": 787, "motivo_guarderia": 217}
    hora = sum(has_code(trips.problemas, c)
               for c in ("hora_invertida", "hora_nocturna", "hora_anterior", "hora_repetida", "hora_traslapada"))
    assert (hora <= 1).all() and (trips.problemas != "").sum() == 4_660 + 491
    # hab and trips stay in step: viajes_contados follows the 38 dropped duplicates and the legs follow the trips
    counted = trips.groupby(level=PERSON).size()
    assert (hab.viajes_contados == counted.reindex(hab.index).fillna(0)).all()
    assert hab.viajes_contados.sum() == load_eod(DATA_DIR, clean_chains=False).hab.viajes_contados.sum() - 38
    assert legs.index.droplevel("folio_traslado").isin(trips.index).all()
    # the categoricals survive the imputation, with the imputed values inside their levels
    assert str(trips.motivo_viaje.dtype) == "category" and str(trips.tipo_lugar_destino.dtype) == "category"


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_eod_can_return_the_survey_as_shipped():
    viv, hab, trips, legs = load_eod(DATA_DIR, clean_chains=False)
    assert (len(hab), len(trips), len(legs)) == (58_061, 154_662, 170_509)
    assert trips.ponderador.sum() == 11_796_386        # the published trip total
    assert trips.hora_inicio_h.isna().sum() == 325
    assert "ajustes" not in trips.columns and not non_trips(trips).any()


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


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_eod_flags_repeated_diaries_and_drops_none():
    viv, hab, trips, legs = load_eod(DATA_DIR, clean_chains=False)
    assert hab.diario_repetido.dtype == bool
    assert hab.diario_repetido.sum() == 1_793
    assert len(hab) == 58_061 and len(trips) == 154_662
    # the twins that surfaced the pattern: 879/3 repeats 9530/3 whole; 879/7 adds two trips to 9560/4's diary
    assert hab.loc[(9530, 3), "diario_repetido"] and hab.loc[(879, 3), "diario_repetido"]
    assert not hab.loc[(9560, 4), "diario_repetido"] and not hab.loc[(879, 7), "diario_repetido"]
    # the same rule reads the legs table once the traslado columns are gone
    assert flag_repeated_diaries(trips, hab, viv, legs).equals(hab.diario_repetido)
    # the flag is set on the survey as shipped and survives the chain rules
    cleaned = load_eod(DATA_DIR)
    assert cleaned.hab.diario_repetido.sum() == 1_793
