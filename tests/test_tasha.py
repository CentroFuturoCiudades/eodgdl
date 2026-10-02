"""Model output schema: the mappings must stay in step with the contract."""
import copy
from pathlib import Path

import pandas as pd
import pytest

import eodgdl
from eodgdl.chains import PERSON, days_past_midnight, has_code, mark_issues, non_trips
from eodgdl import tasha
from eodgdl.eod import EODTables
from eodgdl.tasha import _schema
from eodgdl._resources import taz_ids
from eodgdl.taz import ACCESS_POINTS


ZONE = 412                   # a TAZ (eodgdl/taz_ids.csv): every zone column holds one
OTHER_ZONE = 1873


def test_mappings_agree_with_schema():
    # Every column mapped, every code produced legal under model_schema.yaml.
    assert tasha.check_mappings() == []


def test_mappings_agree_with_the_survey():
    # Every `source` is a real eodgdl column and every lookup key a real answer.
    survey = tasha.survey_columns()
    assert {"modo_principal", "ageb", "traslado_min", "folio_vivienda"} <= set(survey)
    assert "Acompañante" in survey["rol_en_vehiculo"]["trips"]


def test_check_catches_a_source_that_no_longer_exists(monkeypatch):
    broken = copy.deepcopy(_schema.load_mappings())
    broken["trips"]["Mode"]["source"] = "modo_principial"
    monkeypatch.setattr(_schema, "load_mappings", lambda: broken)
    assert any("not a column of" in p for p in tasha.check_mappings())


def test_check_catches_a_lookup_key_the_survey_cannot_answer(monkeypatch):
    # A typo'd Spanish answer maps to NaN silently; the checker must say so.
    broken = copy.deepcopy(_schema.load_mappings())
    broken["trips"]["PurposeDestination"]["values"]["Estudier"] = "S"
    monkeypatch.setattr(_schema, "load_mappings", lambda: broken)
    assert any("has no answer 'Estudier'" in p for p in tasha.check_mappings())


def test_check_reports_an_imputed_lookup_without_its_pipeline(monkeypatch):
    # the entry format before the pipeline tasha (a chain's name) is a problem the checker names, not a crash
    broken = copy.deepcopy(_schema.load_mappings())
    broken["households"]["IncomeClass"]["imputed"] = {"chain": "nse", "column": "nse_calibrado"}
    monkeypatch.setattr(_schema, "load_mappings", lambda: broken)
    assert any("IncomeClass: imputed names ['chain', 'column']: it needs a pipeline and a column" in p for p in tasha.check_mappings())


def test_schema_bundled():
    # Both YAML files ship in the package and load without network access.
    assert set(tasha.tables()) == {"households", "people", "trips"}
    assert tasha.domain("Sex") == {"M": "Male", "F": "Female"}


def test_build_map():
    modes = tasha.build_map("Mode")
    assert modes["A PIE"] == "W"
    assert modes["CAMIÓN O AUTOBÚS"] == "B"
    # The passenger rule is an override, not part of the literal lookup.
    assert "P" not in set(modes.values())
    assert tasha.mapping("Mode")["override"]["values"] == {"Acompañante": "P"}


def test_ambiguous_column_needs_a_table():
    with pytest.raises(KeyError):
        tasha.find_table("HouseholdId")
    assert tasha.column_spec("HouseholdId", "people")["role"] == "key"


def test_gaps_are_declared():
    gaps = tasha.gaps()
    assert set(gaps.status) <= {"imputed", "assumed", "not_surveyed", "pending"}
    assert {"IncomeClass", "EmploymentStatus", "Formality", "Occupation"} <= set(gaps.column[gaps.status == "imputed"])
    assert "DwellingType" in set(gaps.column)


def test_validate_catches_bad_tables():
    trips = pd.DataFrame({
        "HouseholdId": [0], "PersonNumber": [1], "TripNumber": [2],
        "StartTime": [800], "Mode": ["9"], "PurposeOrigin": ["R"],
        "ZoneOrigin": [-3], "PurposeDestination": ["H"],
        "ZoneDestination": [ZONE], "Junk": [1],
    })
    problems = " | ".join(tasha.validate(trips, "trips"))
    assert "Mode" in problems                 # 9 is not a legal mode
    assert "Junk" in problems                 # column not in the schema
    assert "ZoneOrigin" in problems           # -3 is no TAZ
    assert "PurposeOrigin is R or C" in problems
    assert "first TripNumber is not 1" in problems


def test_start_times_run_past_midnight_but_not_past_a_second_one():
    trips = pd.DataFrame({
        "HouseholdId": [0] * 4, "PersonNumber": [1] * 4, "TripNumber": [1, 2, 3, 4],
        "StartTime": [2100, 3000, 4759, 2475], "Mode": ["W"] * 4, "PurposeOrigin": ["H", "W", "H", "W"],
        "ZoneOrigin": [ZONE] * 4, "PurposeDestination": ["W", "H", "W", "H"], "ZoneDestination": [ZONE] * 4,
    })
    problems = tasha.validate(trips, "trips")                 # 30:00 and 47:59 are hours of a diary past midnight...
    assert problems == ["trips: 1 StartTime values whose last two digits are not minutes 00-59"]   # ...24:75 is no time
    trips.loc[3, "StartTime"] = 4800                          # a second midnight is past the range
    assert tasha.validate(trips, "trips") == ["trips.StartTime: 1 values above 4759"]


def test_validate_rejects_home_to_home_trips():
    # A return home made from home is not a trip; the builder must drop it.
    trips = pd.DataFrame({
        "HouseholdId": [0, 0], "PersonNumber": [1, 1], "TripNumber": [1, 2],
        "StartTime": [800, 900], "Mode": ["W", "W"], "PurposeOrigin": ["H", "H"],
        "ZoneOrigin": [ZONE, ZONE], "PurposeDestination": ["H", "M"],
        "ZoneDestination": [ZONE, ZONE],
    })
    assert any("from H to H" in p for p in tasha.validate(trips, "trips"))


def test_chain_report_counts_what_validate_cannot_demand():
    other = OTHER_ZONE
    trips = pd.DataFrame({
        "HouseholdId": [0, 0, 0], "PersonNumber": [1, 1, 1], "TripNumber": [1, 2, 3],
        "StartTime": [800, 730, 730], "Mode": ["W", "W", "W"],
        "PurposeOrigin": ["H", "W", "M"], "ZoneOrigin": [ZONE, other, ZONE],
        "PurposeDestination": ["W", "M", "E"], "ZoneDestination": [other, ZONE, other],
    })
    assert tasha.validate(trips, "trips") == []          # conforms...
    report = " | ".join(tasha.chain_report(trips))       # ...but is not clean
    assert "1 trips (1 people) start earlier" in report   # 730 after 800
    assert "1 trips start at the same minute" in report
    assert "do not start in the zone" not in report      # zones do chain
    assert "1 people whose last trip does not end at home" in report
    trips.loc[1, "ZoneOrigin"] = ZONE                     # now trip 2 starts elsewhere
    assert any("do not start in the zone" in p for p in tasha.chain_report(trips))


def test_validate_accepts_a_clean_table():
    trips = pd.DataFrame({
        "HouseholdId": [0], "PersonNumber": [1], "TripNumber": [1],
        "StartTime": [800], "Mode": ["W"], "PurposeOrigin": ["H"],
        "ZoneOrigin": [OTHER_ZONE], "PurposeDestination": ["W"],
        "ZoneDestination": [ZONE],
    })
    assert tasha.validate(trips, "trips") == []


def test_zone_ids_must_be_taz_integers():
    # A zone column holds the integer TAZ; an AGEB code left as text, or 0 outside a sentinel column, is caught.
    trips = pd.DataFrame({
        "HouseholdId": [0], "PersonNumber": [1], "TripNumber": [1],
        "StartTime": [800], "Mode": ["W"], "PurposeOrigin": ["H"],
        "ZoneOrigin": ["1409700251418"], "PurposeDestination": ["W"],
        "ZoneDestination": [0],
    })
    problems = " | ".join(tasha.validate(trips, "trips"))
    assert "ZoneOrigin: expected integers" in problems
    assert "ZoneDestination: zone ids that are not positive TAZ integers, e.g. [0]" in problems
    assert tasha.zone_columns("trips") == ["ZoneOrigin", "ZoneDestination"]


HOUSEHOLD = {
    "HouseholdZone": ZONE, "NumberOfPersons": 2, "DwellingType": 1,
    "Vehicles": 0, "IncomeClass": 7, "ExpansionFactor": 1.0,
}
TRIP = {
    "StartTime": 800, "Mode": "W", "PurposeOrigin": "H", "ZoneOrigin": ZONE,
    "PurposeDestination": "W", "ZoneDestination": ZONE,
}


def test_household_ids_must_be_dense_and_zero_based():
    h = pd.DataFrame([{"HouseholdId": i, **HOUSEHOLD} for i in (0, 1, 5)])
    assert any("dense and 0-based" in p for p in tasha.validate(h, "households"))
    h.HouseholdId = [0, 1, 2]
    assert tasha.validate(h, "households") == []


def test_trip_numbers_must_be_consecutive():
    # A chain of 1, 2, 4 starts at 1 and has no duplicates, but skips 3.
    t = pd.DataFrame([{"HouseholdId": 0, "PersonNumber": 1, "TripNumber": n, **TRIP}
                      for n in (1, 2, 4)])
    assert any("not consecutive" in p for p in tasha.validate(t, "trips"))
    t.TripNumber = [1, 2, 3]
    assert tasha.validate(t, "trips") == []


def test_number_of_persons_may_exceed_but_not_undercount_the_person_rows():
    h = pd.DataFrame([{"HouseholdId": 0, **HOUSEHOLD}])
    p = pd.DataFrame([
        {"HouseholdId": 0, "PersonNumber": n, "Age": 30, "Sex": "M", "License": "Y",
         "TransitPass": "N", "EmploymentStatus": "O", "Formality": "O",
         "Occupation": "O", "FreeParking": "O", "StudentStatus": "O",
         "EmploymentZone": 0, "SchoolZone": 0, "ExpansionFactor": 1.0}
        for n in (1, 2, 3)
    ])
    # 3 person rows against a reported size of 2 is a real contradiction...
    assert any("below the number of rows" in q for q in tasha.validate_all(h, p))
    # ...but a size above the row count is normal: the EOD interviews ages 6+.
    h.NumberOfPersons = 5
    assert tasha.validate_all(h, p) == []


def test_purpose_origin_is_spelled_correctly():
    # The downstream model spells it "PuposeOrigin"; eodgdl never does.
    assert "PurposeOrigin" in tasha.columns("trips")
    text = "".join(
        (Path(__file__).resolve().parent.parent / "src/eodgdl/tasha" / name).read_text(
            encoding="utf-8"
        )
        for name in ("build.py", "_schema.py", "mappings.yaml")
    )
    assert "PuposeOrigin" not in text


def test_build_refuses_the_survey_as_shipped(stages):
    with pytest.raises(ValueError, match="no start time"):
        eodgdl.tasha.build(stages.shipped)


def test_build_conforms(stages):
    trips = stages.revised.trips
    od = eodgdl.tasha.build(stages.revised)

    assert tasha.validate_all(*od) == []
    assert len(od.households) == 17_901
    # every person, and every trip load_eod returns but the non-trips (tests/test_review.py pins the counts)
    assert len(od.people) == 58_061
    assert len(od.trips) == len(trips) - int(non_trips(trips).sum())

    # Every zone column is the TAZ of the survey's code (eodgdl/taz_ids.csv); 0 only for no work or school zone.
    taz = taz_ids().set_index("AGEB").TAZ.astype(int)
    assert (od.households.HouseholdZone.to_numpy() == stages.revised.viv.ageb.astype(str).map(taz).to_numpy()).all()
    kept = trips[~non_trips(trips)].sort_index()
    assert (od.trips.ZoneOrigin.to_numpy() == kept.origen.astype(str).map(taz).to_numpy()).all()
    assert (od.trips.ZoneDestination.to_numpy() == kept.destino.astype(str).map(taz).to_numpy()).all()
    assert od.trips.ZoneDestination.isin(taz[list(ACCESS_POINTS)]).any()
    assert set(od.people.EmploymentZone) - {0} <= set(taz) and (od.people.EmploymentZone == 0).any()
    # Renumbered over load_eod's gaps, and never from H to H.
    assert (od.trips.groupby(["HouseholdId", "PersonNumber"]).TripNumber.min() == 1).all()
    assert not ((od.trips.PurposeOrigin == "H") & (od.trips.PurposeDestination == "H")).any()

    # What load_eod's cleaning does not repair is reported: no zone break, and no trip that starts before
    # the one it follows, since a day that passes midnight counts its hours on past 2400 from there
    report = " | ".join(tasha.chain_report(od.trips))
    assert "do not start in the zone" not in report and "start earlier" not in report
    kept = trips[~non_trips(trips)].sort_index()
    next_day = (days_past_midnight(kept) > 0).to_numpy()
    assert (next_day == (od.trips.StartTime >= 2400).to_numpy()).all()
    assert next_day.sum() > has_code(kept.problemas, "hora_nocturna").sum() > 0     # the trip that passes it, and later ones
    assert od.trips.StartTime.max() <= 4759
    # ...and a day's last start comes less than 24 hours after its first
    start = (od.trips.StartTime // 100) * 60 + od.trips.StartTime % 100
    span = start.groupby([od.trips.HouseholdId, od.trips.PersonNumber]).agg(lambda s: s.max() - s.min())
    assert (span < 24 * 60).all()


def test_a_first_trip_that_leaves_another_zone_does_not_start_at_home(stages):
    # the rules start at home most activity days that answer 'Su casa' (origen:respuesta) and give a day that opens
    # with a return from elsewhere another answer (tipo_origen:regreso); the first trips that go to the home zone
    # itself, and hand edits, can still leave 'Su casa' on a first trip from another zone, and read as H it would
    # put home in the wrong zone, or go H to H
    viv, hab, trips, legs = stages.rules
    kept = trips[~non_trips(trips)]
    first = kept.groupby(level=PERSON).head(1)
    home = viv.ageb.astype(str).reindex(first.index.get_level_values(0)).to_numpy()
    away = first.index[first.origen.astype(str).to_numpy() != home].droplevel("folio_viaje")
    assert len(away) == 376
    trips = trips.copy()
    trips.loc[trips.index.droplevel("folio_viaje").isin(away), "tipo_lugar_origen"] = "Su casa"
    od = eodgdl.tasha.build(EODTables(viv, hab, trips, legs))
    assert tasha.validate_all(*od) == []
    first = od.trips[od.trips.TripNumber == 1].merge(od.households[["HouseholdId", "HouseholdZone"]], on="HouseholdId")
    away = first.ZoneOrigin != first.HouseholdZone
    assert not (away & (first.PurposeOrigin == "H")).any() and int(away.sum()) == 376

def test_a_daycare_trip_from_age_12_is_an_escort(stages):
    # the chain rules recode every such trip (motivo:guarderia), so the model build's age rule is seen at work on the
    # rules' output with the survey's motive put back
    viv, hab, trips, legs = stages.rules
    trips = trips.copy()
    trips.loc[has_code(trips.ajustes, "motivo:guarderia").to_numpy(), "motivo_viaje"] = "Guardería"
    tables = EODTables(viv, hab, trips, legs)
    od = eodgdl.tasha.build(tables)
    trips = tables.trips[~non_trips(tables.trips)]
    daycare = (trips.motivo_viaje == "Guardería").to_numpy()
    age = tables.hab.edad.reindex(trips.index.droplevel("folio_viaje")).to_numpy()
    purpose = od.trips.PurposeDestination.to_numpy()
    assert (purpose[daycare & (age >= 12)] == "F").all() and (daycare & (age >= 12)).any()
    assert set(purpose[daycare & (age < 12)]) <= {"S", "C"} and (daycare & (age < 12)).any()
    # an adult's daycare trip does not make them a student
    people = od.people.set_axis(tables.hab.index)            # build_people keeps hab's row order
    persons = trips.index.droplevel("folio_viaje")
    hab = tables.hab
    declared = hab.index[(hab.ocupacion == "Estudiante") | (hab.trabajo_semana_pasada == "Es estudiante")]
    studied = persons[(trips.motivo_viaje == "Estudiar").to_numpy()].unique()
    plain = persons[daycare & (age >= 12)].unique().difference(declared).difference(studied)
    assert len(plain) > 30 and (people.loc[plain, "StudentStatus"] == "O").all()
    # load_eod's tables, and the rules' alone: every daycare trip left is a child's
    for kept in (stages.rules, stages.revised):
        left = kept.trips[kept.trips.motivo_viaje == "Guardería"]
        assert len(left) and (kept.hab.edad.reindex(left.index.droplevel("folio_viaje")) < 12).all()


def test_a_return_that_misses_the_home_zone_is_not_h(stages):
    # the chain rules read every return left away from home (_short_returns), so the model build's rule is seen at
    # work on the rules' output with the returns they sent to 'Otros' put back: the return lands elsewhere, and so
    # does the purpose; the trip after it leaves from O, so a second return from there is a trip home, not H to H
    viv, hab, trips, legs = stages.rules
    trips = trips.copy()
    other = has_code(trips.ajustes, "motivo:otros").to_numpy()
    trips.loc[other, "motivo_viaje"] = "Regresar a Casa"
    trips.loc[other, "tipo_lugar_destino"] = "Su casa"
    trips["problemas"] = mark_issues(trips, viv, legs)
    tables = EODTables(viv, hab, trips, legs)
    od = eodgdl.tasha.build(tables)
    trips = tables.trips[~non_trips(tables.trips)]
    missed = has_code(trips.problemas, "regreso_sin_llegar").to_numpy()
    assert missed.sum() == 21 and (od.trips.PurposeDestination.to_numpy()[missed] == "O").all()
    after = pd.Series(missed, index=trips.index).groupby(level=["folio_vivienda", "folio_habitante"]).shift(1)
    assert (od.trips.PurposeOrigin.to_numpy()[after.fillna(False).to_numpy(bool)] == "O").all()
    assert not ((od.trips.PurposeOrigin == "H") & (od.trips.PurposeDestination == "H")).any()


def test_no_trip_starts_before_the_previous_one_arrives(stages):
    # TASHA cannot schedule an overlap: on load_eod's tables every trip starts once the previous one has arrived
    od = eodgdl.tasha.build(stages.revised).trips
    minutes = (od.StartTime // 100) * 60 + od.StartTime % 100
    arrival = (minutes + od.Duration).groupby([od.HouseholdId, od.PersonNumber]).shift(1)
    assert not (minutes < arrival).any()
    assert not any("could have arrived" in line for line in tasha.chain_report(od))


def test_build_round_trips_through_csv(stages, tmp_path):
    od = eodgdl.tasha.build(stages.revised)
    path = tmp_path / "od_households.csv"
    od.households.to_csv(path, index=False)

    back = pd.read_csv(path)
    assert tasha.validate(back, "households") == []
    assert back.HouseholdZone.equals(od.households.HouseholdZone)


def test_imputed_columns_read_the_completed_dataset(stages):
    # IncomeClass is the dwelling's AMAI level; a person without an employment answer takes the drawn situation; a
    # worker is P where the draw is informal, with Formality I; a worker without a reported giro takes the drawn one;
    # an unanswered person drawn as a student is S. Without a completed dataset they fall to the mapping's defaults.
    viv, hab = stages.revised.viv, stages.revised.hab
    reported = hab.trabajo_semana_pasada.isin(["Tiempo completo", "Medio tiempo", "Tenía trabajo, pero no trabajó"])
    unanswered = hab.trabajo_semana_pasada.isna()
    levels = ["e", "d", "d_mas", "c_menos", "c", "c_mas", "ab"]
    nse = pd.DataFrame({"nse_calibrado": [levels[i % 7] for i in range(len(viv))]}, index=viv.index)
    rows = hab.index[reported | unanswered]
    n = len(rows)
    situation = pd.Series("trabaja", index=rows)
    drawn = situation.index.isin(hab.index[unanswered])
    situation[drawn] = (["trabaja", "estudiante", "hogar"] * n)[: int(drawn.sum())]
    labour = pd.DataFrame({"situacion_laboral": situation, "giro": "educacion",
                           "informalidad": (["informal", "formal"] * n)[:n]}, index=rows)
    od = eodgdl.tasha.build(stages.revised, completed={"tasha": {"dwellings": nse, "persons": labour}})
    assert tasha.validate_all(*od) == []
    expected = pd.Series(nse.nse_calibrado.to_numpy()).map(tasha.mapping("IncomeClass")["values"])
    assert (od.households.IncomeClass.to_numpy() == expected.to_numpy()).all() and 7 not in set(od.households.IncomeClass)
    p = od.people.set_index(hab.index)
    workers = situation[situation == "trabaja"].index
    assert (p.loc[workers, "EmploymentStatus"] == labour.loc[workers, "informalidad"].map({"informal": "P", "formal": "F"})).all()
    assert ((p.EmploymentStatus == "P") == (p.Formality == "I")).all() and (p.Formality[p.EmploymentStatus == "O"] == "O").all()
    others = situation.index[drawn & (situation != "trabaja").to_numpy()]
    assert (p.loc[others, "EmploymentStatus"] == "O").all() and (p.loc[others, "Occupation"] == "O").all()
    students = situation.index[drawn & (situation == "estudiante").to_numpy()]
    assert len(students) and (p.loc[students, "StudentStatus"] == "S").all()
    assert (p.loc[hab.index[reported & hab.giro_empresa.isna()], "Occupation"] == "G").all()     # educacion -> G
    assert (p.loc[hab.index[reported & hab.giro_empresa.eq("Industria")], "Occupation"] == "M").all()   # the answer wins

    # every row that reads a drawn value must have one: completed datasets of other rows (a stale or partial run) fail
    worker = hab.index[reported][0]
    for dwellings, persons, column in ((nse.iloc[1:], labour, "nse_calibrado"), (nse, labour.drop(index=hab.index[unanswered][0]), "situacion_laboral"),
                                       (nse, labour.drop(index=worker), "informalidad")):
        with pytest.raises(ValueError, match=rf"tasha.{column}: no completed value for 1 rows that read one"):
            eodgdl.tasha.build(stages.revised, completed={"tasha": {"dwellings": dwellings, "persons": persons}})

    plain = eodgdl.tasha.build(stages.revised)
    assert (plain.households.IncomeClass == 7).all() and not (plain.people.EmploymentStatus == "P").any()
    assert (plain.people.Formality == "O").all() and tasha.validate_all(*plain) == []
    assert (plain.people.EmploymentStatus[unanswered.to_numpy()] == "O").all()


def test_load_completed_reads_one_draw_of_every_level(tmp_path, monkeypatch):
    # <root>/tasha/<level>/completions.parquet, the draw asked for, indexed by the level's keys; a draw outside the
    # pipeline's fails, and so does a run with its aggregates off or drawn on other data than the source reads now
    import json

    from eodgdl.impute import sources

    monkeypatch.setitem(sources._SOURCES, "test.level", sources.Source("test.level", None, {"keys": ["folio_vivienda"], "weight": "w", "group": "g"},
                                                                       lambda context, config: {"data": "v1", "eodgdl": "9.9"}, lambda column: [], __file__))
    folder = tmp_path / "tasha"
    persons = pd.DataFrame({"folio_vivienda": [1, 1, 2, 1, 1, 2], "folio_habitante": [1, 2, 1, 1, 2, 1], "completion": [0, 0, 0, 1, 1, 1],
                            "weight": 0.5, "situacion_laboral": ["trabaja", "hogar", "trabaja", "trabaja", "trabaja", "estudiante"]})
    dwellings = pd.DataFrame({"folio_vivienda": [1, 2, 1, 2], "completion": [0, 0, 1, 1], "weight": 0.5, "nse_calibrado": ["c", "d", "c_mas", "d"]})
    for level, frame in (("persons", persons), ("dwellings", dwellings)):
        (folder / level).mkdir(parents=True)
        frame.to_parquet(folder / level / "completions.parquet", index=False)
    level = {"source": "test.level", "versions": {"data": "v1", "eodgdl": "0.1", "features": {}}}      # the eodgdl version aside
    provenance = {"pipeline": "tasha", "draws": 2, "aggregate": True, "levels": {"persons": level, "dwellings": level}}
    (folder / "provenance.json").write_text(json.dumps(provenance))
    completed = tasha.load_completed(tmp_path, draw=1)
    assert set(completed) == {"tasha"} and set(completed["tasha"]) == {"persons", "dwellings"}
    assert completed["tasha"]["persons"]["situacion_laboral"].to_dict() == {(1, 1): "trabaja", (1, 2): "trabaja", (2, 1): "estudiante"}
    assert completed["tasha"]["dwellings"]["nse_calibrado"].to_dict() == {1: "c_mas", 2: "d"}
    assert completed["tasha"]["dwellings"].attrs["draw"] == 1
    with pytest.raises(ValueError, match="outside its 2 completed datasets"):
        tasha.load_completed(tmp_path, draw=2)
    for changed, message in (({"aggregate": False}, r"aggregates switched off \(--no-aggregate"),
                             ({"levels": {"persons": level, "dwellings": {**level, "versions": {"data": "v0"}}}}, r"other data than the survey's now \(dwellings: data changed\)"),
                             ({"levels": {"persons": {}, "dwellings": level}}, r"\(persons: no data versions recorded\)")):
        (folder / "provenance.json").write_text(json.dumps({**provenance, **changed}))
        with pytest.raises(ValueError, match=message):
            tasha.load_completed(tmp_path, draw=0)
