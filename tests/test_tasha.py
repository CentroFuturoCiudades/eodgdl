"""Model output schema: the mappings must stay in step with the contract."""
import copy
from pathlib import Path

import pandas as pd
import pytest

import eodgdl
from eodgdl.chains import days_past_midnight, has_code, non_trips
from eodgdl import tasha
from eodgdl.tasha import _schema


AGEB = "1409700251418"       # a real 13-character AGEB CVEGEO
LOCALITY = "140390001"       # a real 9-character locality id


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
    assert set(gaps.status) <= {"assumed", "not_surveyed", "pending"}
    assert "DwellingType" in set(gaps.column)


def test_validate_catches_bad_tables():
    trips = pd.DataFrame({
        "HouseholdId": [0], "PersonNumber": [1], "TripNumber": [2],
        "StartTime": [800], "Mode": ["9"], "PurposeOrigin": ["R"],
        "ZoneOrigin": ["14097"], "PurposeDestination": ["H"],
        "ZoneDestination": [AGEB], "Junk": [1],
    })
    problems = " | ".join(tasha.validate(trips, "trips"))
    assert "Mode" in problems                 # 9 is not a legal mode
    assert "Junk" in problems                 # column not in the schema
    assert "ZoneOrigin" in problems           # 14097 is not a 9- or 13-char id
    assert "PurposeOrigin is R or C" in problems
    assert "first TripNumber is not 1" in problems


def test_start_times_run_past_midnight_but_not_past_a_second_one():
    trips = pd.DataFrame({
        "HouseholdId": [0] * 4, "PersonNumber": [1] * 4, "TripNumber": [1, 2, 3, 4],
        "StartTime": [2100, 3000, 4759, 2475], "Mode": ["W"] * 4, "PurposeOrigin": ["H", "W", "H", "W"],
        "ZoneOrigin": [AGEB] * 4, "PurposeDestination": ["W", "H", "W", "H"], "ZoneDestination": [AGEB] * 4,
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
        "ZoneOrigin": [AGEB, AGEB], "PurposeDestination": ["H", "M"],
        "ZoneDestination": [AGEB, AGEB],
    })
    assert any("from H to H" in p for p in tasha.validate(trips, "trips"))


def test_chain_report_counts_what_validate_cannot_demand():
    other = "1409700251419"
    trips = pd.DataFrame({
        "HouseholdId": [0, 0, 0], "PersonNumber": [1, 1, 1], "TripNumber": [1, 2, 3],
        "StartTime": [800, 730, 730], "Mode": ["W", "W", "W"],
        "PurposeOrigin": ["H", "W", "M"], "ZoneOrigin": [AGEB, other, AGEB],
        "PurposeDestination": ["W", "M", "E"], "ZoneDestination": [other, AGEB, other],
    })
    assert tasha.validate(trips, "trips") == []          # conforms...
    report = " | ".join(tasha.chain_report(trips))       # ...but is not clean
    assert "1 trips (1 people) start earlier" in report   # 730 after 800
    assert "1 trips start at the same minute" in report
    assert "do not start in the zone" not in report      # zones do chain
    assert "1 people whose last trip does not end at home" in report
    trips.loc[1, "ZoneOrigin"] = AGEB                     # now trip 2 starts elsewhere
    assert any("do not start in the zone" in p for p in tasha.chain_report(trips))


def test_validate_accepts_a_clean_table():
    trips = pd.DataFrame({
        "HouseholdId": [0], "PersonNumber": [1], "TripNumber": [1],
        "StartTime": [800], "Mode": ["W"], "PurposeOrigin": ["H"],
        "ZoneOrigin": [LOCALITY], "PurposeDestination": ["W"],
        "ZoneDestination": [AGEB],
    })
    assert tasha.validate(trips, "trips") == []


def test_zone_ids_parsed_as_numbers_are_caught():
    # The ids are mostly all-digit, so a naive read_csv can produce int64.
    trips = pd.DataFrame({
        "HouseholdId": [0], "PersonNumber": [1], "TripNumber": [1],
        "StartTime": [800], "Mode": ["W"], "PurposeOrigin": ["H"],
        "ZoneOrigin": [int(AGEB)], "PurposeDestination": ["W"],
        "ZoneDestination": [AGEB],
    })
    assert any("not strings" in p for p in tasha.validate(trips, "trips"))
    assert tasha.zone_columns("trips") == ["ZoneOrigin", "ZoneDestination"]


HOUSEHOLD = {
    "HouseholdZone": AGEB, "NumberOfPersons": 2, "DwellingType": 1,
    "Vehicles": 0, "IncomeClass": 7, "ExpansionFactor": 1.0,
}
TRIP = {
    "StartTime": 800, "Mode": "W", "PurposeOrigin": "H", "ZoneOrigin": AGEB,
    "PurposeDestination": "W", "ZoneDestination": AGEB,
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
         "EmploymentZone": "0", "SchoolZone": "0", "ExpansionFactor": 1.0}
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

    # Zone ids come through as the survey's own codes, not a renumbering.
    assert od.households.HouseholdZone.str.len().isin([9, 13]).all()
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


def test_a_first_trip_that_leaves_another_zone_does_not_start_at_home(stages):
    # the rules alone, before the hand passes: 714 first trips answer 'Su casa' but leave another
    # zone, 75 of them returns home; read as H they would put home in the wrong zone, or go H to H
    od = eodgdl.tasha.build(stages.rules)
    assert tasha.validate_all(*od) == []
    first = od.trips[od.trips.TripNumber == 1].merge(od.households[["HouseholdId", "HouseholdZone"]], on="HouseholdId")
    away = first.ZoneOrigin != first.HouseholdZone
    assert not (away & (first.PurposeOrigin == "H")).any() and int(away.sum()) >= 714

def test_a_daycare_trip_from_age_12_is_an_escort(stages):
    tables = stages.revised
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

def test_build_round_trips_through_csv(stages, tmp_path):
    od = eodgdl.tasha.build(stages.revised)
    path = tmp_path / "od_households.csv"
    od.households.to_csv(path, index=False)

    back = pd.read_csv(path, dtype=dict.fromkeys(tasha.zone_columns("households"), str))
    assert tasha.validate(back, "households") == []
    assert back.HouseholdZone.equals(od.households.HouseholdZone)
