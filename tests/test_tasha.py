"""Model output schema: the mappings must stay in step with the contract."""
import copy
from pathlib import Path

import pandas as pd
import pytest

import eodgdl
from eodgdl import tasha
from eodgdl.tasha import _schema

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "IMEPLAN_Base_Viajes_Master.csv").exists()

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


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_build_refuses_the_survey_as_shipped():
    with pytest.raises(ValueError, match="no start time"):
        eodgdl.tasha.build(eodgdl.load_eod(DATA_DIR, clean_chains=False))


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_build_conforms():
    od = eodgdl.tasha.build(eodgdl.load_eod(DATA_DIR))

    assert tasha.validate_all(*od) == []
    assert len(od.households) == 17_901
    # load_eod excluded the 281 people whose day has an untimed trip.
    assert len(od.people) == 58_061 - 281
    assert len(od.trips) == 153_052

    # Zone ids come through as the survey's own codes, not a renumbering.
    assert od.households.HouseholdZone.str.len().isin([9, 13]).all()
    # Renumbered over load_eod's gaps, and never from H to H.
    assert (od.trips.groupby(["HouseholdId", "PersonNumber"]).TripNumber.min() == 1).all()
    assert not ((od.trips.PurposeOrigin == "H") & (od.trips.PurposeDestination == "H")).any()

    # What load_eod's cleaning does not repair is reported, at these levels.
    report = " | ".join(tasha.chain_report(od.trips))
    assert "152 trips (150 people) do not start in the zone" in report
    assert "1411 trips (1322 people) start earlier" in report


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_build_round_trips_through_csv(tmp_path):
    od = eodgdl.tasha.build(eodgdl.load_eod(DATA_DIR))
    path = tmp_path / "od_households.csv"
    od.households.to_csv(path, index=False)

    back = pd.read_csv(path, dtype=dict.fromkeys(tasha.zone_columns("households"), str))
    assert tasha.validate(back, "households") == []
    assert back.HouseholdZone.equals(od.households.HouseholdZone)
