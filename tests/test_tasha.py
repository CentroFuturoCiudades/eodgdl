"""Model output schema: the mappings must stay in step with the contract."""
from pathlib import Path

import pandas as pd
import pytest

import eodgdl
from eodgdl import tasha

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "IMEPLAN_Base_Viajes_Master.csv").exists()

AGEB = "1409700251418"       # a real 13-character AGEB CVEGEO
LOCALITY = "140390001"       # a real 9-character locality id


def test_mappings_agree_with_schema():
    # Every column mapped, every code produced legal under model_schema.yaml.
    assert tasha.check_mappings() == []


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


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_build_conforms_and_warns():
    with pytest.warns(UserWarning, match="no reported start time"):
        od = eodgdl.tasha.build(eodgdl.load_eod(DATA_DIR))

    assert tasha.validate_all(*od) == []
    assert len(od.households) > 0 and len(od.people) > 0 and len(od.trips) > 0

    # Zone ids come through as the survey's own codes, not a renumbering.
    assert od.households.HouseholdZone.str.len().isin([9, 13]).all()
    # Dropping untimed trips must leave every chain starting at 1.
    assert (od.trips.groupby(["HouseholdId", "PersonNumber"]).TripNumber.min() == 1).all()


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_build_round_trips_through_csv(tmp_path):
    od = eodgdl.tasha.build(eodgdl.load_eod(DATA_DIR))
    path = tmp_path / "od_households.csv"
    od.households.to_csv(path, index=False)

    back = pd.read_csv(path, dtype=dict.fromkeys(tasha.zone_columns("households"), str))
    assert tasha.validate(back, "households") == []
    assert back.HouseholdZone.equals(od.households.HouseholdZone)
