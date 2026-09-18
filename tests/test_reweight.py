"""Inputs for TMG.SurveyReweight: the spec, the records, the targets, the written set."""
import copy
from pathlib import Path

import pandas as pd
import pytest

import eodgdl
from eodgdl import reweight
from eodgdl.reweight import _spec

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "IMEPLAN_Base_Viajes_Master.csv").exists()

SCHEMAS = {
    "households": eodgdl.viv_schema,
    "persons": eodgdl.hab_schema,
    "trips": eodgdl.trips_schema,
}


def test_spec_is_sound():
    assert reweight.check_spec() == []


def test_spec_names_real_survey_answers():
    # Every plain source is a column of its table and every lookup key an answer it has.
    for table, schema in SCHEMAS.items():
        for name, entry in reweight.attributes(table).items():
            source = entry.get("source")
            if not isinstance(source, str) or "." in source:
                continue
            assert source in schema.columns, f"{table}.{name}: {source}"
            dtype = schema.columns[source].dtype.type
            if "values" in entry and isinstance(dtype, pd.CategoricalDtype):
                unknown = set(entry["values"]) - set(dtype.categories)
                assert not unknown, f"{table}.{name}: {unknown}"


def test_check_spec_catches_a_constraint_on_a_missing_attribute(monkeypatch):
    broken = copy.deepcopy(_spec.load_spec())
    broken["constraints"]["persons"]["Male"]["attributes"] = ["Mail"]
    monkeypatch.setattr(_spec, "load_spec", lambda: broken)
    assert any("'Mail'" in p for p in reweight.check_spec())


def test_constraint_index_places_every_target_once():
    idx = reweight.constraint_index(2020)
    assert not idx.duplicated(["file", "target_column"]).any()
    assert (idx.groupby("file").column_index.min() == 1).all()
    assert set(idx.map_column) == {"Zone", "Municipality", "Region"}


@pytest.fixture(scope="module")
def files():
    if not HAS_DATA:
        pytest.skip("in-repo data/ not present")
    return reweight.build(eodgdl.load_eod(DATA_DIR), data_dir=DATA_DIR)


def test_records_have_the_survey_s_rows_and_integer_keys(files):
    hh, pp, tt = files.households, files.persons, files.trips
    assert len(hh) == 17_901
    assert len(pp) == 58_061
    assert len(tt) == 154_662 - 38 - 491  # load_eod drops the duplicates, build leaves out the non-trips
    for df, key in ((hh, "HouseholdID"), (pp, "PersonID")):
        assert pd.api.types.is_integer_dtype(df[key]) and not df[key].duplicated().any()
    assert tt.PersonID.isin(pp.PersonID).all()
    assert pp.HouseholdID.isin(hh.HouseholdID).all()


def test_employed_is_tasha_s_worker(files):
    # One worker definition across the package: tasha's EmploymentStatus F or P is
    # reweight's Employed, person by person (both read trabajo_semana_pasada).
    from eodgdl import tasha

    tables = eodgdl.load_eod(DATA_DIR)
    people = tasha.build_people(tables.hab, tables.trips, tables.viv)
    worker = people.EmploymentStatus.isin(["F", "P"]).to_numpy()
    assert (worker == (files.persons.Employed.to_numpy() == 1)).all()
    assert (files.persons.Employed + files.persons.NotEmployed == (files.persons.Age6_11 == 0)).all()


def test_zone_system_is_the_cells_of_the_survey(files):
    zs = files.zone_system
    assert len(zs) == 90
    assert not zs.TAZ.duplicated().any()
    assert set(zs.Municipality) == set(eodgdl.load_zm_muns())
    assert zs.Zone.between(1, 71).all() and (zs.Region == 1).all()
    assert (zs.TAZ == zs.Zone * 1000 + zs.Municipality).all()
    assert files.households.HouseholdTAZ.isin(zs.TAZ).all()
    assert files.zone_labels.sampled_dwellings.sum() == len(files.households)


def test_dummies_partition(files):
    pp, tt = files.persons, files.trips
    ages = ["Age6_11", "Age12_14", "Age15_17", "Age18_24", "Age25_59", "Age60p"]
    assert (pp.Male + pp.Female == pp.Persons).all()
    assert (pp[ages].sum(axis=1) == pp.Persons).all()
    assert (pp.Employed + pp.Unemployed + pp.Inactive <= pp.Persons).all()
    assert ((pp.Employed + pp.NotEmployed) == (pp.Age6_11 == 0).astype(float)).all()  # partitions the 12+
    assert pp.Cyclist.sum() == 561
    boardings = ["BusBoardings", "RailBoardings", "BRTBoardings", "SitrenBoardings", "OtherTransitBoardings"]
    assert (tt[boardings].sum(axis=1) == tt.TransitBoardings).all()
    assert tt.BicycleTrip.isin([0, 1]).all()


def test_targets_are_consistent_and_pinned(files):
    c = files.constraints
    hh20 = c["HouseholdConstraintsByZone_2020.csv"]
    pp20 = c["PersonConstraintsByZone_2020.csv"]
    mun20 = c["PersonConstraintsByMunicipality_2020.csv"]
    assert len(hh20) == len(pp20) == 64 and len(mun20) == 9
    assert round(hh20.Dwellings.sum()) == 1_454_608
    assert round(pp20.Persons.sum()) == 4_635_931
    assert round(mun20.Cyclist.sum()) == 100_453  # 101,907 whole-municipality cyclists, universe share applied
    assert c["TripConstraintsByRegion.csv"].BusBoardings.iloc[0] == 1_851_750
    ages = ["Age6_11", "Age12_14", "Age15_17", "Age18_24", "Age25_59", "Age60p"]
    assert (pp20[ages].sum(axis=1) - pp20.Persons).abs().max() < 0.5
    # INEGI suppresses some sex-split cells but not the totals, so the sexed targets fall
    # short of the unsexed ones by a few dozen persons in a handful of zones (427 in all).
    assert (pp20.Male + pp20.Female - pp20.Persons).abs().max() < 50
    assert abs((pp20.Male + pp20.Female).sum() - pp20.Persons.sum()) < 500
    for band in ages:
        cross = pp20[f"Male_{band}"] + pp20[f"Female_{band}"]
        assert (cross - pp20[band]).abs().max() < 50
    # The 2023 set is the 2020 set scaled by CONAPO's municipal growth: the study area grows,
    # Guadalajara's zones shrink (CONAPO has the municipality losing population), and the
    # older bands grow faster than the younger ones.
    hh23 = c["HouseholdConstraintsByZone_2023.csv"]
    pp23 = c["PersonConstraintsByZone_2023.csv"]
    assert hh23.Dwellings.sum() > hh20.Dwellings.sum()
    assert pp23.Persons.sum() > pp20.Persons.sum()
    guadalajara_only = files.zone_labels.groupby("centralidad").CVE_MUN.agg(set).eq({39})
    zone_of = dict(zip(files.zone_labels.centralidad, files.zone_labels.TAZ // 1000))
    shrinking = [zone_of[z] for z, only in guadalajara_only.items() if only]
    assert (hh23.set_index("Zone").Dwellings.loc[shrinking] < hh20.set_index("Zone").Dwellings.loc[shrinking]).all()
    assert (pp23.Age60p / pp20.Age60p).mean() > (pp23.Age6_11 / pp20.Age6_11).mean()


def test_census_reconciles_with_imeplan_s_crosswalk(files):
    # mxcensus is the source of every value; IMEPLAN's table only says which rows make a zone.
    universe = reweight.census_universe(DATA_DIR)
    assert reweight.reconcile(universe, data_dir=DATA_DIR) == []
    assert len(universe) == 2154 and set(universe.Zone) == set(files.zone_system.Zone) - set()  # 64 zones
    assert universe.Zone.nunique() == 64


def test_coverage_says_which_municipalities_are_whole(files):
    share = files.coverage.POBTOT_share
    assert share.loc[39] > 0.999 and share.loc[98] > 0.998  # Guadalajara, Tlaquepaque
    assert 0.3 < share.loc[51] < 0.35  # Juanacatlán: a third
    assert (share <= 1.0001).all()


def test_written_set_checks_clean_and_check_catches_breakage(files, tmp_path):
    reweight.write(files, tmp_path)
    assert reweight.check(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["ZoneSystem.csv", "ZoneLabels.csv", "HouseholdRecords.csv", "PersonRecords.csv",
         "TripRecords.csv", "Constraints", "README.md"])

    # A geography missing from a constraint file: the tool would target 0 there.
    path = tmp_path / "Constraints" / "PersonConstraintsByMunicipality_2020.csv"
    pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    assert any("no row for Municipality" in p for p in reweight.check(tmp_path))

    # A person record with an empty cell: the tool cannot parse it.
    reweight.write(files, tmp_path)
    path = tmp_path / "PersonRecords.csv"
    df = pd.read_csv(path)
    df.loc[0, "Male"] = None
    df.to_csv(path, index=False)
    assert any("empty cells" in p for p in reweight.check(tmp_path))


def test_diagnostic_reads_the_shipped_weights_against_the_targets(files):
    d = files.diagnostics[2020]
    dwellings = d[d.target_column == "Dwellings"]
    assert round(dwellings.survey_at_design_weight.sum()) == 1_476_347
    assert (dwellings.ratio - 1).abs().median() < 0.08  # the report's 6.5 % median deviation
    assert d.loc[d.target_column == "BusBoardings", "ratio"].iloc[0] > 2


def test_census_cyclists_reproduce_the_spec_constants():
    mxcensus = pytest.importorskip("mxcensus")
    entry = reweight.constraints("persons")["Cyclist"]
    p = mxcensus.load_extended_personas(state=14)
    p = p[p.MUN.astype(int).isin(eodgdl.load_zm_muns())]
    bike = (p["MED_TRASLADO_TRAB_Bicicleta"] == 1) | (p["MED_TRASLADO_ESC_Bicicleta"] == 1)
    counts = p.FACTOR.astype(float).where(bike, 0).groupby(p.MUN.astype(int)).sum().round()
    assert counts.to_dict() == {int(k): float(v) for k, v in entry["constant"].items()}
