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


@pytest.fixture(scope="module")
def tables():
    if not HAS_DATA:
        pytest.skip("in-repo data/ not present")
    return eodgdl.load_eod(DATA_DIR)


@pytest.fixture(scope="module")
def viv(tables):
    return tables.viv


@pytest.fixture(scope="module")
def trips(tables):
    return tables.trips


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
    assert round(hh20.Dwellings.sum()) == 1_462_391
    assert round(pp20.Persons.sum()) == 4_680_094
    assert round(mun20.Cyclist.sum()) == 100_870  # 101,907 whole-municipality cyclists, universe share applied
    assert c["TripConstraintsByRegion.csv"].BusBoardings.iloc[0] == 1_851_750
    ages = ["Age6_11", "Age12_14", "Age15_17", "Age18_24", "Age25_59", "Age60p"]
    assert (pp20[ages].sum(axis=1) - pp20.Persons).abs().max() < 0.5
    # INEGI suppresses some sex-split cells but not the totals, so the sexed targets fall
    # short of the unsexed ones by up to 41 persons in a zone (469 in all).
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
    # 2023 keeps the census's partitions: bands sum to Persons, each sex's bands to its
    # total, the two activity targets to the 12+ bands; the sexes fall short of the
    # totals only by INEGI's suppressed cells, as in 2020.
    assert round(pp23.Persons.sum()) == 4_850_841
    for y, p in ((2020, pp20), (2023, pp23)):
        assert (p[ages].sum(axis=1) - p.Persons).abs().max() < 1e-6, y
        for sex in ("Male", "Female"):
            assert (p[[f"{sex}_{b}" for b in ages]].sum(axis=1) - p[sex]).abs().max() < 1e-6, (y, sex)
        assert (p.Employed + p.NotEmployed - p[ages[1:]].sum(axis=1)).abs().max() < 1e-6, y
        assert (p.Male + p.Female - p.Persons).abs().max() < 50, y
        for b in ages:
            assert (p[f"Male_{b}"] + p[f"Female_{b}"] - p[b]).abs().max() < 50, (y, b)


def test_census_universe_is_the_assigned_units(files, viv, trips):
    # mxcensus is the source of every value; zoning.assign_units says which rows make a zone.
    universe = reweight.census_universe(viv, trips)
    assert len(universe) == 2308 and set(universe.Zone) == set(files.zone_system.Zone)  # 64 zones
    assert universe.Zone.nunique() == 64


def test_every_census_unit_gets_at_most_one_zone_and_the_survey_s(files, viv, trips):
    a = files.zone_assignment.set_index("CVEGEO")
    assert a.index.is_unique and reweight.check_assignment(a, viv, trips) == []
    assert a.groupby(["unit", "rule"], observed=True).size().to_dict() == {
        ("ageb", "survey"): 993, ("ageb", "survey over polygon"): 26,
        ("ageb", "trip ends"): 616, ("ageb", "trip ends over polygon"): 15,
        ("ageb", "polygon"): 356, ("ageb", "majority"): 20, ("ageb", "nearest polygon"): 13,
        ("locality", "survey"): 212, ("locality", "trip ends"): 48, ("locality", "polygon"): 9,
        ("locality", "outside"): 514, ("rural ageb", "trip ends"): 2,
    }
    # The survey never splits an AGEB, and every sampled one carries the zone it coded.
    code, zone = viv.ageb.astype(str), viv.centralidad.astype(str)
    assert (zone.groupby(code).nunique() == 1).all()
    urban = code.str.len() == 13
    assert (code[urban].map(a.zone) == zone[urban]).all()
    # An unsampled AGEB takes the zone the survey coded most for its trip ends, else it goes
    # whole to the polygon holding most of its population.
    travelled = a[(a.unit == "ageb") & a.survey_zone.isna() & a.trip_zone.notna()]
    assert (travelled.zone == travelled.trip_zone).all()
    rest = a[(a.unit == "ageb") & a.survey_zone.isna() & a.trip_zone.isna() & a.polygon_zone.notna()]
    assert (rest.zone == rest.polygon_zone).all() and (rest.polygon_share > 0.4).all()
    # Every urban AGEB is in a zone; one mostly outside every polygon goes to the polygon
    # holding the most of the rest, or the nearest (Puente Grande's prisons join 45).
    assert a[a.unit == "ageb"].zone.notna().all()
    assert a.loc["1407001161467", ["zone", "rule"]].tolist() == ["45", "nearest polygon"]
    # 141200001098A: a campus the survey codes 68B (968 trip ends), though 82 % of its
    # residents live in 44's polygon.
    assert a.loc["141200001098A", ["zone", "polygon_zone", "rule"]].tolist() == ["68B", "44", "trip ends over polygon"]
    # La Aurora (Juanacatlán) lies outside every polygon; the survey coded its dwellings 49F.
    aurora = a.loc[["1405100020181", "1405100020196", "1405100020209", "1405100020213"]]
    assert (aurora.zone == "49F").all() and aurora.polygon_zone.isna().all()
    # A rural AGEB the survey records sits in the zone it coded, one zone per code, with
    # its localities in the frame (inside a polygon) and those its sample needs.
    locs = a[a.unit == "locality"]
    recorded = locs[locs.rural_ageb.isin(set(viv.ageb.astype(str)) | set(trips.destino.astype(str)))]
    assert (recorded.dropna(subset=["zone"]).groupby("rural_ageb").zone.nunique() == 1).all()
    assert recorded.loc[recorded.polygon_zone.notna(), "zone"].notna().all()
    # 140970123: 80 dwellings coded 51F, its localities inside polygons hold 26; the
    # nearest left-out ones join until they can hold the sample, Cuexcomatitlán among them
    assert a.loc["140970014", ["zone", "rule"]].tolist() == ["51F", "survey"]
    assert a.loc["140970004", "zone"] is pd.NA or pd.isna(a.loc["140970004", "zone"])  # San Juan Evangelista


def test_check_assignment_catches_a_misplaced_rural_ageb(files, viv, trips):
    a = files.zone_assignment.set_index("CVEGEO")
    a.loc[(a.unit == "locality") & (a.rural_ageb == "140970123") & a.zone.notna(), "zone"] = "59"
    problems = reweight.check_assignment(a, viv, trips)
    assert problems and all("rural AGEB 140970123" in p for p in problems)


def test_every_trip_end_is_in_its_ageb_s_zone(files, viv, trips):
    # The survey codes a trip end's zone by where it lies; every AGEB it records is in the
    # zone system, in the zone it coded (a rural AGEB, in a zone holding one of its
    # localities). Only the access points, the airport and six road gateways, are not.
    a = files.zone_assignment.set_index("CVEGEO")
    agreement = reweight.trip_end_agreement(a, viv, trips)
    assert agreement.to_dict() == {"in zone": 305_172, "access point": 3_094}
    # the survey's zone is a function of the AGEB code: one zone per code, across its
    # dwellings, origins and destinations
    codes = reweight.survey_codes(viv, trips)
    assert (codes.groupby("code").zone.nunique() == 1).all() and codes.code.nunique() == 1694
    # the chain rule that moves an origin home moves its zone too
    moved = trips[trips.ajustes.astype(str).str.contains("origen:casa")].reset_index()
    home = viv.centralidad.astype(str)
    assert (moved.zona_origen.astype(str) == moved.folio_vivienda.map(home).to_numpy()).all()


def test_redrawn_zones_follow_the_assignment(files):
    a = files.zone_assignment.set_index("CVEGEO")
    shapes = reweight.zone_shapes(a)
    assert len(shapes) == 64 and shapes.is_valid.all()
    units = reweight.unit_shapes()
    for code in ["1405100020196", "1412000010640"]:  # La Aurora; 93 % in 21A's polygon, surveyed 24B
        inside = shapes.geometry.intersection(units.geometry[code]).area / units.geometry[code].area
        assert inside.idxmax() == a.zone[code] and inside.max() > 0.99


def test_coverage_says_which_municipalities_are_whole(files):
    share = files.coverage.POBTOT_share
    assert share.loc[39] > 0.999 and share.loc[98] > 0.998  # Guadalajara, Tlaquepaque
    assert 0.84 < share.loc[51] < 0.87  # Juanacatlán, with La Aurora in 49F
    assert 0.70 < share.loc[124] < 0.73  # Zapotlanejo: the least covered
    assert (share <= 1.0001).all()


def test_written_set_checks_clean_and_check_catches_breakage(files, tmp_path):
    reweight.write(files, tmp_path)
    assert reweight.check(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["ZoneSystem.csv", "ZoneLabels.csv", "ZoneAssignment.csv", "SampledAGEBs.csv",
         "HouseholdRecords.csv", "PersonRecords.csv", "TripRecords.csv", "Constraints", "README.md"])

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


def test_row_corrections_make_zones_consistent(files, viv, trips):
    # Zone 56 (63F) has 17.5 % of its residents without a stated age (dwellings without
    # occupant information whose details INEGI did not impute), zone 43 (45) a third of
    # its residents in collective quarters (Puente Grande's prisons). Corrected, persons 6+
    # are 85-96 % of private-dwelling occupants in every zone; the survey has ~93 %.
    pp = files.constraints["PersonConstraintsByZone_2020.csv"].set_index("Zone")
    hh = files.constraints["HouseholdConstraintsByZone_2020.csv"].set_index("Zone")
    assert (pp.Persons / hh.Occupants).between(0.8, 1.0).all()
    assert (hh[["HasCar", "HasMoto", "HasBike", "HasInternet"]].max(axis=1) <= hh.Dwellings).all()
    universe = reweight.census_universe(viv, trips)
    ps, ds = reweight.person_scale(universe), reweight.dwelling_scale(universe)
    assert ps.notna().all() and (ps >= 0).all() and ds.notna().all() and (ds >= 1).all()
    six_plus = ["P_6A11", "P_12A14", "P_15A17", "P_18YMAS"]
    z43 = universe.Zone == 43
    raw = universe.loc[z43, six_plus].astype(float).sum(axis=1)
    persons = reweight.row_targets(universe, "persons", "Persons")
    assert pp.Persons.loc[43] == pytest.approx(persons[z43].sum())
    # where every row states ages the target is just the corrected census
    aged = raw > 0
    assert persons[z43 & aged.reindex(universe.index, fill_value=False)].sum() == pytest.approx((raw[aged] * ps[z43][aged]).sum())
    z56 = universe.Zone == 56
    assert pp.Persons.loc[56] / universe.loc[z56, six_plus].astype(float).sum().sum() > 1.15


def test_motorcycles_grow_with_the_registered_fleet(files):
    # 2023 HasMoto = 2020 HasMoto x each municipality's growth of registered private
    # motorcycles (INEGI VMRC); everything else household-level grows with population.
    vmrc = reweight.load_vmrc(DATA_DIR / "VMRC_AMG.csv")
    for m, lo, hi in [(39, 1.32, 1.34), (97, 1.63, 1.65)]:
        assert lo < vmrc.ratio("MOTO_PARTICULAR", m, 2020, 2023) < hi
    c = files.constraints
    h20, h23 = c["HouseholdConstraintsByZone_2020.csv"], c["HouseholdConstraintsByZone_2023.csv"]
    moto_growth = h23.HasMoto.sum() / h20.HasMoto.sum()
    pop_growth = h23.Dwellings.sum() / h20.Dwellings.sum()
    assert 1.40 < moto_growth < 1.55 and 1.02 < pop_growth < 1.04
    assert h23.HasCar.sum() / h20.HasCar.sum() == pytest.approx(pop_growth, abs=0.01)
    assert (h23[["HasCar", "HasMoto", "HasBike", "HasInternet"]].max(axis=1) <= h23.Dwellings).all()
    idx = files.index.set_index(["target_column", "year"])
    assert bool(idx.loc[("HasMoto", 2023), "scaled"])


def test_check_spec_rejects_two_growth_sources(monkeypatch):
    spec = copy.deepcopy(reweight.load_spec())
    spec["constraints"]["households"]["HasMoto"]["conapo"] = "total"
    monkeypatch.setattr(eodgdl.reweight._spec, "load_spec", lambda: spec)
    assert any("one growth source" in p for p in reweight.check_spec())


def test_each_year_lists_the_whole_constraint_set(files):
    # BusBoardings has no year suffix (a 2023 figure already); it must still be part of the
    # 2023 set a reader gets by filtering constraints.csv on `year`.
    by_year = files.index.groupby("year").target_column.apply(set)
    assert by_year.loc[2020] == by_year.loc[2023]
    assert "BusBoardings" in by_year.loc[2023]
    assert not files.index.duplicated(["year", "target_column"]).any()


def test_internet_grows_with_endutih(files):
    rates = reweight.load_rates(DATA_DIR)
    assert rates["endutih_internet"] == pytest.approx(1.193, abs=0.002)
    c = files.constraints
    h20, h23 = c["HouseholdConstraintsByZone_2020.csv"], c["HouseholdConstraintsByZone_2023.csv"]
    pop_growth = h23.Dwellings.sum() / h20.Dwellings.sum()
    growth = h23.HasInternet.sum() / h20.HasInternet.sum() / pop_growth
    assert 1.12 < growth < 1.193 + 1e-9          # the per-row cap only ever lowers it
    assert (h23.HasInternet <= h23.Dwellings + 1e-6).all()


def test_check_spec_requires_conapo_under_a_rate(monkeypatch):
    spec = copy.deepcopy(reweight.load_spec())
    del spec["constraints"]["households"]["HasInternet"]["conapo"]
    monkeypatch.setattr(eodgdl.reweight._spec, "load_spec", lambda: spec)
    assert any("on top of conapo" in p for p in reweight.check_spec())


def test_row_targets_add_up_to_the_zone_targets(files, viv, trips):
    universe = reweight.census_universe(viv, trips)
    conapo = reweight.load_conapo(DATA_DIR / "CONAPO_proyecciones_AMG.csv")
    for year, scaling in ((2020, None), (2023, conapo)):
        for name, table, file in (("Persons", "persons", "PersonConstraintsByZone"),
                                  ("Dwellings", "households", "HouseholdConstraintsByZone")):
            rows = reweight.row_targets(universe, table, name, scaling)
            zone = files.constraints[f"{file}_{year}.csv"].set_index("Zone")[name]
            assert (rows.groupby(universe.Zone).sum() - zone).abs().max() < 1e-6


def test_sampled_agebs_carry_their_zone_taz_and_targets(files, viv, trips):
    s = files.sampled_agebs
    code = viv.ageb.astype(str)
    assert len(s) == code.nunique() == 1046 and s.AGEB.is_unique
    assert s.unit.value_counts().to_dict() == {"urban": 1019, "rural": 27}
    assert set(s.TAZ) <= set(files.zone_system.TAZ) and (s.TAZ == s.Zone * 1000 + s.Municipality).all()
    assert (s.centralidad == s.AGEB.map(viv.centralidad.astype(str).groupby(code).first())).all()
    assert (s.sampled_dwellings == s.AGEB.map(code.value_counts())).all()
    # An urban AGEB's values are its own census row, the zone targets' own arithmetic.
    universe = reweight.census_universe(viv, trips)
    urban = s[s.unit == "urban"].set_index("AGEB")
    persons = reweight.row_targets(universe, "persons", "Persons")
    assert (urban.Persons6plus_2020 - persons.reindex(urban.index)).abs().max() < 1e-6
    assert (urban.Dwellings_2020 == universe.TVIVPARHAB.reindex(urban.index).astype(float)).all()
    # No AGEB exceeds its zone's target; a rural one sums its localities in the coded zone.
    zone = files.constraints["HouseholdConstraintsByZone_2023.csv"].set_index("Zone").Dwellings
    per_zone = s.groupby("Zone").Dwellings_2023.sum()
    assert (per_zone <= zone.reindex(per_zone.index) + 1e-6).all()
    assert (s[s.unit == "rural"].localities > 0).all()
    assert (s.Persons6plus_2020 > 0).all()
    assert list(s.columns[-6:]) == [f"{t}_{y}" for y in (2020, 2023) for t in ("Persons6plus", "Occupants", "Dwellings")]
    occupants = universe.OCUPVIVPAR.reindex(urban.index).astype(float).fillna(0)
    assert (urban.Occupants_2020 == occupants).all() and (urban.Occupants_2020 >= urban.Persons6plus_2020 - 1e-6).mean() > 0.95


def test_rows_without_a_stated_age_take_their_zone_s_mix(viv, trips):
    # AGEB 1412000013876 (centralidad 16) has 179 residents and no stated age: its person
    # counts are its zone's per private-dwelling resident, not 0.
    universe = reweight.census_universe(viv, trips)
    persons = reweight.row_targets(universe, "persons", "Persons")
    row = universe.loc["1412000013876"]
    assert float(row[["P_6A11", "P_12A14", "P_15A17", "P_18YMAS"]].astype(float).sum()) == 0
    zone = universe.Zone == row.Zone
    stated = universe.loc[zone, ["P_0A2", "P_3A5", "P_6A11", "P_12A14", "P_15A17", "P_18YMAS"]].astype(float).sum(axis=1) > 0
    private = (universe.POBTOT - universe.POBCOL.fillna(0)).astype(float)
    mix = persons[zone & stated.reindex(universe.index, fill_value=False)].sum() / private[zone & stated.reindex(universe.index, fill_value=False)].sum()
    assert persons.loc["1412000013876"] == pytest.approx(mix * private.loc["1412000013876"])
    # every age band, sex and activity count of the row comes from the same mix, so the
    # bands still partition the persons and the sexes add up to them
    bands = [reweight.row_targets(universe, "persons", b).loc["1412000013876"]
             for b in ["Age6_11", "Age12_14", "Age15_17", "Age18_24", "Age25_59", "Age60p"]]
    assert sum(bands) == pytest.approx(persons.loc["1412000013876"])
