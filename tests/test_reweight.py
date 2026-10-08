"""Inputs for TMG.SurveyReweight: the spec, the records, the targets, the written set."""
import copy
from pathlib import Path

import pandas as pd
import pytest

import eodgdl
from eodgdl import reweight
from eodgdl.chains import non_trips
from eodgdl.reweight import _spec
from eodgdl._resources import taz_ids
from eodgdl.reweight.records import zone_index
from eodgdl.taz import ACCESS_POINTS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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
    assert set(idx.map_column) == {"MTAZ", "Municipality", "Region"}
    # the TAZ set: every constraint of the MTAZMun set, plus the targets on the TAZ, its base category
    taz = reweight.constraint_index(2020, "taz")
    assert taz.merge(idx, how="left", indicator=True).query("_merge == 'left_only'").target_column.tolist() == ["TAZDwellings"]
    assert taz.set_index("target_column").loc["TAZDwellings", ["file", "map_column", "matching_attributes"]].tolist() == [
        "HouseholdConstraintsByTAZ_2020.csv", "TAZ", "Dwellings"]


def test_check_spec_holds_taz_targets_to_an_ageb_column(monkeypatch):
    broken = copy.deepcopy(_spec.load_spec())
    broken["constraints"]["households"]["TAZDwellings"]["rows_of"] = "HasCar"          # not in ZoneSystemAGEBs.csv
    broken["constraints"]["persons"]["Persons"]["rows_of"] = "Persons"                  # not a taz target
    broken["constraints"]["persons"]["TAZPersons"] = {"geography": "taz", "census": "POBTOT"}   # no rows_of
    monkeypatch.setattr(_spec, "load_spec", lambda: broken)
    problems = reweight.check_spec()
    assert any("TAZDwellings: rows_of 'HasCar'" in p for p in problems)
    assert any("persons.Persons: a taz target is rows_of" in p for p in problems)
    assert any("persons.TAZPersons: a taz target is rows_of" in p for p in problems)


@pytest.fixture(scope="module")
def tables(stages):
    return stages.revised


@pytest.fixture(scope="module")
def files(tables):
    return reweight.build(tables, data_dir=DATA_DIR)


@pytest.fixture(scope="module")
def viv(tables):
    return tables.viv


@pytest.fixture(scope="module")
def trips(tables):
    return tables.trips


def _assignment(files):
    """The assignment as zoning.assign_units returns it, from ZoneAssignment.csv's frame (its MTAZ names put back)."""
    back = {new: old for old, new in reweight.zoning.OUTPUT_ZONE_COLUMNS.items()}
    return files.zone_assignment.set_index("CVEGEO").drop(columns=["TAZ", "MTAZ"]).rename(columns=back)


def test_records_have_the_survey_s_rows_and_integer_keys(files, trips):
    hh, pp, tt = files.households, files.persons, files.trips
    assert len(hh) == 17_901
    assert len(pp) == 58_061
    assert len(tt) == len(trips) - int(non_trips(trips).sum())  # every trip load_eod returns but the non-trips
    for df, key in ((hh, "HouseholdID"), (pp, "PersonID")):
        assert pd.api.types.is_integer_dtype(df[key]) and not df[key].duplicated().any()
    assert tt.PersonID.isin(pp.PersonID).all()
    assert pp.HouseholdID.isin(hh.HouseholdID).all()


def test_employed_is_tasha_s_worker(files, tables):
    # One worker definition across the package: tasha's EmploymentStatus F or P is
    # reweight's Employed, person by person (both read trabajo_semana_pasada), for those
    # who answered; the 4,370 who did not take P(trabaja) from the pinned design-weight
    # empleo bundle (the probability, not a draw), and NotEmployed its complement.
    import numpy as np

    from eodgdl import tasha
    from eodgdl.data import resolve
    from eodgdl.impute.run import score_task
    from eodgdl.impute.sources import Context

    pp, hab = files.persons, tables.hab
    people = tasha.build_people(hab, tables.trips, tables.viv)
    answered = hab.trabajo_semana_pasada.notna().to_numpy()
    worker = people.EmploymentStatus.isin(["F", "P"]).to_numpy()
    assert (worker[answered] == (pp.Employed.to_numpy()[answered] == 1)).all()
    assert np.isclose(pp.Employed + pp.NotEmployed, (pp.Age6_11 == 0).astype(float), rtol=0, atol=1e-12).all()

    unanswered = ~answered & (hab.edad >= 12).to_numpy()
    assert unanswered.sum() == 4_370 and (hab.edad[unanswered] >= 16).all()
    scores = score_task("empleo", path=resolve("od_empleo_design_model.joblib"), context=Context(tables=tables))
    p = scores.set_index(["folio_vivienda", "folio_habitante"])["prob_situacion_laboral_trabaja"].reindex(hab.index[unanswered])
    assert (pp.Employed.to_numpy()[unanswered] == p.to_numpy()).all() and (pp.NotEmployed.to_numpy()[unanswered] == 1 - p.to_numpy()).all()
    assert ((pp.Employed > 0) & (pp.Employed < 1)).sum() == 4_370      # every unanswered person, and no one else, fractional
    assert (pp.Unemployed.to_numpy()[unanswered] == 0).all() and (pp.Inactive.to_numpy()[unanswered] == 0).all()


def test_the_pinned_design_weight_empleo_bundle_is_v0_9_0_s():
    # stage 1 reads the empleo bundle trained on the survey's design weight, pinned under a
    # name of its own: v0.9.0's od_empleo_model.joblib, byte for byte
    import joblib

    from eodgdl.data import resolve
    from eodgdl.impute.bundle import training_weight
    from eodgdl.manifest import sha256

    assert sha256(resolve("od_empleo_design_model.joblib")) == "247d18fc175b98ebd0698648af6cb36dda671a822205889925141a815f6e97f2"
    assert training_weight(joblib.load(resolve("od_empleo_design_model.joblib"))["metadata"]) == "design"


def test_zone_system_is_the_cells_of_the_survey(files):
    zs = files.zone_system
    assert len(zs) == 90
    assert not zs.MTAZMun.duplicated().any()
    assert set(zs.Municipality) == set(eodgdl.load_zm_muns())
    assert zs.MTAZ.between(1, 71).all() and (zs.Region == 1).all()
    assert (zs.MTAZMun == zs.MTAZ * 1000 + zs.Municipality).all()
    assert files.households.HouseholdMTAZMun.isin(zs.MTAZMun).all()
    # each household's TAZ is its surveyed AGEB's, in its own cell, and the AGEB table counts them
    hh, agebs = files.households, files.zone_system_agebs.set_index("TAZ")
    assert hh.HouseholdTAZ.map(agebs.MTAZMun).eq(hh.HouseholdMTAZMun).all()
    assert hh.HouseholdTAZ.value_counts().eq(agebs.sampled_dwellings[agebs.sampled_dwellings > 0]).all()
    assert hh.HouseholdTAZ.nunique() == 1_046
    assert files.zone_labels.sampled_dwellings.sum() == len(files.households)


def test_dummies_partition(files):
    pp, tt = files.persons, files.trips
    ages = ["Age6_11", "Age12_14", "Age15_17", "Age18_24", "Age25_59", "Age60p"]
    assert (pp.Male + pp.Female == pp.Persons).all()
    assert (pp[ages].sum(axis=1) == pp.Persons).all()
    assert (pp.Employed + pp.Unemployed + pp.Inactive <= pp.Persons).all()
    assert ((pp.Employed + pp.NotEmployed) - (pp.Age6_11 == 0).astype(float)).abs().max() < 1e-12  # partitions the 12+
    assert pp.Cyclist.sum() == 561   # the survey's bicycle commuters with a trip on the chain the model reads
    boardings = ["BusBoardings", "RailBoardings", "BRTBoardings", "SitrenBoardings", "OtherTransitBoardings"]
    assert (tt[boardings].sum(axis=1) == tt.TransitBoardings).all()
    assert tt.BicycleTrip.isin([0, 1]).all()


def test_targets_are_consistent_and_pinned(files):
    c = files.constraints
    hh20 = c["HouseholdConstraintsByMTAZ_2020.csv"]
    pp20 = c["PersonConstraintsByMTAZ_2020.csv"]
    mun20 = c["PersonConstraintsByMunicipality_2020.csv"]
    assert len(hh20) == len(pp20) == 64 and len(mun20) == 9
    assert round(hh20.Dwellings.sum()) == 1_462_360
    assert round(pp20.Persons.sum()) == 4_679_979
    assert round(mun20.Cyclist.sum()) == 100_591  # 101,907 whole-municipality cyclists, universe share applied
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
    hh23 = c["HouseholdConstraintsByMTAZ_2023.csv"]
    pp23 = c["PersonConstraintsByMTAZ_2023.csv"]
    assert hh23.Dwellings.sum() > hh20.Dwellings.sum()
    assert pp23.Persons.sum() > pp20.Persons.sum()
    guadalajara_only = files.zone_labels.groupby("MTAZ_code").CVE_MUN.agg(set).eq({39})
    zone_of = dict(zip(files.zone_labels.MTAZ_code, files.zone_labels.MTAZMun // 1000))
    shrinking = [zone_of[z] for z, only in guadalajara_only.items() if only]
    assert (hh23.set_index("MTAZ").Dwellings.loc[shrinking] < hh20.set_index("MTAZ").Dwellings.loc[shrinking]).all()
    assert (pp23.Age60p / pp20.Age60p).mean() > (pp23.Age6_11 / pp20.Age6_11).mean()
    # 2023 keeps the census's partitions: bands sum to Persons, each sex's bands to its
    # total, the two activity targets to the 12+ bands; the sexes fall short of the
    # totals only by INEGI's suppressed cells, as in 2020.
    assert round(pp23.Persons.sum()) == 4_850_717
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
    assert len(universe) == 2307 and set(universe.MTAZ) == set(files.zone_system.MTAZ)  # 64 zones
    assert universe.MTAZ.nunique() == 64


def test_every_census_unit_gets_at_most_one_zone_and_the_survey_s(files, viv, trips):
    a = _assignment(files)
    assert a.index.is_unique and reweight.check_assignment(a, viv, trips) == []
    assert a.groupby(["unit", "rule"], observed=True).size().to_dict() == {
        ("ageb", "survey"): 993, ("ageb", "survey over polygon"): 26,
        ("ageb", "trip ends"): 615, ("ageb", "trip ends over polygon"): 15,
        ("ageb", "polygon"): 356, ("ageb", "majority"): 20, ("ageb", "nearest polygon"): 12,
        ("locality", "survey"): 213, ("locality", "trip ends"): 48, ("locality", "polygon"): 9,
        ("locality", "outside"): 515, ("rural ageb", "trip ends"): 2,
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
    # holding the most of the rest, or the nearest.
    assert a[a.unit == "ageb"].zone.notna().all()
    # The census tabulates two rural localities of El Salto by block under their rural AGEB's number (1407001161467,
    # 1407001341467); INEGI's map draws no such urban AGEB: they are localities of rural AGEB 140701467, which the
    # survey codes 13. Parques del Triunfo, inside 60's polygon, joins 13; Agua Blanca (Puente Grande's prisons) lies
    # outside every polygon and stays out of the targets.
    assert not a.index.isin(["1407001161467", "1407001341467"]).any()
    assert a.loc["140700134", ["unit", "rural_ageb", "zone", "rule"]].tolist() == ["locality", "140701467", "13", "survey"]
    assert a.loc["140700116", ["unit", "rural_ageb", "rule"]].tolist() == ["locality", "140701467", "outside"]
    assert pd.isna(a.loc["140700116", "zone"])
    # 141200001098A: a campus the survey codes 68B (968 trip ends), though 82 % of its
    # residents live in 44's polygon.
    assert a.loc["141200001098A", ["zone", "polygon_zone", "rule"]].tolist() == ["68B", "44", "trip ends over polygon"]
    # La Aurora (Juanacatlán) lies outside every polygon; the survey coded its dwellings 49F.
    aurora = a.loc[["1405100020181", "1405100020196", "1405100020209", "1405100020213"]]
    assert (aurora.zone == "49F").all() and aurora.polygon_zone.isna().all()
    # A rural AGEB the survey records counts whole in the zone it coded: every locality of it
    # inside some polygon (and those its sample needs), one zone per code.
    locs = a[a.unit == "locality"]
    coded = reweight.survey_codes(viv, trips).query("code.str.len() == 9").groupby("code").zone.first()
    recorded = locs[locs.rural_ageb.isin(coded.index)]
    inside = recorded[recorded.polygon_zone.notna()]
    assert (inside.zone == inside.rural_ageb.map(coded)).all()
    assert (recorded.dropna(subset=["zone"]).groupby("rural_ageb").zone.nunique() == 1).all()
    # 140970123: 80 dwellings coded 51F, its localities inside polygons hold 26; the
    # nearest left-out ones join until they can hold the sample, Cuexcomatitlán among them
    assert a.loc["140970014", ["zone", "rule"]].tolist() == ["51F", "survey"]
    assert a.loc["140970004", "zone"] is pd.NA or pd.isna(a.loc["140970004", "zone"])  # San Juan Evangelista


def test_check_assignment_catches_a_misplaced_rural_ageb(files, viv, trips):
    a = _assignment(files)
    a.loc[(a.unit == "locality") & (a.rural_ageb == "140970123") & a.zone.notna(), "zone"] = "59"
    problems = reweight.check_assignment(a, viv, trips)
    assert problems and all("rural AGEB 140970123" in p for p in problems)


def _taz_ids(files):
    """A clean TAZ table over the built zone system and the access points, and the raw assignment it is checked
    against (ZoneAssignment.csv's columns renamed back)."""
    assignment = _assignment(files)
    codes = sorted(set(reweight.zone_system_codes(assignment))) + sorted(ACCESS_POINTS)
    ids = pd.DataFrame({"AGEB": codes, "TAZ": [str(i) for i in range(1, len(codes) + 1)], "note": "", "source": "test"})
    return ids, assignment


def test_zone_system_codes_are_its_agebs(files):
    codes = reweight.zone_system_codes(_taz_ids(files)[1])
    assert codes.str.len().value_counts().to_dict() == {13: 2037, 9: 270 + 2}
    assert set(codes) == set(files.zone_system_agebs.AGEB)


def test_check_taz_ids_catches_every_kind_of_breakage(files):
    ids, assignment = _taz_ids(files)
    assert reweight.check_taz_ids(ids, assignment) == []
    mapping = reweight.taz_map(assignment, ids)
    assert len(mapping) == 2087 + 7 and mapping.dtype == "int64"
    joined = reweight.with_taz(assignment, ids)
    placed = joined.zone.notna()
    assert joined.TAZ[placed].notna().all() and joined.TAZ[~placed].isna().all()
    loc = joined[joined.unit == "locality"].dropna(subset=["zone"]).iloc[0]
    assert loc.TAZ == mapping[loc.rural_ageb]

    def problems(edit):
        broken = ids.copy()
        edit(broken)
        return reweight.check_taz_ids(broken, assignment)

    assert problems(lambda t: t.drop(columns="note", inplace=True))[0].startswith("taz_ids.csv: columns")
    assert problems(lambda t: t.drop(index=0, inplace=True)) == [f"AGEB {ids.AGEB[0]}: in the zone system, no TAZ"]
    ap = ids.index[ids.AGEB == "99999000A"][0]
    assert problems(lambda t: t.drop(index=ap, inplace=True)) == ["AGEB 99999000A: in the zone system, no TAZ"]
    assert problems(lambda t: t.loc.__setitem__((1, "TAZ"), "1")) == [f"TAZ 1: names 2 codes, {sorted(ids.AGEB[:2])}"]
    for bad in ("0", "-3", "007", "1.0", "x", str(2**31)):
        assert problems(lambda t: t.loc.__setitem__((0, "TAZ"), bad)) == [
            f"AGEB {ids.AGEB[0]}: TAZ {bad!r} is not a positive integer up to {2**31 - 1}"], bad
    short = ids.AGEB[0][:-1]
    assert problems(lambda t: t.loc.__setitem__((0, "AGEB"), short)) == [
        f"AGEB {short!r}: not an INEGI AGEB key of Jalisco (13 characters urban, 9 rural) nor an access point",
        f"AGEB {ids.AGEB[0]}: in the zone system, no TAZ", f"AGEB {short}: has a TAZ, not in the zone system"]
    extra = pd.DataFrame({"AGEB": [ids.AGEB[0]], "TAZ": ["99999"], "note": "", "source": ""})
    assert problems(lambda t: t.__init__(pd.concat([t, extra], ignore_index=True))) == [f"AGEB {ids.AGEB[0]}: 2 rows"]
    with pytest.raises(ValueError, match="disagrees with the zone system"):
        reweight.taz_map(assignment, ids.iloc[1:])


def test_the_committed_taz_ids_fit_the_zone_system(files):
    ids, assignment = _taz_ids(files)
    # the working group's mapping (zonas_2212.shp, 2026-10-02), restricted to our zone system's codes
    assert len(taz_ids()) == 2094 and reweight.check_taz_ids(taz_ids(), assignment) == []
    out = files.zone_assignment
    assert (out.TAZ.notna() == out.MTAZ_code.notna()).all() and (out.MTAZ == out.MTAZ_code.map(zone_index())).all()


def test_every_trip_end_is_in_its_ageb_s_zone(files, viv, trips):
    # The survey codes a trip end's zone by where it lies; every AGEB it records is in the
    # zone system, in the zone it coded (a rural AGEB, in a zone holding one of its
    # localities). Only the access points, the airport and six road gateways, are not.
    a = _assignment(files)
    agreement = reweight.trip_end_agreement(a, viv, trips)
    ends = 2 * int((~non_trips(trips)).sum())
    assert agreement.to_dict() == {"in zone": ends - 3_076, "access point": 3_076}
    # the survey's zone is a function of the AGEB code: one zone per code, across its
    # dwellings, origins and destinations
    codes = reweight.survey_codes(viv, trips)
    # (1,693 codes: 1,694 on the rules' output; one AGEB's only trip end is an origin moved home by hand)
    assert (codes.groupby("code").zone.nunique() == 1).all() and codes.code.nunique() == 1_693
    # the chain rule that moves an origin home moves its zone too
    moved = trips[trips.ajustes.astype(str).str.contains("origen:casa")].reset_index()
    home = viv.centralidad.astype(str)
    assert (moved.zona_origen.astype(str) == moved.folio_vivienda.map(home).to_numpy()).all()


def test_redrawn_zones_follow_the_assignment(files):
    a = _assignment(files)
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
    from eodgdl.manifest import check_manifest, read_manifest

    reweight.write(files, tmp_path, data_dir=DATA_DIR)
    assert reweight.check(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["ZoneSystem.csv", "ZoneLabels.csv", "ZoneAssignment.csv", "ZoneSystemAGEBs.csv",
         "HouseholdRecords.csv", "PersonRecords.csv", "TripRecords.csv", "Constraints", "README.md", "manifest.json"])
    # stage 2's manifest: every file written, the survey, census tables and the pinned empleo bundle read
    manifest = read_manifest(tmp_path)
    assert manifest["stage"] == "reweight_inputs" and len(manifest["outputs"]) == sum(1 for p in tmp_path.rglob("*") if p.is_file()) - 1
    assert manifest["inputs"]["od_empleo_design_model.joblib"]["sha256"] == "247d18fc175b98ebd0698648af6cb36dda671a822205889925141a815f6e97f2"
    assert {"IMEPLAN_Base_Viviendas_Master.csv", "CONAPO_proyecciones_AMG.csv", "AMG_Zonificacion_para_encuesta.parquet"} <= set(manifest["inputs"])
    assert "mxcensus" in manifest["environment"] and manifest["parameters"]["years"] == [2020, 2023]
    assert check_manifest(tmp_path / "manifest.json") == []

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
    # the activity targets with the 4,370 unanswered counted by their probability of working
    # (2,252,572 and 2,181,263 when they were all NotEmployed)
    activity = d.groupby("target_column").survey_at_design_weight.sum()
    assert round(activity.Employed + activity.NotEmployed) == 2_252_572 + 2_181_263
    assert (round(activity.Employed), round(activity.NotEmployed)) == (2_366_943, 2_066_892)


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
    # occupant information whose details INEGI did not impute); zone 43 (45) held a third of its
    # residents in collective quarters until Puente Grande's prisons (Agua Blanca, a rural locality
    # outside every polygon) left the targets, 2026-10-02, and now has none. Corrected, persons 6+
    # are 85-96 % of private-dwelling occupants in every zone; the survey has ~93 %.
    pp = files.constraints["PersonConstraintsByMTAZ_2020.csv"].set_index("MTAZ")
    hh = files.constraints["HouseholdConstraintsByMTAZ_2020.csv"].set_index("MTAZ")
    assert (pp.Persons / hh.Occupants).between(0.8, 1.0).all()
    assert (hh[["HasCar", "HasMoto", "HasBike", "HasInternet"]].max(axis=1) <= hh.Dwellings).all()
    universe = reweight.census_universe(viv, trips)
    ps, ds = reweight.person_scale(universe), reweight.dwelling_scale(universe)
    assert ps.notna().all() and (ps >= 0).all() and ds.notna().all() and (ds >= 1).all()
    six_plus = ["P_6A11", "P_12A14", "P_15A17", "P_18YMAS"]
    z43 = universe.MTAZ == 43
    raw = universe.loc[z43, six_plus].astype(float).sum(axis=1)
    persons = reweight.row_targets(universe, "persons", "Persons")
    assert pp.Persons.loc[43] == pytest.approx(persons[z43].sum())
    # where every row states ages the target is just the corrected census
    aged = raw > 0
    assert persons[z43 & aged.reindex(universe.index, fill_value=False)].sum() == pytest.approx((raw[aged] * ps[z43][aged]).sum())
    z56 = universe.MTAZ == 56
    assert pp.Persons.loc[56] / universe.loc[z56, six_plus].astype(float).sum().sum() > 1.15


def test_motorcycles_grow_with_the_registered_fleet(files):
    # 2023 HasMoto = 2020 HasMoto x each municipality's growth of registered private
    # motorcycles (INEGI VMRC); everything else household-level grows with population.
    vmrc = reweight.load_vmrc(DATA_DIR / "VMRC_AMG.csv")
    for m, lo, hi in [(39, 1.32, 1.34), (97, 1.63, 1.65)]:
        assert lo < vmrc.ratio("MOTO_PARTICULAR", m, 2020, 2023) < hi
    c = files.constraints
    h20, h23 = c["HouseholdConstraintsByMTAZ_2020.csv"], c["HouseholdConstraintsByMTAZ_2023.csv"]
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
    h20, h23 = c["HouseholdConstraintsByMTAZ_2020.csv"], c["HouseholdConstraintsByMTAZ_2023.csv"]
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
        for name, table, file in (("Persons", "persons", "PersonConstraintsByMTAZ"),
                                  ("Dwellings", "households", "HouseholdConstraintsByMTAZ")):
            rows = reweight.row_targets(universe, table, name, scaling)
            zone = files.constraints[f"{file}_{year}.csv"].set_index("MTAZ")[name]
            assert (rows.groupby(universe.MTAZ).sum() - zone).abs().max() < 1e-6


def test_zone_system_agebs_carry_their_taz_mtaz_and_targets(files, viv, trips):
    s = files.zone_system_agebs
    code = viv.ageb.astype(str)
    assert len(s) == 2087 and s.AGEB.is_unique
    assert s.unit.value_counts().to_dict() == {"urban": 2037, "rural": 50}
    # every placed census unit is in the file, under its own or its rural AGEB's code
    placed = _assignment(files).reset_index()
    placed = placed[placed.zone.notna()]
    assert set(s.AGEB) == set(placed.CVEGEO.where(placed.unit != "locality", placed.rural_ageb))
    assert s.localities.sum() == (placed.unit == "locality").sum() == 270
    assert s.TAZ.is_unique and (s.TAZ == s.AGEB.map(taz_ids().set_index("AGEB").TAZ.astype(int))).all()
    assert (s.MTAZMun == s.MTAZ * 1000 + s.Municipality).all()
    # one cell holds no sampled dwelling, so the tool's zone system lacks it: 26098, one AGEB of Tlaquepaque
    assert sorted(set(s.MTAZMun) - set(files.zone_system.MTAZMun)) == [26098]
    sampled = s[s.sampled_dwellings > 0]
    assert len(sampled) == code.nunique() == 1046
    assert sampled.unit.value_counts().to_dict() == {"urban": 1019, "rural": 27}
    assert (sampled.MTAZ_code == sampled.AGEB.map(viv.centralidad.astype(str).groupby(code).first())).all()
    assert (sampled.sampled_dwellings == sampled.AGEB.map(code.value_counts())).all()
    # An urban AGEB's values are its own census row, the zone targets' own arithmetic.
    universe = reweight.census_universe(viv, trips)
    urban = s[s.unit == "urban"].set_index("AGEB")
    persons = reweight.row_targets(universe, "persons", "Persons")
    assert (urban.Persons6plus_2020 - persons.reindex(urban.index)).abs().max() < 1e-6
    assert (urban.Dwellings_2020 == universe.TVIVPARHAB.reindex(urban.index).astype(float)).all()
    # Every zone's AGEBs sum to its targets, both years.
    for year in (2020, 2023):
        for file, target, label in (("PersonConstraintsByMTAZ", "Persons", "Persons6plus"),
                                    ("HouseholdConstraintsByMTAZ", "Dwellings", "Dwellings"),
                                    ("HouseholdConstraintsByMTAZ", "Occupants", "Occupants")):
            zone = files.constraints[f"{file}_{year}.csv"].set_index("MTAZ")[target]
            per_zone = s.groupby("MTAZ")[f"{label}_{year}"].sum()
            assert per_zone.index.equals(zone.index) and (per_zone - zone).abs().max() < 1e-6
    assert (sampled.Persons6plus_2020 > 0).all() and (s[s.unit == "rural"].localities > 0).sum() == 48
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
    zone = universe.MTAZ == row.MTAZ
    stated = universe.loc[zone, ["P_0A2", "P_3A5", "P_6A11", "P_12A14", "P_15A17", "P_18YMAS"]].astype(float).sum(axis=1) > 0
    private = (universe.POBTOT - universe.POBCOL.fillna(0)).astype(float)
    mix = persons[zone & stated.reindex(universe.index, fill_value=False)].sum() / private[zone & stated.reindex(universe.index, fill_value=False)].sum()
    assert persons.loc["1412000013876"] == pytest.approx(mix * private.loc["1412000013876"])
    # every age band, sex and activity count of the row comes from the same mix, so the
    # bands still partition the persons and the sexes add up to them
    bands = [reweight.row_targets(universe, "persons", b).loc["1412000013876"]
             for b in ["Age6_11", "Age12_14", "Age15_17", "Age18_24", "Age25_59", "Age60p"]]
    assert sum(bands) == pytest.approx(persons.loc["1412000013876"])


@pytest.fixture(scope="module")
def taz_files(files):
    return reweight.rebase(files, "taz")


def test_taz_targets_put_each_cell_on_its_sampled_taz(files, taz_files):
    agebs = files.zone_system_agebs.set_index("TAZ")
    sampled = agebs.sampled_dwellings > 0
    for year in (2020, 2023):
        t = taz_files.constraints[f"HouseholdConstraintsByTAZ_{year}.csv"].set_index("TAZ").TAZDwellings
        own = agebs[f"Dwellings_{year}"]
        assert t.index.equals(agebs.index.sort_values()) and len(t) == 2087
        # 0 on the 1,041 TAZ with no sampled dwelling, positive on the 1,046 with one
        assert (t[~sampled] == 0).all() and (t[sampled] > 0).all() and sampled.sum() == 1046
        # every cell keeps its census total on its sampled TAZ, in proportion to their own; the cell with no sampled
        # TAZ (26098) goes to its MTAZ's sampled TAZ (all in 26120), so every MTAZ keeps its total, the MTAZ target
        cell = agebs.MTAZMun
        assert ((t.groupby(cell).sum() - own.groupby(cell).sum()).drop(index=[26098, 26120]).abs() < 1e-6).all()
        assert (t.groupby(agebs.MTAZ).sum() - files.constraints[f"HouseholdConstraintsByMTAZ_{year}.csv"]
                .set_index("MTAZ").Dwellings).abs().max() < 1e-6
        assert t.groupby(cell).sum()[26120] == pytest.approx(own.groupby(cell).sum()[[26120, 26098]].sum())   # Tlaquepaque to Zapopan
        factor = (t / own)[sampled].groupby(cell[sampled])
        assert ((factor.max() - factor.min()) < 1e-9).all()


def test_taz_set_is_the_mtazmun_set_on_the_taz(files, taz_files, tmp_path):
    from eodgdl.manifest import read_manifest

    zs = taz_files.zone_system
    assert list(zs.columns) == ["TAZ", "MTAZ", "Municipality", "Region"] and len(zs) == 2087 and zs.TAZ.is_unique
    assert list(taz_files.households.columns[:3]) == ["HouseholdID", "HouseholdTAZ", "HouseholdMTAZMun"]
    assert taz_files.households.drop(columns="HouseholdTAZ").equals(files.households.drop(columns="HouseholdTAZ"))
    assert taz_files.persons is files.persons and taz_files.trips is files.trips
    for name, frame in files.constraints.items():
        assert taz_files.constraints[name] is frame
    assert set(taz_files.constraints) - set(files.constraints) == {f"HouseholdConstraintsByTAZ_{y}.csv" for y in (2020, 2023)}
    assert (taz_files.index.groupby("year").size() == files.index.groupby("year").size() + 1).all()
    # the diagnostic reads the same weights through the TAZ: identical above it, and the TAZ rows besides
    d, d0 = taz_files.diagnostics[2020], files.diagnostics[2020]
    taz_rows = d[d.target_column == "TAZDwellings"]
    assert len(taz_rows) == 2087 and (taz_rows.records > 0).sum() == 1046
    assert d[d.target_column != "TAZDwellings"].drop(columns="TAZ").reset_index(drop=True).equals(d0)
    with pytest.raises(ValueError, match="rebase goes from the mtazmun set"):
        reweight.rebase(taz_files, "mtazmun")

    reweight.write(taz_files, tmp_path, data_dir=DATA_DIR)
    assert reweight.check(tmp_path) == []
    assert read_manifest(tmp_path)["parameters"] == {"years": [2020, 2023], "base": "taz"}
    # a TAZ target on a TAZ with no record makes the tool abort
    path = tmp_path / "Constraints" / "HouseholdConstraintsByTAZ_2020.csv"
    df = pd.read_csv(path)
    df.loc[df.TAZDwellings == 0, "TAZDwellings"] = 1.0
    df.to_csv(path, index=False)
    assert any("TAZDwellings > 0 but no households record" in p for p in reweight.check(tmp_path))
    # the records read with the other base first
    pd.read_csv(tmp_path / "HouseholdRecords.csv")[list(files.households.columns)].to_csv(tmp_path / "HouseholdRecords.csv", index=False)
    assert any("the loader reads HouseholdID and the base category, HouseholdTAZ" in p for p in reweight.check(tmp_path))
