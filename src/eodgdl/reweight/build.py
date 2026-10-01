"""Build and write the reweighting input set."""
from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from eodgdl.data._catalog import CENSUS_FILES, CONAPO_CSV, SURVEY_FILES, VMRC_CSV, ZONE_FILES
from eodgdl.reweight._spec import constraint_index, load_spec
from eodgdl.reweight.records import (
    build_households,
    build_people,
    build_trips,
    build_zones,
    imputed_scores,
    record_geography,
)
from eodgdl.reweight.targets import (
    build_constraints,
    census_universe,
    coverage,
    load_conapo,
    load_rates,
    load_vmrc,
    sampled_agebs,
)
from eodgdl.reweight.zoning import assign_units

RECORD_FILES = {
    "households": "HouseholdRecords.csv",
    "persons": "PersonRecords.csv",
    "trips": "TripRecords.csv",
}


class ReweightFiles(NamedTuple):
    zone_system: pd.DataFrame
    zone_labels: pd.DataFrame
    households: pd.DataFrame
    persons: pd.DataFrame
    trips: pd.DataFrame
    constraints: dict[str, pd.DataFrame]  # file name -> frame, every year together
    index: pd.DataFrame  # one row per (year, target column): Constraints/constraints.csv
    diagnostics: dict[str, pd.DataFrame]  # per year: target vs survey at the design weight
    coverage: pd.DataFrame  # the survey universe as a share of each whole municipality
    zone_assignment: pd.DataFrame = pd.DataFrame()  # every census unit's zone and the rule that set it
    sampled_agebs: pd.DataFrame = pd.DataFrame()  # every sampled AGEB's zone, TAZ and census targets, by year


def years():
    c = load_spec()["conapo"]
    return (c["base_year"], c["target_year"])


def build(tables, data_dir=None):
    """Everything the tool reads, from `load_eod`'s tables, the imputed attributes' models
    (spec.yaml's `imputed`) and the census."""
    viv, hab, trips, legs = tables
    zone_system, zone_labels = build_zones(viv)
    records = ReweightFiles(
        zone_system, zone_labels,
        build_households(viv), build_people(hab, trips, legs, imputed_scores(tables)), build_trips(trips, legs),
        {}, pd.DataFrame(), {}, pd.DataFrame(),
    )
    universe = census_universe(viv, trips)  # raises if the zone assignment disagrees with the survey
    shares = coverage(universe)
    conapo = load_conapo(Path(data_dir) / CONAPO_CSV if data_dir is not None else None)
    vmrc = load_vmrc(Path(data_dir) / VMRC_CSV if data_dir is not None else None)
    rates = load_rates(data_dir)
    base, target = years()
    constraints = {}
    index = []
    diagnostics = {}
    for year in (base, target):
        frames = build_constraints(universe, year, conapo, shares, vmrc, rates)
        constraints.update(frames)
        idx = constraint_index(year).assign(year=year)
        index.append(idx)
        diagnostics[year] = diagnostic(records, frames, idx)
    # Every year lists its whole set, the unscaled files included: a reader that filters
    # on `year` must get every constraint (de-duplicating across years once left the
    # 2023 set without BusBoardings).
    index = pd.concat(index, ignore_index=True)
    assignment = assign_units(viv, trips)
    return records._replace(constraints=constraints, index=index, diagnostics=diagnostics, coverage=shares,
                            zone_assignment=assignment.reset_index(),
                            sampled_agebs=sampled_agebs(universe, viv, assignment, conapo, vmrc, rates))


def diagnostic(records, constraints, index):
    """Target vs. the survey expanded at the dwelling design weight, per constraint and geography.

    What the tool's constraint report would say before it moves anything, had it started
    from viv.ponderador instead of 1.0: the shipped weights' distance from every target.
    """
    weight = records.households.set_index("HouseholdID").ExpansionFactor
    rows = []
    for row in index.itertuples(index=False):
        frame = getattr(records, row.table)
        w = frame.HouseholdID.map(weight)
        value = frame[row.matching_attributes.split(";")].prod(axis=1)
        geo = record_geography(records, row.table, records.zone_system, row.geography)
        survey = (w * value).groupby(geo).sum()
        n = (value > 0).groupby(geo).sum()
        target = constraints[row.file].set_index(row.map_column)[row.target_column]
        for g, t in target.items():
            s = float(survey.get(g, 0.0))
            rows.append({
                "file": row.file, "target_column": row.target_column, row.map_column: g,
                "target": t, "survey_at_design_weight": s, "records": int(n.get(g, 0)),
                "ratio": s / t if t else float("nan"),
            })
    out = pd.DataFrame(rows)
    geo_cols = [c for c in ("Zone", "Municipality", "Region") if c in out.columns]
    return out[["file", "target_column", *geo_cols, "target", "survey_at_design_weight", "records", "ratio"]]


def write(files, out_dir, data_dir=None):
    """Write the set under `out_dir` with its manifest (:func:`write_manifest`); returns the
    paths written."""
    out_dir = Path(out_dir)
    (out_dir / "Constraints").mkdir(parents=True, exist_ok=True)
    written = []

    def put(name, df, **kw):
        path = out_dir / name
        df.to_csv(path, index=False, **kw)
        written.append(path)

    put("ZoneSystem.csv", files.zone_system)
    put("ZoneLabels.csv", files.zone_labels)
    for table, name in RECORD_FILES.items():
        df = getattr(files, table)
        ints = df.columns[[pd.api.types.is_integer_dtype(df[c]) for c in df.columns]]
        put(name, df.astype({c: "int64" for c in ints}))
    for name, df in files.constraints.items():
        put(f"Constraints/{name}", df, float_format="%.6f")
    put("Constraints/constraints.csv", files.index)
    for year, df in files.diagnostics.items():
        put(f"Constraints/diagnostic_{year}.csv", df, float_format="%.4f")
    put("Constraints/coverage.csv", files.coverage.reset_index(), float_format="%.4f")
    put("ZoneAssignment.csv", files.zone_assignment, float_format="%.4f")
    put("SampledAGEBs.csv", files.sampled_agebs, float_format="%.4f")
    readme = resources.files("eodgdl.reweight") / "README.md"
    with resources.as_file(readme) as src:
        shutil.copy(src, out_dir / "README.md")
        written.append(out_dir / "README.md")
    written.append(write_manifest(out_dir, data_dir, written))
    return written


def write_manifest(out_dir, data_dir, outputs):
    """Stage 2's manifest (eodgdl.manifest): the survey files, the zone polygons, the CONAPO /
    VMRC / ENDUTIH tables and the imputed attributes' model files read, the survey's data versions, the
    constraint years and the sha256 of `outputs`. The census comes through mxcensus, whose
    version and registry the manifest's `environment` records."""
    from eodgdl.data import resolve
    from eodgdl.impute.sources.eod import survey_versions
    from eodgdl.manifest import write_manifest as write
    from eodgdl.reweight._spec import TABLES, attributes

    local = lambda name: Path(data_dir) / name if data_dir is not None and (Path(data_dir) / name).exists() else resolve(name)
    inputs = {name: local(name) for name in [*SURVEY_FILES, *ZONE_FILES, *CENSUS_FILES]}
    for table in TABLES:
        for entry in attributes(table).values():
            if "imputed" in entry:
                inputs[entry["imputed"]["bundle"]] = resolve(entry["imputed"]["bundle"])
    return write(out_dir, "reweight_inputs", inputs=inputs, parameters={"years": list(years())},
                 versions=survey_versions(), outputs=outputs)
