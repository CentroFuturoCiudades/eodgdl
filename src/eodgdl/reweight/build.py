"""Build and write the reweighting input set."""
from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from eodgdl.data._catalog import CONAPO_CSV
from eodgdl.reweight._spec import constraint_index, load_spec
from eodgdl.reweight.records import (
    build_households,
    build_people,
    build_trips,
    build_zones,
    record_geography,
)
from eodgdl.reweight.targets import (
    build_constraints,
    census_universe,
    coverage,
    load_conapo,
    reconcile,
)

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


def years():
    c = load_spec()["conapo"]
    return (c["base_year"], c["target_year"])


def build(tables, data_dir=None):
    """Everything the tool reads, from `load_eod`'s tables and the census."""
    viv, hab, trips, legs = tables
    zone_system, zone_labels = build_zones(viv)
    records = ReweightFiles(
        zone_system, zone_labels,
        build_households(viv), build_people(hab, trips, legs), build_trips(trips, legs),
        {}, pd.DataFrame(), {}, pd.DataFrame(),
    )
    universe = census_universe(data_dir)
    disagreements = reconcile(universe, data_dir=data_dir)
    if disagreements:
        raise ValueError("the census and IMEPLAN's crosswalk disagree: " + "; ".join(disagreements))
    shares = coverage(universe)
    conapo = load_conapo(Path(data_dir) / CONAPO_CSV if data_dir is not None else None)
    base, target = years()
    constraints = {}
    index = []
    diagnostics = {}
    for year in (base, target):
        frames = build_constraints(universe, year, conapo, shares)
        constraints.update(frames)
        idx = constraint_index(year).assign(year=year)
        index.append(idx)
        diagnostics[year] = diagnostic(records, frames, idx)
    index = pd.concat(index, ignore_index=True)
    index = index.drop_duplicates(["file", "target_column"]).reset_index(drop=True)
    return records._replace(constraints=constraints, index=index, diagnostics=diagnostics, coverage=shares)


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


def write(files, out_dir):
    """Write the set under `out_dir`; returns the paths written."""
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
    readme = resources.files("eodgdl.reweight") / "README.md"
    with resources.as_file(readme) as src:
        shutil.copy(src, out_dir / "README.md")
        written.append(out_dir / "README.md")
    return written
