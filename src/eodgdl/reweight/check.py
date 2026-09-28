"""Check a written reweighting input set the way TMG.SurveyReweight will read it.

Imports nothing from the rest of the package: the point is that the files stand on their
own. Every problem is a string; the list is empty when the set is loadable and every
constraint is feasible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = {
    "households": ("HouseholdRecords.csv", ["HouseholdID", "HouseholdTAZ"]),
    "persons": ("PersonRecords.csv", ["PersonID", "HouseholdID"]),
    "trips": ("TripRecords.csv", ["HouseholdID", "PersonID"]),
}
PARTITIONS = {
    "persons": [
        (["Male", "Female"], "Persons", "=="),
        (["Age6_11", "Age12_14", "Age15_17", "Age18_24", "Age25_59", "Age60p"], "Persons", "=="),
        (["Employed", "Unemployed", "Inactive"], "Persons", "<="),
        (["Employed", "NotEmployed"], "Persons", "<="),
    ],
}


def _read(path):
    return pd.read_csv(path)


def check(out_dir):
    """Problems with the files under `out_dir`; empty when there are none."""
    out_dir = Path(out_dir)
    problems: list[str] = []

    def need(name):
        path = out_dir / name
        if not path.exists():
            problems.append(f"{name}: missing")
            return None
        return _read(path)

    zones = need("ZoneSystem.csv")
    records = {t: need(f) for t, (f, _) in KEYS.items()}
    index = need("Constraints/constraints.csv")
    if zones is None or any(r is None for r in records.values()) or index is None:
        return problems

    # --- zone system: ints, unique TAZ, every map column present
    for col in ["TAZ", "Zone", "Municipality", "Region"]:
        if col not in zones.columns:
            problems.append(f"ZoneSystem.csv: no column {col}")
        elif not pd.api.types.is_integer_dtype(zones[col]):
            problems.append(f"ZoneSystem.csv: {col} is not integer")
    if "TAZ" in zones.columns and zones.TAZ.duplicated().any():
        problems.append("ZoneSystem.csv: duplicated TAZ")
    if problems:
        return problems

    # --- records: integer keys, unique ids, joins, numeric attributes
    for table, (file, keys) in KEYS.items():
        df = records[table]
        for k in keys:
            if k not in df.columns:
                problems.append(f"{file}: no column {k}")
            elif not pd.api.types.is_integer_dtype(df[k]):
                problems.append(f"{file}: {k} is not integer")
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                problems.append(f"{file}: {c} is not numeric")
            elif df[c].isna().any():
                problems.append(f"{file}: {c} has {int(df[c].isna().sum())} empty cells")
    if problems:
        return problems
    hh, pp, tt = records["households"], records["persons"], records["trips"]
    if hh.HouseholdID.duplicated().any():
        problems.append("HouseholdRecords.csv: duplicated HouseholdID")
    if pp.PersonID.duplicated().any():
        problems.append("PersonRecords.csv: duplicated PersonID")
    if not hh.HouseholdTAZ.isin(zones.TAZ).all():
        problems.append("HouseholdRecords.csv: a HouseholdTAZ is not in ZoneSystem.csv")
    if not pp.HouseholdID.isin(hh.HouseholdID).all():
        problems.append("PersonRecords.csv: a HouseholdID has no household record")
    if not tt.HouseholdID.isin(hh.HouseholdID).all():
        problems.append("TripRecords.csv: a HouseholdID has no household record")
    if not tt.PersonID.isin(pp.PersonID).all():
        problems.append("TripRecords.csv: a PersonID has no person record")
    if not tt.PersonID.map(pp.set_index("PersonID").HouseholdID).eq(tt.HouseholdID).all():
        problems.append("TripRecords.csv: a trip's HouseholdID is not its person's")

    # --- dummies partition what they should
    for table, rules in PARTITIONS.items():
        df = records[table]
        for parts, total, op in rules:
            if not all(c in df.columns for c in parts + [total]):
                continue
            s = df[parts].sum(axis=1)
            bad = ~(s == df[total]) if op == "==" else ~(s <= df[total])
            if bad.any():
                problems.append(f"{KEYS[table][0]}: {'+'.join(parts)} {op} {total} fails on {int(bad.sum())} rows")

    # --- constraints: every target column present, every map geography covered, feasible
    taz_of = {
        "households": hh.HouseholdTAZ,
        "persons": pp.HouseholdID.map(hh.set_index("HouseholdID").HouseholdTAZ),
        "trips": tt.HouseholdID.map(hh.set_index("HouseholdID").HouseholdTAZ),
    }
    files: dict[str, pd.DataFrame] = {}
    for row in index.itertuples(index=False):
        if row.file not in files:
            files[row.file] = need(f"Constraints/{row.file}")
        cf = files[row.file]
        if cf is None:
            continue
        geo = cf.columns[0]
        if geo != row.map_column:
            problems.append(f"{row.file}: first column is {geo}, expected {row.map_column}")
            continue
        if row.target_column not in cf.columns:
            problems.append(f"{row.file}: no column {row.target_column}")
            continue
        if list(cf.columns).index(row.target_column) != row.column_index:
            problems.append(f"{row.file}: {row.target_column} is not at column {row.column_index}")
        if cf[geo].duplicated().any():
            problems.append(f"{row.file}: duplicated {geo}")
        wanted = set(zones[row.map_column])
        missing = wanted - set(cf[geo])
        if missing:
            problems.append(f"{row.file}: no row for {geo} {sorted(missing)} (the tool would target 0 there)")
        target = cf.set_index(geo)[row.target_column]
        if target.isna().any() or (target < 0).any():
            problems.append(f"{row.file}: {row.target_column} has empty or negative targets")
            continue
        df = records[row.table]
        attrs = row.matching_attributes.split(";")
        absent = [a for a in attrs if a not in df.columns]
        if absent:
            problems.append(f"{row.file}: {row.target_column} matches {absent}, not in {KEYS[row.table][0]}")
            continue
        value = df[attrs].prod(axis=1)
        geo_of = taz_of[row.table].map(zones.set_index("TAZ")[row.map_column])
        present = set(geo_of[value > 0])
        infeasible = sorted(g for g, t in target.items() if t > 0 and g not in present)
        if infeasible:
            problems.append(
                f"{row.file}: {row.target_column} > 0 but no {row.table} record with "
                f"{'*'.join(attrs)} > 0 in {geo} {infeasible}"
            )
    return problems
