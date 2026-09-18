"""spec.yaml: what each record attribute and each constraint target is.

The spec ships inside the package. Nothing else in ``eodgdl.reweight`` holds a lookup:
``records.py`` realises the ``attributes`` half, ``targets.py`` the ``constraints`` half,
and ``check_spec()`` keeps the two consistent with each other.
"""
from __future__ import annotations

import functools
import re
from importlib import resources

import pandas as pd
import yaml

TABLES = ("households", "persons", "trips")
TABLE_LABEL = {"households": "Household", "persons": "Person", "trips": "Trip"}
GEOGRAPHIES = ("zone", "municipality", "region")
GEOGRAPHY_COLUMN = {"zone": "Zone", "municipality": "Municipality", "region": "Region"}
KEYS = {
    "households": ["HouseholdID", "HouseholdTAZ"],
    "persons": ["PersonID", "HouseholdID"],
    "trips": ["HouseholdID", "PersonID"],
}
_TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


@functools.cache
def load_spec():
    """spec.yaml as nested dicts. Cached."""
    text = (resources.files("eodgdl.reweight") / "spec.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def attributes(table):
    """{attribute: entry} for one record table, in spec order."""
    return dict(load_spec()["attributes"][table])


def constraints(table):
    """{target column: entry} for one record table, in spec order."""
    return dict(load_spec()["constraints"].get(table) or {})


def matching_attributes(target, table):
    """The record attributes a constraint multiplies (the tool's MatchingAttributes)."""
    return list(constraints(table)[target].get("attributes") or [target])


def census_columns(expression):
    """The census column names an expression refers to."""
    return sorted(set(_TOKEN.findall(expression)))


def constraint_file(table, geography, year=None):
    """File name for one (table, geography) group, with the year when it is scaled."""
    name = f"{TABLE_LABEL[table]}ConstraintsBy{GEOGRAPHY_COLUMN[geography]}"
    return f"{name}_{year}.csv" if year is not None else f"{name}.csv"


def is_scaled(table, geography):
    """Whether the file for (table, geography) has a year suffix: any target with `conapo`."""
    return any(
        "conapo" in e for e in constraints(table).values() if e["geography"] == geography
    )


def constraint_index(year):
    """One row per target column, for the year's constraint set: where it is and what it matches.

    Columns: file, target_column, column_index (0-based, the tool's TargetColumn), table,
    geography, map_column (the ZoneSystem.csv column the CategoryMap maps to; the geography
    column of the file is always column 0), matching_attributes (';'-joined), scaled.
    """
    rows = []
    for table in TABLES:
        groups: dict[str, list[str]] = {}
        for target, entry in constraints(table).items():
            groups.setdefault(entry["geography"], []).append(target)
        for geography, targets in groups.items():
            scaled = is_scaled(table, geography)
            file = constraint_file(table, geography, year if scaled else None)
            for i, target in enumerate(targets):
                rows.append({
                    "file": file,
                    "target_column": target,
                    "column_index": i + 1,
                    "table": table,
                    "geography": geography,
                    "map_column": GEOGRAPHY_COLUMN[geography],
                    "matching_attributes": ";".join(matching_attributes(target, table)),
                    "scaled": bool(scaled and "conapo" in constraints(table)[target]),
                })
    return pd.DataFrame(rows)


def check_spec():
    """Inconsistencies inside spec.yaml, as human-readable strings. Empty when it is sound."""
    spec = load_spec()
    problems = []
    bands = set(spec["conapo"]["bands"])
    for table in TABLES:
        attrs = attributes(table)
        for name, entry in attrs.items():
            kinds = [k for k in ("constant", "values", "range", "count", "sum") if k in entry]
            if len(kinds) > 1:
                problems.append(f"{table}.{name}: has {kinds}; one way to build it only")
            if not kinds and "source" not in entry:
                problems.append(f"{table}.{name}: neither a source nor a constant")
            for other in entry.get("sum") or []:
                if other not in attrs:
                    problems.append(f"{table}.{name}: sums {other!r}, not an attribute")
        for target, entry in constraints(table).items():
            if entry.get("geography") not in GEOGRAPHIES:
                problems.append(f"{table}.{target}: geography {entry.get('geography')!r} is not one of {GEOGRAPHIES}")
            for attr in matching_attributes(target, table):
                if attr not in attrs:
                    problems.append(f"{table}.{target}: matches {attr!r}, not a {table} attribute")
            if ("census" in entry) == ("constant" in entry):
                problems.append(f"{table}.{target}: needs exactly one of census / constant")
            if "constant" in entry:
                if not all(isinstance(k, int) for k in entry["constant"]):
                    problems.append(f"{table}.{target}: constant keys must be integer geography ids")
                if "provenance" not in entry:
                    problems.append(f"{table}.{target}: a constant needs a provenance")
            if "universe_share" in entry and ("constant" not in entry or entry.get("geography") != "municipality"):
                problems.append(f"{table}.{target}: universe_share needs a municipality constant")
            if "conapo" in entry and entry["conapo"] not in bands:
                problems.append(f"{table}.{target}: conapo band {entry['conapo']!r} is not defined")
            if entry.get("sex") not in (None, "F", "M"):
                problems.append(f"{table}.{target}: sex must be F or M")
    for name in spec["attributes"]:
        if name not in TABLES:
            problems.append(f"attributes.{name}: not one of {TABLES}")
    for name in spec["constraints"]:
        if name not in TABLES:
            problems.append(f"constraints.{name}: not one of {TABLES}")
    return problems
