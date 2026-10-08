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
GEOGRAPHIES = ("taz", "mtaz", "municipality", "region")
GEOGRAPHY_COLUMN = {"taz": "TAZ", "mtaz": "MTAZ", "municipality": "Municipality", "region": "Region"}
# The tool's base category per set (spec.yaml's zones.bases): its column in ZoneSystem.csv, and the
# geographies its constraints may read. Only the TAZ base reads the TAZ, its own category.
BASE_GEOGRAPHIES = {"mtazmun": ("mtaz", "municipality", "region"), "taz": GEOGRAPHIES}
_TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def bases():
    """{base: {column, directory}}: the two sets, spec.yaml's zones.bases."""
    return dict(load_spec()["zones"]["bases"])


def base_column(base):
    """The base category's column: ZoneSystem.csv's first, and HouseholdRecords.csv's second as Household<column>."""
    if base not in bases():
        raise ValueError(f"base {base!r} is not one of {sorted(bases())}")
    return bases()[base]["column"]


def keys(base="mtazmun"):
    """The key columns of each record table in the set of `base`, in the order the tool's loader reads them: the
    household's base category second, the other set's category third (an attribute no constraint reads)."""
    other = [c["column"] for b, c in bases().items() if b != base]
    return {
        "households": ["HouseholdID", *(f"Household{c}" for c in [base_column(base), *other])],
        "persons": ["PersonID", "HouseholdID"],
        "trips": ["HouseholdID", "PersonID"],
    }


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


def base_constraints(table, base="mtazmun"):
    """{target column: entry} for one record table in the set of `base`: the constraints whose geography it maps to."""
    return {t: e for t, e in constraints(table).items() if e["geography"] in BASE_GEOGRAPHIES[base]}


def census_columns(expression):
    """The census column names an expression refers to."""
    return sorted(set(_TOKEN.findall(expression)))


def constraint_file(table, geography, year=None):
    """File name for one (table, geography) group, with the year when it is scaled."""
    name = f"{TABLE_LABEL[table]}ConstraintsBy{GEOGRAPHY_COLUMN[geography]}"
    return f"{name}_{year}.csv" if year is not None else f"{name}.csv"


GROWTH_KEYS = ("conapo", "vmrc", "sum_of", "rows_of")   # rows_of: a census target's values, grown with it


def is_scaled(table, geography):
    """Whether the file for (table, geography) has a year suffix: any target with a growth
    source (`conapo` or `vmrc`, or the census target `rows_of` names)."""
    return any(
        any(k in e for k in GROWTH_KEYS)
        for e in constraints(table).values() if e["geography"] == geography
    )


def constraint_index(year, base="mtazmun"):
    """One row per target column, for the year's constraint set of `base`: where it is and what it matches.

    Columns: file, target_column, column_index (0-based, the tool's TargetColumn), table,
    geography, map_column (the ZoneSystem.csv column the CategoryMap maps to, or the base
    category's own column for a constraint on it, which takes no CategoryMap; the geography
    column of the file is always column 0), matching_attributes (';'-joined), scaled.
    """
    rows = []
    for table in TABLES:
        groups: dict[str, list[str]] = {}
        for target, entry in base_constraints(table, base).items():
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
                    "scaled": bool(scaled and any(k in constraints(table)[target] for k in GROWTH_KEYS)),
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
            if "imputed" in entry:
                from eodgdl.data import MODEL_SUFFIX, models_dir

                imputed = entry["imputed"]
                if "values" not in entry or not isinstance(entry.get("source"), str):
                    problems.append(f"{table}.{name}: imputed fills a `values` attribute's missing answers")
                if not isinstance(imputed, dict) or not {"task", "bundle", "class"} <= set(imputed) or set(imputed) - {"task", "bundle", "class", "complement"}:
                    problems.append(f"{table}.{name}: imputed is {{task, bundle, class, complement}}")
                elif not imputed["bundle"].endswith(MODEL_SUFFIX) or not (models_dir() / imputed["bundle"]).exists():
                    problems.append(f"{table}.{name}: imputed bundle {imputed['bundle']!r} is not a model file under models/")
        for target, entry in constraints(table).items():
            if entry.get("geography") not in GEOGRAPHIES:
                problems.append(f"{table}.{target}: geography {entry.get('geography')!r} is not one of {GEOGRAPHIES}")
            for attr in matching_attributes(target, table):
                if attr not in attrs:
                    problems.append(f"{table}.{target}: matches {attr!r}, not a {table} attribute")
            if sum(k in entry for k in ("census", "constant", "rows_of")) != 1:
                problems.append(f"{table}.{target}: needs exactly one of census / constant / rows_of")
            if (entry.get("geography") == "taz") != ("rows_of" in entry):
                problems.append(f"{table}.{target}: a taz target is rows_of an mtaz target, and only a taz target is")
            if "rows_of" in entry:
                from eodgdl.reweight.targets import AGEB_TARGETS

                of = constraints(table).get(entry["rows_of"])
                if entry["rows_of"] not in AGEB_TARGETS or AGEB_TARGETS[entry["rows_of"]][0] != table or of is None:
                    problems.append(f"{table}.{target}: rows_of {entry['rows_of']!r} is not a {table} target "
                                    "ZoneSystemAGEBs.csv carries (targets.AGEB_TARGETS)")
                elif of.get("geography") != "mtaz" or matching_attributes(target, table) != matching_attributes(entry["rows_of"], table):
                    problems.append(f"{table}.{target}: rows_of {entry['rows_of']!r} must be an mtaz target matching the same attributes")
            if "constant" in entry:
                if not all(isinstance(k, int) for k in entry["constant"]):
                    problems.append(f"{table}.{target}: constant keys must be integer geography ids")
                if "provenance" not in entry:
                    problems.append(f"{table}.{target}: a constant needs a provenance")
            if "row_scale" in entry and (entry["row_scale"] not in ("persons", "dwellings") or "census" not in entry):
                problems.append(f"{table}.{target}: row_scale must be persons or dwellings, on a census target")
            if "universe_share" in entry and ("constant" not in entry or entry.get("geography") != "municipality"):
                problems.append(f"{table}.{target}: universe_share needs a municipality constant")
            if "conapo" in entry and entry["conapo"] not in bands:
                problems.append(f"{table}.{target}: conapo band {entry['conapo']!r} is not defined")
            if "rate" in entry:
                if entry["rate"] not in (spec.get("rates") or {}):
                    problems.append(f"{table}.{target}: rate {entry['rate']!r} is not in rates")
                if "conapo" not in entry or "census" not in entry:
                    problems.append(f"{table}.{target}: a rate scales a census target on top of conapo")
            if "vmrc" in entry:
                if "conapo" in entry:
                    problems.append(f"{table}.{target}: one growth source only (conapo or vmrc)")
                if "census" not in entry:
                    problems.append(f"{table}.{target}: vmrc growth applies to census targets only")
                if entry["vmrc"] not in spec["vmrc"]["columns"]:
                    problems.append(f"{table}.{target}: vmrc column {entry['vmrc']!r} is not listed")
            if entry.get("sex") not in (None, "F", "M"):
                problems.append(f"{table}.{target}: sex must be F or M")
            peers = spec["constraints"][table]
            if "sum_of" in entry:
                if "conapo" in entry or "vmrc" in entry or "census" not in entry:
                    problems.append(f"{table}.{target}: sum_of grows a census target from its parts, with no growth of its own")
                for part in entry["sum_of"]:
                    p = peers.get(part)
                    if p is None or p.get("geography") != entry.get("geography") or "census" not in p:
                        problems.append(f"{table}.{target}: sum_of names {part!r}, not a census target of its geography")
                    elif "sum_of" in p or "grow_within" in p:
                        problems.append(f"{table}.{target}: sum_of part {part!r} is itself derived")
            if "grow_with" in entry:
                if "conapo" not in entry or "census" not in entry:
                    problems.append(f"{table}.{target}: grow_with regrows a census target grown by conapo")
                for part in entry["grow_with"]:
                    p = peers.get(part)
                    if p is None or p.get("geography") != entry.get("geography") or "census" not in p:
                        problems.append(f"{table}.{target}: grow_with names {part!r}, not a census target of its geography")
                    elif any(k in p for k in ("sum_of", "grow_within", "grow_with")):
                        problems.append(f"{table}.{target}: grow_with part {part!r} is itself derived")
            if "grow_within" in entry:
                g = entry["grow_within"]
                if "conapo" not in entry or "census" not in entry:
                    problems.append(f"{table}.{target}: grow_within rescales a census target grown by conapo")
                if target not in g.get("group", []):
                    problems.append(f"{table}.{target}: grow_within's group must include the target itself")
                for part in [*g.get("group", []), *g.get("total", [])]:
                    p = peers.get(part)
                    if p is None or p.get("geography") != entry.get("geography") or "census" not in p:
                        problems.append(f"{table}.{target}: grow_within names {part!r}, not a census target of its geography")
                for part in g.get("group", []):
                    if part in peers and peers[part].get("grow_within") != g:
                        problems.append(f"{table}.{target}: {part!r} must carry the same grow_within")
    for name in spec["attributes"]:
        if name not in TABLES:
            problems.append(f"attributes.{name}: not one of {TABLES}")
    for name in spec["constraints"]:
        if name not in TABLES:
            problems.append(f"constraints.{name}: not one of {TABLES}")
    return problems
