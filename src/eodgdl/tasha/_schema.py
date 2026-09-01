"""The TASHA model's target tables, and how the EOD GDL 2023 survey fills them.

Three pieces, two of them data:

    model_schema.yaml   the contract - which columns, which codes, what each
                        one means. Says nothing about the survey.
    mappings.yaml       one entry per output column: the eodgdl source column,
                        the {raw answer: code} lookup, and the caveats.
    _schema.py          this file - loads both, hands you .map()-ready dicts,
                        checks the mappings against the contract, and checks a
                        produced table against the contract.

Both YAML files ship inside the package, so this works offline and from an
installed wheel. See ``eodgdl/tasha/README.md`` for the full guide.

Typical use::

    from eodgdl import tasha

    tasha.domain("PurposeDestination")  # {'H': 'Home', 'W': 'Work - first...', ...}
    tasha.mapping("Mode")               # source column, lookup, override, notes
    tasha.gaps()                        # what is assumed, constant or unresolved

    trips["Mode"] = raw.modo_principal.map(tasha.build_map("Mode"))

    for problem in tasha.validate_all(households, people, trips):
        print(problem)

From the shell::

    eodgdl tasha check                  # mappings vs. contract
    eodgdl tasha validate output/       # produced CSVs vs. contract
"""

import functools
from importlib import resources

import pandas as pd
import yaml


def _read(name):
    text = (resources.files("eodgdl.tasha") / name).read_text(encoding="utf-8")
    return yaml.safe_load(text)


@functools.cache
def load_schema():
    """The contract, as nested dicts. Cached."""
    return _read("model_schema.yaml")


@functools.cache
def load_mappings():
    """The EOD GDL mappings, as nested dicts. Cached."""
    return _read("mappings.yaml")


# ---------------------------------------------------------------- the contract


def tables():
    return list(load_schema()["tables"])


def columns(table):
    """Column names for `table`, in schema order."""
    return list(load_schema()["tables"][table]["columns"])


def find_table(column):
    """The table a column belongs to. Raises if the name is ambiguous."""
    hits = [t for t in tables() if column in load_schema()["tables"][t]["columns"]]
    if not hits:
        raise KeyError(f"{column!r} is not in the schema")
    if len(hits) > 1:
        raise KeyError(f"{column!r} appears in {hits}; pass table= to disambiguate")
    return hits[0]


def column_spec(column, table=None):
    table = table or find_table(column)
    return load_schema()["tables"][table]["columns"][column]


def domain(column, table=None):
    """Legal codes for a column as {code: label}, or {} if it is not coded."""
    return dict(column_spec(column, table).get("domain") or {})


def required_columns(table):
    return [c for c in columns(table) if column_spec(c, table).get("required")]


def zone_columns(table):
    """Columns holding a zone id. Read these with dtype=str.

    Most ids are all-digit, so a plain ``read_csv`` can turn a column into int64
    depending on which rows are present, making the dtype vary with the data.
    """
    return [c for c in columns(table) if column_spec(c, table).get("role") == "zone"]


# ---------------------------------------------------------------- the mappings


def mapping(column, table=None):
    """The mapping entry for one output column."""
    table = table or find_table(column)
    try:
        return load_mappings()[table][column]
    except KeyError:
        raise KeyError(f"no mapping for {table}.{column} in mappings.yaml") from None


def build_map(column, table=None):
    """{raw survey answer: output code} for one column, ready for .map().

    Only the literal lookup. Anything the mapping expresses as `derivation`,
    `constant` or `override` prose is not in here - read mapping() for those.
    """
    return dict(mapping(column, table).get("values") or {})


def gaps():
    """Every column whose mapping is assumed, constant or unresolved.

    Also flags schema columns that mappings.yaml does not mention at all.
    """
    rows = []
    for table in tables():
        entries = load_mappings().get(table, {})
        for column in columns(table):
            entry = entries.get(column)
            if entry is None:
                rows.append({"table": table, "column": column,
                             "status": "unmapped", "note": ""})
                continue
            status = entry.get("status", "ok")
            if status != "ok":
                rows.append({"table": table, "column": column, "status": status,
                             "note": (entry.get("note") or "").strip()})
    return pd.DataFrame(rows, columns=["table", "column", "status", "note"])


def check_mappings():
    """Check mappings.yaml against model_schema.yaml.

    Returns a list of human-readable problems; empty means they agree.
    """
    problems = []
    for table in tables():
        entries = load_mappings().get(table, {})
        for extra in set(entries) - set(columns(table)):
            problems.append(f"{table}.{extra}: mapped but not in the schema")
        for column in columns(table):
            entry = entries.get(column)
            if entry is None:
                problems.append(f"{table}.{column}: no mapping entry")
                continue
            if not any(k in entry for k in ("values", "constant", "derivation")):
                problems.append(
                    f"{table}.{column}: mapping has no values, constant or derivation"
                )
            legal = domain(column, table)
            if not legal:
                continue
            produced = set((entry.get("values") or {}).values())
            produced |= set(((entry.get("override") or {}).get("values") or {}).values())
            for key in ("default", "constant"):
                if key in entry:
                    produced.add(entry[key])
            illegal = sorted(str(c) for c in produced if str(c) not in map(str, legal))
            if illegal:
                problems.append(
                    f"{table}.{column}: mapping produces codes outside the domain {illegal}"
                )
    return problems


# ---------------------------------------------------------------- validation


def _is_integral(s):
    s = s.dropna()
    if s.empty or pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        return bool((s == s.round()).all())
    return False


def validate(df, table, *, check_dtypes=True):
    """Check one produced table against the contract.

    Returns a list of human-readable problems; an empty list means it conforms.
    Nothing is raised and nothing is modified.
    """
    problems = []
    p = problems.append
    spec = load_schema()["tables"][table]

    missing = [c for c in required_columns(table) if c not in df.columns]
    if missing:
        p(f"{table}: missing required columns {missing}")
    unknown = [c for c in df.columns if c not in columns(table)]
    if unknown:
        p(f"{table}: columns not in the schema {unknown}")

    key = spec.get("key", [])
    if all(k in df.columns for k in key):
        dupes = int(df.duplicated(subset=key).sum())
        if dupes:
            p(f"{table}: {dupes} rows duplicate the key {key}")

    id_lengths = set(load_schema()["zones"]["id_lengths"])

    for column in columns(table):
        if column not in df.columns:
            continue
        cspec = column_spec(column, table)
        s = df[column]

        nulls = int(s.isna().sum())
        if nulls and cspec.get("required"):
            p(f"{table}.{column}: {nulls} null values in a required column")

        sentinels = {str(k) for k in (cspec.get("sentinel") or {})}

        legal = domain(column, table)
        if legal:
            seen = set(s.dropna().astype(str).unique())
            illegal = sorted(seen - {str(c) for c in legal} - sentinels)
            if illegal:
                counts = {v: int((s.astype(str) == v).sum()) for v in illegal[:8]}
                p(f"{table}.{column}: values outside the domain {counts}")

        if check_dtypes:
            dtype = cspec.get("dtype")
            if dtype == "int" and not _is_integral(s):
                p(f"{table}.{column}: expected integers, got {s.dtype}")
            elif dtype == "float" and not pd.api.types.is_numeric_dtype(s):
                p(f"{table}.{column}: expected a numeric column, got {s.dtype}")

        if cspec.get("role") == "zone":
            # Zone ids are INEGI codes, not numbers: check their shape, and warn
            # if a reader has parsed the all-digit ids into integers.
            ids = s.dropna().astype(str)
            if pd.api.types.is_numeric_dtype(s):
                p(
                    f"{table}.{column}: zone ids read as {s.dtype}, not strings; "
                    "re-read with dtype=str, or ids that carry a trailing letter "
                    "will not compare equal to ids that do not"
                )
            bad_shape = sorted(set(ids[~ids.str.len().isin(id_lengths) & ~ids.isin(sentinels)]))
            if bad_shape:
                p(
                    f"{table}.{column}: {len(bad_shape)} zone ids are not "
                    f"{sorted(id_lengths)} characters, e.g. {bad_shape[:5]}"
                )
        elif pd.api.types.is_numeric_dtype(s):
            lo, hi = cspec.get("range") or (None, None)
            allowed = (
                s.isin([float(k) for k in sentinels])
                if sentinels
                else pd.Series(False, index=s.index)
            )
            if lo is not None:
                bad = int(((s < lo) & ~allowed).sum())
                if bad:
                    p(f"{table}.{column}: {bad} values below {lo}")
            if hi is not None:
                bad = int(((s > hi) & ~allowed).sum())
                if bad:
                    p(f"{table}.{column}: {bad} values above {hi}")

    problems.extend(_invariants(df, table))
    return problems


def _invariants(df, table):
    """Cross-column rules the schema states in prose."""
    out = []
    if table == "people" and {"EmploymentStatus", "Occupation"} <= set(df.columns):
        mismatch = int(((df.EmploymentStatus == "O") != (df.Occupation == "O")).sum())
        if mismatch:
            out.append(
                f"people: {mismatch} rows where EmploymentStatus=='O' and "
                "Occupation=='O' disagree; a non-worker must have no sector"
            )
    if table == "trips":
        if "PurposeOrigin" in df.columns:
            leaked = int(df.PurposeOrigin.isin(["R", "C"]).sum())
            if leaked:
                out.append(
                    f"trips: {leaked} rows where PurposeOrigin is R or C; the "
                    "return codes belong to PurposeDestination only"
                )
        if {"HouseholdId", "PersonNumber", "TripNumber"} <= set(df.columns):
            first = df.groupby(["HouseholdId", "PersonNumber"]).TripNumber.min()
            if (first != 1).any():
                out.append(
                    f"trips: {int((first != 1).sum())} people whose first "
                    "TripNumber is not 1"
                )
    return out


def validate_all(households=None, people=None, trips=None):
    """Validate the three tables together, including the joins between them."""
    problems = []
    for table, df in [("households", households), ("people", people), ("trips", trips)]:
        if df is not None:
            problems.extend(validate(df, table))

    if households is not None and people is not None:
        orphans = int((~people.HouseholdId.isin(households.HouseholdId)).sum())
        if orphans:
            problems.append(
                f"people: {orphans} rows whose HouseholdId is not in the "
                "household table"
            )
    if people is not None and trips is not None:
        known = set(zip(people.HouseholdId, people.PersonNumber))
        orphans = set(zip(trips.HouseholdId, trips.PersonNumber)) - known
        if orphans:
            problems.append(
                f"trips: {len(orphans)} (HouseholdId, PersonNumber) pairs with "
                "no matching person"
            )
    return problems
