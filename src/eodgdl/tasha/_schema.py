"""The TASHA model's target tables, and how the EOD GDL 2023 survey fills them.

Three pieces, two of them data:

    model_schema.yaml   the contract - which columns, which codes, what each
                        one means. Says nothing about the survey.
    mappings.yaml       one entry per output column: the eodgdl source column,
                        the {raw answer: code} lookup, and the caveats.
    _schema.py          this file - loads both, hands you .map()-ready dicts,
                        checks the mappings against the contract AND against the
                        survey's own pandera schemas, and checks a produced table
                        against the contract.

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


# load_eod unpivots trips.traslado{1..5}_* into these three columns on `legs`;
# they take their dtype from the first leg's column in trips_schema.
_LEG_COLUMNS = {
    "traslado_medio": "traslado1_medio",
    "traslado_min": "traslado1_min",
    "traslado_pago": "traslado1_pago",
}
# Index levels rather than columns on the tables load_eod returns.
_INDEX_LEVELS = ("folio_vivienda", "folio_habitante", "folio_viaje", "folio_traslado")


@functools.cache
def survey_columns():
    """{source column: {grain: levels or None}} over the pandera survey schemas.

    The keys are every name a mappings.yaml `source` may legally use: the
    columns of viv, hab and trips, the three columns ``load_eod`` unpivots onto
    legs, and the four folio_* index levels. The value records the column's
    categorical levels where it has them, so a lookup key can be checked against
    the answers the survey can actually produce.
    """
    from eodgdl.schemas import hab_schema, trips_schema, viv_schema

    def levels(column):
        cats = getattr(column.dtype, "categories", None)
        return None if cats is None else frozenset(map(str, cats))

    found = {}
    for grain, schema in [("viv", viv_schema), ("hab", hab_schema), ("trips", trips_schema)]:
        for name, column in schema.columns.items():
            found.setdefault(name, {})[grain] = levels(column)
    for name, first_leg in _LEG_COLUMNS.items():
        found.setdefault(name, {})["legs"] = levels(trips_schema.columns[first_leg])
    for level in _INDEX_LEVELS:
        found.setdefault(level, {}).setdefault("index", None)
    return found


def _lookups(entry):
    """The (label, source names, values) pairs an entry defines: main, override."""
    override = entry.get("override") or {}
    for label, source, values in [
        ("source", entry.get("source"), entry.get("values")),
        ("override source", override.get("source"), override.get("values")),
    ]:
        if source is None:
            continue
        # A source may be written bare (`ageb`) or grain-qualified (`legs.traslado_min`).
        names = [source] if isinstance(source, str) else list(source)
        yield label, [n.split(".")[-1] for n in names], (values or {})


def _check_sources(table, column, entry):
    """The survey half of the check: the sources exist, the keys are real answers.

    ``check_mappings`` compares a mapping against the contract; this compares the
    same mapping against the survey it claims to read. It catches the two ways a
    hand-edited mappings.yaml silently stops working: a source column that the
    rename map or the pandera schema no longer produces, and a lookup key that is
    not one of the answers that column can take (a typo, or a level the survey
    dropped) — which `.map()` would quietly turn into NaN rather than an error.
    """
    problems = []
    survey = survey_columns()
    for label, names, values in _lookups(entry):
        for name in names:
            if name not in survey:
                problems.append(
                    f"{table}.{column}: {label} {name!r} is not a column of "
                    "viv, hab, trips or legs"
                )
        for key in values:
            for name in names:
                # Every grain the column appears in must accept the key; a
                # non-categorical column (a free string, a count) constrains nothing.
                coded = [lv for lv in survey.get(name, {}).values() if lv is not None]
                if coded and not all(str(key) in lv for lv in coded):
                    problems.append(
                        f"{table}.{column}: {label} {name} has no answer {key!r}, "
                        "so the lookup entry can never match"
                    )
    return problems


def check_mappings():
    """Check mappings.yaml against model_schema.yaml and against the survey.

    Two directions. Against the contract: every column mapped, no mapping for a
    column the contract lacks, no code produced outside a column's domain.
    Against the survey: every `source` is a real column of viv / hab / trips /
    legs, and every `values` key is an answer that column can actually take.

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
            problems.extend(_check_sources(table, column, entry))
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
    if table == "households" and "HouseholdId" in df.columns:
        # "Must be dense and 0-based; people and trips join on it."
        ids = df.HouseholdId.dropna()
        if len(ids):
            lo, hi = int(ids.min()), int(ids.max())
            if lo != 0 or hi != len(ids) - 1:
                out.append(
                    f"households: HouseholdId must be dense and 0-based, but "
                    f"{len(ids)} rows run {lo}..{hi}"
                )
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
        if {"PurposeOrigin", "PurposeDestination"} <= set(df.columns):
            # "A return home made from home is not a trip": the chain says the
            # person is already at home, so the row cannot be scheduled.
            home_to_home = int(((df.PurposeOrigin == "H") & (df.PurposeDestination == "H")).sum())
            if home_to_home:
                out.append(
                    f"trips: {home_to_home} rows go from H to H; a return home made "
                    "from home is not a trip, and the builder must drop it"
                )
        if {"HouseholdId", "PersonNumber", "TripNumber"} <= set(df.columns):
            chains = df.groupby(["HouseholdId", "PersonNumber"]).TripNumber.agg(
                ["min", "max", "count", "nunique"]
            )
            broken = int((chains["min"] != 1).sum())
            if broken:
                out.append(f"trips: {broken} people whose first TripNumber is not 1")
            # "Consecutive from 1 within each person": with distinct numbers
            # starting at 1, running 1..n is exactly max == count.
            gapped = int(
                ((chains["max"] != chains["count"]) & (chains["nunique"] == chains["count"])).sum()
            )
            if gapped:
                out.append(
                    f"trips: {gapped} people whose TripNumber is not consecutive; "
                    "the contract requires 1..n with no gaps"
                )
    return out


_CHAIN_COLUMNS = (
    "HouseholdId", "PersonNumber", "TripNumber", "StartTime",
    "PurposeOrigin", "PurposeDestination", "ZoneOrigin", "ZoneDestination",
)


def chain_report(trips):
    """Data-quality facts about the trip chains that the contract cannot demand.

    ``validate`` checks what a conforming table MUST satisfy. This reports, as
    counts, what a survey can leave imperfect and a builder cannot repair
    without inventing data: zone continuity (each trip starts where the
    previous one ended), start-time order along the chain, and tours that do
    not begin or end at home. Read it to judge a build, not to fail it.

    Returns a list of human-readable lines; empty means every chain is clean.
    Nothing is raised and nothing is modified.
    """
    missing = [c for c in _CHAIN_COLUMNS if c not in trips.columns]
    if missing:
        return [f"trips: chain report needs the columns {missing}"]
    person = ["HouseholdId", "PersonNumber"]
    t = trips.sort_values(person + ["TripNumber"], kind="stable")
    g = t.groupby(person, sort=False)
    prev_zone = g.ZoneDestination.shift(1)
    prev_time = g.StartTime.shift(1)
    has_prev = prev_zone.notna()

    def people(mask):
        return int(t.loc[mask, person].drop_duplicates().shape[0])

    lines = []
    breaks = has_prev & (t.ZoneOrigin.astype(str) != prev_zone.astype(str))
    if breaks.any():
        lines.append(
            f"trips: {int(breaks.sum())} trips ({people(breaks)} people) do not "
            "start in the zone the previous trip ended in"
        )
    earlier = has_prev & (t.StartTime < prev_time)
    if earlier.any():
        lines.append(
            f"trips: {int(earlier.sum())} trips ({people(earlier)} people) start "
            "earlier than the trip before them; StartTime is not in chain order "
            "for those people"
        )
    if "Duration" in t.columns:
        minutes = (t.StartTime // 100) * 60 + t.StartTime % 100
        prev_arrival = (minutes + t.Duration).groupby([t.HouseholdId, t.PersonNumber], sort=False).shift(1)
        early = prev_arrival.notna() & (minutes < prev_arrival)
        if early.any():
            lines.append(
                f"trips: {int(early.sum())} trips ({people(early)} people) start before the "
                "previous trip could have arrived, its StartTime plus its Duration"
            )
    ties = has_prev & (t.StartTime == prev_time)
    if ties.any():
        lines.append(
            f"trips: {int(ties.sum())} trips start at the same minute as the trip "
            "before them"
        )
    first = t[g.cumcount() == 0]
    away = int((first.PurposeOrigin != "H").sum())
    if away:
        lines.append(f"trips: {away} people whose first trip does not start at home")
    last = t[g.cumcount(ascending=False) == 0]
    out = int((last.PurposeDestination != "H").sum())
    if out:
        lines.append(f"trips: {out} people whose last trip does not end at home")
    return lines


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
        # NumberOfPersons is the reported size RAISED to the observed member
        # count, so it may exceed the person rows but must never fall below them.
        if "NumberOfPersons" in households.columns and households.HouseholdId.is_unique:
            observed = people.groupby("HouseholdId").size()
            reported = households.set_index("HouseholdId").NumberOfPersons
            short = int((reported.reindex(observed.index).fillna(0) < observed).sum())
            if short:
                problems.append(
                    f"households: {short} households whose NumberOfPersons is "
                    "below the number of rows they have in the person table"
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
