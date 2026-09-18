"""Review sheets: the trip chains pending a fix as a CSV to edit by hand, and the edits read back.

A review sheet is the chain browser's table (``notebooks/chain_browser.ipynb``)
as a CSV: one row per shipped trip of the persons selected, in chain order,
every cell reading ``shipped → cleaned`` where the chain rules changed the
value and the value alone where they did not; a row ``load_eod`` dropped
keeps its shipped values and reads ``dropped`` under ``status``; the
household's zone reads ``home`` wherever it appears. The person's home zone,
sex, age, occupation and repeated-diary flag repeat on every row, so the
sheet stands on its own in a spreadsheet.

Next to every column that may take a change stands an empty ``new …``
column: ``new start``, ``new motive``, ``new dest. type``, ``new orig. type``,
``new origin``, ``new destination``, ``new mode`` and ``new status``. The
hand edits go there — ``start`` as HH:MM, a motive or place type by its
label, a zone by its id or ``home``, a mode by its label; ``dropped`` under
``new status`` to take a row out of the chain, ``restored`` to bring back a
row ``load_eod`` dropped — and why under ``note``. The original columns are
never touched, so the sheet carries its own before and after and needs no
copy to be read back; ``leg min``, ``ajustes`` and ``problemas`` take no new
value (the first belongs to the legs, the other two are recomputed), and
rows cannot be added.

    rows = review.chain_rows(cleaned, shipped)                       # per shipped trip, shipped and cleaned values
    sheet = review.chain_sheet(rows, cleaned.hab, review.pending_persons(rows))
    review.write_sheet(sheet, "chain_review.csv")
    ...                                                              # edit by hand
    edits = review.sheet_edits(review.read_sheet("chain_review.csv"))
    verified = review.verify_edits(cleaned, shipped, edits)          # .tables, .sheet, .persons, .summary

``verify_edits`` applies the edits (:func:`apply_edits`: each one leaves a
``<field>:revision`` code in ``ajustes``) and recomputes ``problemas``
(:func:`eodgdl.eod.mark_issues`), so it says which defects the edits cleared
and which they left or made. ``eodgdl review export`` and ``eodgdl review
verify`` wrap the two halves.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from eodgdl.eod import (
    FIX_FLAG,
    ISSUE_CODES,
    ISSUE_FLAG,
    PERSON,
    EODTables,
    _home_zone,
    _leg_minutes,
    has_code,
    mark_issues,
)

KEYS = ["household", "person", "trip"]
PERSON_COLUMNS = ["home", "sex", "age", "occupation", "repeated"]
ARROW = " → "
MISSING = "—"
HOME = "home"
DROPPED = "dropped"
STATUS = "status"
NOTE = "note"


class Field(NamedTuple):
    key: str  # column stem in chain_rows: <key>_shipped, <key>_clean
    column: str | tuple  # the trips column(s) it reads and writes
    code: str  # the ajustes code an edit leaves


# The editable columns of the sheet, in sheet order.
FIELDS = {
    "start": Field("start", ("hora_inicio_h", "hora_inicio_m"), "hora:revision"),
    "motive": Field("motive", "motivo_viaje", "motivo:revision"),
    "dest. type": Field("dtype", "tipo_lugar_destino", "tipo_destino:revision"),
    "orig. type": Field("otype", "tipo_lugar_origen", "tipo_origen:revision"),
    "origin": Field("origin", "origen", "origen:revision"),
    "destination": Field("destination", "destino", "destino:revision"),
    "mode": Field("mode", "modo_principal", "modo:revision"),
}
READ_ONLY = ["leg min", FIX_FLAG, ISSUE_FLAG]
EDITABLE = list(FIELDS) + [STATUS]
NEW = "new "  # prefix of the columns that take the hand edits
RESTORED = "restored"  # under new status: bring back a row load_eod dropped
RESTORED_CODE = "fila:revision"


def new_column(name: str) -> str:
    """The column that takes the new value of ``name``: ``new start``, ``new status``, ..."""
    return NEW + name


COLUMNS = (
    KEYS[:2]
    + PERSON_COLUMNS
    + KEYS[2:]
    + [c for name in FIELDS for c in (name, new_column(name))]
    + READ_ONLY
    + [STATUS, new_column(STATUS), NOTE]
)

_ARROW_SPLIT = re.compile(r"\s*(?:→|->)\s*")
_HHMM = re.compile(r"^(\d{1,2}):(\d{2})$")


# ------------------------------------------------------------------ building


def _values(trips: pd.DataFrame, column) -> pd.Series:
    """A field of ``trips`` as plain objects: minutes from midnight for the start, NaN where missing."""
    if isinstance(column, tuple):
        hour, minute = column
        return trips[hour].astype(float) * 60 + trips[minute].astype(float)
    return trips[column].astype(object).where(trips[column].notna(), np.nan)


def chain_rows(cleaned: EODTables, shipped: EODTables) -> pd.DataFrame:
    """One row per shipped trip: the shipped and cleaned value of every editable field, and the codes.

    Indexed like ``shipped.trips``. For every field in ``FIELDS`` a
    ``<key>_shipped`` and a ``<key>_clean`` column (the start in minutes
    from midnight, the rest as labels or zone ids; NaN where missing, and on
    every cleaned column of a dropped row), then ``leg_min`` (the legs'
    minutes summed), ``ajustes``, ``problemas``, ``dropped`` (the row is not
    in ``cleaned.trips``) and ``home`` (the household's zone). ``cleaned`` may
    be any table derived from ``shipped`` — ``load_eod()`` or the result of
    :func:`apply_edits` — as long as its rows are a subset of the shipped ones.
    """
    R, C = shipped.trips, cleaned.trips
    Cr = C.reindex(R.index)
    out = {}
    for f in FIELDS.values():
        out[f"{f.key}_shipped"] = _values(R, f.column)
        out[f"{f.key}_clean"] = _values(Cr, f.column)
    out["leg_min"] = _leg_minutes(R, shipped.legs).astype(int)
    out[FIX_FLAG] = Cr[FIX_FLAG].astype(object).fillna("") if FIX_FLAG in Cr else ""
    out[ISSUE_FLAG] = (
        Cr[ISSUE_FLAG].astype(object).fillna("") if ISSUE_FLAG in Cr else ""
    )
    out["dropped"] = ~R.index.isin(C.index)
    out["home"] = _home_zone(R, shipped.viv)
    return pd.DataFrame(out, index=R.index)


def code_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per ``ISSUE_CODES`` entry, ``dropped``, and each ``ajustes`` code present, over ``rows``."""
    cols = {c: has_code(rows[ISSUE_FLAG], c) for c in ISSUE_CODES}
    cols["dropped"] = rows.dropped
    fixes = sorted({c for v in rows[FIX_FLAG] for c in v.split(";") if c})
    cols.update({c: has_code(rows[FIX_FLAG], c) for c in fixes})
    return pd.DataFrame(cols, index=rows.index)


def pending_persons(rows: pd.DataFrame, codes=None) -> pd.MultiIndex:
    """The persons with a row carrying any of ``codes`` (default: every ``problemas`` code), in chain order."""
    codes = list(ISSUE_CODES) if codes is None else list(codes)
    unknown = set(codes) - set(ISSUE_CODES)
    if unknown:
        raise ValueError(f"not a problemas code: {sorted(unknown)}")
    flagged = pd.concat([has_code(rows[ISSUE_FLAG], c) for c in codes], axis=1).any(
        axis=1
    )
    return flagged.groupby(level=PERSON).any().loc[lambda s: s].index


def _hhmm(minutes: pd.Series) -> pd.Series:
    return minutes.map(
        lambda m: MISSING if pd.isna(m) else f"{int(m) // 60:02d}:{int(m) % 60:02d}"
    )


def _text(values: pd.Series) -> pd.Series:
    return values.map(lambda v: MISSING if pd.isna(v) else str(v))


def _zone(values: pd.Series, home: pd.Series) -> pd.Series:
    text = _text(values)
    return text.where(text != home.astype(str), HOME)


def _cell(shipped: pd.Series, clean: pd.Series, dropped: pd.Series) -> pd.Series:
    """``shipped → clean`` where the two differ, the value alone otherwise; the shipped value on a dropped row."""
    same = (shipped == clean) | dropped
    return shipped.where(same, shipped + ARROW + clean)


def _person_info(hab: pd.DataFrame | None, persons: pd.MultiIndex) -> pd.DataFrame:
    """Sex, age, occupation and the repeated-diary flag of ``persons``, blank where ``hab`` lacks them."""
    info = pd.DataFrame(index=persons)
    src = {
        "sex": "sexo_nacimiento",
        "age": "edad",
        "occupation": "ocupacion",
        "repeated": "diario_repetido",
    }
    for col, name in src.items():
        if hab is not None and name in hab.columns:
            values = hab[name].reindex(persons)
            if name == "diario_repetido":
                values = values.map({True: "yes", False: ""})
            info[col] = values.astype(object).where(values.notna(), "").astype(str)
        else:
            info[col] = ""
    return info


def chain_sheet(
    rows: pd.DataFrame, hab: pd.DataFrame | None = None, persons=None
) -> pd.DataFrame:
    """The review sheet of ``persons`` (every person when None): the browser's table, one row per shipped trip.

    Columns ``COLUMNS``: the keys, the person columns, then the editable
    fields formatted as the browser shows them (``shipped → cleaned`` where
    the rules changed a value, HH:MM starts, ``home`` for the household's
    zone, ``—`` for a missing value) each followed by its empty ``new …``
    column, ``leg min``, ``ajustes``, ``problemas``, ``status`` (``dropped``,
    ``changed`` or blank) with its ``new status``, and an empty ``note``.
    """
    t = rows
    if persons is not None:
        keep = pd.MultiIndex.from_arrays(
            [t.index.get_level_values(PERSON[0]), t.index.get_level_values(PERSON[1])]
        ).isin(list(persons))
        t = t[keep]
    home = t.home.astype(str)
    sheet = pd.DataFrame(index=t.index)
    sheet["household"] = t.index.get_level_values(PERSON[0])
    sheet["person"] = t.index.get_level_values(PERSON[1])
    who = pd.MultiIndex.from_arrays([sheet.household, sheet.person])
    sheet["home"] = home.to_numpy()
    info = _person_info(hab, who.unique()).reindex(who)
    for col in PERSON_COLUMNS[1:]:
        sheet[col] = info[col].to_numpy()
    sheet["trip"] = t.index.get_level_values("folio_viaje")
    for name, f in FIELDS.items():
        shipped, clean = t[f"{f.key}_shipped"], t[f"{f.key}_clean"]
        if f.key == "start":
            shipped, clean = _hhmm(shipped), _hhmm(clean)
        elif f.key in ("origin", "destination"):
            shipped, clean = _zone(shipped, home), _zone(clean, home)
        else:
            shipped, clean = _text(shipped), _text(clean)
        sheet[name] = _cell(shipped, clean, t.dropped)
        sheet[new_column(name)] = ""
    sheet["leg min"] = t.leg_min.to_numpy()
    sheet[FIX_FLAG] = t[FIX_FLAG].to_numpy()
    sheet[ISSUE_FLAG] = t[ISSUE_FLAG].to_numpy()
    sheet[STATUS] = np.where(
        t.dropped, DROPPED, np.where(t[FIX_FLAG] != "", "changed", "")
    )
    sheet[new_column(STATUS)] = ""
    sheet[NOTE] = ""
    return sheet[COLUMNS].reset_index(drop=True)


def write_sheet(sheet: pd.DataFrame, path) -> Path:
    """Write a review sheet as UTF-8 CSV with a byte-order mark, so spreadsheets keep the arrows and accents."""
    path = Path(path)
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def read_sheet(path) -> pd.DataFrame:
    """Read a review sheet back: every column as text, blanks as empty strings, the keys as integers."""
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in KEYS if c not in sheet.columns]
    if missing:
        raise ValueError(f"{path}: not a review sheet, missing column(s) {missing}")
    for c in KEYS:
        sheet[c] = pd.to_numeric(sheet[c].str.strip()).astype(int)
    return sheet


# ------------------------------------------------------------------ the edits


def _read_value(cells: pd.Series) -> pd.Series:
    """The value a cell holds: the text after the last arrow, stripped."""
    return cells.astype(str).map(lambda s: _ARROW_SPLIT.split(s.strip())[-1].strip())


def _status(cells: pd.Series) -> pd.Series:
    return cells.astype(str).str.strip().str.lower().where(lambda s: s == DROPPED, "")


def sheet_edits(sheet: pd.DataFrame) -> pd.DataFrame:
    """The hand edits on a review sheet: one row per filled ``new …`` cell that differs from the value beside it.

    Columns ``household``, ``person``, ``trip``, ``field`` (the column
    edited: a ``FIELDS`` name, ``status``, or ``note``), ``before`` (what the
    sheet showed: the text after the last arrow), ``after`` (the new cell,
    stripped) and ``note`` (the row's note). A row with a note and nothing
    else is one edit with field ``note``. A new value equal to the one
    beside it is not an edit, nor is a ``new status`` that would not change
    the row (``dropped`` on a dropped row, ``restored`` on a kept one). A
    value that cannot be applied — an unknown label or zone, a bad time, a
    status other than those two — is an edit here; :func:`apply_edits`
    refuses it. A ``new …`` column the sheet does not read is an error.
    """
    s = sheet.set_index(KEYS)
    if s.index.has_duplicates:
        raise ValueError("the sheet repeats a (household, person, trip) key")
    unknown = [
        c for c in s.columns if c.startswith(NEW) and c[len(NEW) :] not in EDITABLE
    ]
    if unknown:
        raise ValueError(
            f"not a column the sheet reads back: {unknown}; the new columns are "
            f"{[new_column(c) for c in EDITABLE]}"
        )
    notes = (
        s[NOTE].astype(str).str.strip() if NOTE in s else pd.Series("", index=s.index)
    )
    edits, changed = [], pd.Series(False, index=s.index)
    for col in EDITABLE:
        if col not in s or new_column(col) not in s:
            continue
        after = s[new_column(col)].astype(str).str.strip()
        if col == STATUS:
            before, wanted = _status(s[col]), after.str.lower()
            diff = (
                ((wanted == DROPPED) & (before != DROPPED))
                | ((wanted == RESTORED) & (before == DROPPED))
                | ~wanted.isin(["", DROPPED, RESTORED])
            )
        else:
            before = _read_value(s[col])
            diff = (after != "") & (after != before)
        for key in s.index[diff.to_numpy()]:
            edits.append((*key, col, before[key], after[key], notes[key]))
        changed |= diff
    for key in s.index[(notes != "").to_numpy() & ~changed.to_numpy()]:
        edits.append((*key, NOTE, "", notes[key], notes[key]))
    out = pd.DataFrame(edits, columns=KEYS + ["field", "before", "after", NOTE])
    order = {c: i for i, c in enumerate(EDITABLE + [NOTE])}
    out["_order"] = out.field.map(order)
    return (
        out.sort_values(KEYS + ["_order"]).drop(columns="_order").reset_index(drop=True)
    )


def _known_zones(cleaned: EODTables, shipped: EODTables | None) -> set:
    tables = [cleaned] + ([shipped] if shipped is not None else [])
    zones = set(cleaned.viv.ageb.astype(str))
    for t in tables:
        zones |= set(t.trips.origen.astype(str)) | set(t.trips.destino.astype(str))
    return zones


def apply_edits(
    cleaned: EODTables, edits: pd.DataFrame, shipped: EODTables | None = None
) -> EODTables:
    """The cleaned tables with the sheet's edits applied and ``problemas`` recomputed.

    Every field edit is written to the trips column ``FIELDS`` names and
    leaves its ``<field>:revision`` code in ``ajustes``; a zone reads
    ``home`` as the household's zone. A ``status`` of ``dropped`` removes the
    row and its legs and lowers the person's ``viajes_contados``; one of
    ``restored`` on a row ``load_eod`` dropped brings the shipped row back
    (so ``shipped`` is required then) under ``fila:revision``. An edit whose
    value already holds is skipped, so a sheet exported before a rule change
    still applies cleanly. ``problemas`` is then recomputed over the whole
    table (:func:`eodgdl.eod.mark_issues`). Notes are not applied. Raises
    ``ValueError`` listing every edit that cannot be applied — an unknown
    label, zone or time, a trip the table lacks, a status other than the two
    — and applies nothing in that case.
    """
    trips = cleaned.trips.copy()
    levels = {
        f.column: set(trips[f.column].cat.categories)
        if hasattr(trips[f.column], "cat")
        else None
        for f in FIELDS.values()
        if not isinstance(f.column, tuple)
    }
    zones = _known_zones(cleaned, shipped)
    home = cleaned.viv.ageb.astype(str)
    shipped_index = shipped.trips.index if shipped is not None else pd.Index([])

    problems, plan = [], []  # plan: (key, action, column, value, code)
    for r in edits.itertuples(index=False):
        key = (int(r.household), int(r.person), int(r.trip))
        where = f"household {key[0]}, person {key[1]}, trip {key[2]}, {r.field}"
        value = str(r.after).strip()
        if r.field == NOTE:
            continue
        if r.field == STATUS:
            status = value.lower()
            if status == DROPPED:
                if key in trips.index:
                    plan.append((key, "drop", None, None, None))
                elif key not in shipped_index:
                    problems.append(f"{where}: no such trip")
            elif status == RESTORED:
                if key in trips.index or key in shipped_index:
                    if key not in trips.index:
                        plan.append((key, "restore", None, None, None))
                elif shipped is None:
                    problems.append(
                        f"{where}: restoring a dropped row needs the shipped tables"
                    )
                else:
                    problems.append(f"{where}: no such trip to restore")
            else:
                problems.append(
                    f"{where}: new status must be '{DROPPED}' or '{RESTORED}', not '{value}'"
                )
            continue
        if r.field not in FIELDS:
            problems.append(f"{where}: not a column that takes a new value")
            continue
        if key not in trips.index:
            problems.append(f"{where}: no such trip in the cleaned table")
            continue
        f = FIELDS[r.field]
        if f.key == "start":
            m = _HHMM.match(value)
            if not m or int(m[1]) > 23 or int(m[2]) > 59:
                problems.append(f"{where}: '{value}' is not a time HH:MM")
                continue
            new_value = (int(m[1]), int(m[2]))
            current = (trips.at[key, f.column[0]], trips.at[key, f.column[1]])
        elif f.key in ("origin", "destination"):
            new_value = home[key[0]] if value.lower() == HOME else value
            if new_value not in zones:
                problems.append(f"{where}: '{value}' is not a zone the survey uses")
                continue
            current = str(trips.at[key, f.column])
        else:
            if levels[f.column] is not None and value not in levels[f.column]:
                problems.append(f"{where}: '{value}' is not a level of {f.column}")
                continue
            new_value = value
            current = str(trips.at[key, f.column])
        if new_value != current:  # a value that already holds is not a change
            plan.append((key, "set", f.column, new_value, f.code))
    if problems:
        raise ValueError("edits that cannot be applied:\n  " + "\n  ".join(problems))

    hab, legs = cleaned.hab.copy(), cleaned.legs
    dropped = [key for key, action, *_ in plan if action == "drop"]
    restored = [key for key, action, *_ in plan if action == "restore"]
    for key, action, column, value, code in plan:
        if action != "set":
            continue
        if isinstance(column, tuple):
            trips.loc[key, column[0]], trips.loc[key, column[1]] = value
        else:
            trips.loc[key, column] = value
        flags = trips.at[key, FIX_FLAG]
        trips.loc[key, FIX_FLAG] = code if not flags else flags + ";" + code
    if dropped:
        trips = trips.drop(index=dropped)
        if legs is not None:
            legs = legs[~legs.index.droplevel("folio_traslado").isin(dropped)]
    if restored:
        rows = shipped.trips.loc[restored].reindex(columns=trips.columns)
        rows[FIX_FLAG], rows[ISSUE_FLAG] = RESTORED_CODE, ""
        trips = pd.concat([trips, rows])
        if legs is not None:
            legs = pd.concat(
                [
                    legs,
                    shipped.legs[
                        shipped.legs.index.droplevel("folio_traslado").isin(restored)
                    ],
                ]
            )
    for keys, sign in ((dropped, -1), (restored, 1)):
        for hh, person in (
            pd.MultiIndex.from_tuples([k[:2] for k in keys], names=PERSON).unique()
            if keys
            else []
        ):
            n = sum(1 for k in keys if k[:2] == (hh, person))
            hab.loc[(hh, person), "viajes_contados"] += sign * n
    trips, legs = trips.sort_index(), (legs.sort_index() if legs is not None else None)
    trips[ISSUE_FLAG] = mark_issues(trips, cleaned.viv, legs)
    return EODTables(cleaned.viv, hab, trips, legs)


class Verification(NamedTuple):
    """What :func:`verify_edits` returns."""

    tables: (
        EODTables  # the cleaned tables with the edits applied and problemas recomputed
    )
    sheet: pd.DataFrame  # the review sheet of the edited persons after the edits: shipped → revised, codes recomputed
    persons: (
        pd.DataFrame
    )  # one row per edited person: what was edited, the codes before and after
    summary: dict  # counts over the edited persons


def _codes_of(flags: pd.Series) -> pd.Series:
    """The codes a person's rows carry, ';'-joined and sorted, per person."""
    return flags.groupby(level=PERSON).agg(
        lambda s: ";".join(sorted({c for v in s for c in v.split(";") if c}))
    )


def verify_edits(
    cleaned: EODTables, shipped: EODTables, edits: pd.DataFrame
) -> Verification:
    """Apply the edits and say what they did to the defects of the persons they touch.

    ``persons`` has one row per edited person: ``cells`` (field edits),
    ``dropped``, ``restored``, ``notes``, ``before`` and ``after`` (the
    ``problemas`` codes over the person's rows, ';'-joined) and ``cleared``
    (nothing left). ``summary`` counts persons, cells, rows dropped and
    restored, rows with a defect before and after, persons cleared and
    persons with a defect the edits made (a code absent before). ``sheet``
    is the review sheet of those persons rebuilt from the revised tables.
    """
    revised = apply_edits(cleaned, edits, shipped)
    status = edits.after.astype(str).str.strip().str.lower()
    persons = (
        edits.assign(
            cells=edits.field.isin(list(FIELDS)),
            dropped=(edits.field == STATUS) & (status == DROPPED),
            restored=(edits.field == STATUS) & (status == RESTORED),
            notes=edits[NOTE].astype(str).str.strip() != "",
        )
        .groupby(KEYS[:2])[["cells", "dropped", "restored", "notes"]]
        .sum()
        .astype(int)
        .sort_index()
        .reset_index()
    )
    who = pd.MultiIndex.from_frame(persons[KEYS[:2]], names=PERSON)
    before_rows = chain_rows(cleaned, shipped)
    after_rows = chain_rows(revised, shipped)
    sel = lambda rows: rows[
        pd.MultiIndex.from_arrays(
            [rows.index.get_level_values(0), rows.index.get_level_values(1)]
        ).isin(who)
    ]
    before, after = sel(before_rows), sel(after_rows)
    persons["before"] = _codes_of(before[ISSUE_FLAG]).reindex(who).fillna("").to_numpy()
    persons["after"] = _codes_of(after[ISSUE_FLAG]).reindex(who).fillna("").to_numpy()
    persons["cleared"] = persons.after == ""
    made = [
        bool(set(a.split(";")) - set(b.split(";")) - {""})
        for a, b in zip(persons.after, persons.before)
    ]
    summary = {
        "persons": len(persons),
        "cells": int(persons.cells.sum()),
        "rows_dropped": int(persons.dropped.sum()),
        "rows_restored": int(persons.restored.sum()),
        "defects_before": int((before[ISSUE_FLAG] != "").sum()),
        "defects_after": int((after[ISSUE_FLAG] != "").sum()),
        "persons_cleared": int(persons.cleared.sum()),
        "persons_with_new_defects": int(sum(made)),
    }
    return Verification(
        revised, chain_sheet(after_rows, revised.hab, who), persons, summary
    )
