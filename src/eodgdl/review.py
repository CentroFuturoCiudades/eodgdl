"""Review sheets: the trip chains pending a fix as a CSV to edit by hand, and the edits read back.

A review sheet is the chain browser's table (``notebooks/chain_browser.ipynb``)
as a CSV: one row per shipped trip of the persons selected, in chain order,
every cell reading ``shipped → cleaned`` where the chain rules or an earlier
hand pass changed the value and the value alone where nothing did; a row the
tables no longer hold keeps its shipped values and reads ``dropped`` under
``status``; the household's zone reads ``home`` wherever it appears. The
person's home zone, sex, age, occupation and repeated-diary flag repeat on
every row, so the sheet stands on its own in a spreadsheet. ``orig. type`` is
the person's one answer to where the day's first trip started, so it shows on
the person's first trip only.

Next to every column that may take a change stands an empty ``new …``
column: ``new start``, ``new motive``, ``new dest. type``, ``new orig. type``,
``new origin``, ``new destination``, ``new mode`` and ``new status``. The
hand edits go there — ``start`` as HH:MM, a motive or place type by its
label, a zone by its id or ``home``, a mode by its label; ``dropped`` under
``new status`` to take a row out of the chain, ``restored`` to bring back a
row the tables no longer hold — and why under ``note``. The original columns
are never touched, so the sheet carries its own before and after and needs no
copy to be read back; ``leg min``, ``ajustes`` and ``problemas`` take no new
value (the first belongs to the legs, the other two are recomputed), and
rows cannot be added.

    stages = load_stages()                                            # shipped, rules and revised from one read
    rows = review.chain_rows(stages.revised, stages.shipped)          # per shipped trip, shipped and current values
    sheet = review.chain_sheet(rows, stages.revised.hab, review.pending_persons(rows))
    review.write_sheet(sheet, "chain_review.csv")
    ...                                                               # edit by hand
    edits = review.sheet_edits(review.read_sheet("chain_review.csv"))
    verified = review.verify_edits(stages.revised, stages.shipped, edits, stages.rules)
    frozen = review.freeze_edits(edits, stages.revised, stages.shipped, stages.rules)
    review.write_pass(frozen, review.next_pass_path())                # the next pass load_eod applies

``verify_edits`` applies the edits (:func:`apply_edits`) and says which
defects they cleared and which they left or made; ``freeze_edits`` keeps the
edits that change something, in canonical form, for a new pass under
``eodgdl/revisions/``. An edit is checked against the tables it is applied
to: the value it says the sheet showed must be the one they hold, so a sheet
cannot overwrite a value its editor never saw. ``eodgdl review export``,
``verify`` and ``freeze`` wrap the three steps.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from eodgdl.chains import (
    FIX_FLAG,
    ISSUE_CODES,
    ISSUE_GROUPS,
    ISSUE_FLAG,
    PERSON,
    _home_zone,
    _leg_minutes,
    has_code,
    mark_issues,
)
from eodgdl.eod import EODTables

KEYS = ["household", "person", "trip"]
PERSON_COLUMNS = ["home", "sex", "age", "occupation", "repeated"]
ARROW = " → "
MISSING = "—"
HOME = "home"
DROPPED = "dropped"
STATUS = "status"
NOTE = "note"
NOTE_ALIASES = ("comments", "comment", "notes")  # read as note: the first pass came back with "comments"
ORIGIN_TYPE = "tipo_lugar_origen"  # the person's answer to where the day started, the same on every row


class Field(NamedTuple):
    key: str  # column stem in chain_rows: <key>_shipped, <key>_clean
    column: str | tuple  # the trips column(s) it reads and writes
    code: str  # the ajustes code a hand edit leaves


# The editable columns of the sheet, in sheet order.
FIELDS = {
    "start": Field("start", ("hora_inicio_h", "hora_inicio_m"), "hora:revision"),
    "motive": Field("motive", "motivo_viaje", "motivo:revision"),
    "dest. type": Field("dtype", "tipo_lugar_destino", "tipo_destino:revision"),
    "orig. type": Field("otype", ORIGIN_TYPE, "tipo_origen:revision"),
    "origin": Field("origin", "origen", "origen:revision"),
    "destination": Field("destination", "destino", "destino:revision"),
    "mode": Field("mode", "modo_principal", "modo:revision"),
}
READ_ONLY = ["leg min", FIX_FLAG, ISSUE_FLAG]
EDITABLE = list(FIELDS) + [STATUS]
NEW = "new "  # prefix of the columns that take the hand edits
RESTORED = "restored"  # under new status: bring back a row the tables no longer hold
RESTORED_CODE = "fila:revision"
# The trips columns behind each ajustes prefix: every code names the field it touches before the colon.
_PREFIX_COLUMNS = {
    f.code.split(":")[0]: f.column if isinstance(f.column, tuple) else (f.column,)
    for f in FIELDS.values()
}


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
_HHMM = re.compile(r"^(\d{1,2}):(\d{2})(?::00)?$")  # spreadsheets add :00 seconds and drop the leading zero


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
    """The persons with a row carrying any of ``codes`` (default: every ``problemas`` code), in chain order.

    A group name from ``ISSUE_GROUPS`` (``breaking``, ``tolerated``) stands for its codes.
    """
    codes = list(ISSUE_CODES) if codes is None else list(codes)
    codes = [c for code in codes for c in ISSUE_GROUPS.get(code, (code,))]
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
    shipped, clean = shipped.astype(str), clean.astype(str)  # an empty selection comes as floats
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
    a value changed, HH:MM starts, ``home`` for the household's zone, ``—``
    for a missing value) each followed by its empty ``new …`` column, ``leg
    min``, ``ajustes``, ``problemas``, ``status`` (``dropped``, ``changed`` or
    blank) with its ``new status``, and an empty ``note``. ``orig. type``, the
    person's one answer to where the day started, is filled on the person's
    first kept trip only; a dropped row shows every shipped value.
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
    kept = ~t.dropped.to_numpy()
    seen = pd.Series(kept.astype(int), index=t.index).groupby(level=PERSON).cumsum().to_numpy()
    sheet.loc[kept & (seen > 1), "orig. type"] = ""  # the answer shows on the first kept trip
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
    """Read a review sheet back: every column as text, blanks as empty strings, the keys as integers.

    A sheet with no ``note`` column but one of ``NOTE_ALIASES`` has that column read as the note.
    """
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    alias = next((c for c in NOTE_ALIASES if c in sheet.columns), None)
    if NOTE not in sheet.columns and alias:
        sheet = sheet.rename(columns={alias: NOTE})
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


def _time(text) -> tuple[int, int] | None:
    """(hour, minute) of an HH:MM cell, seconds of :00 and a missing leading zero allowed; None if it is not one."""
    m = _HHMM.match(str(text).strip())
    if not m or int(m[1]) > 23 or int(m[2]) > 59:
        return None
    return int(m[1]), int(m[2])


def _canon(field: str, text, home: str | None) -> str:
    """The form in which two cells of ``field`` compare: HH:MM for a time, the zone id for ``home``, lower case for a label."""
    text = str(text).strip()
    key = FIELDS[field].key
    if key == "start":
        t = _time(text)
        return f"{t[0]:02d}:{t[1]:02d}" if t else text
    if key in ("origin", "destination"):
        return home if home is not None and text.lower() == HOME else text
    return text.lower()


def sheet_edits(sheet: pd.DataFrame) -> pd.DataFrame:
    """The hand edits on a review sheet: one row per filled ``new …`` cell that differs from the value beside it.

    Columns ``household``, ``person``, ``trip``, ``field`` (the column
    edited: a ``FIELDS`` name, ``status``, or ``note``), ``before`` (what the
    sheet showed: the text after the last arrow), ``after`` (the new cell,
    stripped) and ``note`` (the row's note). A row with a note and nothing
    else is one edit with field ``note``. A new value equal to the one
    beside it is not an edit — times compare as times, so ``9:05`` or
    ``09:05:00`` is ``09:05``, labels regardless of case, and ``home`` as the
    row's home zone — nor is a ``new status`` that would not change the row
    (``dropped`` on a dropped row, ``restored`` on a kept one). A value that
    cannot be applied — an unknown label or zone, a bad time, a status other
    than those two — is an edit here; :func:`apply_edits` refuses it. A
    ``new …`` column the sheet does not read is an error.
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
    homes = s["home"].astype(str).to_numpy() if "home" in s else [None] * len(s)
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
            differs = [
                a != "" and _canon(col, a, h) != _canon(col, b, h)
                for a, b, h in zip(after, before, homes)
            ]
            diff = pd.Series(differs, index=s.index, dtype=bool)
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


# The zone column that goes with each place column. The survey codes one zone
# per AGEB (dwellings, origins and destinations agree), so an edit that moves
# a trip end moves its zone too, as the chain rules do (_start_from_home_after_return).
_ZONE_OF = {"origen": "zona_origen", "destino": "zona_destino"}


def _zones(shipped: EODTables) -> tuple[set, dict | None]:
    """The zones an edit may name — the dwellings' AGEBs and every trip end — and AGEB id → the zone the survey
    coded for it (None without zone columns). Read off the survey as shipped: every place the rules or a pass
    can set comes from it, and the survey's zone is a function of the AGEB."""
    trips = shipped.trips
    known = (
        set(shipped.viv.ageb.astype(str))
        | set(trips.origen.astype(str))
        | set(trips.destino.astype(str))
    )
    if not set(_ZONE_OF.values()) <= set(trips.columns):
        return known, None
    pairs = [(trips[place], trips[zone]) for place, zone in _ZONE_OF.items()]
    if "centralidad" in shipped.viv:
        pairs.insert(0, (shipped.viv.ageb, shipped.viv.centralidad))
    frame = pd.concat(
        [pd.DataFrame({"place": p.astype(str).to_numpy(), "zone": z.astype(object).to_numpy()}) for p, z in pairs],
        ignore_index=True,
    )
    frame = frame[frame.zone.notna()].drop_duplicates("place")
    return known, dict(zip(frame.place, frame.zone.astype(str)))


def _restore(trips, legs, keys, rules, shipped):
    """``trips`` and ``legs`` with the rows ``keys`` brought back: the rules' version where the rules kept the
    row, else the shipped one, with the person's current answer to where the day started."""
    answer = (
        trips.groupby(level=PERSON, sort=False)[ORIGIN_TYPE].first()
        if ORIGIN_TYPE in trips
        else None
    )
    in_rules = set(rules.trips.index[rules.trips.index.isin(keys)])
    parts, leg_parts = [trips], [legs]
    for source, picked in (
        (rules, [k for k in keys if k in in_rules]),
        (shipped, [k for k in keys if k not in in_rules]),
    ):
        if not picked:
            continue
        rows = source.trips.loc[picked].reindex(columns=trips.columns)
        rows[FIX_FLAG] = ""  # both recomputed once the edits are in
        rows[ISSUE_FLAG] = ""
        if answer is not None:
            now = answer.reindex(rows.index.droplevel("folio_viaje")).astype(object).to_numpy()
            own = rows[ORIGIN_TYPE].astype(object).to_numpy()
            rows[ORIGIN_TYPE] = pd.Series(np.where(pd.isna(now), own, now), index=rows.index).astype(
                trips[ORIGIN_TYPE].dtype
            )
        parts.append(rows)
        if legs is not None and source.legs is not None:
            leg_parts.append(source.legs[source.legs.index.droplevel("folio_traslado").isin(picked)])
    trips = pd.concat(parts).sort_index()
    legs = pd.concat(leg_parts).sort_index() if legs is not None else None
    return trips, legs


class _Resolution(NamedTuple):
    """What a set of edits does to a table, worked out before anything is applied."""

    outcome: pd.Series  # per edit: 'apply', 'holds' (the value is already there), 'moot' or 'note'
    value: pd.Series  # per applied edit, the value in canonical form: HH:MM, a level's label, a zone or home
    problems: list  # the edits that cannot be applied, as messages
    trips: pd.DataFrame  # the table with the restored rows back, nothing else changed
    legs: pd.DataFrame | None
    restored: list  # keys brought back
    dropped: list  # keys taken out
    sets: list  # (key, Field, value)
    leg_modes: list  # (leg key, mode): a one-leg trip's leg follows its main mode


def _first_rows(index: pd.MultiIndex) -> set:
    """The first key of every person in a sorted trip index."""
    first = pd.Series(0, index=index).groupby(level=PERSON).cumcount().to_numpy() == 0
    return set(index[first])


def _shown(trips: pd.DataFrame, key, f: Field) -> str:
    """The value of field ``f`` on row ``key`` as a sheet shows it: HH:MM, a zone id or a label, ``—`` if missing."""
    if f.key == "start":
        h, m = trips.at[key, f.column[0]], trips.at[key, f.column[1]]
        return MISSING if pd.isna(h) or pd.isna(m) else f"{int(h):02d}:{int(m):02d}"
    v = trips.at[key, f.column]
    return MISSING if pd.isna(v) else str(v)


def _resolve(cleaned, edits, shipped, rules, zones=None) -> _Resolution:
    """Work out every edit against ``cleaned`` without applying any: what it sets, and whether it can apply at all.

    Status edits first: ``dropped`` on a kept row and ``restored`` on a row
    the table lacks apply, the same status on a row already so holds. The
    rows restored are brought back (:func:`_restore`) before the field edits
    are read, so the same edits may set a restored row's fields. A field edit
    is moot on a row the same edits drop, and so is an ``orig. type`` edit
    anywhere but on the person's first trip once the edits are in: the origin
    type is one answer per person, and the first pass typed 6,357 of them per
    trip. A value already there holds. Every other edit must name a value the
    field takes and a trip the table holds, and its ``before`` must be what
    the sheet showed for the table it is applied to — the current value, or
    the shipped one on a row restored by the same edits, since the sheet shows
    a dropped row as shipped — else it was made against another version of
    the tables and is refused. A mode edit on a trip of several legs must
    name one of their modes; on a one-leg trip the leg takes the new mode.
    """
    trips, legs = cleaned.trips, cleaned.legs
    known, zone_of = zones if zones is not None else _zones(shipped)
    home = shipped.viv.ageb.astype(str)
    edits = edits.reset_index(drop=True)
    keys = [(int(h), int(p), int(t)) for h, p, t in zip(edits.household, edits.person, edits.trip)]
    fields = edits.field.astype(str).to_numpy()
    afters = edits.after.astype(str).str.strip().to_numpy()
    befores = edits.before.astype(str).to_numpy()
    outcome = pd.Series("", index=edits.index, dtype=object)
    value = pd.Series(None, index=edits.index, dtype=object)
    problems, restored, dropped, sets, leg_modes = [], [], [], [], []
    where_of = lambda key, field: f"household {key[0]}, person {key[1]}, trip {key[2]}, {field}"

    for i in np.flatnonzero(fields == STATUS):
        key, status = keys[i], afters[i].lower()
        if status == DROPPED:
            if key in trips.index:
                outcome[i], value[i] = "apply", DROPPED
                dropped.append(key)
            elif key in shipped.trips.index:
                outcome[i] = "holds"
            else:
                problems.append(f"{where_of(key, STATUS)}: no such trip")
        elif status == RESTORED:
            if key in trips.index:
                outcome[i] = "holds"
            elif key in shipped.trips.index:
                outcome[i], value[i] = "apply", RESTORED
                restored.append(key)
            else:
                problems.append(f"{where_of(key, STATUS)}: no such trip to restore")
        else:
            problems.append(
                f"{where_of(key, STATUS)}: new status must be '{DROPPED}' or '{RESTORED}', not '{afters[i]}'"
            )
    outcome[fields == NOTE] = "note"
    if restored:
        trips, legs = _restore(trips, legs, restored, rules, shipped)
    now_restored = set(restored)
    dropping = {keys[i] for i in np.flatnonzero((fields == STATUS) & (np.char.lower(afters.astype(str)) == DROPPED))}
    first_after = _first_rows(trips.index[~trips.index.isin(list(dropping))])
    first_before = _first_rows(cleaned.trips.index)
    levels = {  # lower-cased label -> the level, so a label's case is not held against it
        f.column: {str(c).lower(): c for c in trips[f.column].cat.categories}
        if isinstance(trips[f.column].dtype, pd.CategoricalDtype)
        else None
        for f in FIELDS.values()
        if not isinstance(f.column, tuple)
    }

    for i in np.flatnonzero(~np.isin(fields, [STATUS, NOTE])):
        key, name, text = keys[i], fields[i], afters[i]
        where = where_of(key, name)
        if key in dropping:  # a field of a row the edits drop is moot
            outcome[i] = "moot"
            continue
        if name not in FIELDS:
            problems.append(f"{where}: not a column that takes a new value")
            continue
        if key not in trips.index:
            problems.append(f"{where}: no such trip in the cleaned table")
            continue
        f = FIELDS[name]
        if f.key == "otype" and key not in first_after:  # the answer sits on the first trip
            outcome[i] = "moot"
            continue
        if f.key == "start":
            new = _time(text)
            if new is None:
                problems.append(f"{where}: '{text}' is not a time HH:MM")
                continue
            h, m = trips.at[key, f.column[0]], trips.at[key, f.column[1]]
            holds = not pd.isna(h) and not pd.isna(m) and (int(h), int(m)) == new
            canonical = f"{new[0]:02d}:{new[1]:02d}"
        elif f.key in ("origin", "destination"):
            new = home[key[0]] if text.lower() == HOME else text
            if new not in known:
                problems.append(f"{where}: '{text}' is not a zone the survey uses")
                continue
            if zone_of is not None and new not in zone_of:
                problems.append(f"{where}: no zone is coded for '{text}'")
                continue
            holds = str(trips.at[key, f.column]) == new
            canonical = HOME if text.lower() == HOME else new
        else:
            new = text
            if levels[f.column] is not None:
                if text.lower() not in levels[f.column]:
                    problems.append(f"{where}: '{text}' is not a level of {f.column}")
                    continue
                new = levels[f.column][text.lower()]
            holds = str(trips.at[key, f.column]) == str(new)
            canonical = str(new)
        if f.key == "mode" and legs is not None and not holds:
            modes = legs.loc[key].traslado_medio.astype(str).tolist()
            if len(modes) > 1 and new not in modes:
                problems.append(
                    f"{where}: '{text}' is none of the trip's leg modes ({', '.join(modes)}); the sheet does not edit legs"
                )
                continue
            if len(modes) == 1:
                leg_modes.append((key + (legs.loc[key].index[0],), new))
        if holds:
            outcome[i] = "holds"
            continue
        if f.key != "otype" or key in first_before or key in now_restored:
            source = shipped.trips if key in now_restored else trips
            shown, h = _shown(source, key, f), home[key[0]]
            if _canon(name, befores[i], h) != _canon(name, shown, h):
                problems.append(
                    f"{where}: the sheet showed '{befores[i]}' but the tables hold '{shown}': "
                    "it was made against another version of them"
                )
                continue
        outcome[i], value[i] = "apply", canonical
        sets.append((key, f, new))
    return _Resolution(outcome, value, problems, trips, legs, restored, dropped, sets, leg_modes)


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality of two aligned columns, missing equal to missing."""
    a, b = a.astype(object).to_numpy(), b.astype(object).to_numpy()
    na_a, na_b = pd.isna(a), pd.isna(b)
    same = na_a & na_b
    both = ~na_a & ~na_b
    same[both] = a[both] == b[both]
    return same


def _net_fixes(trips: pd.DataFrame, shipped: EODTables, rules: EODTables) -> pd.Series:
    """The ``ajustes`` of a revised table: what each row's values owe to the rules and to the hand passes.

    Field by field (``_PREFIX_COLUMNS``): a value that is the shipped one
    carries no code; one that is the rules' value carries the rules' codes
    for that field; any other was set by hand and carries ``<field>:revision``.
    A row the rules dropped and a pass restored carries ``fila:revision``. So
    a hand edit that puts back the shipped value takes the rule's code off the
    row with it, and a later pass that undoes an earlier one leaves no trace
    on the row; the frozen passes keep the history.
    """
    n = len(trips)
    in_rules = trips.index.isin(rules.trips.index)
    columns = [c for cols in _PREFIX_COLUMNS.values() for c in cols if c in trips]
    S = shipped.trips[columns].reindex(trips.index)
    R = rules.trips[columns + [FIX_FLAG]].reindex(trips.index)
    same_shipped, same_rules = {}, {}
    for prefix, cols in _PREFIX_COLUMNS.items():
        cols = [c for c in cols if c in trips]
        if not cols:
            continue
        same_shipped[prefix] = np.logical_and.reduce([_same(trips[c], S[c]) for c in cols])
        same_rules[prefix] = np.logical_and.reduce([_same(trips[c], R[c]) for c in cols]) & in_rules
    rule_codes = R[FIX_FLAG].astype(object).where(in_rules, "").fillna("").to_numpy()

    def stands(code: str, i: int) -> bool:  # the rule's value is still there, and it is not the shipped one
        p = code.split(":")[0]
        return p not in same_rules or (same_rules[p][i] and not same_shipped[p][i])

    out = np.full(n, "", dtype=object)
    differs = ~np.logical_and.reduce(list(same_rules.values()))
    for i in np.flatnonzero((rule_codes != "") | differs):
        codes = [c for c in rule_codes[i].split(";") if c and stands(c, i)]
        codes += [
            f"{p}:revision" for p in same_rules if not same_rules[p][i] and not same_shipped[p][i]
        ]
        if not in_rules[i]:
            codes.append(RESTORED_CODE)
        out[i] = ";".join(codes)
    return pd.Series(out, index=trips.index, name=FIX_FLAG, dtype=str)


def _apply(cleaned: EODTables, r: _Resolution, shipped: EODTables, rules: EODTables, zone_of) -> EODTables:
    """The tables with a resolution's edits applied, ``ajustes`` and ``problemas`` recomputed; raises on a problem."""
    if r.problems:
        raise ValueError("edits that cannot be applied:\n  " + "\n  ".join(r.problems))
    trips, hab = r.trips.copy(), cleaned.hab.copy()
    legs = r.legs.copy() if r.legs is not None else None
    by_column = {}
    for key, f, new in r.sets:
        i = trips.index.get_loc(key)
        if isinstance(f.column, tuple):
            for c, v in zip(f.column, new):
                by_column.setdefault(c, {})[i] = v
        else:
            by_column.setdefault(f.column, {})[i] = new
            if zone_of is not None and f.column in _ZONE_OF:
                by_column.setdefault(_ZONE_OF[f.column], {})[i] = zone_of[new]
    for column, values in by_column.items():
        trips.iloc[list(values), trips.columns.get_loc(column)] = list(values.values())
    for leg, mode in r.leg_modes:
        legs.loc[leg, "traslado_medio"] = mode
    if r.dropped:
        trips = trips.drop(index=r.dropped)
        if legs is not None:
            legs = legs[~legs.index.droplevel("folio_traslado").isin(r.dropped)]
    for keys, sign in ((r.dropped, -1), (r.restored, 1)):
        for person, n in pd.Series([k[:2] for k in keys], dtype=object).value_counts().items():
            hab.loc[person, "viajes_contados"] += sign * n
    trips = trips.sort_index()
    legs = legs.sort_index() if legs is not None else None
    if ORIGIN_TYPE in trips:  # one answer per person, the one on the first trip
        trips[ORIGIN_TYPE] = trips.groupby(level=PERSON, sort=False)[ORIGIN_TYPE].transform("first")
    trips[FIX_FLAG] = _net_fixes(trips, shipped, rules)
    trips[ISSUE_FLAG] = mark_issues(trips, cleaned.viv, legs)
    return EODTables(cleaned.viv, hab, trips, legs)


def apply_edits(
    cleaned: EODTables,
    edits: pd.DataFrame,
    shipped: EODTables,
    rules: EODTables,
    *,
    zones=None,
) -> EODTables:
    """The cleaned tables with the sheet's edits applied, ``ajustes`` and ``problemas`` recomputed.

    ``rules`` is the chain rules' output (``load_eod(revise_chains=False)``)
    and ``shipped`` the survey as shipped; ``cleaned`` is either, or a table
    earlier edits made from them. Every field edit is written to the trips
    column ``FIELDS`` names; a zone reads ``home`` as the household's zone,
    and a new origin or destination takes the ``zona_origen`` /
    ``zona_destino`` the survey coded for that AGEB, so the AGEB and the zone
    stay one place (tables without zone columns are left as they are). An
    ``orig. type`` edit on the person's first trip changes the person's
    answer, on every row; a mode edit on a one-leg trip changes the leg's mode
    too, so ``modo_principal`` stays one of the trip's leg modes, which the
    model build and the reweighting both read. A ``status`` of ``dropped``
    removes the row and its legs and lowers the person's ``viajes_contados``;
    one of ``restored`` brings back a row the table lacks: from ``rules`` when
    it holds the row, so a row an earlier pass dropped comes back with the
    rules' fixes, else from ``shipped`` (a row the rules dropped). Restores
    come first, so the same edits may change a restored row's fields.

    What is moot or already holds is skipped (see :func:`_resolve`), so
    applying the same edits twice changes nothing the second time; a label is
    matched regardless of case and a time may carry ``:00`` seconds, as
    spreadsheets write them. Each edit's ``before`` must be what the sheet
    showed for ``cleaned``. ``ajustes`` is then recomputed from the values
    (:func:`_net_fixes`) and ``problemas`` over the whole table
    (:func:`eodgdl.chains.mark_issues`). Notes are not applied. Raises
    ``ValueError`` listing every edit that cannot be applied — an unknown
    label, zone or time, a trip the table lacks, a status other than the two,
    a ``before`` the tables no longer hold — and applies nothing in that case.
    ``zones`` is the result of ``_zones(shipped)``, passed in by
    :func:`apply_revisions` so a run of passes reads the survey's zones once.
    """
    zones = zones if zones is not None else _zones(shipped)
    return _apply(cleaned, _resolve(cleaned, edits, shipped, rules, zones), shipped, rules, zones[1])


def apply_revisions(rules: EODTables, shipped: EODTables, passes=None) -> EODTables:
    """The chain rules' tables with the frozen hand passes applied in order (:func:`apply_edits` each).

    ``rules`` is the rules' output (``load_eod(revise_chains=False)``) and is
    also where a later pass restores a row an earlier one dropped from.
    ``passes`` is a list of ``(name, edits)``; by default the ones that ship
    under ``eodgdl/revisions/`` (:func:`eodgdl._resources.chain_revisions`),
    which is what ``load_eod`` applies.
    """
    from eodgdl._resources import chain_revisions

    zones = _zones(shipped)
    tables = rules
    for name, edits in chain_revisions() if passes is None else passes:
        try:
            tables = apply_edits(tables, edits, shipped, rules, zones=zones)
        except ValueError as err:
            raise ValueError(f"revision pass {name}: {err}") from None
    return tables


def freeze_edits(
    edits: pd.DataFrame, cleaned: EODTables, shipped: EODTables, rules: EODTables
) -> pd.DataFrame:
    """The edits a new frozen pass holds: the ones that change ``cleaned``, in canonical form, one note per trip.

    ``edits`` is what :func:`sheet_edits` recovers from a sheet exported from
    ``cleaned`` (the tables ``load_eod()`` returns). Edits that hold or are
    moot are left out, and so are notes with no edit; ``after`` is written the
    way a pass stores it — ``HH:MM``, a level's own label, a zone id or
    ``home``, ``dropped`` or ``restored`` — and a trip's note stays on its
    first edit only. Raises ``ValueError`` as :func:`apply_edits` does if any
    edit cannot be applied.
    """
    r = _resolve(cleaned, edits, shipped, rules)
    if r.problems:
        raise ValueError("edits that cannot be applied:\n  " + "\n  ".join(r.problems))
    kept = edits.reset_index(drop=True)
    applied = (r.outcome == "apply").to_numpy()
    kept = kept[applied].assign(after=r.value[applied].to_numpy())
    kept[NOTE] = kept[NOTE].astype(str).str.strip().where(~kept.duplicated(KEYS), "")
    return kept[KEYS + ["field", "before", "after", NOTE]].reset_index(drop=True)


def write_pass(edits: pd.DataFrame, path) -> Path:
    """Write a pass's edits as the gzipped CSV ``load_eod`` reads, byte for byte the same for the same edits.

    The gzip header carries no file name and no time, so a pass written again is the file already committed.
    """
    import gzip

    path = Path(path)
    with open(path, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
        gz.write(edits.to_csv(index=False).encode("utf-8"))
    return path


def next_pass_path() -> Path:
    """Where the next frozen pass goes: ``chains_<n+1>.csv.gz`` beside the others under ``eodgdl/revisions/``."""
    from importlib import resources

    from eodgdl._resources import pass_files, pass_number

    n = max((pass_number(f.name) for f in pass_files()), default=0)
    return Path(str(resources.files("eodgdl") / "revisions")) / f"chains_{n + 1}.csv.gz"


class Verification(NamedTuple):
    """What :func:`verify_edits` returns."""

    tables: EODTables  # the cleaned tables with the edits applied and problemas recomputed
    sheet: pd.DataFrame  # the review sheet of the edited persons after the edits: shipped → revised, codes recomputed
    persons: pd.DataFrame  # one row per edited person: what was edited, the codes before and after
    summary: dict  # counts over the edited persons


def _codes_of(flags: pd.Series) -> pd.Series:
    """The codes a person's rows carry, ';'-joined and sorted, per person."""
    return flags.groupby(level=PERSON).agg(
        lambda s: ";".join(sorted({c for v in s for c in v.split(";") if c}))
    )


def verify_edits(
    cleaned: EODTables,
    shipped: EODTables,
    edits: pd.DataFrame,
    rules: EODTables,
) -> Verification:
    """Apply the edits and say what they did to the defects of the persons they touch.

    ``persons`` has one row per edited person: ``cells`` (field edits),
    ``dropped``, ``restored``, ``notes``, ``before`` and ``after`` (the
    ``problemas`` codes over the person's rows, ';'-joined) and ``cleared``
    (nothing left). ``summary`` counts persons, cells, rows dropped and
    restored, edits that already held or were moot (an ``orig. type`` off the
    first trip, a field of a row the edits drop), rows with a defect before
    and after, persons cleared and persons with a defect the edits made (a
    code absent before). ``sheet`` is the review sheet of those persons
    rebuilt from the revised tables. ``rules`` (the rules' output) is where a
    restored row comes from when the rules kept it (see :func:`apply_edits`).
    """
    zones = _zones(shipped)
    resolution = _resolve(cleaned, edits, shipped, rules, zones)
    revised = _apply(cleaned, resolution, shipped, rules, zones[1])
    status = edits.after.astype(str).str.strip().str.lower().to_numpy()
    applied = (resolution.outcome == "apply").to_numpy()  # what changed the tables, not what was asked
    field = edits.field.astype(str).to_numpy()
    persons = (
        edits.assign(
            cells=np.isin(field, list(FIELDS)) & applied,
            dropped=(field == STATUS) & (status == DROPPED) & applied,
            restored=(field == STATUS) & (status == RESTORED) & applied,
            notes=(edits[NOTE].astype(str).str.strip() != "").to_numpy(),
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
        "edits_already_holding": int((resolution.outcome == "holds").sum()),
        "edits_moot": int((resolution.outcome == "moot").sum()),
        "defects_before": int((before[ISSUE_FLAG] != "").sum()),
        "defects_after": int((after[ISSUE_FLAG] != "").sum()),
        "persons_cleared": int(persons.cleared.sum()),
        "persons_with_new_defects": int(sum(made)),
    }
    return Verification(
        revised, chain_sheet(after_rows, revised.hab, who), persons, summary
    )
