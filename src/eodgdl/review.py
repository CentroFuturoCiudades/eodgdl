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
the person's first trip only, and a decision keeps it under trip
``PERSON_TRIP`` (0): it belongs to the person, whichever trip is first.

Next to every column that may take a change stands an empty ``new …``
column: ``new start``, ``new motive``, ``new dest. type``, ``new orig. type``,
``new origin``, ``new destination``, ``new mode`` and ``new status``. The
hand edits go there — ``start`` as HH:MM, a motive or place type by its
label, a zone by its id or ``home``, a mode by its label; ``dropped`` under
``new status`` to take a row out of the chain, ``restored`` to bring back a
row the tables no longer hold — and why under ``note``. The original columns
are never touched, so the sheet carries its own before and after and needs
no copy to be read back; ``leg min`` (the trip's travel minutes, which the
rules read, so a review corrects them before the rules, in
``eodgdl/revisions/leg_minutes.csv.gz``), ``ajustes`` and ``problemas`` take
no new value, and rows cannot be added. A note typed where nothing else
changes gives the decisions already on that trip a new reason.

    stages = load_stages()                                            # shipped, rules and revised from one read
    rows = review.chain_rows(stages.revised, stages.shipped)          # per shipped trip, shipped and current values
    sheet = review.chain_sheet(rows, stages.revised.hab, review.pending_persons(rows))
    review.write_sheet(sheet, "chain_review.csv")
    ...                                                               # edit by hand
    sheet = review.read_sheet("chain_review.csv")
    edits = review.sheet_edits(sheet)
    done = review.freeze_round(edits, stages, "chain_review_17", shown=review.sheet_keys(sheet))
    done.verified.summary                                             # what the edits did (eodgdl review verify)
    review.write_decisions(done.decisions, review.decisions_path())   # what load_eod applies, in one pass

``freeze_round`` keeps the edits that change something, in canonical form
(:func:`freeze_edits`), folds them into the hand decisions ``load_eod``
applies — one per trip and field, each against the rules' output
(``eodgdl/revisions/chains.csv.gz``, :func:`merge_decisions`) — checks that
the merged decisions, applied once, give what the sheet gives, and says which
defects the edits cleared and which they left or made, and how the stays and
the days of the persons they touch changed against the rules' output and
against the sheet's starting point (:func:`verify_edits`, :func:`screens`).
``eodgdl review verify`` is that without the write, so a sheet that verifies
freezes. An edit is checked against the tables it is applied to: the value
it says the sheet showed must be the one they hold, so a sheet cannot
overwrite a value its editor never saw. A decision the rules' output no
longer supports — a change to the rules moved the value it was made against —
is stale (:func:`stale_decisions`): ``load_stages(skip_stale=True)`` sets it
aside, and a round that shows its trip decides it again. A value the rules
gave and a round kept holds no decision, so a change to the rules that moves
it is seen by comparing two :func:`snapshot` tables, taken before and after
(``eodgdl review snapshot``, ``eodgdl review export --since``).
``eodgdl review export``, ``verify`` and ``freeze`` wrap the steps.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from eodgdl.chains import (
    _COMPANION_TOL,
    _EARLY_START,
    _HALF_DAY,
    _NIGHT_SHIFT_RETURN,
    ESCORT,
    FIX_FLAG,
    HOME_MOTIVE,
    ISSUE_CODES,
    ISSUE_FLAG,
    ISSUE_GROUPS,
    MANDATORY_MOTIVES,
    PERSON,
    WORK_MOTIVE,
    _clock_gap,
    _companion_links,
    _home_zone,
    _leg_minutes,
    _start_minutes,
    days_past_midnight,
    has_code,
    mark_issues,
    non_trips,
)
from eodgdl.eod import EODStages, EODTables

log = logging.getLogger(__name__)

KEYS = ["household", "person", "trip"]
PERSON_COLUMNS = ["home", "sex", "age", "occupation", "repeated"]
ARROW = " → "
MISSING = "—"
HOME = "home"
DROPPED = "dropped"
STATUS = "status"
NOTE = "note"
ORIGIN_TYPE = "tipo_lugar_origen"  # the person's answer to where the day started, the same on every row
PERSON_TRIP = 0  # the trip of a decision about the person, not one of their trips: the day-start answer


class Field(NamedTuple):
    key: str  # column stem in chain_rows: <key>_shipped, <key>_clean
    column: str | tuple  # the trips column(s) it reads and writes; for leg min, the legs column
    code: str  # the ajustes code a hand edit leaves


LEG_MINUTES = "traslado_min"  # the legs column behind leg min, summed over the trip's legs
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
# The columns the sheet shows, shipped → cleaned, that take no new value: the travel minutes the rules read,
# which a review corrects before the rules (eodgdl/revisions/leg_minutes.csv.gz, eodgdl.chains._correct_minutes).
SHOWN = {"leg min": Field("legmin", LEG_MINUTES, "minutos:revision")}
READ_ONLY = [FIX_FLAG, ISSUE_FLAG]
EDITABLE = list(FIELDS) + [STATUS]
NEW = "new "  # prefix of the columns that take the hand edits
RESTORED = "restored"  # under new status: bring back a row the tables no longer hold
RESTORED_CODE = "fila:revision"


def new_column(name: str) -> str:
    """The column that takes the new value of ``name``: ``new start``, ``new status``, ..."""
    return NEW + name


COLUMNS = (
    KEYS[:2]
    + PERSON_COLUMNS
    + KEYS[2:]
    + [c for name in FIELDS for c in (name, new_column(name))]
    + list(SHOWN)
    + READ_ONLY
    + [STATUS, new_column(STATUS), NOTE]
)

_ARROW_SPLIT = re.compile(r"\s*(?:→|->)\s*")
_HHMM = re.compile(r"^(\d{1,2}):(\d{2})(?::00)?$")  # spreadsheets add :00 seconds and drop the leading zero


# ------------------------------------------------------------------ building


def _trip_minutes(trips: pd.DataFrame, legs: pd.DataFrame | None, index) -> pd.Series:
    """Each trip's travel minutes, its legs' summed, over ``index``; NaN for a trip with no legs.

    Without a legs table the ``traslado*_min`` columns of ``trips`` are summed instead.
    """
    if legs is not None:
        return legs[LEG_MINUTES].astype(float).groupby(level=[0, 1, 2]).sum().reindex(index)
    cols = [c for c in trips.columns if c.startswith("traslado") and c.endswith("_min")]
    if cols:
        return trips[cols].astype(float).sum(axis=1, min_count=1).reindex(index)
    return pd.Series(np.nan, index=index)


def _values(trips: pd.DataFrame, legs: pd.DataFrame | None, f: Field, index) -> pd.Series:
    """Field ``f`` of the trips in ``index`` as plain objects: minutes from midnight for the start, the travel
    minutes for leg min, labels or zone ids for the rest; NaN where missing."""
    if f.key == "legmin":
        return _trip_minutes(trips, legs, index)
    if isinstance(f.column, tuple):
        hour, minute = (trips[c].reindex(index) for c in f.column)
        return hour.astype(float) * 60 + minute.astype(float)
    values = trips[f.column].reindex(index).astype(object)
    return values.where(values.notna(), np.nan)


def chain_rows(cleaned: EODTables, shipped: EODTables) -> pd.DataFrame:
    """One row per shipped trip: the shipped and cleaned value of every field a sheet shows, and the codes.

    Indexed like ``shipped.trips``. For every field in ``FIELDS`` and ``SHOWN``
    a ``<key>_shipped`` and a ``<key>_clean`` column (the start in minutes
    from midnight, the travel minutes summed over the legs, the rest as
    labels or zone ids; NaN where missing, and on every cleaned column of a
    dropped row), then ``ajustes``, ``problemas``, ``dropped`` (the row is
    not in ``cleaned.trips``) and ``home`` (the household's zone). ``cleaned``
    may be any table derived from ``shipped`` — ``load_eod()`` or the result
    of :func:`apply_edits` — as long as its rows are a subset of the shipped ones.
    """
    R, C = shipped.trips, cleaned.trips
    Cr = C.reindex(R.index)
    kept = pd.Series(R.index.isin(C.index), index=R.index)
    out = {}
    for f in (*FIELDS.values(), *SHOWN.values()):
        out[f"{f.key}_shipped"] = _values(R, shipped.legs, f, R.index)
        out[f"{f.key}_clean"] = _values(C, cleaned.legs, f, R.index).where(kept)
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


def marked_persons(tables: EODTables) -> pd.MultiIndex:
    """The persons whose day ``tables`` (the rules' output) marks: a ``problemas`` code, or an ``ajustes`` one.

    The hand decisions sit on these persons only: the survey that nothing marks keeps its values
    (user, 2026-09-26). A repair is a mark as much as a code is, now that the rules close on their
    own what hand passes closed before them (a copied first origin, an overlap within the
    tolerance, since 2026-09-26).
    """
    t = tables.trips
    marked = (t[ISSUE_FLAG] != "") | (t[FIX_FLAG] != "")
    return marked.groupby(level=PERSON).any().loc[lambda s: s].index


def _hhmm(minutes: pd.Series) -> pd.Series:
    return minutes.map(
        lambda m: MISSING if pd.isna(m) else f"{int(m) // 60:02d}:{int(m) % 60:02d}"
    )


def _text(values: pd.Series) -> pd.Series:
    return values.map(lambda v: MISSING if pd.isna(v) else str(v))


def _whole(values: pd.Series) -> pd.Series:
    return values.map(lambda v: MISSING if pd.isna(v) else str(int(v)))


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
    for a missing value) each followed by its empty ``new …`` column, the
    whole travel minutes (``leg min``, shown only), ``ajustes``,
    ``problemas``, ``status`` (``dropped``, ``changed`` or blank) with its
    ``new status``, and an empty ``note``.
    ``orig. type``, the person's one answer to where the day started, is
    filled on the person's first kept trip only; a dropped row shows every
    shipped value.
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
    for name, f in (*FIELDS.items(), *SHOWN.items()):
        shipped, clean = t[f"{f.key}_shipped"], t[f"{f.key}_clean"]
        if f.key == "start":
            shipped, clean = _hhmm(shipped), _hhmm(clean)
        elif f.key in ("origin", "destination"):
            shipped, clean = _zone(shipped, home), _zone(clean, home)
        elif f.key == "legmin":
            shipped, clean = _whole(shipped), _whole(clean)
        else:
            shipped, clean = _text(shipped), _text(clean)
        sheet[name] = _cell(shipped, clean, t.dropped)
        if name in FIELDS:
            sheet[new_column(name)] = ""
    kept = ~t.dropped.to_numpy()
    seen = pd.Series(kept.astype(int), index=t.index).groupby(level=PERSON).cumsum().to_numpy()
    sheet.loc[kept & (seen > 1), "orig. type"] = ""  # the answer shows on the first kept trip
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

    Rows with no key at all, which spreadsheets leave at the end, are dropped.
    """
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in KEYS if c not in sheet.columns]
    if missing:
        raise ValueError(f"{path}: not a review sheet, missing column(s) {missing}")
    sheet = sheet[~(sheet[KEYS].apply(lambda c: c.str.strip()) == "").all(axis=1)].reset_index(drop=True)
    for c in KEYS:
        sheet[c] = pd.to_numeric(sheet[c].str.strip()).astype(int)
    return sheet


def sheet_keys(sheet: pd.DataFrame) -> set:
    """The (household, person, trip) of every row a review sheet shows."""
    return set(zip(sheet.household.astype(int), sheet.person.astype(int), sheet.trip.astype(int)))


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
    """The form in which two cells of ``field`` compare: HH:MM for a time, the zone id for ``home``, lower case for
    a label."""
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
    else is one edit with field ``note``, which :func:`freeze_round` makes the
    new reason of the decisions on that trip. A new value equal to the one
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
# a trip end moves its zone too, as the chain rules do (_start_where_previous_ended).
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

    outcome: pd.Series  # per edit: 'apply', 'holds' (already there), 'moot', 'note', 'stale' or 'problem'
    value: pd.Series  # per applied edit, the value in canonical form: HH:MM, a level's label, a zone or home
    problem_at: dict  # edit position -> why it cannot be applied
    trips: pd.DataFrame  # the table with the restored rows back, nothing else changed
    legs: pd.DataFrame | None
    restored: list  # keys brought back
    dropped: list  # keys taken out
    sets: list  # (key, Field, value)
    leg_modes: list  # (leg key, mode): a one-leg trip's leg follows its main mode

    @property
    def problems(self) -> list:
        """The edits that cannot be applied, as messages, in edit order."""
        return [self.problem_at[i] for i in sorted(self.problem_at)]


def _first_rows(index: pd.MultiIndex) -> set:
    """The first key of every person in a sorted trip index."""
    first = pd.Series(0, index=index).groupby(level=PERSON).cumcount().to_numpy() == 0
    return set(index[first])


def _answer_rows(index: pd.MultiIndex) -> dict:
    """Person -> the row a sheet shows the person's answer on: their first trip in a sorted trip index."""
    return {key[:2]: key for key in _first_rows(index)}


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
    is moot on a row the same edits drop. ``orig. type`` is the person's one
    answer to where the day started, not a trip's value: a decision gives it
    with trip ``PERSON_TRIP`` (0), a sheet on the row it shows the answer on,
    the person's first trip in ``cleaned`` — even when the same sheet drops
    that trip — and anywhere else it is moot (the first pass typed 6,357 of
    them per trip); it is moot too for a person the edits leave without a
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
    problem_at, restored, dropped, sets, leg_modes = {}, [], [], [], []

    def where_of(key, field):
        trip = "the person" if key[2] == PERSON_TRIP else f"trip {key[2]}"
        return f"household {key[0]}, person {key[1]}, {trip}, {field}"

    def problem(i, message, stale=False):
        """``stale``: made against another version of the tables (a value or a trip that is no longer there)."""
        outcome[i] = "stale" if stale else "problem"
        problem_at[i] = message

    for i in np.flatnonzero(fields == STATUS):
        key, status = keys[i], afters[i].lower()
        if status == DROPPED:
            if key in trips.index:
                outcome[i], value[i] = "apply", DROPPED
                dropped.append(key)
            elif key in shipped.trips.index:
                outcome[i] = "holds"
            else:
                problem(i, f"{where_of(key, STATUS)}: no such trip", stale=True)
        elif status == RESTORED:
            if key in trips.index:
                outcome[i] = "holds"
            elif key in shipped.trips.index:
                outcome[i], value[i] = "apply", RESTORED
                restored.append(key)
            else:
                problem(i, f"{where_of(key, STATUS)}: no such trip to restore", stale=True)
        else:
            problem(i, f"{where_of(key, STATUS)}: new status must be '{DROPPED}' or '{RESTORED}', not '{afters[i]}'")
    outcome[fields == NOTE] = "note"
    if restored:
        trips, legs = _restore(trips, legs, restored, rules, shipped)
    now_restored = set(restored)
    dropping = {keys[i] for i in np.flatnonzero((fields == STATUS) & (np.char.lower(afters.astype(str)) == DROPPED))}
    # the person's answer: the row a sheet shows it on, the persons with a row left once the edits are in
    shows_answer = _answer_rows(cleaned.trips.index)
    keeping = set(trips.index[~trips.index.isin(list(dropping))].droplevel("folio_viaje"))
    answer_now = trips.groupby(level=PERSON, sort=False)[ORIGIN_TYPE].first() if ORIGIN_TYPE in trips else None
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
        if name not in FIELDS:
            problem(i, f"{where}: not a column that takes a new value")
            continue
        f = FIELDS[name]
        if f.key == "otype":  # the person's answer to where the day started, not a trip's value
            person = key[:2]
            if key[2] != PERSON_TRIP and shows_answer.get(person) != key:  # off the row the sheet shows it on
                outcome[i] = "moot"
                continue
            if answer_now is None or person not in answer_now.index:
                problem(i, f"{where}: no such person in the cleaned table", stale=True)
                continue
            if person not in keeping:  # the edits drop every trip the person has: no answer to give
                outcome[i] = "moot"
                continue
            new = levels[f.column].get(text.lower()) if levels[f.column] is not None else text
            if new is None:
                problem(i, f"{where}: '{text}' is not a level of {f.column}")
                continue
            now = answer_now[person]
            if str(now) == str(new):
                outcome[i] = "holds"
                continue
            shown, h = (MISSING if pd.isna(now) else str(now)), home[key[0]]
            if _canon(name, befores[i], h) != _canon(name, shown, h):
                problem(i, f"{where}: the sheet showed '{befores[i]}' but the tables hold '{shown}': "
                           "it was made against another version of them", stale=True)
                continue
            outcome[i], value[i] = "apply", str(new)
            sets.append(((*person, PERSON_TRIP), f, new))
            continue
        if key in dropping:  # a field of a row the edits drop is moot
            outcome[i] = "moot"
            continue
        if key not in trips.index:
            problem(i, f"{where}: no such trip in the cleaned table", stale=True)
            continue
        if f.key == "start":
            new = _time(text)
            if new is None:
                problem(i, f"{where}: '{text}' is not a time HH:MM")
                continue
            h, m = trips.at[key, f.column[0]], trips.at[key, f.column[1]]
            holds = not pd.isna(h) and not pd.isna(m) and (int(h), int(m)) == new
            canonical = f"{new[0]:02d}:{new[1]:02d}"
        elif f.key in ("origin", "destination"):
            new = home[key[0]] if text.lower() == HOME else text
            if new not in known:
                problem(i, f"{where}: '{text}' is not a zone the survey uses")
                continue
            if zone_of is not None and new not in zone_of:
                problem(i, f"{where}: no zone is coded for '{text}'")
                continue
            holds = str(trips.at[key, f.column]) == new
            canonical = HOME if text.lower() == HOME else new
        else:
            new = text
            if levels[f.column] is not None:
                if text.lower() not in levels[f.column]:
                    problem(i, f"{where}: '{text}' is not a level of {f.column}")
                    continue
                new = levels[f.column][text.lower()]
            holds = str(trips.at[key, f.column]) == str(new)
            canonical = str(new)
        if f.key == "mode" and legs is not None and not holds:
            modes = legs.loc[key].traslado_medio.astype(str).tolist()
            if len(modes) > 1 and new not in modes:
                problem(i, f"{where}: '{text}' is none of the trip's leg modes ({', '.join(modes)}); "
                           "the sheet does not edit the modes of several legs")
                continue
            if len(modes) == 1:
                leg_modes.append((key + (legs.loc[key].index[0],), new))
        if holds:
            outcome[i] = "holds"
            continue
        shown = _shown(shipped.trips if key in now_restored else trips, key, f)
        h = home[key[0]]
        if _canon(name, befores[i], h) != _canon(name, shown, h):
            problem(i, f"{where}: the sheet showed '{befores[i]}' but the tables hold '{shown}': "
                       "it was made against another version of them", stale=True)
            continue
        outcome[i], value[i] = "apply", canonical
        sets.append((key, f, new))
    return _Resolution(outcome, value, problem_at, trips, legs, restored, dropped, sets, leg_modes)


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality of two aligned columns, missing equal to missing."""
    a, b = a.astype(object).to_numpy(), b.astype(object).to_numpy()
    na_a, na_b = pd.isna(a), pd.isna(b)
    same = na_a & na_b
    both = ~na_a & ~na_b
    same[both] = a[both] == b[both]
    return same


def _net_fixes(trips: pd.DataFrame, legs: pd.DataFrame | None, shipped: EODTables, rules: EODTables) -> pd.Series:
    """The ``ajustes`` of a revised table: what each row's values owe to the rules and to the hand decisions.

    Field by field (``FIELDS``, the code's prefix naming the field): a value
    that is the shipped one carries no code; one that is the rules' value
    carries the rules' codes for that field; any other was set by hand and
    carries ``<field>:revision``. A row the rules dropped and a pass restored
    carries ``fila:revision``. So a hand edit that puts back the shipped value
    takes the rule's code off the row with it, and a decision undone leaves
    no trace on the row; git keeps the history of the decisions.
    """
    n = len(trips)
    index = trips.index
    in_rules = index.isin(rules.trips.index)
    same_shipped, same_rules = {}, {}
    for f in FIELDS.values():
        if not all(c in trips for c in (f.column if isinstance(f.column, tuple) else (f.column,))):
            continue
        prefix = f.code.split(":")[0]
        now = _values(trips, legs, f, index)
        same_shipped[prefix] = _same(now, _values(shipped.trips, shipped.legs, f, index))
        same_rules[prefix] = _same(now, _values(rules.trips, rules.legs, f, index)) & in_rules
    rule_codes = rules.trips[FIX_FLAG].reindex(index).astype(object).where(in_rules, "").fillna("").to_numpy()

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
    by_column, answers = {}, {}
    for key, f, new in r.sets:
        if f.key == "otype":  # the person's answer, set on every row they keep below
            answers[key[:2]] = new
            continue
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
    if ORIGIN_TYPE in trips:  # one answer per person: the one the edits give, else the one on the first trip
        answer = trips.groupby(level=PERSON, sort=False)[ORIGIN_TYPE].transform("first").astype(object)
        if answers:
            given = pd.Series(list(trips.index.droplevel("folio_viaje")), index=trips.index).map(answers)
            answer = answer.where(given.isna(), given)
        trips[ORIGIN_TYPE] = answer.astype(trips[ORIGIN_TYPE].dtype)
    trips[FIX_FLAG] = _net_fixes(trips, legs, shipped, rules)
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
    ``orig. type`` edit — a decision with trip ``PERSON_TRIP``, or a sheet's
    on the person's first trip — changes the person's answer, on every row;
    a mode edit on a one-leg trip changes the leg's mode
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
    ``zones`` is the result of ``_zones(shipped)``, for a caller that applies
    several sets of edits to read the survey's zones once.
    """
    zones = zones if zones is not None else _zones(shipped)
    return _apply(cleaned, _resolve(cleaned, edits, shipped, rules, zones), shipped, rules, zones[1])


def _decisions(decisions) -> pd.DataFrame:
    """``decisions``, or the table that ships as ``eodgdl/revisions/chains.csv.gz`` when None."""
    from eodgdl._resources import chain_decisions

    return chain_decisions() if decisions is None else decisions


def decision_outcomes(rules: EODTables, shipped: EODTables, decisions=None) -> pd.DataFrame:
    """The hand decisions, each with what it does to the rules' output: ``outcome`` and ``problem``.

    ``outcome`` is ``apply`` for a decision that changes the rules' output as
    it should, ``holds`` when the rules already give its value, ``moot`` when
    it cannot take effect (a field of a row the decisions drop, the answer of
    a person they leave without a trip), ``stale`` when its ``before`` is no
    longer the rules' value or its trip is gone — a change to the rules moved
    what it was made against, and a review round decides it again — and
    ``problem`` when it names what no table can take (an unknown label, zone
    or time), with the reason under ``problem``. Indexed like ``decisions``.
    """
    decisions = _decisions(decisions)
    r = _resolve(rules, decisions, shipped, rules)
    out = decisions.copy()
    out["outcome"] = r.outcome.to_numpy()
    out["problem"] = [r.problem_at.get(i, "") for i in range(len(decisions))]
    return out


def stale_decisions(rules: EODTables, shipped: EODTables, decisions=None) -> pd.DataFrame:
    """The hand decisions the rules' output no longer supports, each with its ``problem`` (see :func:`decision_outcomes`)."""
    outcomes = decision_outcomes(rules, shipped, decisions)
    return outcomes[outcomes.outcome == "stale"]


def apply_revisions(rules: EODTables, shipped: EODTables, decisions=None, *, skip_stale: bool = False) -> EODTables:
    """The chain rules' tables with the hand decisions applied, all at once (:func:`apply_edits`).

    ``rules`` is the rules' output (``load_eod(revise_chains=False)``).
    ``decisions`` defaults to the table that ships as
    ``eodgdl/revisions/chains.csv.gz`` (:func:`eodgdl._resources.chain_decisions`),
    which is what ``load_eod`` applies: one decision per trip and field, each
    made against the rules' output, so they apply in one pass and in no
    particular order. A review round merges into it (:func:`freeze_round`).

    A stale decision (:func:`stale_decisions`) raises ``ValueError``, naming
    it; with ``skip_stale`` the stale ones are set aside, with a warning, and
    the rest applied, so the review tools still run and a round can decide
    the stale ones again (``eodgdl review export --stale``). A decision no
    table can take (an unknown label, zone or time) always raises.
    """
    decisions = _decisions(decisions)
    if skip_stale:
        stale = stale_decisions(rules, shipped, decisions)
        while len(stale):  # setting a decision aside can leave another without its row
            log.warning("%d hand decisions no longer apply to the rules' output and are set aside", len(stale))
            decisions = decisions.drop(index=stale.index)
            stale = stale_decisions(rules, shipped, decisions)
    zones = _zones(shipped)
    r = _resolve(rules, decisions, shipped, rules, zones)
    if r.problems:
        stale = (r.outcome == "stale").any()
        why = ("A change to the chain rules moved what these decisions were made against; `eodgdl review export "
               "--stale` puts them on a sheet to decide again." if stale and not (r.outcome == "problem").any()
               else "The table holds values no table can take; it was changed outside `eodgdl review freeze`.")
        raise ValueError("the hand decisions cannot be applied:\n  " + "\n  ".join(r.problems) + "\n" + why)
    return _apply(rules, r, shipped, rules, zones[1])


def freeze_edits(
    edits: pd.DataFrame, cleaned: EODTables, shipped: EODTables, rules: EODTables
) -> pd.DataFrame:
    """The edits a review round adds to the decisions: the ones that change ``cleaned``, in canonical form, one note per trip.

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
    # a trip's note, wherever it sits among its edits: the edit that carries it may be one left out
    kept[NOTE] = kept.groupby(KEYS)[NOTE].transform(lambda s: next((n.strip() for n in s.astype(str) if n.strip()), ""))
    applied = (r.outcome == "apply").to_numpy()
    kept = kept[applied].assign(after=r.value[applied].to_numpy())
    kept[NOTE] = kept[NOTE].where(~kept.duplicated(KEYS), "")
    return kept[KEYS + ["field", "before", "after", NOTE]].reset_index(drop=True)


DECISION_COLUMNS = KEYS + ["field", "before", "after", NOTE, "source"]


def decision_frame(rows) -> pd.DataFrame:
    """Decisions as a table in ``DECISION_COLUMNS``, sorted by trip and then by field in sheet order."""
    order = {c: i for i, c in enumerate(EDITABLE)}
    out = pd.DataFrame(list(rows), columns=DECISION_COLUMNS).astype({k: int for k in KEYS})
    out["_order"] = out.field.map(order)
    return out.sort_values(KEYS + ["_order"]).drop(columns="_order").reset_index(drop=True)


def merge_decisions(
    decisions: pd.DataFrame,
    frozen: pd.DataFrame,
    rules: EODTables,
    shipped: EODTables,
    source: str,
    notes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The hand decisions with a review round's edits merged in: still one decision per trip and field.

    ``frozen`` is what :func:`freeze_edits` keeps of a sheet exported from the
    tables the decisions give. Every decision is made against the rules'
    output (its ``before`` is the rules' value, or the shipped one on a row
    the rules dropped): a new value replaces the field's decision, and one
    that is the rules' own removes it. ``dropped`` on a row the rules keep is
    a decision, and the row's other decisions go with it; ``restored`` on such
    a row removes the drop, and on a row the rules dropped is a decision of its
    own. The person's answer to where the day started is a decision about the
    person, trip ``PERSON_TRIP``, whichever trip it was typed on, so no drop or
    restore moves it. ``source`` names the review round; ``note`` keeps the
    sheet's reason on every decision it made. ``notes`` (``household, person,
    trip, note``: a note typed where nothing else changed, trip ``PERSON_TRIP``
    for the answer's) gives a new reason to every decision already on that
    trip, keeping each decision's value and source.
    """
    R, S = rules.trips, shipped.trips
    home = shipped.viv.ageb.astype(str)
    answers = R.groupby(level=PERSON, sort=False)[ORIGIN_TYPE].first() if ORIGIN_TYPE in R else None

    def made_against(key, f):  # the rules' value as a sheet shows it, the shipped one on a row the rules dropped
        if f.key == "otype":
            v = answers.get(key[:2], np.nan)
            return MISSING if pd.isna(v) else str(v)
        v = _shown(R if key in R.index else S, key, f)
        return HOME if f.key in ("origin", "destination") and v == home[key[0]] else v

    table = {
        (int(r[0]), int(r[1]), int(r[2]), r[3]): [int(r[0]), int(r[1]), int(r[2]), *r[3:]]
        for r in decisions[DECISION_COLUMNS].itertuples(index=False)
    }
    reasons = frozen.groupby(KEYS, sort=False)[NOTE].agg(lambda s: next((n for n in s if n), "")).to_dict()
    for e in frozen.itertuples(index=False):
        key = (int(e.household), int(e.person), int(e.trip))
        note = reasons.get(key, "")
        if e.field == STATUS:
            if e.after == DROPPED:
                if key in R.index:
                    table[(*key, STATUS)] = [*key, STATUS, "", DROPPED, note, source]
                else:
                    table.pop((*key, STATUS), None)
                for name in FIELDS:  # the row's decisions go with it
                    table.pop((*key, name), None)
            elif key in R.index:
                table.pop((*key, STATUS), None)
            else:
                table[(*key, STATUS)] = [*key, STATUS, DROPPED, RESTORED, note, source]
            continue
        f = FIELDS[e.field]
        if f.key == "otype":  # the person's answer
            key = (*key[:2], PERSON_TRIP)
        before = made_against(key, f)
        if _canon(e.field, e.after, home[key[0]]) == _canon(e.field, before, home[key[0]]):
            table.pop((*key, e.field), None)
        else:
            table[(*key, e.field)] = [*key, e.field, before, e.after, note, source]

    if notes is not None and len(notes):  # a new reason for the decisions already on the trip
        by_trip = {}
        for k in table:
            by_trip.setdefault(k[:3], []).append(k)
        for n in notes.itertuples(index=False):
            text = str(n.note).strip()
            for k in by_trip.get((int(n.household), int(n.person), int(n.trip)), []) if text else []:
                table[k][6] = text
    return decision_frame(table.values())


def decision_changes(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """How many decisions a merge added, changed (a new value) and removed, and how many took a new note."""
    key = KEYS + ["field"]
    old = before.set_index(key)[["after", NOTE]]
    new = after.set_index(key)[["after", NOTE]]
    both = old.index.intersection(new.index)
    return {
        "added": len(new.index.difference(old.index)),
        "changed": int((old.loc[both, "after"] != new.loc[both, "after"]).sum()),
        "removed": len(old.index.difference(new.index)),
        "notes": int(((old.loc[both, "after"] == new.loc[both, "after"]) & (old.loc[both, NOTE] != new.loc[both, NOTE])).sum()),
    }


def _write_gzip_csv(frame: pd.DataFrame, path) -> Path:
    """``frame`` as a gzipped CSV whose header carries no file name and no time: the same table, the same bytes."""
    import gzip

    path = Path(path)
    with open(path, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
        gz.write(frame.to_csv(index=False).encode("utf-8"))
    return path


def write_decisions(decisions: pd.DataFrame, path) -> Path:
    """Write the decisions as the gzipped CSV ``load_eod`` reads, byte for byte the same for the same decisions.

    The gzip header carries no file name and no time, so the file written again is the file already committed.
    The cached table (:func:`eodgdl._resources.chain_decisions`) is dropped, so a ``load_eod`` in the same
    process reads what was written.
    """
    from eodgdl._resources import chain_decisions

    path = _write_gzip_csv(decision_frame(decisions[DECISION_COLUMNS].itertuples(index=False)), path)
    chain_decisions.cache_clear()
    return path


def decisions_path() -> Path:
    """Where the decisions live: ``eodgdl/revisions/chains.csv.gz`` in the source tree."""
    from importlib import resources

    return Path(str(resources.files("eodgdl") / "revisions")) / "chains.csv.gz"


LEG_MINUTE_COLUMNS = KEYS + ["before", "after", NOTE, "source"]


def write_leg_minutes(table: pd.DataFrame, path) -> Path:
    """Write the travel minutes a review corrected as the gzipped CSV ``load_eod`` reads before the rules.

    One row per trip, sorted, in ``LEG_MINUTE_COLUMNS`` (see :func:`eodgdl._resources.leg_minutes`); byte for byte
    the same for the same table, and the cached table is dropped, as :func:`write_decisions` does.
    """
    from eodgdl._resources import leg_minutes

    frame = table[LEG_MINUTE_COLUMNS].astype({k: int for k in KEYS + ["before", "after"]})
    path = _write_gzip_csv(frame.sort_values(KEYS).reset_index(drop=True), path)
    leg_minutes.cache_clear()
    return path


def leg_minutes_path() -> Path:
    """Where the corrected travel minutes live: ``eodgdl/revisions/leg_minutes.csv.gz`` in the source tree."""
    return decisions_path().with_name("leg_minutes.csv.gz")


# ------------------------------------------------------------------ snapshots


def snapshot(tables: EODTables, shipped: EODTables) -> pd.DataFrame:
    """Every shipped trip's value of every ``FIELDS`` and ``SHOWN`` column in ``tables``, as a sheet shows it, and ``dropped``.

    What :func:`changed_persons` compares a later load with. The stale check
    guards the decisions, but a round that let the rules' value stand leaves
    no decision behind: a change to the rules that moves such a value is seen
    only by comparing the tables before and after it (``eodgdl review
    snapshot`` before the change, ``eodgdl review export --since`` after).
    """
    rows = chain_rows(tables, shipped)
    shown = {"start": _hhmm, "legmin": _whole}
    out = pd.DataFrame(
        {name: shown.get(f.key, _text)(rows[f"{f.key}_clean"]) for name, f in {**FIELDS, **SHOWN}.items()},
        index=rows.index,
    )
    out[STATUS] = np.where(rows.dropped, DROPPED, "")
    return out.rename_axis(KEYS)


def write_snapshot(snap: pd.DataFrame, path) -> Path:
    """Write a :func:`snapshot` as a gzipped CSV."""
    path = Path(path)
    snap.reset_index().to_csv(path, index=False, compression="gzip")
    return path


def read_snapshot(path) -> pd.DataFrame:
    """Read back what :func:`write_snapshot` wrote."""
    snap = pd.read_csv(path, dtype=str, keep_default_na=False, compression="gzip")
    return snap.astype({k: int for k in KEYS}).set_index(KEYS)


def changed_persons(before: pd.DataFrame, after: pd.DataFrame) -> pd.MultiIndex:
    """The persons with a trip whose value of some ``FIELDS`` column, or status, differs between two snapshots."""
    keys = before.index.union(after.index)
    a, b = before.reindex(keys).fillna(""), after.reindex(keys).fillna("")
    differs = (a[b.columns] != b).any(axis=1)
    return pd.MultiIndex.from_tuples(sorted({k[:2] for k in keys[differs.to_numpy()]}), names=PERSON)


# ------------------------------------------------------------------ companions

# A household member who made a trip with the person is the one check on a start
# reading that the chain itself does not supply. Decided 2026-09-26 in a review of
# the readings against household companions (the evidence is in
# eodgdl.chains._repair_start_times): where a companion reported another start, it
# matched the typo search's 12-hour readings 73 times and its readings of an extra
# or missing leading 1 never. The link is eodgdl.chains._companion_links, which the
# rules read too (hora_acompanante_12h), with its tolerances.


def companions(tables: EODTables, shipped: EODTables) -> pd.DataFrame:
    """Each trip beside every household member's version of it: the check a companion gives a start.

    Over the chain the model reads (the rows marked as non-trips left out), a
    companion of a trip is another member of the household who made the trip
    next to it with the person, as the survey reports both: the previous
    trip (``anchor`` ``previous``) or the next one (``next``) goes from the
    same zone to the same zone for the same motive in both chains, is no
    escort (after a drop-off the two part), and was reported starting within
    ``_COMPANION_TOL`` minutes in both, the two on the road together (each
    leaves no later than the other arrives). The link is the reports', so a
    reading that moves the trip next to it does not undo it. The companion's
    version of the trip is the one beside that anchor in their chain, with
    the same zones and motive, unless either trip is one entry with another
    trip of the other's (:func:`eodgdl.chains._companion_links`: a split
    shift beside a straight one). One row per trip and companion trip, both ways
    round: ``household, person, trip``, ``companion, companion_trip``,
    ``start`` and ``companion_start`` (minutes from midnight, as ``tables``
    hold them), ``reported`` and ``companion_reported`` (as ``shipped``
    reports them; NaN where it reports none), ``anchor``, and three verdicts:
    ``same_entry`` (the two reported one start, to ``_COMPANION_TOL``: one
    entry, whose two readings should agree), ``apart`` (one entry read to
    starts more than ``_COMPANION_TOL`` apart) and ``half_day`` (two reports,
    whose starts as ``tables`` hold them sit 12 hours apart to
    ``_COMPANION_TOL``: one of the two is a 12-hour slip). Household 3627,
    person 5, is the case in point: to school at 19:00, which the rules read
    as 09:00 until 2026-09-26, while person 6 left for the same school at
    07:00 and both came home at 13:00.
    """
    t = tables.trips[~non_trips(tables.trips)].sort_index()
    s = shipped.trips.reindex(t.index)
    start = _start_minutes(t)
    reported = (s.hora_inicio_h.astype(float) * 60 + s.hora_inicio_m.astype(float)).to_numpy()
    frame = pd.DataFrame(
        {"origin": t.origen.astype(str).to_numpy(), "destination": t.destino.astype(str).to_numpy(),
         "motive": t.motivo_viaje.astype(str).to_numpy(), "link": reported,  # an imputed start (no report) links nothing
         "travel": _leg_minutes(t, tables.legs)},
        index=t.index,
    )
    pairs = _companion_links(frame)
    i, j = pairs.i.to_numpy(), pairs.j.to_numpy()
    out = pairs.drop(columns=["i", "j", "anchor"]).assign(
        start=start[i], companion_start=start[j], reported=reported[i], companion_reported=reported[j],
        anchor=pairs.anchor.to_numpy(),
    )
    gap = _clock_gap(out.start, out.companion_start)
    out["same_entry"] = _clock_gap(out.reported, out.companion_reported) <= _COMPANION_TOL
    out["apart"] = out.same_entry & (gap > _COMPANION_TOL)
    out["half_day"] = ~out.same_entry & (np.abs(gap - _HALF_DAY) <= _COMPANION_TOL)
    return out.sort_values(KEYS + ["companion", "companion_trip"]).reset_index(drop=True)


# ------------------------------------------------------------------ screens

# What a consistent chain can still get implausibly wrong, for choosing whom to review: the
# problemas codes see the order of the times and where the day starts and ends, not how long
# anything lasts, nor what the household members who made a trip together report. Each is a
# threshold on the chain the model reads, read person by person; the two companion screens
# need the shipped tables (:func:`companions`).
SCREENS = {
    "zero_stay": "a trip leaves the minute the previous one arrives: a stay of no minutes",
    "zero_work": "a trip leaves work the minute it got there: no time at work",
    "short_work": "a trip leaves work or school less than 30 minutes after arriving",
    "long_workday": "more than 14 hours at work in the day",
    "long_day": "more than 20 hours from the day's first start to its last arrival",
    "early_start": "the day's first trip, to anything but work, starts before 05:00",
    "long_errand": "a stay of 8 hours or more at a place that is not home, work or school",
    "companion_apart": "a trip made with a household member, reported at the same start, is read more than 20 minutes "
                       "from theirs",
    "companion_12h": "a trip made with a household member starts 12 hours from theirs: one of the two is a 12-hour slip",
}
_SHORT_WORK = 30  # minutes
_LONG_WORKDAY = 14 * 60
_LONG_DAY = 20 * 60
_LONG_ERRAND = 8 * 60


def screens(tables: EODTables, shipped: EODTables | None = None) -> pd.DataFrame:
    """Per person with trips, how long the stays and the day last on the chain the model reads.

    The rows marked as non-trips are left out and a day that passes midnight
    counts on past it (:func:`eodgdl.chains.days_past_midnight`), as the model
    build does. Columns: ``zero_stays`` (trips that leave the minute the
    previous one arrives), ``zero_work_stays`` (those that leave work so),
    ``short_work_stays`` (those that leave work or school within half an
    hour), ``zero_away_stays`` (stays of no minutes anywhere else but at home
    and at an escort's drop-off), ``work_minutes`` (the stays at a work destination
    summed), ``day_minutes`` (the first start to the last arrival),
    ``long_errands`` (stays of eight hours or more at a place that is not
    home, work or school), ``night_shift_end`` (the day ends with a return
    from work the next morning, before 10:00, as a night shift does; until
    2026-09-26 any hour of the next day counted, and days that went to work
    again after a night passed as night shifts), ``companion_apart_trips`` and
    ``companion_12h_trips`` (the person's trips that one entry a household
    member made with them reads to another start, and those that start 12
    hours from the member's: :func:`companions`, given ``shipped``, the survey
    as shipped, to tell a reported start from a read one; 0 without it), one
    boolean column per ``SCREENS`` entry, and ``problems``: the stays of no
    minutes away from home, work and school, the short stays at work or
    school, the early start, the long errands, a day of more than 20 hours
    that does not end with a night shift's morning return and the trips a
    companion contradicts, summed — the count a review round reads a day by
    (``scripts/revisions``). Indexed by person.
    """
    t = tables.trips[~non_trips(tables.trips)].sort_index()
    start = _start_minutes(t) + 1440 * days_past_midnight(t).to_numpy()
    arrive = start + _leg_minutes(t, tables.legs)
    person = t.index.droplevel("folio_viaje")
    stay = pd.Series(start, index=t.index).groupby(level=PERSON).shift(-1).to_numpy() - arrive
    motive = t.motivo_viaje.astype(object).to_numpy()
    work = (motive == WORK_MOTIVE) & ~np.isnan(stay)
    mandatory = np.isin(motive, MANDATORY_MOTIVES) & ~np.isnan(stay)
    first = t.groupby(level=PERSON).cumcount().to_numpy() == 0
    last = t.groupby(level=PERSON).cumcount(ascending=False).to_numpy() == 0
    after_work = pd.Series(motive == WORK_MOTIVE, index=t.index).groupby(level=PERSON).shift(1).fillna(False).to_numpy(bool)
    next_morning = (start >= 1440) & (start < 1440 + _NIGHT_SHIFT_RETURN)
    frame = pd.DataFrame(
        {"zero": stay == 0, "zero_work": work & (stay == 0), "short_work": mandatory & (stay < _SHORT_WORK),
         "zero_away": (stay == 0) & ~np.isin(motive, [HOME_MOTIVE, ESCORT, *MANDATORY_MOTIVES]),  # a drop-off takes no time
         "work": np.where(work, np.clip(stay, 0, None), 0.0),
         "long_errand": (stay >= _LONG_ERRAND) & ~np.isin(motive, [HOME_MOTIVE, *MANDATORY_MOTIVES]),
         "early": first & ~np.isin(motive, [WORK_MOTIVE, HOME_MOTIVE]) & (start % 1440 < _EARLY_START),
         "night_end": last & (motive == HOME_MOTIVE) & after_work & next_morning,
         "start": start, "arrive": arrive},
        index=person,
    )
    g = frame.groupby(level=PERSON)
    out = pd.DataFrame(
        {"zero_stays": g.zero.sum().astype(int), "zero_work_stays": g.zero_work.sum().astype(int),
         "short_work_stays": g.short_work.sum().astype(int), "zero_away_stays": g.zero_away.sum().astype(int),
         "work_minutes": g.work.sum(), "day_minutes": g.arrive.max() - g.start.min(),
         "long_errands": g.long_errand.sum().astype(int), "early": g.early.any(), "night_shift_end": g.night_end.any()}
    )
    out["zero_stay"] = out.zero_stays > 0
    out["zero_work"] = out.zero_work_stays > 0
    out["short_work"] = out.short_work_stays > 0
    out["long_workday"] = out.work_minutes > _LONG_WORKDAY
    out["long_day"] = out.day_minutes > _LONG_DAY
    out["early_start"] = out.early
    out["long_errand"] = out.long_errands > 0
    out["companion_apart_trips"] = out["companion_12h_trips"] = 0
    if shipped is not None:
        pairs = companions(tables, shipped)
        for column, verdict in (("companion_apart_trips", "apart"), ("companion_12h_trips", "half_day")):
            flagged = pairs.loc[pairs[verdict], KEYS].drop_duplicates()
            per = flagged.groupby(["household", "person"]).size().rename_axis(PERSON)
            out[column] = per.reindex(out.index, fill_value=0).astype(int).to_numpy()
    out["companion_apart"] = out.companion_apart_trips > 0
    out["companion_12h"] = out.companion_12h_trips > 0
    out["problems"] = (out.zero_away_stays + out.short_work_stays + out.early.astype(int) + out.long_errands
                       + (out.long_day & ~out.night_shift_end).astype(int)
                       + out.companion_apart_trips + out.companion_12h_trips)
    return out


def screened_persons(tables: EODTables, names, shipped: EODTables | None = None) -> pd.MultiIndex:
    """The persons any of the ``SCREENS`` in ``names`` picks out, in chain order; the companion screens need
    ``shipped`` (:func:`screens`)."""
    names = list(names)
    unknown = set(names) - set(SCREENS)
    if unknown:
        raise ValueError(f"not a screen: {sorted(unknown)}; the screens are {list(SCREENS)}")
    if shipped is None and {"companion_apart", "companion_12h"} & set(names):
        raise ValueError("the companion screens need the survey as shipped, to tell a reported start from a read one")
    s = screens(tables, shipped)
    return s.index[s[names].any(axis=1).to_numpy()]


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
    row that shows the answer, a field of a row the edits drop), rows with a
    defect before and after, persons cleared and persons with a defect the
    edits made (a code absent before), and, over the same persons, the stays
    of no minutes and the persons each of the ``SCREENS`` picks out on the
    rules' output, before the edits and after them (:func:`screens`), with
    the persons a screen picks out after the edits but not before, and after
    the edits but not on the rules' output — a day the hand made implausible.
    ``sheet`` is the review sheet of those persons rebuilt from the revised
    tables. ``rules`` (the rules' output) is also where a restored row comes
    from when the rules kept it (see :func:`apply_edits`).
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
    sr, sb, sa = (screens(t, shipped).reindex(who) for t in (rules, cleaned, revised))
    flags = list(SCREENS)
    on = lambda frame: frame[flags].fillna(False).to_numpy(bool)
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
        **{f"zero_stays_{when}": int(frame.zero_stays.fillna(0).sum())
           for when, frame in (("rules", sr), ("before", sb), ("after", sa))},
        **{f"{name}_{when}": int(frame[name].fillna(False).sum())
           for name in flags for when, frame in (("rules", sr), ("before", sb), ("after", sa))},
        "persons_with_new_screens": int((on(sa) & ~on(sb)).any(axis=1).sum()),
        "persons_with_screens_not_in_rules": int((on(sa) & ~on(sr)).any(axis=1).sum()),
    }
    return Verification(
        revised, chain_sheet(after_rows, revised.hab, who), persons, summary
    )


# ------------------------------------------------------------------ a round


class Round(NamedTuple):
    """What :func:`freeze_round` returns."""

    decisions: pd.DataFrame  # the hand decisions with the round merged in
    frozen: pd.DataFrame  # the sheet's edits that change the tables, in canonical form
    changes: dict  # decisions added, changed and removed
    verified: Verification | None  # the edits applied to the tables the sheet was exported from; None without edits
    stale: pd.DataFrame  # decisions the rules no longer supported: the sheet decided them again or let them go
    dead: pd.DataFrame  # decisions that changed nothing any more (the rules give their value): removed


def _same_tables(a: EODTables, b: EODTables) -> bool:
    legs = a.legs.equals(b.legs) if a.legs is not None and b.legs is not None else a.legs is b.legs
    return a.trips.equals(b.trips) and a.hab.equals(b.hab) and legs


def note_edits(edits: pd.DataFrame, tables: EODTables) -> pd.DataFrame:
    """The notes typed where nothing else changed (:func:`sheet_edits` field ``note``), as ``household, person,
    trip, note``; on the row a sheet exported from ``tables`` shows the person's answer on, the note is the
    answer's too (trip ``PERSON_TRIP``)."""
    notes = edits.loc[edits.field.astype(str) == NOTE, KEYS + [NOTE]].copy()
    notes[NOTE] = notes[NOTE].astype(str).str.strip()
    notes = notes[notes[NOTE] != ""].astype({k: int for k in KEYS})
    shows = _answer_rows(tables.trips.index)
    answer = notes[[shows.get((h, p)) == (h, p, t) for h, p, t in zip(notes.household, notes.person, notes.trip)]]
    return pd.concat([notes, answer.assign(trip=PERSON_TRIP)], ignore_index=True)


def freeze_round(edits: pd.DataFrame, stages: EODStages, source: str, *, decisions=None, shown=None) -> Round:
    """A review round merged into the hand decisions, checked: what ``eodgdl review freeze`` writes.

    ``edits`` is what :func:`sheet_edits` recovers from a sheet exported from
    ``stages.revised``, the tables the decisions give (``decisions``, default
    the shipped table). The edits that change something are kept in canonical
    form (:func:`freeze_edits`) and merged (:func:`merge_decisions`) under
    ``source``, the round's name; a note typed where nothing else changed is
    the new reason of the decisions already on that trip (:func:`note_edits`).
    A decision that no longer changes anything (the rules now give its value)
    is removed. A stale decision — one the rules' output no longer supports,
    which ``load_stages(skip_stale=True)`` set aside — must sit on a row the
    sheet shows (``shown``, the sheet's keys, :func:`sheet_keys`; for the
    person's answer, any of the person's rows): the sheet's edit on that
    field, if any, is the decision now, and none lets the rules' value stand;
    a stale decision the sheet does not show is an error. Nothing is returned
    unless the merged decisions, applied once to the rules' output, give
    exactly what the edits give on top of ``stages.revised``. Raises
    ``ValueError`` on an edit that cannot be applied, on a decision no table
    can take, on stale decisions the sheet does not show, and when there is
    nothing to merge. ``eodgdl review verify`` is this without the write.
    """
    decisions = _decisions(decisions)
    outcomes = decision_outcomes(stages.rules, stages.shipped, decisions)
    bad = outcomes[outcomes.outcome == "problem"]
    if len(bad):
        raise ValueError("hand decisions no table can take:\n  " + "\n  ".join(bad.problem))
    stale = outcomes[outcomes.outcome == "stale"]
    dead = outcomes[outcomes.outcome.isin(["holds", "moot"])]
    if len(stale):
        seen = set(shown or ())
        persons = {k[:2] for k in seen}
        unseen = [k for k in zip(stale.household, stale.person, stale.trip)
                  if (k[:2] not in persons if k[2] == PERSON_TRIP else k not in seen)]
        if unseen:
            raise ValueError(
                f"{len(stale)} hand decisions no longer apply to the rules' output and {len(unseen)} of their "
                "trips are not on this sheet; `eodgdl review export --stale` puts them on one"
            )
    frozen = freeze_edits(edits, stages.revised, stages.shipped, stages.rules)
    notes = note_edits(edits, stages.revised)
    kept = decisions.drop(index=stale.index.union(dead.index))
    merged = merge_decisions(kept, frozen, stages.rules, stages.shipped, source, notes)
    changes = decision_changes(decisions, merged)
    if frozen.empty and stale.empty and dead.empty and not changes["notes"]:
        raise ValueError("nothing to freeze: every edit already holds or is moot")
    once = apply_revisions(stages.rules, stages.shipped, merged)
    stacked = apply_edits(stages.revised, frozen, stages.shipped, stages.rules)
    if not _same_tables(once, stacked):
        raise ValueError("the merged decisions do not give what the sheet's edits give; nothing merged")
    verified = verify_edits(stages.revised, stages.shipped, edits, stages.rules) if len(frozen) else None
    return Round(merged, frozen, changes, verified, stale.drop(columns="outcome"), dead.drop(columns="problem"))


def prefill_stale(sheet: pd.DataFrame, stale: pd.DataFrame) -> pd.DataFrame:
    """``sheet`` with every stale decision on a row it shows written into its ``new …`` cell, and why into the note.

    ``stale`` is :func:`stale_decisions`. The value the decision set stands
    ready to be kept, changed or cleared; the note says what it was made
    against and carries its reason. :func:`freeze_round` reads the cell as
    it finds it: kept, the decision is made again against today's rules;
    cleared, the rules' value stands.
    """
    sheet = sheet.copy()
    at = {k: i for i, k in zip(sheet.index, zip(sheet.household, sheet.person, sheet.trip))}
    for i, h, p, status in zip(sheet.index, sheet.household, sheet.person, sheet[STATUS]):
        if status != DROPPED:  # the answer shows on the person's first kept row
            at.setdefault((int(h), int(p), PERSON_TRIP), i)
    for d in stale.itertuples(index=False):
        i = at.get((int(d.household), int(d.person), int(d.trip)))
        if i is None:
            continue
        column = new_column(STATUS if d.field == STATUS else d.field)
        sheet.loc[i, column] = d.after
        why = (f"Decisión vencida ({d.source}): se tomó contra «{d.before}» y la regla ya no da ese valor; "
               f"se propone de nuevo «{d.after}». Nota original: {d.note}")
        sheet.loc[i, NOTE] = f"{sheet.at[i, NOTE]} | {why}" if sheet.at[i, NOTE] else why
    return sheet
