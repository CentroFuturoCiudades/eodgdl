"""Round 15 over the trip chains: the hand decisions made again on the chain rules of the fifth review.

On 2026-09-26 the fifth review of the chains changed the rules under round 14's decisions: the
travel minutes round 9 corrected are read before the rules, since the rules read the minutes; a
day read past midnight must hold together after it (no trip to work or school after the night,
and less than 24 hours from the day's first start to its last); the typo search runs once more
after the move to the arrival; and two of round 14's criteria became rules, the rotated first
trips and the returns that stop short of home. This round makes the decisions again against that
output, from round 14's own inputs: the decisions as rounds 1–13 left them and a snapshot of the
tables they gave (``PREVIOUS``, ``SNAPSHOT``). Round 14's table was retired whole, as round 14
retired the one before it.

- **Leg minutes** (``leg_minutes``): round 9's corrections of the travel minutes, every one — a
  motorised trip reporting twice the survey's typical minutes for its mode over its straight-line
  distance takes the typical minutes, and so do the person's other trips over the same route —
  written as the table the rules read before anything else
  (``eodgdl/revisions/leg_minutes.csv.gz``, ``review.write_leg_minutes``). Round 14 carried them as
  decisions applied after the rules, which had read the reported minutes.
- **Carried** (``carried``): the decisions of passes 2–6 on any field but the start — the rotated
  first trips and separate starts of pass 5, the staff bus of pass 4, the swaps and restores of
  pass 2 — where the rules' output still holds the value each was made against and not yet the
  value it sets, on a person the rules' output marks. A drop of a row the rules already mark as no
  trip is left out, and so are the day-start answers and the night shifts' last rows, which rules
  give. Pass 5's rotated first trips are completed where no rule starts that day at home.
- **Start times** (``readings``): the day of every person the rules' output marks whose start
  times the earlier tables read otherwise, or who still has a trip that starts before the
  previous one arrives. Three readings compete: the rules' (made again on the chain as the carried
  decisions leave it: ``eodgdl.chains._read_start_times``), the earlier tables' (the first pass's
  reading and its corrections: a suggestion), and, where the rules' reading still overlaps, the
  fewest minutes moved in all that let every trip leave once the previous one has arrived (an
  isotonic fit; ``fewest_minutes``). The reading kept is the one with no overlap and the fewest
  problems (``eodgdl.review.screens``: stays of no minutes away from home, under half an hour at
  work or school, a first non-work trip before 05:00, eight hours or more at an errand, a day of
  more than 20 hours that is no night shift), then the one that changes the fewest starts and
  moves the fewest minutes from the rules' reading.

Round 14's other two steps, the rotated first trips that answered 'Su casa' and the returns that
stop short, are the rules ``_rotated_first_trips`` and ``_short_returns`` now.

A second run finds nothing: the leg minutes are written again as they are, the target is computed
from the rules' output and the two inputs, and the tables the decisions give then hold it.

    uv run python scripts/revisions/chains_15.py --data data --out notebooks/chain_review_15.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from eodgdl import load_stages, review
from eodgdl._resources import leg_minutes as committed_leg_minutes
from eodgdl.chains import (
    BREAKING_ISSUES, HOME_MOTIVE, PERSON, _leg_minutes, _read_start_times, _start_minutes, days_past_midnight,
    has_code, non_trips,
)
from eodgdl.eod import EODTables

PREVIOUS = "notebooks/revisions/decisions_13.csv.gz"  # the hand decisions as rounds 1-13 left them
SNAPSHOT = "notebooks/revisions/snapshot_13.csv.gz"  # the tables they gave: eodgdl review snapshot, before round 14's rules
ROUND = "Decimoquinta revisión"
LEG_MINUTES = "chain_review_9"  # the round whose leg minutes the rules read
ROUND_9 = "Novena revisión"  # how round 9's notes begin its own reason
CARRIED = {  # the rounds whose decisions on other fields than the start carry over, and how the notes name them
    "chain_review_2": "la segunda revisión", "chain_review_3": "la tercera revisión",
    "chain_review_4": "la cuarta revisión", "chain_review_5": "la quinta revisión",
    "chain_review_6": "la sexta revisión",
}
NAMES = {"chain_review": "la primera revisión", **CARRIED, "chain_review_8": "la octava revisión",
         "chain_review_9": "la novena revisión", "chain_review_10": "la décima revisión",
         "chain_review_11": "la undécima revisión", "chain_review_12": "la duodécima revisión"}
PROBLEMS = {  # review.screens columns that make up its problems count, and this round's own, as the notes name them
    "small_hours": "una salida movida a la madrugada",
    "zero_away_stays": "una estancia de cero minutos fuera de casa",
    "short_work_stays": "menos de media hora en el trabajo o la escuela",
    "early": "un primer viaje antes de las 05:00",
    "long_errands": "una estancia de 8 horas o más en un mandado o una visita",
    "long_day": "un día de más de 20 horas",
}


def hhmm(minutes) -> str:
    minutes = int(round(minutes)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def breaking(tables: EODTables) -> pd.Series:
    """Per person: does a row carry a breaking code?"""
    t = tables.trips
    flag = pd.concat([has_code(t.problemas, c) for c in BREAKING_ISSUES], axis=1).any(axis=1)
    return flag.groupby(level=PERSON).any()


# ------------------------------------------------------------------ leg minutes


def leg_minutes(previous: pd.DataFrame) -> pd.DataFrame:
    """Round 9's corrections of the travel minutes, as the table the rules read (``review.LEG_MINUTE_COLUMNS``).

    The note is round 9's reason; the text of the trip's earlier decisions that its note carried (the first pass's
    hour readings, retired) is left out."""
    rows = previous[(previous.source == LEG_MINUTES) & (previous.field == "leg min")]
    rows = rows.assign(note=[n[n.index(ROUND_9):] if ROUND_9 in n else n for n in rows.note])
    return rows[review.LEG_MINUTE_COLUMNS].astype({"before": int, "after": int}).reset_index(drop=True)


def write_leg_minutes(table: pd.DataFrame) -> bool:
    """Write the leg minutes where the committed table differs from ``table``; True if it was written."""
    path = review.leg_minutes_path()
    now = committed_leg_minutes() if path.exists() else None
    order = lambda t: t[review.LEG_MINUTE_COLUMNS].sort_values(review.KEYS).reset_index(drop=True).astype(str)  # noqa: E731
    if now is not None and order(now).equals(order(table)):
        return False
    review.write_leg_minutes(table, path)
    return True


# ------------------------------------------------------------------ carried decisions


def carried(previous: pd.DataFrame, stages) -> pd.DataFrame:
    """The earlier decisions that carry over: see the module docstring."""
    rules, shipped = stages.rules, stages.shipped
    candidates = previous[previous.source.isin(CARRIED) & (previous.field != "start")].reset_index(drop=True)
    outcome = review.decision_outcomes(rules, shipped, candidates).outcome
    keys = list(zip(candidates.household, candidates.person, candidates.trip))
    not_trip = non_trips(rules.trips)
    drop = ((candidates.field == "status") & (candidates.after == review.DROPPED)).to_numpy()
    redundant = drop & np.array([bool(not_trip.get(k, False)) for k in keys], dtype=bool)
    marked = set(review.marked_persons(rules))
    on_marked = np.array([k[:2] in marked for k in keys], dtype=bool)
    # the sixth pass recoded trips that arrive at 'Su casa' in the home zone; where only the first pass had put
    # the trip there, the premise is gone with it
    t = rules.trips.reindex(keys)
    home = shipped.viv.ageb.astype(str).reindex([k[0] for k in keys]).to_numpy()
    arrives = (t.destino.astype(str).to_numpy() == home) & (t.tipo_lugar_destino.astype(str).to_numpy() == "Su casa")
    premise = ~((candidates.source == "chain_review_6").to_numpy() & ~arrives)
    kept = candidates[(outcome == "apply").to_numpy() & ~redundant & on_marked & premise]
    kept = kept.assign(note=[f"{ROUND}: se conserva la decisión de {CARRIED[s]} ({s}). {n}"
                             for s, n in zip(kept.source, kept.note)])
    return pd.concat([kept, rotations(kept, stages)], ignore_index=True)


def rotations(kept: pd.DataFrame, stages) -> pd.DataFrame:
    """The fifth pass's rotated first trips, completed where no rule starts the day at home.

    The fifth pass read a first trip recorded from a place to home, followed by one from home onwards, as the
    trip from home to that place, and set its destination and the next trip's origin to the place; the first
    pass had already moved the first trip's origin home and made the answer 'Su casa'. The rule of rotated first
    trips reads most such days by itself; where the next trip goes somewhere else, or back to the place, it
    does not, and the origin and the answer are set here, or the trip would go from the place to it.
    """
    rules, home = stages.rules.trips, stages.shipped.viv.ageb.astype(str)
    rows = []
    for d in kept[(kept.source == "chain_review_5") & (kept.field == "destination")].itertuples(index=False):
        key = (d.household, d.person, d.trip)
        first = rules.loc[key[:2]].index[0] == d.trip
        if not first or str(rules.at[key, "origen"]) == home[d.household]:
            continue
        note = (f"{ROUND}: el viaje 1 se registró de {rules.at[key, 'origen']} a casa y el siguiente de casa en "
                f"adelante; la quinta revisión leyó que {d.after} es adonde fue, no de donde salió, sobre la lectura de "
                "la primera revisión, que empezaba el día en casa. Ninguna regla empieza hoy este día en casa, así que "
                "el origen del viaje 1 es la vivienda y la respuesta de dónde empezó el día, «Su casa».")
        rows.append((*key, "origin", str(rules.at[key, "origen"]), review.HOME, note, "chain_review_5"))
        answer = str(rules.at[key, "tipo_lugar_origen"])
        if answer != "Su casa":
            rows.append((d.household, d.person, review.PERSON_TRIP, "orig. type", answer, "Su casa", note, "chain_review_5"))
    return pd.DataFrame(rows, columns=review.DECISION_COLUMNS)


# ------------------------------------------------------------------ start times


def fewest_minutes(start: np.ndarray, minutes: np.ndarray, away: np.ndarray) -> np.ndarray:
    """The starts nearest ``start`` (fewest minutes moved in all) with every trip leaving once the previous arrived.

    With ``u_i`` the start less the minutes of the trips before, the constraint is that ``u`` never decreases:
    an isotonic fit in the absolute distance, whose optimum takes values among the recorded ones. Ties go to
    the reading that moves the fewest trips, then leaves the fewest stays of no minutes where ``away`` (the
    trip ends away from home), then moves the fewest minutes earlier. (Round 12's, 2026-09-26.)
    """
    n = len(start)
    before = np.r_[0, np.cumsum(minutes[:-1])]
    v = start - before
    values = np.unique(v)
    best = [{} for _ in range(n)]
    for k, u in enumerate(values):
        best[0][k] = ((abs(u - v[0]), int(u != v[0]), 0, max(0.0, v[0] - u)), None)
    for i in range(1, n):
        for k, u in enumerate(values):
            step = (abs(u - v[i]), int(u != v[i]), 0, max(0.0, v[i] - u))
            options = [
                (tuple(a + b for a, b in zip(cost, step[:2] + (int(values[j] == u and away[i - 1]),) + step[3:])), j)
                for j, (cost, _) in best[i - 1].items() if values[j] <= u
            ]
            if options:
                best[i][k] = min(options)
    k = min(best[n - 1], key=lambda j: best[n - 1][j][0])
    fit = np.zeros(n)
    for i in range(n - 1, -1, -1):
        fit[i] = values[k]
        k = best[i][k][1]
    return fit + before


def chain_frame(tables: EODTables) -> pd.DataFrame:
    """The chain the model reads, one row per trip: start (absolute minutes), minutes, motive, away from home."""
    t = tables.trips[~non_trips(tables.trips)].sort_index()
    home = tables.viv.ageb.astype(str).reindex(t.index.get_level_values(0)).to_numpy()
    is_return = (t.motivo_viaje == HOME_MOTIVE).to_numpy()
    return pd.DataFrame({
        "start": _start_minutes(t) + 1440 * days_past_midnight(t).to_numpy(),
        "minutes": _leg_minutes(t, tables.legs), "motive": t.motivo_viaje.astype(str),
        "away": ~(is_return & (t.destino.astype(str).to_numpy() == home)),
    }, index=t.index)


def shown_starts(tables: EODTables) -> pd.Series:
    """Every trip's start as a sheet shows it, HH:MM."""
    t = tables.trips
    return pd.Series([hhmm(h * 60 + m) for h, m in zip(t.hora_inicio_h.astype(int), t.hora_inicio_m.astype(int))],
                     index=t.index)


def rules_reading(tables: EODTables, stages, persons: set) -> pd.Series:
    """The rules' start times for the chains the carried decisions change: the rules' time stage
    (``_read_start_times``) run again from the survey's hours on the chain as the carried decisions leave it — a
    row gone, a motive recoded, an origin moved. An imputed start keeps the rules' value. Only for ``persons``;
    returns the new start (HH:MM) of every trip that changes."""
    t = tables.trips[~non_trips(tables.trips)].sort_index()
    t = t[t.index.droplevel("folio_viaje").isin(list(persons))]
    if t.empty:
        return pd.Series(dtype=object)
    s = stages.shipped.trips.reindex(t.index)
    imputed = (s.hora_inicio_h.isna() | s.hora_inicio_m.isna()).to_numpy()
    t = t.copy()
    for column in ("hora_inicio_h", "hora_inicio_m"):
        t[column] = pd.array(np.where(imputed, t[column].astype(float), s[column].astype(float)).astype(int), dtype="Int64")
    t, _, _, _ = _read_start_times(t, tables.legs, imputed)
    new = shown_starts(EODTables(tables.viv, tables.hab, t, tables.legs))
    now = shown_starts(tables).reindex(new.index)
    return new[new != now]


def apply_starts(tables: EODTables, starts: pd.Series, stages) -> EODTables:
    """``tables`` with the starts in ``starts`` (key -> HH:MM) set, problemas recomputed."""
    if starts.empty:
        return tables
    now = shown_starts(tables).reindex(starts.index)
    edits = pd.DataFrame({
        "household": [k[0] for k in starts.index], "person": [k[1] for k in starts.index],
        "trip": [k[2] for k in starts.index], "field": "start", "before": now.to_numpy(), "after": starts.to_numpy(),
        "note": "",
    })
    edits = edits[edits.before != edits.after]
    return review.apply_edits(tables, edits, stages.shipped, stages.rules)


def readings(tables: EODTables, stages, snapshot: pd.DataFrame, previous: pd.DataFrame,
             carried_persons: set) -> tuple[pd.Series, dict]:
    """The start times the days in question take (key -> HH:MM, where they change from ``tables``) and each one's note."""
    marked = set(review.marked_persons(stages.rules))
    now = shown_starts(tables)
    chain = chain_frame(tables)
    kept = snapshot[snapshot[review.STATUS] != review.DROPPED]
    common = chain.index[chain.index.isin(kept.index)]
    earlier = pd.Series(kept.reindex(common)["start"].to_numpy(), index=common)
    differs = earlier.to_numpy() != now.reindex(common).to_numpy()
    broken = breaking(tables)
    persons = sorted((set(common[differs].droplevel("folio_viaje")) | set(broken[broken].index) | carried_persons)
                     & marked)

    base = rules_reading(tables, stages, set(persons) & carried_persons)  # the rules' reading of the chain carried
    base_tables = apply_starts(tables, base, stages)
    base_now = shown_starts(base_tables)
    base_broken = breaking(base_tables)
    base_chain = chain_frame(base_tables)
    fit = {}
    for person in persons:
        if base_broken.get(person, False):
            c = base_chain.loc[person]
            new = fewest_minutes(c.start.to_numpy(), c.minutes.to_numpy(), c.away.to_numpy())
            fit.update({(*person, trip): hhmm(v) for trip, v in zip(c.index, new)})
    fit = pd.Series(fit, dtype=object)
    if len(fit):
        fit.index = pd.MultiIndex.from_tuples(fit.index, names=tables.trips.index.names)

    def only(starts: pd.Series) -> pd.Series:  # the starts of the persons in question that differ from the base reading
        if starts.empty:
            return starts
        starts = starts[starts.index.droplevel("folio_viaje").isin(persons)]
        return starts[starts.to_numpy() != base_now.reindex(starts.index).to_numpy()]

    who = pd.MultiIndex.from_tuples(persons, names=PERSON)
    candidates = {"rules": pd.Series(dtype=object), "earlier": only(earlier), "fit": only(fit)}
    departure = tables.trips.motivo_viaje != HOME_MOTIVE
    scored = {}
    for name, starts in candidates.items():
        t = apply_starts(base_tables, starts, stages)
        # a departure a reading moves into the small hours: the rules read no hour to before 05:00 either
        small = pd.Series([_minutes(v) < SMALL_HOURS and bool(departure[k]) for k, v in starts.items()], dtype=int)
        if len(starts):
            small.index = starts.index
        moved = pd.Series([abs(_minutes(v) - _minutes(base_now[k])) for k, v in starts.items()], dtype=float)
        if len(starts):
            moved.index = starts.index
        s = review.screens(t).reindex(who)
        s["small_hours"] = (small.groupby(level=PERSON).sum() if len(starts) else pd.Series(dtype=int)).reindex(who).fillna(0)
        s["problems"] = s.problems.fillna(0) + s.small_hours
        scored[name] = {
            "starts": starts,
            "screens": s,
            "broken": breaking(t).reindex(who).fillna(False),
            "changed": (pd.Series(1, index=starts.index).groupby(level=PERSON).sum() if len(starts)
                        else pd.Series(dtype=int)).reindex(who).fillna(0),
            "moved": (moved.groupby(level=PERSON).sum() if len(starts) else pd.Series(dtype=float)).reindex(who).fillna(0.0),
            "persons": set(starts.index.droplevel("folio_viaje")) if len(starts) else set(),
        }
    chosen = {}
    for person in persons:
        options = []
        for order, name in enumerate(("rules", "earlier", "fit")):
            c = scored[name]
            if name != "rules" and person not in c["persons"]:
                continue
            key = (bool(c["broken"][person]), int(c["screens"].problems.fillna(0)[person]), int(c["changed"][person]),
                   float(c["moved"][person]), order)
            options.append((key, name))
        chosen[person] = min(options)[1]

    final, notes = {}, {}
    sources = previous[previous.field == "start"].groupby(["household", "person"]).source.agg(lambda s: sorted(set(s)))
    for key, value in base.items():
        final[key] = value
        notes[key] = (f"{ROUND}: la salida queda a las {value}, la que dan las reglas al leer de nuevo las horas del "
                      "día tal como lo dejan las decisiones que se conservan (un renglón quitado, un motivo "
                      f"recodificado, un origen movido); sobre el día registrado daban las {now[key]}.")
    for person, name in chosen.items():
        if name == "rules":
            continue
        c = scored[name]
        mine = c["starts"][c["starts"].index.droplevel("folio_viaje") == person]
        detail = ", ".join(f"viaje {k[2]} {now[k]} → {v}" for k, v in mine.items())
        rules_text = (OVERLAP if scored["rules"]["broken"][person] else problem_text(scored["rules"]["screens"], person))
        if name == "earlier":
            by = sources.get(person)
            by = " y ".join(NAMES.get(s, s) for s in by) if by is not None else "las reglas anteriores"
            note = (f"{ROUND}: se toma la lectura de las horas de las tablas anteriores ({by}), una sugerencia entre "
                    f"otras: con las reglas de hoy el día queda con {rules_text}, y con esa lectura con "
                    f"{problem_text(c['screens'], person)}. Cambia: {detail}.")
        else:
            if person in scored["earlier"]["persons"]:
                earlier_text = (OVERLAP if scored["earlier"]["broken"][person]
                                else problem_text(scored["earlier"]["screens"], person))
            else:
                earlier_text = "lo mismo"
            note = (f"{ROUND}: ninguna lectura de las horas cierra el día: con las reglas queda {rules_text}, con la "
                    f"de las tablas anteriores {earlier_text}. Las salidas se mueven lo menos posible "
                    f"({int(c['moved'][person])} min en total) para que ningún viaje salga antes de que llegue el "
                    f"anterior; queda {problem_text(c['screens'], person)}. Cambia: {detail}.")
        for k, v in mine.items():
            final[k] = v
            notes[k] = note
    out = pd.Series(final, dtype=object)
    if len(out):
        out.index = pd.MultiIndex.from_tuples(list(out.index), names=tables.trips.index.names)
        out = out[out.to_numpy() != now.reindex(out.index).to_numpy()]
    counts = pd.Series(chosen, dtype=object).value_counts().to_dict()
    print(f"days read: {len(persons)} {counts}; starts the rules read again on the chains carried: {len(base)}")
    return out.sort_index(), {k: notes[k] for k in out.index}


OVERLAP = "un viaje que sale antes de que llegue el anterior"
SMALL_HOURS = 5 * 60  # minutes: a reading that moves a departure before this counts a problem


def _minutes(text: str) -> int:
    return int(text[:2]) * 60 + int(text[3:5])


def problem_text(s: pd.DataFrame, person) -> str:
    """The problems of ``person``'s day in ``s`` (``review.screens``), as a note names them."""
    if person not in s.index or s.loc[person].isna().all():
        return "ningún problema"
    row = s.loc[person]
    parts = [label for col, label in PROBLEMS.items()
             if (bool(row.long_day) and not bool(row.night_shift_end) if col == "long_day" else bool(row[col]))]
    return "; ".join(parts) if parts else "ningún problema"


# ------------------------------------------------------------------ the sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    parser.add_argument("--previous", default=PREVIOUS, help="The decisions as rounds 1-13 left them")
    parser.add_argument("--snapshot", default=SNAPSHOT, help="The tables they gave")
    parser.add_argument("--out", default="notebooks/chain_review_15.csv", help="Where to write the sheet")
    args = parser.parse_args()

    previous = pd.read_csv(args.previous, dtype=str, keep_default_na=False).astype(
        {k: int for k in review.KEYS})
    minutes = leg_minutes(previous)
    written = write_leg_minutes(minutes)  # before the rules run: they read these minutes
    print(f"{'wrote' if written else 'unchanged:'} {review.leg_minutes_path()} ({len(minutes)} trips)")
    stages = load_stages(args.data)
    snapshot = review.read_snapshot(args.snapshot)
    rules, shipped, revised = stages.rules, stages.shipped, stages.revised

    kept = carried(previous, stages)
    tables = review.apply_revisions(rules, shipped, kept)
    starts, start_why = readings(tables, stages, snapshot, previous, set(zip(kept.household, kept.person)))
    tables = apply_starts(tables, starts, stages)
    left = breaking(tables)
    if left.any():
        raise SystemExit(f"{int(left.sum())} persons still carry a breaking code: {list(left[left].index)[:10]}")

    # the target, against what the tables the decisions give now hold
    target, current = review.snapshot(tables, shipped), review.snapshot(revised, shipped)
    diff = (target != current.reindex(target.index)) & ~(target.isna() & current.reindex(target.index).isna())
    notes = defaultdict(list)
    for d in kept.itertuples(index=False):
        notes[(d.household, d.person, d.trip)].append(d.note)
    for key, note in start_why.items():
        notes[key].append(note)

    def answer(snap):  # one answer per person, off the rows kept
        return snap.loc[snap[review.STATUS] != review.DROPPED, "orig. type"].groupby(level=[0, 1]).first()

    was, now = answer(current), answer(target)
    answers = now[now != was.reindex(now.index)]
    fields = [f for f in diff.columns if f in review.FIELDS and f != "orig. type"] + [review.STATUS]
    status_changes = diff[review.STATUS]
    rows = diff.index[diff[fields].any(axis=1).to_numpy()]
    persons = sorted(set(answers.index) | {k[:2] for k in rows})
    sheet = review.chain_sheet(review.chain_rows(revised, shipped), revised.hab,
                               pd.MultiIndex.from_tuples(persons, names=PERSON))
    at = {k: i for i, k in enumerate(zip(sheet.household, sheet.person, sheet.trip))}
    first_kept = {}
    for i, (h, p, status) in enumerate(zip(sheet.household, sheet.person, sheet[review.STATUS])):
        if status != review.DROPPED:
            first_kept.setdefault((h, p), i)
    home = shipped.viv.ageb.astype(str)
    cells = 0
    for person, value in answers.items():
        row = first_kept[person]
        sheet.loc[row, review.new_column("orig. type")] = value
        person_notes = [n for (h, p, t), ns in notes.items() if (h, p) == person and t == review.PERSON_TRIP for n in ns]
        if person_notes:
            sheet.loc[row, review.NOTE] = " | ".join(dict.fromkeys(person_notes))
        cells += 1
    for key in rows:
        # a row dropped or restored takes its new status alone: the sheet shows a dropped row as shipped
        changed = [review.STATUS] if status_changes[key] else [f for f in fields if f != review.STATUS and diff.at[key, f]]
        for field in changed:
            value = target.at[key, field]
            if field == review.STATUS:
                value = review.DROPPED if value == review.DROPPED else review.RESTORED
            elif field in ("origin", "destination") and value == home[key[0]]:
                value = review.HOME
            sheet.loc[at[key], review.new_column(field)] = value
            cells += 1
        if notes.get(key):
            mine = " | ".join(dict.fromkeys(notes[key]))
            there = sheet.at[at[key], review.NOTE]
            sheet.loc[at[key], review.NOTE] = f"{there} | {mine}" if there else mine
    review.write_sheet(sheet, args.out)
    print(f"wrote {args.out}: {len(persons)} persons, {len(sheet)} rows, {cells} cells "
          f"({len(kept)} carried, {len(starts)} starts)")


if __name__ == "__main__":
    main()
