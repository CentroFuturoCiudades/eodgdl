"""Round 12 over the trip chains: the days the first pass's hour readings made implausible.

The third review of the chains (2026-09-26) found that the first hand pass, closing the time
contradictions the rules left, often made the day implausible where the rules' output was
not: a first trip read twelve hours back into the small hours (a doctor at 00:30, school at
02:00), a return slid to the minute its trip to work arrives (no time at work), a shopping
trip that lasts ten hours, a day of more than twenty. This round rereads those days from
the rules' output, the survey's own hours as the rules read them.

**Which days.** Those of the persons for whom the tables ``load_eod`` returns show more of
these than the rules' output does (``problems``):

- a stay of no minutes at work or at school;
- a first trip, to anything but work, that starts before 05:00 (``hora_madrugada``);
- a stay of eight hours or more at a place that is not home, work or school;
- a day of more than twenty hours.

Round 10 looked at the persons in ``ROUND_10`` one by one and kept the first pass's reading,
so this round leaves them.

**The reading.** Every other decision on the day stands: its drops, minutes, places and
motives. Its starts go back to the rules' and move the fewest minutes in all that let every
trip start once the previous one has arrived (``fewest_minutes``). Among readings that move
the same minutes, the one moving the fewest trips wins, then the one with the fewest stays of
no minutes away from home, then the one moving later trips rather than earlier ones. Where
the day's last trip is a return home recorded at 12:xx that starts before the previous trip
arrives, and that trip started at or after noon, the return can also be read on the 12-hour
clock's midnight, 00:xx of the next day (``midnight``). The fewest minutes then place the
rest.

**When.** The new reading replaces the current one only if it leaves fewer of the problems
above, or as many while moving fewer minutes from the rules' hours. Where it would leave
more (a day whose recorded hours hold two tours that cannot both fit), the current reading
stands.

A second run finds nothing: the days changed then hold the reading this round chose.

    uv run python scripts/revisions/chains_12.py --data data --out notebooks/chain_review_12.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from eodgdl import load_stages, review
from eodgdl.chains import (
    BREAKING_ISSUES, HOME_MOTIVE, PERSON, _EARLY_START, _leg_minutes, _next_day, _start_minutes, days_past_midnight,
    has_code, mark_issues, non_trips,
)

# the persons round 10 read one by one and kept as the first pass read them (a double shift is no error in itself;
# no same-day reading fits the others' recorded hours): scripts/revisions/chains_10.py
ROUND_10 = {(16714, 1), (11165, 1), (13088, 3), (7776, 3), (8201, 1), (1771, 1), (11303, 1), (11303, 2)}
LONG_STAY = 8 * 60  # minutes at a place that is not home, work or school
LONG_DAY = 20 * 60  # minutes from the day's first start to its last arrival
MOST_MOVED = 4 * 60  # minutes: a start this far off is an hour typed wrong, not minutes to close
MANDATORY = ("Trabajar", "Estudiar")
ESCORT = "Llevar o recoger a alguien"  # dropping someone off takes no time


def problems(start: np.ndarray, minutes: np.ndarray, motives: np.ndarray) -> np.ndarray:
    """How many of each problem a reading of one day holds: stays of no minutes anywhere but home (an escort's
    drop-off apart), an early first trip to anything but work, stays of eight hours or more elsewhere than home,
    work or school, a day of more than twenty hours (unless it ends with the morning return of a night shift)."""
    stay = np.r_[start[1:] - (start[:-1] + minutes[:-1]), np.nan]
    night_shift = len(start) > 1 and motives[-1] == HOME_MOTIVE and motives[-2] == "Trabajar" and start[-1] >= 1440
    return np.array([
        int(((stay == 0) & ~np.isin(motives, [HOME_MOTIVE, ESCORT])).sum()),
        int(motives[0] not in ("Trabajar", HOME_MOTIVE) and start[0] % 1440 < _EARLY_START),
        int(((stay >= LONG_STAY) & ~np.isin(motives, [HOME_MOTIVE, *MANDATORY])).sum()),
        int(start[-1] + minutes[-1] - start[0] > LONG_DAY and not night_shift),
    ])


def fewest_minutes(start: np.ndarray, minutes: np.ndarray, away: np.ndarray) -> np.ndarray:
    """The starts nearest ``start`` (fewest minutes moved in all) with every trip leaving once the previous arrived.

    With ``u_i`` the start less the minutes of the trips before, the constraint is that ``u`` never decreases:
    an isotonic fit in the absolute distance, whose optimum takes values among the recorded ones. Ties go to
    the reading that moves fewest trips, then leaves the fewest stays of no minutes where ``away`` (the trip
    ends away from home), then moves the fewest minutes earlier.
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


def midnight(start: np.ndarray, minutes: np.ndarray, motives: np.ndarray) -> np.ndarray | None:
    """The day with its last return read on the 12-hour clock's midnight, where that reading applies; else None."""
    if len(start) < 2 or motives[-1] != HOME_MOTIVE or not 720 <= start[-1] % 1440 < 780:
        return None
    if start[-2] % 1440 < 720 or start[-1] >= start[-2] + minutes[-2]:
        return None
    out = start.copy()
    out[-1] += 720
    return out


def day(tables, person, rules_starts):
    """One person's chain as the model reads it: keys, starts (absolute minutes), minutes, motives, away."""
    t = tables.trips[~non_trips(tables.trips)].xs(person, level=[0, 1], drop_level=False).sort_index()
    home = str(tables.viv.ageb.astype(str).loc[person[0]])
    if rules_starts:
        s = rules_starts
        s = np.array([s[k] for k in t.index], dtype=float)
        is_return = (t.motivo_viaje == HOME_MOTIVE).to_numpy()
        start = s + 1440 * np.cumsum(_next_day(s, np.r_[np.nan, s[:-1]], is_return))
    else:
        start = _start_minutes(t) + 1440 * days_past_midnight(t).to_numpy()
    motives = t.motivo_viaje.astype(str).to_numpy()
    away = ~((motives == HOME_MOTIVE) & (t.destino.astype(str).to_numpy() == home))
    return list(t.index), start, _leg_minutes(t, tables.legs), motives, away


def hhmm(minutes) -> str:
    minutes = int(round(minutes)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def problem_counts(tables) -> pd.DataFrame:
    """``problems`` for every person with trips, as columns of counts, on the chain the model reads."""
    t = tables.trips[~non_trips(tables.trips)].sort_index()
    start = _start_minutes(t) + 1440 * days_past_midnight(t).to_numpy()
    minutes = _leg_minutes(t, tables.legs)
    person = t.index.droplevel("folio_viaje")
    nxt = pd.Series(start, index=t.index).groupby(level=PERSON).shift(-1).to_numpy()
    stay = nxt - (start + minutes)
    motives = t.motivo_viaje.astype(str).to_numpy()
    first = t.groupby(level=PERSON).cumcount().to_numpy() == 0
    last = t.groupby(level=PERSON).cumcount(ascending=False).to_numpy() == 0
    prev_work = pd.Series(motives == "Trabajar", index=t.index).groupby(level=PERSON).shift(1).fillna(False).to_numpy(bool)
    frame = pd.DataFrame({
        "zero": (stay == 0) & ~np.isin(motives, [HOME_MOTIVE, ESCORT]),
        "early": first & ~np.isin(motives, ["Trabajar", HOME_MOTIVE]) & (start % 1440 < _EARLY_START),
        "long_stay": (stay >= LONG_STAY) & ~np.isin(motives, [HOME_MOTIVE, *MANDATORY]),
        "night_shift": last & (motives == HOME_MOTIVE) & prev_work & (start >= 1440),
        "start": start, "arrive": start + minutes,
    }, index=person)
    g = frame.groupby(level=PERSON)
    out = g[["zero", "early", "long_stay"]].sum().astype(int)
    out["long_day"] = (((g.arrive.max() - g.start.min()) > LONG_DAY) & ~g.night_shift.any()).astype(int)
    return out


def readings(stages) -> dict:
    """person -> (keys, current starts, new starts, how, problems now, problems then, minutes moved now and then)."""
    shipped, rules, revised = stages
    r = rules.trips
    rules_start = dict(zip(r.index, r.hora_inicio_h.astype(float) * 60 + r.hora_inicio_m.astype(float)))
    now_counts, floor_counts = problem_counts(revised), problem_counts(rules)
    floor_counts = floor_counts.reindex(now_counts.index).fillna(0).astype(int)
    over = (now_counts > floor_counts).any(axis=1)
    out = {}
    for person in now_counts.index[over.to_numpy()]:
        if person in ROUND_10:
            continue
        keys, now, minutes, motives, away = day(revised, person, None)
        floor = floor_counts.loc[person].to_numpy()
        have = np.maximum(problems(now, minutes, motives) - floor, 0)
        if not have.any():
            continue
        _, recorded, _, _, _ = day(revised, person, rules_start)
        candidates = [("fewest", fewest_minutes(recorded, minutes, away))]
        read = midnight(recorded, minutes, motives)
        if read is not None:
            candidates.append(("midnight", fewest_minutes(read, minutes, away)))

        def moved(starts, how):
            gap = np.abs(starts - recorded)
            if how == "midnight":  # an hour reading, not minutes moved
                gap[-1] = abs(starts[-1] - recorded[-1] - 720)
            return gap

        def score(starts, how):
            return (int(problems(starts, minutes, motives).sum()), float(moved(starts, how).sum()))

        how, new = min(candidates, key=lambda c: score(c[1], c[0]))
        then, before = problems(new, minutes, motives), problems(now, minutes, motives)
        if then.sum() < before.sum() and moved(new, how).max() <= MOST_MOVED and (np.round(new) != np.round(now)).any():
            out[person] = (keys, now, new, how, before, then, float(np.abs(now - recorded).sum()),
                           float(moved(new, how).sum()), recorded)
    return out


LABELS = ["una estancia de cero minutos fuera de casa", "un primer viaje antes de las 05:00",
          "una estancia de 8 horas o más fuera de casa, del trabajo y de la escuela", "un día de más de 20 horas"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    parser.add_argument("--out", default="notebooks/chain_review_12.csv", help="Where to write the sheet")
    args = parser.parse_args()

    stages = load_stages(args.data)
    shipped, rules, revised = stages
    calls = readings(stages)
    persons = pd.MultiIndex.from_tuples(sorted(calls), names=PERSON)
    sheet = review.chain_sheet(review.chain_rows(revised, shipped), revised.hab, persons)
    at = {k: i for i, k in enumerate(zip(sheet.household, sheet.person, sheet.trip))}
    for person, (keys, now, new, how, have, left, moved_now, moved_then, recorded) in calls.items():
        what = "; ".join(LABELS[i] for i in np.flatnonzero(have))
        rest = "; queda " + "; ".join(LABELS[i] for i in np.flatnonzero(left)) if left.any() else ""
        reading = (" El último regreso, registrado a las 12, se lee a medianoche (00 h del día siguiente, el reloj de "
                   "12 horas)." if how == "midnight" else "")
        changed = [(k, n, w) for k, n, w in zip(keys, now, new) if hhmm(n) != hhmm(w)]
        detail = ", ".join(f"viaje {k[2]} {hhmm(n)} → {hhmm(w)}" for k, n, w in changed)
        note = (
            f"Duodécima revisión: las horas de las revisiones anteriores dejaban {what}. El día vuelve a las horas "
            f"de las reglas, movidas lo menos posible ({int(round(moved_then))} min en total, frente a "
            f"{int(round(moved_now))}) para que ningún viaje salga antes de que llegue el anterior."
            f"{reading} Cambia: {detail}{rest}."
        )
        for k, n, w in changed:
            sheet.loc[at[k], review.new_column("start")] = hhmm(w)
            sheet.loc[at[k], review.NOTE] = note
    review.write_sheet(sheet, args.out)
    print(f"wrote {args.out}: {len(persons)} persons, {len(sheet)} rows, "
          f"{int(sheet[review.new_column('start')].ne('').sum())} starts")


if __name__ == "__main__":
    main()
