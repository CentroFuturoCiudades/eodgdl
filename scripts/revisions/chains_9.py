"""Round 9 over the trip chains: the review sheet for leg minutes, overlaps and daycare escorts.

Written 2026-09-26 after a review of the chain fixes. Each criterion below is read off the
tables ``load_eod`` returns before this round, so the script is the round's evidence and a
second run finds nothing left to do. It needs the census geography (``uv sync --extra
reweight``) for the straight-line distance between two AGEBs. It touches only persons the
rules' output flags — a ``problemas`` code on any of their trips, the persons the first pass
reviewed — since a stay of no minutes is no code (user, 2026-09-26: the survey a code does
not flag keeps its values).

- **Leg minutes that explain a stay of no minutes.** A first-pass rule moved 868 starts to
  the minute the previous trip arrives, so the stay there lasts no minutes; the survey's own
  times hold more such stays. Where the previous trip is motorised and reports at least
  twice the travel minutes the survey typically reports for its mode over its straight-line
  distance (the median over the survey's trips of that mode in that distance band, at
  least 30 of them, AGEB centroid to centroid), the reported minutes are the error, not the
  clock: the trip takes the typical minutes, spread over its legs, and so do the person's
  other trips over the same route in the same mode that are as slow. On those persons the
  clock then goes back where the minutes had pushed it: a start the typo search or a hand
  decision moved by three hours or less returns to the survey's own value when that now fits
  with a stay of a minute at least on each side, or else, where a hand decision had moved it
  to the old arrival, to the rules' value (ten- and twelve-hour readings are readings of the
  hour and stay). Distances of half a kilometre or less, walks and
  bicycle rides are left alone: a centroid says little about them. Taken across the survey,
  trips that overlap the next start are about as fast for their distance as any others
  (their median speed is about their mode's median), so this reads a minority of the stays, the
  implausibly slow trips, and the start moves stand elsewhere. On flagged persons most such
  stays are the first pass's moves; a stay of no minutes in the survey's own times, on a
  person nothing flags, is left as recorded.
- **Overlaps left.** TASHA cannot schedule a trip that starts before the previous one
  arrives, so the overlaps within the old 15-minute tolerance (``hora_traslapada``) are
  now breaking. The ones the leg minutes do not explain are closed as the first pass closed
  the rest: the start moves to the minute the previous trip arrives.
- **Daycare escorts.** The first pass recoded the 'Guardería' trips of persons aged 12 or
  more as 'Llevar o recoger a alguien' — the survey uses the motive both for whoever attends
  and for whoever takes or fetches a child — and left 46. They take the same motive, so every
  daycare trip left is a child's.

    uv run python scripts/revisions/chains_9.py --data data --out notebooks/chain_review_9.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from eodgdl import load_stages, review
from eodgdl.chains import _START_TIME_TOLERANCE, PERSON, _leg_minutes, _start_minutes, days_past_midnight, non_trips
from eodgdl.tasha.build import DAYCARE, ESCORT_FROM_AGE

ESCORT = "Llevar o recoger a alguien"
NOT_BY_DISTANCE = {"A PIE", "Bicicleta", "Mi Bici"}  # a centroid says little about a walk or a ride
BANDS = [0.5, 1, 2, 3, 5, 8, 12, 20, 1000]  # km, straight line between AGEB centroids
MIN_CELL = 30  # trips of a mode in a band for its median to stand as the typical minutes
SLOW = 2.0  # reported minutes at least this many times the typical ones
TOLERANCE = _START_TIME_TOLERANCE  # how far short of the arrival an earlier pass may have left a moved start
WINDOW = 180  # minutes: a start moved this far or less may go back to the survey's; 10 and 12 hours are hour readings


def centroids() -> pd.DataFrame:
    """x, y in metres (UTM 13N) of every AGEB of Jalisco, urban and rural, by CVEGEO."""
    import geopandas as gpd
    from mxcensus.data import POOCH

    points = pd.concat(
        gpd.read_parquet(POOCH.fetch(f"mg_{layer}_14.parquet")).to_crs(32613).set_index("CVEGEO").geometry.representative_point()
        for layer in ("a", "ar")
    )
    points = points[~points.index.duplicated()]
    return pd.DataFrame({"x": points.x, "y": points.y})


def distances(trips: pd.DataFrame, xy: pd.DataFrame) -> np.ndarray:
    """Straight-line km between each trip's origin and destination AGEB; NaN where either has no centroid."""
    o = xy.reindex(trips.origen.astype(str)).to_numpy()
    d = xy.reindex(trips.destino.astype(str)).to_numpy()
    return np.hypot(o[:, 0] - d[:, 0], o[:, 1] - d[:, 1]) / 1000


def typical_minutes(shipped, xy) -> pd.DataFrame:
    """The survey's median travel minutes by main mode and distance band, and how many trips each rests on."""
    t = shipped.trips
    frame = pd.DataFrame({"mode": t.modo_principal.astype(str).to_numpy(), "km": distances(t, xy),
                          "minutes": _leg_minutes(t, shipped.legs)})
    frame = frame[frame.km > BANDS[0]]
    frame["band"] = pd.cut(frame.km, BANDS)
    return frame.groupby(["mode", "band"], observed=True).minutes.agg(typical="median", trips="size")


def hhmm(minutes) -> str:
    minutes = int(round(minutes)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def stays(revised, rules, shipped, xy, typical) -> pd.DataFrame:
    """Every trip of the chain the model reads with the stay that follows it and what the distance says of it."""
    t = revised.trips[~non_trips(revised.trips)].sort_index()
    start = _start_minutes(t) + 1440 * days_past_midnight(t).to_numpy()
    minutes = _leg_minutes(t, revised.legs)
    f = pd.DataFrame({"mode": t.modo_principal.astype(str).to_numpy(), "minutes": minutes, "start": start,
                      "km": distances(t, xy), "o": t.origen.astype(str).to_numpy(), "d": t.destino.astype(str).to_numpy()},
                     index=t.index)
    after = pd.Series(list(t.index), index=t.index).groupby(level=PERSON).shift(-1)
    f["next_key"] = after
    f["next_start"] = f.start.groupby(level=PERSON).shift(-1)
    f["stay"] = f.next_start - (f.start + f.minutes)
    f["band"] = pd.cut(f.km, BANDS)
    f = f.join(typical, on=["mode", "band"])
    # the rules' and the survey's start of each trip, on the same day count as the revised one
    near = lambda value: value + 1440 * np.floor((f.start - value + 720) / 1440)
    f["rules"] = near(pd.Series(_start_minutes(rules.trips), index=rules.trips.index).reindex(t.index))
    f["shipped"] = near(pd.Series(_start_minutes(shipped.trips), index=shipped.trips.index).reindex(t.index))
    f["next_rules"] = f.rules.groupby(level=PERSON).shift(-1)
    coded = (rules.trips.problemas != "").groupby(level=PERSON).any()
    f["flagged"] = coded.reindex(t.index.droplevel("folio_viaje")).fillna(False).to_numpy(bool)
    return f


def leg_minutes(f: pd.DataFrame) -> pd.DataFrame:
    """The slow trips behind a stay of no minutes or an overlap, and the same person's other equally slow trips
    over the same route (the same two AGEBs either way, the same mode), with the typical minutes they take instead;
    ``trigger`` tells the first from the second."""
    slow = (
        f.flagged
        & ~f["mode"].isin(NOT_BY_DISTANCE)
        & (f.km > BANDS[0])
        & (f.trips >= MIN_CELL)
        & (f.minutes >= SLOW * f.typical)
    )
    trigger = slow & (f.stay <= 0)
    route = pd.Series(
        [(k[0], k[1], *sorted((o, d)), m) for k, o, d, m in zip(f.index, f.o, f.d, f["mode"])], index=f.index
    )
    fix = f[slow & (trigger | route.isin(set(route[trigger])))].copy()
    fix["trigger"] = trigger[fix.index]
    fix["new_minutes"] = np.floor(fix.typical + 0.5).astype(int)
    return fix


def new_starts(f: pd.DataFrame, fix: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Every start once the minutes are corrected, walking each chain in order.

    On a person whose minutes this round corrects, a start the rules or a hand decision moved by up to
    ``WINDOW`` minutes goes back to the survey's own value when that now falls after the previous trip's
    arrival and lets it arrive before the next trip's start, with a stay of a minute at least on both sides:
    the minutes were the error, and the clock can stand. Failing that, a
    start a hand decision had moved to the old arrival, or to within the old tolerance of it, goes back to
    the rules' value when that fits. Any start still before the previous trip's arrival moves to it. Moves of
    ten or twelve hours are readings of the hour, not of the minutes, and stay. Returns the starts and, for
    each start put back, where it came from (``shipped`` or ``rules``).
    """
    minutes = f.minutes.copy()
    minutes[fix.index] = fix.new_minutes
    start = f.start.copy()
    back = {}
    fixed = {k[:2] for k in fix.index}
    touched = {k[:2] for k in f.index[((f.stay < 0) & f.flagged).to_numpy()]} | fixed
    chains = pd.Series(list(f.index), index=f.index).groupby(level=PERSON).agg(list)
    for person in sorted(touched):
        keys = chains[person]
        for i, k in enumerate(keys):
            arrive = start[keys[i - 1]] + minutes[keys[i - 1]] if i else -np.inf
            leave = start[keys[i + 1]] if i + 1 < len(keys) else np.inf
            options = []
            if person in fixed and start[k] != f.shipped[k] and abs(start[k] - f.shipped[k]) <= WINDOW:
                options.append(("shipped", f.shipped[k]))
            if i and keys[i - 1] in fix.index:
                p = f.loc[keys[i - 1]]
                if -TOLERANCE <= p.stay <= 0 and f.rules[k] < start[k]:
                    options.append(("rules", f.rules[k]))
            for name, value in options:
                if value > arrive and value + minutes[k] < leave:  # a stay of a minute at least, either side
                    start[k] = value
                    back[k] = name
                    break
            if start[k] < arrive:
                start[k] = arrive
    return start, back


def daycare_escorts(revised) -> pd.DataFrame:
    """'Guardería' trips of persons aged ESCORT_FROM_AGE or more, with their age."""
    t = revised.trips
    age = revised.hab.edad.reindex(t.index.droplevel("folio_viaje")).to_numpy()
    pick = (t.motivo_viaje == DAYCARE).to_numpy() & (age >= ESCORT_FROM_AGE)
    return pd.DataFrame({"age": age[pick]}, index=t.index[pick])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    parser.add_argument("--out", default="notebooks/chain_review_9.csv", help="Where to write the sheet")
    args = parser.parse_args()

    stages = load_stages(args.data, skip_stale=True)
    xy = centroids()
    typical = typical_minutes(stages.shipped, xy)
    f = stays(stages.revised, stages.rules, stages.shipped, xy, typical)
    fix = leg_minutes(f)
    start, back = new_starts(f, fix)
    moved = f.index[(start != f.start).to_numpy()]
    closed = [k for k in moved if k not in back]
    daycare = daycare_escorts(stages.revised)

    keys = list(fix.index) + list(moved) + list(daycare.index)
    persons = pd.MultiIndex.from_tuples(sorted({k[:2] for k in keys}), names=PERSON)
    rows = review.chain_rows(stages.revised, stages.shipped)
    sheet = review.chain_sheet(rows, stages.revised.hab, persons)
    at = {k: i for i, k in enumerate(zip(sheet.household, sheet.person, sheet.trip))}
    previous = pd.Series(list(f.index), index=f.index).groupby(level=PERSON).shift(1)

    def put(key, column, value, note):
        i = at[key]
        sheet.loc[i, review.new_column(column)] = value
        sheet.loc[i, review.NOTE] = f"{sheet.at[i, review.NOTE]} {note}".strip()

    for key, r in fix.iterrows():
        arrive = start[key] + r.new_minutes
        why = (f"Novena revisión: {int(r.minutes)} min en {r['mode']} para {r.km:.1f} km en línea recta es más del "
               f"doble de lo que la encuesta suele reportar para ese modo a esa distancia ({r.new_minutes} min, la "
               f"mediana de {int(r.trips)} viajes): los minutos son el error, no la hora.")
        if not isinstance(r.next_key, tuple):
            then = " Es el último viaje del día."
        else:
            after = start[r.next_key]
            then = (f" Con {r.new_minutes} min llega a las {hhmm(arrive)} y el viaje siguiente sale a las {hhmm(after)}"
                    + (f", {int(after - arrive)} min después." if after > arrive else ", en cuanto llega."))
        if not r.trigger:
            then += " Es el mismo recorrido, en el mismo modo, que otro viaje del día cuyos minutos chocaban con la salida siguiente."
        put(key, "leg min", str(r.new_minutes), why + then)
    for key, source in back.items():
        p = previous[key]
        arrive = start[p] + (fix.new_minutes[p] if p in fix.index else f.minutes[p])
        if source == "shipped":
            who = ("la regla de horas" if f.rules[key] == f.start[key] and f.rules[key] != f.shipped[key]
                   else "una revisión anterior")
            put(key, "start", hhmm(start[key]), (
                f"Novena revisión: la salida vuelve a las {hhmm(start[key])}, la hora que reporta la encuesta: con los "
                f"minutos de traslado que corrige esta revisión cabe después de la llegada del viaje anterior "
                f"({hhmm(arrive)}); {who} la había movido a las {hhmm(f.start[key])} para dejar sitio a los minutos de más."
            ))
        else:
            put(key, "start", hhmm(start[key]), (
                f"Novena revisión: la salida vuelve a las {hhmm(start[key])}, la hora que dan las reglas; la primera "
                f"revisión la había recorrido a las {hhmm(f.start[key])}, la llegada del viaje anterior con sus "
                f"{int(f.minutes[p])} min, que esta revisión corrige a {int(fix.new_minutes[p])}."
            ))
    for key in closed:
        p = previous[key]
        minutes = int(fix.new_minutes[p]) if p in fix.index else int(f.minutes[p])
        put(key, "start", hhmm(start[key]), (
            f"Novena revisión: sale a las {hhmm(start[key])}, la hora a la que llega el viaje anterior "
            f"({hhmm(start[p])} + {minutes} min), en vez de las {hhmm(f.start[key])}: TASHA no admite un viaje "
            "que sale antes de que llegue el anterior"
            + (", y sus minutos ya son los típicos para la distancia." if p in fix.index else
               f", y {minutes} min en {f['mode'][p]} no son excesivos para su distancia, así que se mueve la "
               "salida y no los minutos.")
        ))
    for key, r in daycare.iterrows():
        put(key, "motive", ESCORT, (
            f"Novena revisión: el motivo «{DAYCARE}» se registra como acompañamiento, como la primera revisión lo "
            f"hizo con los demás viajes de personas de {ESCORT_FROM_AGE} años o más: la encuesta lo usa tanto para "
            f"quien asiste como para quien lleva o recoge, y la persona tiene {int(r.age)} años."
        ))
    review.write_sheet(sheet, args.out)
    print(f"wrote {args.out}: {len(persons)} persons, {len(sheet)} rows; {len(fix)} leg minutes "
          f"({len(back)} with the next start put back), {len(closed)} overlaps closed, {len(daycare)} daycare motives")


if __name__ == "__main__":
    main()
