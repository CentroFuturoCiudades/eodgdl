"""Round 11 over the trip chains: the hand decisions carried over the chain rules of 2026-09-26.

On 2026-09-26, after the third review of the chains, three rules took over what the hand
passes had done trip by trip (``eodgdl.chains``):

- ``_copied_first_origins``: a first trip that leaves its own destination's AGEB starts at
  home, and the person's answer to where the day started is 'Su casa';
- ``_start_at_arrival``: a start that falls before the previous trip's arrival by the
  tolerance or less moves to that arrival;
- the typo search also reads a day whose first trip, to anything but work, starts before
  05:00 with a stay of eight hours or more after it.

1,957 hand decisions now give what the rules give, and ``eodgdl review freeze`` removes
them. This round keeps what the tables the decisions give were before the rules changed,
except where the third rule reads a day nobody had reviewed:

- **Stale decisions** (45 starts): each was made against the old rules' value, which the
  new rules moved. The decided value stands, made again against the new one.
- **Starts the slide moves where no decision sat** (109): an earlier round had resolved
  the overlap another way, by cutting the previous trip's minutes (round 9), moving its
  hour or dropping a row. It kept the recorded start, and the rule, which reads the rules'
  minutes, would push it to the arrival. The start goes back to what the rules gave before
  the slide (the survey's hour, or its typo reading), wherever that still comes after the
  previous trip's arrival in the tables the decisions give. That start is the shipped hour
  plus the typo reading in ``ajustes``, except for the one imputed start, listed below.
  A start decision the rules now give guards nothing, so the same holds where the first
  pass had slid a start against the rules' hour of the trip before and then moved that
  hour itself (3: 5108/2, 9502/2, 13587/2), leaving a slide the day no longer needs.

A second run finds nothing: every such start then holds a decision.

    uv run python scripts/revisions/chains_11.py --data data --out notebooks/chain_review_11.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from eodgdl import load_stages, review
from eodgdl._resources import chain_decisions
from eodgdl.chains import _START_TIME_EDITS, PERSON, _leg_minutes, days_past_midnight, has_code, non_trips

# the imputed start the slide moved (hora:vecinos): its value before the slide, which the
# shipped hour cannot give; the rules imputed 18:07 and slid it to the old arrival, 18:10
IMPUTED_BEFORE_SLIDE = {(13587, 2, 3): 18 * 60 + 7}


def hhmm(minutes) -> str:
    minutes = int(minutes) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def before_slide(rules: pd.DataFrame, shipped: pd.DataFrame, key) -> int:
    """A slid start's value before the slide: the shipped hour with the typo reading ``ajustes`` names."""
    if key in IMPUTED_BEFORE_SLIDE:
        return IMPUTED_BEFORE_SLIDE[key]
    h, m = shipped.at[key, "hora_inicio_h"], shipped.at[key, "hora_inicio_m"]
    if pd.isna(h) or pd.isna(m):
        raise ValueError(f"{key}: an imputed start the slide moved; add it to IMPUTED_BEFORE_SLIDE")
    delta = {f"hora:{name}": d * 60 for name, _, _, _, d in _START_TIME_EDITS}
    return int(h * 60 + m + sum(delta.get(c, 0) for c in rules.at[key, "ajustes"].split(";")))


def put_backs(stages, decisions, stale) -> dict:
    """key -> (start before the slide, the slid start, the previous arrival): the starts to put back.

    ``stale`` are the stale decisions, which ``stages.revised`` sets aside: the walk reads their value, which
    this round makes again."""
    shipped, rules, revised = stages
    # a start decision the rules now give (holds) is removed by the freeze, so it guards nothing
    live = review.decision_outcomes(rules, shipped, decisions).outcome.isin(["apply", "stale"])
    starts = decisions[live.to_numpy() & (decisions.field == "start").to_numpy()]
    decided = set(zip(*(starts[c] for c in review.KEYS)))
    again = {(int(d.household), int(d.person), int(d.trip)): int(d.after[:2]) * 60 + int(d.after[3:])
             for d in stale.itertuples(index=False)}
    slid = rules.trips.index[has_code(rules.trips.ajustes, "hora:llegada").to_numpy()]
    candidates = {k for k in slid if k not in decided}
    t = revised.trips[~non_trips(revised.trips)].sort_index()
    days = days_past_midnight(t)
    start = pd.Series(t.hora_inicio_h.astype(float) * 60 + t.hora_inicio_m.astype(float) + 1440 * days, index=t.index)
    minutes = pd.Series(_leg_minutes(t, revised.legs), index=t.index)
    for key, value in again.items():  # the stale decisions, made again
        if key in start.index:
            start[key] = value + 1440 * np.round((start[key] - value) / 1440)
    out = {}
    persons = sorted({k[:2] for k in candidates if k in t.index})
    for person in persons:
        keys = list(t.loc[person].index)
        for i, trip in enumerate(keys):
            key = (*person, trip)
            if key not in candidates or i == 0:
                continue
            prev = (*person, keys[i - 1])
            arrival = start[prev] + minutes[prev]
            value = before_slide(rules.trips, shipped.trips, key) + 1440 * days[key]
            assert abs(value - start[key]) < 720, key  # the same calendar day
            if value != start[key] and value >= arrival:
                out[key] = (value, start[key], arrival)
                start[key] = value  # the next trip reads this start
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    parser.add_argument("--out", default="notebooks/chain_review_11.csv", help="Where to write the sheet")
    args = parser.parse_args()

    stages = load_stages(args.data, skip_stale=True)
    shipped, rules, revised = stages
    decisions = chain_decisions()
    stale = review.stale_decisions(rules, shipped, decisions)
    back = put_backs(stages, decisions, stale)
    persons = pd.MultiIndex.from_tuples(
        sorted({k[:2] for k in back} | set(zip(stale.household, stale.person))), names=PERSON
    )
    sheet = review.chain_sheet(review.chain_rows(revised, shipped), revised.hab, persons)
    at = {k: i for i, k in enumerate(zip(sheet.household, sheet.person, sheet.trip))}
    sources = decisions.groupby(["household", "person"]).source.agg(lambda s: ", ".join(sorted(set(s))))
    rules_start = lambda key: hhmm(rules.trips.at[key, "hora_inicio_h"] * 60 + rules.trips.at[key, "hora_inicio_m"])

    for d in stale.itertuples(index=False):
        key = (int(d.household), int(d.person), int(d.trip))
        assert d.field == "start", d
        i = at[key]
        sheet.loc[i, review.new_column("start")] = d.after
        sheet.loc[i, review.NOTE] = (
            f"{d.note} [Undécima revisión: la decisión de {d.source} se mantiene. Se tomó contra «{d.before}», "
            f"que las reglas del 26 de septiembre de 2026 dan ahora como «{rules_start(key)}».]"
        )
    for key, (value, slid, arrival) in back.items():
        i = at[key]
        sheet.loc[i, review.new_column("start")] = hhmm(value)
        sheet.loc[i, review.NOTE] = (
            f"Undécima revisión: la salida vuelve a las {hhmm(value)}. La regla del 26 de septiembre de 2026 recorre a "
            f"la llegada del viaje anterior la salida que la adelanta por 15 minutos o menos, y con los minutos de las "
            f"reglas la llevaba a las {hhmm(slid)}; pero las decisiones ya tomadas sobre este día ({sources[key[:2]]}) "
            f"resolvieron ese traslape de otra forma y el viaje anterior llega a las {hhmm(arrival)}, así que la salida "
            "registrada cabe y se conserva."
        )
    review.write_sheet(sheet, args.out)
    print(f"wrote {args.out}: {len(persons)} persons, {len(sheet)} rows; {len(stale)} stale decisions kept, "
          f"{len(back)} starts put back")


if __name__ == "__main__":
    main()
