"""Round 13 over the trip chains: first-pass notes that tell of an origin type that did not change.

The first pass typed the origin type trip by trip ("El tipo de origen pasa de «Su casa» a
«Oficina»; el origen se encadena con el destino del viaje 1 …"). The origin type is the
person's one answer to where the day started, so those per-trip types were moot, and the
answer changed only where a decision on the person gives it. The clause stayed in the notes
of the decisions made on the same trips: 855 of them tell of a change of type that the
tables do not hold (the third review of the chains, 2026-09-26). This round rewrites those
notes and changes no value:

- the change of type goes, unless the person's answer is that type;
- the chaining of the origin ("el origen se encadena con el destino del viaje N (…)") stays
  where the trip's origin is that AGEB, since that part of the reading holds.

A note typed on a row where nothing else changes gives a new reason to every decision
already on that trip (``eodgdl.review.merge_decisions``), and to the person's answer on the
row the sheet shows it on, so a trip whose decisions carry different notes takes them all,
each rewritten, in one. A second run finds nothing.

    uv run python scripts/revisions/notes_13.py --data data --out notebooks/chain_review_13.csv
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from eodgdl import load_stages, review
from eodgdl._resources import chain_decisions
from eodgdl.chains import PERSON

CLAUSE = re.compile(
    r"El tipo de origen pasa de “(?P<old>[^”]*)” a “(?P<new>[^”]*)”"
    r"(?: y la AGEB de origen pasa de (?P<was>[^;]+?) a (?P<now>[^;]+?))?; "
    r"(?:el origen se encadena con el destino del viaje (?P<trip>\d+) \((?P<place>.*?) en "
    r"(?P<where>la zona del hogar|[0-9A-Z]+)\): la encuesta pregunta el tipo de origen una sola vez y lo repite en "
    r"todos los renglones de la persona\.|se ajusta para que el viaje salga de donde terminó el anterior\.)\s*"
)
HOME = "la zona del hogar"


def rewrite(note: str, answer: str, origin: str | None, home: str) -> str:
    """``note`` without the changes of type that did not happen; what it says of the origin's AGEB kept where the
    trip's origin (``origin``; None for the person's answer, which has none) is that AGEB."""
    def holds(ageb: str | None) -> bool:
        return origin is not None and ageb is not None and origin == (home if ageb == HOME else ageb)

    def one(m: re.Match) -> str:
        if m["new"] == answer:
            return m[0]
        parts = []
        if m["was"] and holds(m["now"]):
            parts.append(f"la AGEB de origen pasa de {m['was']} a {m['now']}")
        if m["trip"] and holds(m["where"]):
            parts.append(f"el origen se encadena con el destino del viaje {m['trip']} ({m['place']} en {m['where']})")
        return ("; ".join(parts)[:1].upper() + "; ".join(parts)[1:] + ". ") if parts else ""
    return CLAUSE.sub(one, note).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    parser.add_argument("--out", default="notebooks/chain_review_13.csv", help="Where to write the sheet")
    args = parser.parse_args()

    shipped, rules, revised = load_stages(args.data)
    t = revised.trips
    decisions = chain_decisions()
    answer = t.groupby(level=PERSON).tipo_lugar_origen.first().astype(str)
    home = revised.viv.ageb.astype(str)
    shows = {k[:2]: k for k in review._first_rows(t.index)}  # the row the sheet shows the answer on
    # the notes a sheet row gives: every decision on its trip — a dropped row too, whose note tells of a row the
    # tables no longer hold — and on the answer's row the answer's too
    by_row = {}
    for d in decisions.sort_values(["source", "field"]).itertuples(index=False):
        person = (int(d.household), int(d.person))
        answer_decision = int(d.trip) == review.PERSON_TRIP
        row = shows.get(person) if answer_decision else (*person, int(d.trip))
        if row is None:
            continue
        origin = str(t.at[row, "origen"]) if row in t.index and not answer_decision else None
        note = rewrite(d.note, answer.get(person, ""), origin, home[person[0]])
        by_row.setdefault(row, []).append((d.note, note))
    new_notes = {}
    for row, notes in by_row.items():
        if all(old == new for old, new in notes):
            continue
        rewritten = list(dict.fromkeys(new for _, new in notes if new))  # distinct, in order
        if not rewritten:
            raise ValueError(f"{row}: every note would be empty")
        new_notes[row] = " ".join(rewritten)
    persons = pd.MultiIndex.from_tuples(sorted({k[:2] for k in new_notes}), names=PERSON)
    sheet = review.chain_sheet(review.chain_rows(revised, shipped), revised.hab, persons)
    at = {k: i for i, k in enumerate(zip(sheet.household, sheet.person, sheet.trip))}
    for key, note in new_notes.items():
        sheet.loc[at[key], review.NOTE] = note
    review.write_sheet(sheet, args.out)
    print(f"wrote {args.out}: {len(persons)} persons, {len(sheet)} rows, {len(new_notes)} notes rewritten")


if __name__ == "__main__":
    main()
