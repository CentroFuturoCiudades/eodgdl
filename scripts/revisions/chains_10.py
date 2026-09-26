"""Round 10 over the trip chains: evening returns for the split shifts the first pass read as night shifts.

The calls are made chain by chain (Claude, 2026-09-26, at the user's request to adjust better the start
times of the days the first pass read as a day shift followed by a night shift); this script writes them
into the round's sheet and reads each one's evidence off the survey, so the notes carry figures anyone
can recompute. A call is written only while the person's last return still sits in the small hours, so
a second run finds nothing left to do. The readings:

- **The evening return of the template.** 61 diaries share "work, home, work, home" with the first
  departure recorded at night, after the next return. The first pass read that hour with 12 hours too
  many as the morning departure and, in 39 of them, the last return in the evening; in 12 it took the
  last return to the small hours of the next day, a night shift after the day's. In the ones here the
  first row's own hour fits after the second trip to work arrives — "left at 9, back at 9" — and the
  workplace's other workers come home between 19:00 and 23:00, hardly ever between 01:00 and 06:00.
  The last return is that hour. Three six-trip diaries of the same shape after a morning walk read the
  same way.
- **8763/1** also leaves at 08:00 (its recorded 20:00 less 12 hours, where the first pass had read a
  digit slip, 10:00) and goes back to work at its recorded 17:02.
- **13583/1** reads a 1 keyed as a 2 in 22:00 and 23:30: a split shift, 08:30-12:00 and 13:30-15:00,
  ending at its recorded 15:00.

The first pass's other day-plus-night readings stand, a double shift being no error in itself: no
same-day reading fits the recorded hours of 16714/1, 11165/1 and 13088/3, 7776/3 goes to a second job
by bicycle, and 8201/1's trips to an access point take what its zone reports.

    uv run python scripts/revisions/chains_10.py --data data --out notebooks/chain_review_10.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from eodgdl import load_stages, review
from eodgdl.chains import PERSON, non_trips

# person: (the trip whose recorded hour is the evening, the last return)
TEMPLATE = {
    (2987, 3): (1, 4), (7993, 2): (1, 4), (8763, 1): (1, 4), (13218, 2): (1, 4), (13478, 2): (1, 4),
    (17056, 1): (1, 4), (11846, 1): (3, 6), (11942, 2): (3, 6), (12047, 2): (3, 6),
}
FAMILY, EVENING_READINGS = 61, 39  # the template's diaries, and the first pass's evening readings among them


def hhmm(h, m) -> str:
    return f"{int(h):02d}:{int(m):02d}"


def returns_from(trips: pd.DataFrame, ageb: str, person) -> tuple[int, int, int]:
    """Other persons' returns home from ``ageb`` starting 19-23 h, 1-6 h and 12-15 h, on the rules' output."""
    r = trips[(trips.motivo_viaje == "Regresar a Casa") & (trips.origen.astype(str) == ageb)
              & (trips.index.droplevel("folio_viaje") != person)]
    h = r.hora_inicio_h.astype(int)
    return int(h.between(19, 23).sum()), int(h.between(1, 6).sum()), int(h.between(12, 15).sum())


def calls(shipped, rules, revised) -> dict:
    """key -> [(field, value, note)]: the round's readings, for the persons still read with a night return."""
    t = rules.trips[~non_trips(rules.trips)]
    now = lambda key: revised.trips.loc[key]
    overnight = lambda key: int(now(key).hora_inicio_h) < 7
    out = {}
    for person, (first, last) in TEMPLATE.items():
        if not overnight(person + (last,)):
            continue
        s1, s4 = shipped.trips.loc[person + (first,)], shipped.trips.loc[person + (last,)]
        evening = hhmm(s1.hora_inicio_h, s1.hora_inicio_m)
        morning = hhmm(int(s1.hora_inicio_h) - 12, s1.hora_inicio_m)
        ageb = str(s1.destino)
        late, night, _ = returns_from(t, ageb, person)
        was = now(person + (last,))
        out[person + (last,)] = [("start", evening, (
            f"Décima revisión: el regreso se lee a las {evening}, la hora de la noche que registra el viaje {first}. El "
            "diario sigue la plantilla «trabajo, casa, trabajo, casa» cuya primera salida trae una hora de la noche "
            f"anterior a la del regreso siguiente ({FAMILY} días así en la encuesta): la primera revisión leyó esa hora con "
            f"12 horas de más, como la salida de la mañana ({morning}), y en {EVENING_READINGS} de esos días leyó el último "
            f"regreso por la tarde o la noche; aquí lo había llevado a las {hhmm(was.hora_inicio_h, was.hora_inicio_m)} "
            "del día siguiente, un turno nocturno tras la jornada, porque la hora registrada para él "
            f"({hhmm(s4.hora_inicio_h, s4.hora_inicio_m)}) queda antes de la segunda salida al trabajo. «Salió a las "
            f"{int(morning[:2])} y regresó a las {int(evening[:2]) - 12}»: la hora del viaje {first} sirve a los dos "
            f"extremos del día, y de ese lugar de trabajo ({ageb}) regresan {late} viajes de otras personas entre las "
            f"19 y las 23 h y {night} entre la 1 y las 6 h."
        ))]
    person = (8763, 1)
    if person + (4,) in out:
        ageb = str(shipped.trips.loc[person + (1,)].destino)
        w = t[(t.motivo_viaje == "Trabajar") & (t.destino.astype(str) == ageb)
              & (t.index.droplevel("folio_viaje") != person)].hora_inicio_h.astype(int)
        out[person + (1,)] = [("start", "08:00", (
            "Décima revisión: la salida se lee a las 08:00, la hora registrada (20:00) con 12 horas de más, como en los "
            "demás días de esta plantilla; la primera revisión la había leído a las 10:00 por un error de dígito. Al "
            f"mismo lugar de trabajo salen {int(w.between(6, 8).sum())} viajes entre las 6 y las 8 h y "
            f"{int((w == 10).sum())} a las 10."
        ))]
        out[person + (3,)] = [("start", "17:02", (
            "Décima revisión: la segunda salida al trabajo vuelve a las 17:02 que registra la encuesta; la primera "
            "revisión la había movido dos horas, a las 19:02, para abrir un turno nocturno que el regreso de la noche "
            "hace innecesario."
        ))]
    person = (13583, 1)
    if overnight(person + (4,)):
        _, night, midday = returns_from(t, str(shipped.trips.loc[person + (1,)].destino), person)
        note = (
            "Décima revisión: 22:00 y 23:30 se leen 12:00 y 13:30, un 1 tecleado como 2 en la hora, y el regreso vuelve "
            "a las 15:00 que registra la encuesta: una jornada partida, de 08:30 a 12:00 y de 13:30 a 15:00, en vez de 13 "
            "horas de trabajo seguidas de otras 5 de madrugada, la lectura de la primera revisión, que llevó el regreso a "
            f"las 05:00 del día siguiente. De ese lugar de trabajo regresan {midday} viajes de otras personas entre las 12 "
            f"y las 15 h y {night} entre la 1 y las 6 h."
        )
        for trip, value in ((2, "12:00"), (3, "13:30"), (4, "15:00")):
            out[person + (trip,)] = [("start", value, note)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    parser.add_argument("--out", default="notebooks/chain_review_10.csv", help="Where to write the sheet")
    args = parser.parse_args()

    stages = load_stages(args.data)
    edits = calls(*stages)
    persons = pd.MultiIndex.from_tuples(sorted({key[:2] for key in edits}), names=PERSON)
    sheet = review.chain_sheet(review.chain_rows(stages.revised, stages.shipped), stages.revised.hab, persons)
    at = {key: i for i, key in enumerate(zip(sheet.household, sheet.person, sheet.trip))}
    for key, items in edits.items():
        for field, value, note in items:
            sheet.loc[at[key], review.new_column(field)] = value
            sheet.loc[at[key], review.NOTE] = note
    review.write_sheet(sheet, args.out)
    print(f"wrote {args.out}: {len(persons)} persons, {len(sheet)} rows, {sum(len(v) for v in edits.values())} edits")


if __name__ == "__main__":
    main()
