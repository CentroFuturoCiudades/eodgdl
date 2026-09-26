"""Pass 7 over the trip chains: the review sheet that undoes three kinds of first-pass edit.

Found in a review of the chain fixes (2026-09-25); each criterion below is
read off the tables ``load_eod`` returns before this pass, so the script is
the pass's evidence and a second run finds nothing left to do.

- **Home-to-home tours put back.** The first pass dropped 15 activity trips
  that go from the home AGEB to the home AGEB, and the 8 returns closing them
  (13 persons; 7 go to work by light rail, bus or car, 20-75 minutes each
  way), as "no corresponde a un viaje real". The sixth pass keeps 375 trips of
  that shape: the place of the activity was never recorded, so the activity
  stays in the home zone for the model. The rows come back as the chain rules
  left them.
- **Children's daycare trips.** The first pass recoded 23 'Guardería' trips of
  students aged 6-11 as 'Llevar o recoger a alguien', the reading it gave the
  adults. A child in daycare attends it; the model reads 'Guardería' under 12
  as school (``tasha.build.ESCORT_FROM_AGE``). The motive goes back.
- **Modes off their legs.** The first pass changed the main mode of two
  one-leg trips with no evidence, leaving it apart from the leg the
  reweighting reads. The mode goes back to the survey's, and the leg with it.

It was frozen as pass 7 (``notebooks/revisions/chains_7.csv.gz``, archived). Every one
of its edits undoes a pass-1 decision, so merged into the hand decisions
(``scripts/revisions/squash_passes.py``) it removes 48 and adds none; run on today's
tables the script finds nothing left to do.

    uv run python scripts/revisions/chains_7.py --data data --out notebooks/chain_review_7.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from eodgdl import load_stages, review
from eodgdl.chains import HOME_MOTIVE, PERSON, has_code
from eodgdl.tasha.build import DAYCARE, ESCORT_FROM_AGE

ESCORT = "Llevar o recoger a alguien"


def dropped_home_tours(shipped, rules, revised) -> pd.Series:
    """The rows the hand passes dropped that are home-to-home activity trips or the returns closing them: key -> note."""
    t = rules.trips
    home = rules.viv.ageb.astype(str).reindex(t.index.get_level_values("folio_vivienda")).to_numpy()
    at_home = (t.origen.astype(str).to_numpy() == home) & (t.destino.astype(str).to_numpy() == home)
    activity = (t.motivo_viaje != HOME_MOTIVE).to_numpy()
    non_trip = has_code(t.problemas, "regreso_en_casa").to_numpy()
    after_tour = pd.Series(at_home & activity, index=t.index).groupby(level=PERSON).shift(1).fillna(False).to_numpy(bool)
    gone = ~t.index.isin(revised.trips.index)
    minutes = rules.legs.traslado_min.groupby(level=[0, 1, 2]).sum().reindex(t.index)
    notes = {}
    for key in t.index[gone & at_home & ~non_trip & (activity | after_tour)]:
        r = t.loc[key]
        how = f"{r.modo_principal}, {int(minutes[key])} min"
        if r.motivo_viaje != HOME_MOTIVE:
            notes[key] = (
                f"Séptima revisión: se restituye. La primera revisión descartó esta salida de casa a casa "
                f"(«{r.motivo_viaje}», {how}) como «no corresponde a un viaje real», pero la sexta conserva "
                "375 viajes de actividad de la misma forma: el lugar de la actividad no se registró y la "
                "actividad queda en la zona del hogar para el modelo."
            )
        else:
            notes[key] = (
                f"Séptima revisión: se restituye el regreso ({how}) que cierra la salida de casa a casa del "
                f"viaje anterior, restituida en esta misma revisión."
            )
    return pd.Series(notes, dtype=object)


def children_s_daycare(shipped, rules, revised) -> pd.Series:
    """Daycare trips of children under ESCORT_FROM_AGE that a pass recoded as escorting: key -> note."""
    t = revised.trips
    was = rules.trips.motivo_viaje.reindex(t.index).astype(object)
    age = revised.hab.edad.reindex(t.index.droplevel("folio_viaje")).to_numpy()
    pick = (was == DAYCARE).to_numpy() & (t.motivo_viaje == ESCORT).to_numpy() & (age < ESCORT_FROM_AGE)
    occupation = revised.hab.ocupacion.reindex(t.index.droplevel("folio_viaje")).astype(object).to_numpy()
    notes = {
        key: (
            f"Séptima revisión: el motivo vuelve a «{DAYCARE}». La persona tiene {int(a)} años"
            f"{' y es estudiante' if o == 'Estudiante' else ''}: asiste a la guardería, no acompaña a nadie. "
            "La primera revisión le dio la lectura de las personas adultas; el modelo lee «Guardería» antes "
            "de los 12 años como escuela."
        )
        for key, a, o in zip(t.index[pick], age[pick], occupation[pick])
    }
    return pd.Series(notes, dtype=object)


def modes_off_their_legs(shipped, rules, revised) -> pd.DataFrame:
    """Trips whose main mode a pass moved off every mode the survey recorded for their legs: key -> (mode, note)."""
    t = revised.trips
    legs = shipped.legs.traslado_medio.astype(str).groupby(level=[0, 1, 2]).agg(list)
    minutes = shipped.legs.traslado_min.groupby(level=[0, 1, 2]).sum()
    was = shipped.trips.modo_principal.reindex(t.index).astype(str)
    now = t.modo_principal.astype(str)
    out = {}
    for key in t.index[(now != was).to_numpy()]:
        if now[key] in legs[key]:
            continue
        out[key] = (
            was[key],
            f"Séptima revisión: el modo vuelve a «{was[key]}», el que la encuesta registra en su "
            f"{'único tramo' if len(legs[key]) == 1 else 'tramo'} ({int(minutes[key])} min). La primera "
            f"revisión lo cambió a «{now[key]}» sin más evidencia y dejó el modo principal fuera de los "
            "tramos, que son lo que lee la reponderación.",
        )
    return pd.DataFrame.from_dict(out, orient="index", columns=["mode", "note"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    parser.add_argument("--out", default="notebooks/chain_review_7.csv", help="Where to write the sheet")
    args = parser.parse_args()

    stages = load_stages(args.data)
    tours = dropped_home_tours(*stages)
    daycare = children_s_daycare(*stages)
    modes = modes_off_their_legs(*stages)
    keys = list(tours.index) + list(daycare.index) + list(modes.index)
    persons = pd.MultiIndex.from_tuples(sorted({k[:2] for k in keys}), names=PERSON)
    rows = review.chain_rows(stages.revised, stages.shipped)
    sheet = review.chain_sheet(rows, stages.revised.hab, persons)
    at = {k: i for i, k in enumerate(zip(sheet.household, sheet.person, sheet.trip))}
    for key, note in tours.items():
        sheet.loc[at[key], [review.new_column(review.STATUS), review.NOTE]] = [review.RESTORED, note]
    for key, note in daycare.items():
        sheet.loc[at[key], [review.new_column("motive"), review.NOTE]] = [DAYCARE, note]
    for key, (mode, note) in modes.iterrows():
        sheet.loc[at[key], [review.new_column("mode"), review.NOTE]] = [mode, note]
    review.write_sheet(sheet, args.out)
    print(
        f"wrote {args.out}: {len(persons)} persons, {len(sheet)} rows; {len(tours)} rows restored, "
        f"{len(daycare)} daycare motives, {len(modes)} modes"
    )


if __name__ == "__main__":
    main()
