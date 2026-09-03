"""Load and clean the IMEPLAN Guadalajara EOD 2023 origin-destination survey."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from eodgdl._resources import imeplan_rename_map
from eodgdl.data._catalog import HABITANTES_CSV, VIAJES_CSV, VIVIENDAS_CSV
from eodgdl.schemas import hab_schema, trips_schema, viv_schema

log = logging.getLogger(__name__)

PERSON = ["folio_vivienda", "folio_habitante"]

# Manual fixes for known data-entry errors in the "viajes" (trips) table, keyed by
# (folio_vivienda, folio_habitante, folio_viaje) → (column, corrected value). Source:
# manual review against the survey instrument; see README "Data provenance".
_VIAJES_MEDIO_FIXES = {
    (9560, 4, 1): ("traslado1_medio", "A PIE"),
    (9560, 4, 2): ("traslado4_medio", "CAMIÓN O AUTOBÚS"),
    (9530, 3, 3): ("traslado5_medio", "CAMIÓN O AUTOBÚS"),
}

# A 'Regresar a Casa' whose destination zone is not the household's did not go
# home, so its motive is wrong; the reported destination type is the best label
# there is for what the trip did. This table turns that type into the motive the
# survey would have recorded. Decided 2026-09-01 from the chain diagnosis behind
# clean_trip_chains(): 108 returns contradict their destination type, 24 of them
# land in the household's zone (motive corroborated, type stale) and keep their
# motive; the other 84 are recoded (22 Estudiar, 13 Trabajar, 22 Compras, 27 other).
_MOTIVO_POR_TIPO_LUGAR_DESTINO = {
    "Escuela": "Estudiar",
    "Oficina": "Trabajar",
    "Fábrica o taller": "Trabajar",
    "Comercio, mercado, tienda o centro comercial": "Compras (bienes, productos y servicios)",
    "Deportivo, gimnasio": "Deportes o recreación",
    "Centro cultural o área recreativa": "Deportes o recreación",
    "Hospital, clínica, consultorio, laboratorio clínico": "Al médico o atención de salud",
    "Otra vivienda": "Visitar a alguien",
    "Restaurante, bar, cafetería": "Otros (especifique)",
    "Otros (especifique)": "Otros (especifique)",
}

HOME_MOTIVE = "Regresar a Casa"
HOME_PLACE = "Su casa"

# Start-time repair. Each entry is one way a start hour can have been mistyped,
# with the cost the search minimises: (name, cost, lowest hour it applies to,
# highest hour, hours added). The costs rank the mechanisms by how common they
# are — the 12-hour clock is the dominant one — and break ties between readings
# that would otherwise be equally cheap. Decided 2026-09-03 from the diagnosis
# behind _repair_start_times().
_START_TIME_EDITS = (
    ("+12h", 1.0, 0, 11, 12),        # 12-hour clock without AM/PM
    ("-10h", 1.2, 15, 19, -10),      # an extra leading 1, only where it yields a morning hour
    ("+10h", 1.5, 0, 9, 10),         # a missing leading 1
    ("-10h+12h", 2.0, 10, 19, 2),    # an extra leading 1 on a 12-hour-clock entry
)
_START_TIME_TOLERANCE = 15   # minutes: an inversion this small is minute noise, not an hour error
_START_TIME_MAX_COST = 4.0   # give up rather than rewrite a day
START_TIME_FLAG = "hora_inicio_ajuste"   # column load_eod adds: the edit applied, "" if none


class EODTables(NamedTuple):
    """The four linked levels of the cleaned EOD survey."""

    viv: pd.DataFrame  # dwellings, indexed by folio_vivienda
    hab: pd.DataFrame  # persons, indexed by (folio_vivienda, folio_habitante)
    trips: pd.DataFrame  # trips, indexed by (folio_vivienda, folio_habitante, folio_viaje)
    legs: pd.DataFrame  # trip legs, indexed by (..., folio_viaje, folio_traslado)


def rename_imeplan(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Rename raw IMEPLAN columns to snake_case.

    ``table`` in {"habitantes", "viajes", "viviendas"}.
    """
    rename_map = imeplan_rename_map()[table]
    cols_present = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=cols_present).copy()


def clean_eod(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Normalize INEGI missing tokens to NaN, parse the date column, apply manual fixes.

    ``level`` in {"habitantes", "viviendas", "viajes"}.
    """
    df = df.copy()

    # Normalize common missing tokens
    df = df.replace({"N/D": np.nan, "ND": np.nan, "": np.nan})
    # Dates
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d-%b-%y")

    if level == "viajes":
        for idx, (col, value) in _VIAJES_MEDIO_FIXES.items():
            df.loc[idx, col] = value
    return df


# --------------------------------------------------------------- trip chains
# The survey records each person's trips for one weekday ("el día de ayer o el
# día hábil más reciente") as a chain. folio_viaje is a generated sequence
# number and carries no information of its own, but the zones and the times pin
# the sequence down — for 99.9% of clean days with three or more trips the row
# order is the only permutation that is both zone-continuous and time-monotone —
# and the row order is that sequence: 99.8% of adjacent trips start where the
# previous one ended, whereas sorting the 1,849 people whose start times are
# not monotone by time keeps only 67% of those links and makes 'Regresar a
# Casa' the first trip of 1,191 of them. Where the two disagree the start time
# is the noisy field: re-sequencing whole home-based tours resolves under 10%
# of the inversions and misreads night shifts, so the row order is kept and the
# times are repaired instead, where a unique cheapest reading exists. The rules
# below repair what the chain itself implies, remove what cannot be scheduled,
# mark what they had to guess, and leave the rest for the consumer to count
# (eodgdl.tasha.chain_report).


def _person_ids(trips: pd.DataFrame) -> np.ndarray:
    """An integer label per row, constant within a person; rows in index order."""
    return pd.factorize(trips.index.droplevel("folio_viaje"))[0]


def _drop_incomplete_days(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.MultiIndex]:
    """Drop every trip of a person who has a trip with no start time.

    325 trips report neither hour nor minute; they also lack motivo_viaje and
    tipo_lugar_destino but carry zones, mode and legs, so they are real trips
    whose questionnaire block was lost (71% captured in April 2023), never a
    person's last trip. The row after such a block is a 'Regresar a Casa' at
    18:00–20:59 in 280 of 284 cases with the block's mode and leg minutes:
    after the 239 single-trip blocks it is the outbound trip's own return;
    after the 40 multi-trip blocks ending at home it goes from home to home
    and duplicates the block's last trip with the answers attached. A day with
    an unobservable trip cannot be scheduled, and keeping the person with zero
    trips would recode 281 travellers as non-travellers (no-travel share 9.13%
    to 9.62%), so the person is excluded whole: 0.48% of persons, 0.48% of
    weighted trips; mode shares move by under 0.2 points. Imputing instead was
    prototyped and rejected: it would give 288 trips an invented purpose (82%
    'Otros') and a start time with a median ±3 h window, making those days
    30% 'other' against 5.4% overall.

    Returns the surviving trips and the persons dropped.
    """
    untimed = trips.hora_inicio_h.isna() | trips.hora_inicio_m.isna()
    persons = trips.index[untimed].droplevel("folio_viaje").unique()
    keep = ~trips.index.droplevel("folio_viaje").isin(persons)
    return trips[keep], persons


def _recode_returns_by_destination(trips: pd.DataFrame, home: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Give a 'Regresar a Casa' that did not reach the home zone its destination type's motive.

    See ``_MOTIVO_POR_TIPO_LUGAR_DESTINO``. The zone is the arbiter: where it
    says the trip did not go home, the motive is wrong and any activity beats
    it. Some of the 84 labels are probably stale too — 24 repeat the previous
    trip's type and 18 stay in its zone — but each is a trip that ended away
    from home. Returns the trips and the number recoded.
    """
    kind = trips.tipo_lugar_destino
    away = ((trips.motivo_viaje == HOME_MOTIVE) & kind.notna() & (kind != HOME_PLACE)
            & (trips.destino.astype(str).to_numpy() != home))
    if away.any():
        trips = trips.copy()
        trips["motivo_viaje"] = trips.motivo_viaje.mask(
            away, kind.map(_MOTIVO_POR_TIPO_LUGAR_DESTINO).fillna("Otros (especifique)")
        )
    return trips, int(away.sum())


def _drop_home_to_home_returns(trips: pd.DataFrame) -> pd.DataFrame:
    """Drop 'return home' trips made by a person who is already at home.

    Walk each chain with an at-home flag: it starts as whether the first trip
    left from 'Su casa', and after each kept trip it is whether that trip was
    a 'Regresar a Casa'. A return home while the flag is set is not a trip and
    is dropped; the flag stays set, so a run of such rows is dropped whole.
    468 rows in the shipped survey: 158 first trips coded 'Regresar a Casa'
    from 'Su casa' and 310 second and later members of consecutive returns.
    295 start and end in the home zone and their leg minutes usually differ
    from the return they follow, so they are not literal duplicates — a
    within-zone errand coded as 'return home' is the likeliest reading.
    Dropping them removes 26 of the survey's 178 zone-continuity breaks and
    creates none.
    """
    to_home = (trips.motivo_viaje == HOME_MOTIVE).to_numpy()
    from_home = (trips.tipo_lugar_origen == HOME_PLACE).to_numpy()
    pid = _person_ids(trips)
    keep = np.ones(len(trips), dtype=bool)
    at_home = False
    for i in range(len(trips)):
        if i == 0 or pid[i] != pid[i - 1]:
            at_home = from_home[i]
        if to_home[i] and at_home:
            keep[i] = False
        else:
            at_home = to_home[i]
    return trips[keep]


def _start_from_home_after_return(trips: pd.DataFrame, home: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Make a trip that follows a return home start in the home zone.

    A trip whose origin is not where the previous trip ended breaks the chain.
    The breaks are wrong zones, not wrong sequence: of the 176 people with one,
    82 admit no continuous ordering at all and only one of the other 94 has an
    ordering the times agree with. In 147 of the 178 breaks the previous trip
    is a 'Regresar a Casa' that reached the household's zone, and in 95 of
    those the offending origin is a stop visited earlier in the day, as if
    carried over from it. The person was at home, so the origin becomes the
    home zone — the reading a home-based activity model makes anyway. The
    other breaks (a return that did not reach home, an origin at home after a
    trip that went elsewhere) are left, since either side could be wrong.
    Returns the trips and the number of origins repaired.
    """
    prev_return = (trips.motivo_viaje == HOME_MOTIVE).groupby(level=PERSON).shift(1)
    prev_dest = trips.destino.astype(str).groupby(level=PERSON).shift(1)
    fix = (prev_return.fillna(False).astype(bool).to_numpy()
           & (prev_dest.to_numpy() == home) & (trips.origen.astype(str).to_numpy() != home))
    if fix.any():
        trips = trips.copy()
        trips.loc[fix, "origen"] = home[fix]
    return trips, int(fix.sum())


def _leg_minutes(trips: pd.DataFrame, legs: pd.DataFrame | None) -> np.ndarray:
    """Travel minutes per trip, from ``legs`` or from the traslado*_min columns; zeros if neither."""
    if legs is not None:
        return (legs.groupby(level=[0, 1, 2]).traslado_min.sum()
                    .reindex(trips.index).fillna(0).to_numpy(dtype=float))
    cols = [c for c in trips.columns if c.startswith("traslado") and c.endswith("_min")]
    if cols:
        return trips[cols].sum(axis=1, min_count=1).fillna(0).to_numpy(dtype=float)
    return np.zeros(len(trips))


def _overnight(prev: float, this: float) -> bool:
    return prev >= 18 * 60 and this <= 6 * 60


def _chain_needs_repair(start: np.ndarray, travel: np.ndarray) -> bool:
    """Does some trip start before the previous one could have arrived, beyond the tolerance?"""
    tol = _START_TIME_TOLERANCE
    return any(b < a + tr - tol and not _overnight(a, b)
               for a, b, tr in zip(start[:-1], start[1:], travel[:-1]))


def _search_edits(hours: np.ndarray, mins: np.ndarray, travel: np.ndarray):
    """The unique fewest-cost combination of edits that makes one chain feasible, or None.

    Feasible means every trip starts no earlier than the previous trip's
    arrival — its start plus its travel minutes — less the tolerance. Depth-first
    over the trips, pruned by the running cost and by the chain so far; an
    overnight wrap is accepted only between two unedited trips, so the search
    cannot manufacture a night shift. The cheapest reading must be unique.
    """
    tol, cap = _START_TIME_TOLERANCE, _START_TIME_MAX_COST
    options = [[("", 0.0, h)] + [(name, c, h + d) for name, c, lo, hi, d in _START_TIME_EDITS if lo <= h <= hi]
               for h in hours]
    n = len(hours)
    best = [cap + 1e-9]
    found: list[tuple[str, ...]] = []
    chosen: list[str] = [""] * n

    def walk(i, cost, prev_arrival, prev_start, prev_edited):
        if cost > best[0] + 1e-9:
            return
        if i == n:
            if cost < best[0] - 1e-9:
                best[0] = cost
                found.clear()
            found.append(tuple(chosen))
            return
        for name, c, h in options[i]:
            start = h * 60 + mins[i]
            edited = name != ""
            if prev_arrival is not None and not (
                start >= prev_arrival - tol
                or (not edited and not prev_edited and _overnight(prev_start, start))
            ):
                continue
            chosen[i] = name
            walk(i + 1, cost + c, start + travel[i], start, edited)

    walk(0, 0.0, None, None, False)
    return found[0] if len(found) == 1 else None


def _repair_start_times(trips: pd.DataFrame, legs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Repair mistyped start hours with the fewest edits that make a chain monotone.

    The start times are the noisy field, and the noise has a structure: hours
    entered on a 12-hour clock, and an extra or missing leading 1 (19:00 for
    9:00, 8:00 for 18:00), sometimes both on one field. A chain is feasible
    when every trip starts no earlier than the previous trip's arrival — its
    start plus its leg minutes — less ``_START_TIME_TOLERANCE``; an overnight
    wrap between a trip at or after 18:00 and one by 06:00 is not a violation.
    For every infeasible chain the search tries the menu in
    ``_START_TIME_EDITS`` on each trip and keeps the cheapest combination that
    makes the chain feasible, provided it is unique and costs at most
    ``_START_TIME_MAX_COST``. A chain with no such reading is left as it is.
    Every edited row is marked in the ``START_TIME_FLAG`` column with the edit
    applied, so a consumer can treat it as uncertain.

    On the shipped survey 2,314 chains are infeasible; 1,549 get a unique
    reading and 2,029 rows change (+12h 662, an extra leading 1 428, both 787,
    a missing leading 1 152), none of them to an hour before 05:00. The strict
    12-hour rule this replaced moved 579 rows; the search moves 536 of them
    identically, reads 5 differently because the arrival constraint rules out
    the 12-hour reading, and leaves 38 in chains it cannot resolve as a whole.
    838 inversions in 765 people remain, and 883 trips in 765 people still
    start before the previous trip could have arrived by more than the
    tolerance; tasha.chain_report counts both. Household 8, person 3 is the
    worked example: 07:24, 07:37, 19:00, 16:30, 18:02, 18:00, with 30-minute
    drives, becomes feasible by reading the 19:00 as 09:00 and the closing
    18:00 as 20:00 — the two-minute step at the end is a 32-minute
    contradiction once the drive is counted. Returns the trips and a dict
    with ``start_times_edited`` and ``chains_repaired``.
    """
    hours = trips.hora_inicio_h.to_numpy(dtype=int)
    mins = trips.hora_inicio_m.to_numpy(dtype=int)
    travel = _leg_minutes(trips, legs)
    pid = _person_ids(trips)
    first = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    last = np.r_[first[1:], len(trips)]
    delta = {name: d for name, _, _, _, d in _START_TIME_EDITS}
    new_hours = hours.copy()
    flag = np.full(len(trips), "", dtype=object)
    chains = 0
    for a, b in zip(first, last):
        if not _chain_needs_repair(hours[a:b] * 60 + mins[a:b], travel[a:b]):
            continue
        edits = _search_edits(hours[a:b], mins[a:b], travel[a:b])
        if edits is None:
            continue
        chains += 1
        for j, name in enumerate(edits):
            if name:
                new_hours[a + j] += delta[name]
                flag[a + j] = name
    trips = trips.copy()
    trips["hora_inicio_h"] = pd.array(new_hours, dtype="Int64")
    trips[START_TIME_FLAG] = flag
    return trips, {"start_times_edited": int((flag != "").sum()), "chains_repaired": chains}


def _home_zone(trips: pd.DataFrame, viv: pd.DataFrame) -> np.ndarray:
    """The household's zone id on every trip row."""
    return viv.ageb.astype(str).reindex(trips.index.get_level_values("folio_vivienda")).to_numpy()


def clean_trip_chains(
    trips: pd.DataFrame, viv: pd.DataFrame, legs: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.MultiIndex, dict]:
    """Apply the five chain rules to a trip table in row (folio_viaje) order.

    In order: exclude the persons whose day has an untimed trip; recode the
    'Regresar a Casa' trips that did not reach the household's zone from their
    destination type; drop returns home made while already at home; make a
    trip that follows a return home start in the home zone; repair mistyped
    start hours with the fewest edits that let every trip start after the
    previous one arrived. The first two orderings matter (the drop reads the
    recoded motives, the origin
    repair reads the surviving neighbour); the time repair is independent of
    the others. Each rule's evidence is in its docstring. ``viv`` supplies the
    household zone (``ageb``); ``legs`` supplies travel minutes for the time
    repair's tie-break when the table no longer carries the traslado columns.

    Returns the cleaned trips (folio_viaje untouched, so gaps mark the dropped
    rows; a ``hora_inicio_ajuste`` column marks the edited start times), the
    persons excluded, and a dict of counts: ``incomplete_persons``,
    ``incomplete_trips``, ``recoded_returns``, ``home_to_home``,
    ``origins_repaired``, ``start_times_edited``, ``chains_repaired``.
    """
    trips = trips.sort_index()
    n0 = len(trips)
    trips, dropped = _drop_incomplete_days(trips)
    n1 = len(trips)
    trips, recoded = _recode_returns_by_destination(trips, _home_zone(trips, viv))
    trips = _drop_home_to_home_returns(trips)
    n2 = len(trips)
    trips, repaired = _start_from_home_after_return(trips, _home_zone(trips, viv))
    trips, times = _repair_start_times(trips, legs)
    counts = {
        "incomplete_persons": len(dropped),
        "incomplete_trips": n0 - n1,
        "recoded_returns": recoded,
        "home_to_home": n1 - n2,
        "origins_repaired": repaired,
        **times,
    }
    return trips, dropped, counts


# ------------------------------------------------------------------- loading


def _resolve_csv(eod_path: Path | None, filename: str) -> Path:
    if eod_path is not None:
        return Path(eod_path) / filename
    from eodgdl.data import resolve

    return resolve(filename)


def load_eod(
    eod_path: Path | None = None, *, verbose: bool = False, clean_chains: bool = True
) -> EODTables:
    """Load and clean the EOD survey at four linked levels.

    With no argument the three master CSVs are fetched from the data mirror (and cached);
    pass ``eod_path`` (or set ``$EODGDL_DATA_DIR``) to read them from a local directory.

    By default the trip chains are cleaned (:func:`clean_trip_chains`): the 281
    persons whose day has an untimed trip leave ``hab`` and ``trips`` together,
    84 mislabelled 'Regresar a Casa' trips take their destination type's motive,
    468 returns home made from home are dropped, 120 trips that follow a return
    home start in the home zone, 2,029 mistyped start hours are repaired and
    marked in a ``hora_inicio_ajuste`` column, and ``hab.viajes_contados`` is
    recounted. ``folio_viaje`` is left as shipped, so a gap marks a dropped
    row. Pass
    ``clean_chains=False`` for the survey as shipped — the expansion factors
    reconcile to the published totals only on that.

    Returns an :class:`EODTables` named tuple ``(viv, hab, trips, legs)``.
    """
    viv_csv = _resolve_csv(eod_path, VIVIENDAS_CSV)
    hab_csv = _resolve_csv(eod_path, HABITANTES_CSV)
    trips_csv = _resolve_csv(eod_path, VIAJES_CSV)

    df_hab = (
        pd.read_csv(hab_csv, encoding="ISO-8859-1", low_memory=False)
        .pipe(rename_imeplan, table="habitantes")
        .set_index(["folio_vivienda", "folio_habitante"])
        .pipe(clean_eod, level="habitantes")
        .pipe(hab_schema)
        .sort_index()
    )

    df_viv = (
        pd.read_csv(viv_csv, encoding="ISO-8859-1", low_memory=False)
        .pipe(rename_imeplan, table="viviendas")
        .set_index("folio_vivienda")
        .pipe(clean_eod, level="viviendas")
        .pipe(viv_schema)
        .sort_index()
    )

    df_trips = (
        pd.read_csv(trips_csv, encoding="ISO-8859-1", low_memory=False)
        .pipe(rename_imeplan, table="viajes")
        .set_index(["folio_vivienda", "folio_habitante", "folio_viaje"])
        .pipe(clean_eod, level="viajes")
        .pipe(trips_schema)
        .sort_index()
    )

    # Remove shared columns with viv
    cols = df_hab.groupby("folio_vivienda").nunique().max().loc[lambda s: s == 1].index
    assert np.all(df_hab.groupby("folio_vivienda")[cols].first() == df_viv[cols])
    assert np.all(
        df_trips.groupby("folio_vivienda").nunique().max().loc[lambda s: s == 1].index
        == cols
    )
    df_hab = df_hab.drop(columns=cols)
    df_trips = df_trips.drop(columns=cols)
    # Remove shared columns with hab
    cols = (
        df_trips.groupby(["folio_vivienda", "folio_habitante"])
        .nunique()
        .max()
        .loc[lambda s: s == 1]
        .index.drop("tipo_lugar_origen")
    )
    assert (
        df_trips.groupby(["folio_vivienda", "folio_habitante"])[cols]
        .first()
        .equals(df_hab.loc[df_hab.viajes_contados > 0, cols])
    )
    df_trips = df_trips.drop(columns=cols)

    if clean_chains:
        df_trips, dropped, counts = clean_trip_chains(df_trips, df_viv)
        df_hab = df_hab[~df_hab.index.isin(dropped)].copy()
        df_hab["viajes_contados"] = (
            df_trips.groupby(level=PERSON).size().reindex(df_hab.index).fillna(0).astype(int)
        )
        log.info("trip chains cleaned: %s", counts)
        if verbose:
            print("Trip chains cleaned:", counts)

    # Create legs df
    df_legs = (
        pd.concat(
            [
                df_trips[
                    [
                        f"traslado{n_leg}_medio",
                        f"traslado{n_leg}_min",
                        f"traslado{n_leg}_pago",
                    ]
                ]
                .rename(
                    columns={
                        f"traslado{n_leg}_medio": "traslado_medio",
                        f"traslado{n_leg}_min": "traslado_min",
                        f"traslado{n_leg}_pago": "traslado_pago",
                    }
                )
                .dropna(subset="traslado_medio")
                .assign(folio_traslado=n_leg)
                for n_leg in range(1, 6)
            ]
        )
        .set_index("folio_traslado", append=True)
        .sort_index()
    )
    df_trips = df_trips.drop(
        columns=[c for c in df_trips.columns if c.startswith("traslado")]
    )

    if verbose:
        print(
            "DF shapes viv/hab/viaj/legs",
            df_viv.shape,
            df_hab.shape,
            df_trips.shape,
            df_legs.shape,
        )

    return EODTables(df_viv, df_hab, df_trips, df_legs)
