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
# día hábil más reciente") as a chain in folio_viaje order — the order the trips
# were made — and that order is authoritative: in it, 99.8% of adjacent trips
# start in the zone the previous one ended in, whereas sorting the 1,849 people
# whose start times are not monotone by start time keeps only 67% of those
# links and makes 'Regresar a Casa' the first trip of 1,191 of them. The start
# times are the noisy field. The four rules below repair what can be repaired
# without inventing data and remove what cannot be scheduled; the rest is left
# for the consumer to count (eodgdl.tasha.chain_report).


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


def _fix_twelve_hour_clock(trips: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Move forward 12 hours a start time that was entered on a 12-hour clock.

    A trip is moved when it starts earlier than the trip before it, its hour is
    11 or less, and adding 12 hours puts it at or after the previous trip and
    at or before the next one — so the correction never creates an inversion
    it did not find, and a run of consecutive 12-hour entries is left alone.
    The scan is sequential: a moved trip is the "previous" its successor is
    compared against. Of the 2,011 trips in the shipped survey that start
    before their predecessor, 82% are 'Regresar a Casa' and half follow a
    'Trabajar'; this rule moves 579 (a previous-only rule would move 887 but
    overtake the following trip 206 times), about 130 of them after a trip
    starting at or after 17:00, which could be genuine night-shift returns.
    1,411 inversions in 1,322 people remain — about 124 look like night
    shifts, 88 are within 15 minutes, the rest are typos with no safe repair —
    and are kept: dropping those people would remove 3.4% of weighted travel,
    concentrated in 4+ trip days and evening and night workers. Returns the
    trips and the number moved.
    """
    hours = trips.hora_inicio_h.to_numpy(dtype=int)
    start = (hours * 60 + trips.hora_inicio_m.to_numpy(dtype=int)).astype(float)
    pid = _person_ids(trips)
    fixed = start.copy()
    moved = np.zeros(len(trips), dtype=bool)
    n = len(trips)
    for i in range(1, n):
        if pid[i] != pid[i - 1] or fixed[i] >= fixed[i - 1] or hours[i] > 11:
            continue
        candidate = fixed[i] + 12 * 60
        if candidate < fixed[i - 1]:
            continue
        if i + 1 < n and pid[i + 1] == pid[i] and candidate > start[i + 1]:
            continue
        fixed[i] = candidate
        moved[i] = True
    if moved.any():
        trips = trips.copy()
        trips["hora_inicio_h"] = trips.hora_inicio_h.where(~moved, trips.hora_inicio_h + 12)
    return trips, int(moved.sum())


def clean_trip_chains(trips: pd.DataFrame, viv: pd.DataFrame) -> tuple[pd.DataFrame, pd.MultiIndex, dict]:
    """Apply the four chain rules to a trip table in folio_viaje order.

    In order: exclude the persons whose day has an untimed trip; recode the
    'Regresar a Casa' trips that did not reach the household's zone from their
    destination type; drop returns home made while already at home; move
    12-hour-clock start times forward 12 hours. Each rule's evidence is in its
    docstring. ``viv`` supplies the household zone (``ageb``).

    Returns the cleaned trips (folio_viaje untouched, so gaps mark the dropped
    rows), the persons excluded, and a dict of counts: ``incomplete_persons``,
    ``incomplete_trips``, ``recoded_returns``, ``home_to_home``, ``moved_12h``.
    """
    trips = trips.sort_index()
    n0 = len(trips)
    trips, dropped = _drop_incomplete_days(trips)
    n1 = len(trips)
    home = viv.ageb.astype(str).reindex(trips.index.get_level_values("folio_vivienda")).to_numpy()
    trips, recoded = _recode_returns_by_destination(trips, home)
    trips = _drop_home_to_home_returns(trips)
    n2 = len(trips)
    trips, moved = _fix_twelve_hour_clock(trips)
    counts = {
        "incomplete_persons": len(dropped),
        "incomplete_trips": n0 - n1,
        "recoded_returns": recoded,
        "home_to_home": n1 - n2,
        "moved_12h": moved,
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
    468 returns home made from home are dropped, 579 12-hour-clock start times
    are moved forward, and ``hab.viajes_contados`` is recounted. ``folio_viaje``
    is left as recorded, so a gap marks a dropped row. Pass
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
