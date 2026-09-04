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
# (folio_vivienda, folio_habitante, folio_viaje) → (column, corrected value).
# All three cells carry the bare string '17' in a leg-mode field where every
# other leg carries a label. Neither the glossaries nor the technical report
# list mode codes, so what 17 stood for in the field system is undocumented,
# and without a label the trip schema rejects the file, since 17 is outside
# the mode levels. The label comes from a twin record: household 879, in the
# same AGEB (1409700251418, Tlajomulco) and interviewed the same day
# (28-Jan-23), holds two persons (3 and 7) whose diaries duplicate these two
# persons' trip by trip and leg by leg — same zones, motives, minutes and
# fares, start times one to nine minutes apart — and label exactly these
# three legs 'Transporte informal' (5 minutes for 10 pesos, the fare person 3
# of 9530 also reports under that label on another trip). Bus route 17 was
# ruled out: no route numbered 17 stops within 3 km of any end of these
# trips. Decided 2026-09-03; see reports/loading.qmd, "Three cells corrected
# by hand", and reports/duplicate_diaries.qmd for the twin records.
_VIAJES_MEDIO_FIXES = {
    (9560, 4, 1): ("traslado1_medio", "Transporte informal"),
    (9560, 4, 2): ("traslado4_medio", "Transporte informal"),
    (9530, 3, 3): ("traslado5_medio", "Transporte informal"),
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

# Repeated diaries. A person's diary — the trips in row order, each read as its
# origin and destination zone, motive, destination place type and every leg's
# mode, minutes and fare — is also the diary of a person in another household
# of the same AGEB and interview date for some 2,360 persons, and for two
# thirds of them every start time sits within ten minutes of the twin's: the
# differences fill one to ten minutes evenly and stop there (217 paired trips
# differ by exactly ten, 11 by eleven), whereas people who travel together
# agree to the minute (within a household, 69% of paired trips). The twin is a
# different person in 87% of the pairs. That is a diary copied onto another
# record and nudged, not shared travel; which of the pair is the copy cannot
# be told, so both are flagged and neither is dropped — the expansion factors
# reconcile only on the file as shipped. Decided 2026-09-03 from the diagnosis
# in reports/duplicate_diaries.qmd.
DIARY_FLAG = "diario_repetido"          # column load_eod adds to hab: True when the diary repeats
_DIARY_TWIN_MAX_OFFSET = 10             # minutes: the largest start-time difference a twin may show


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
    """Normalize missing tokens to NaN, parse the date column, apply manual fixes.

    ``level`` in {"habitantes", "viviendas", "viajes"}.

    The token rule (``N/D``, ``ND``, the empty string) is inert on the 2023
    release: none of the tokens appears in the shipped files, so every missing
    value is an empty CSV field, and most blanks are skip logic — a question
    not asked, not an unknown answer. ``reports/loading.qmd`` lists every
    column with blanks and the rule it follows. ``fecha`` is the interview
    date, not the travel day; the schema types it as a UTC midnight, so treat
    it as a calendar date.
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
# below impute what the questionnaire lost, repair what the chain itself
# implies, and drop nothing: every change is written to the row's FIX_FLAG
# column and everything left wrong to its ISSUE_FLAG column, so a consumer can
# keep, filter or weight the rows as it sees fit (eodgdl.tasha.build leaves out
# the rows marked as not being trips; tasha.chain_report counts the rest).

FIX_FLAG = "ajustes"       # column load_eod adds to trips: what it changed on the row, ';'-joined, "" if nothing
ISSUE_FLAG = "problemas"   # column load_eod adds to trips: what is still wrong with the row, ';'-joined, "" if nothing

# The vocabulary of the two columns. Each code names the field it touches and
# where the value came from (fixes) or the defect left (issues); the docstring
# of the rule that writes it is the evidence.
FIX_CODES = {
    "hora:duplicado": "start time taken from the home-to-home return that repeats this untimed return",
    "hora:vecinos": "start time imputed from the nearest timed trips: the next trip's start less this "
                    "trip's leg minutes and the donors' median activity duration",
    "motivo:duplicado": "motive and destination type taken from the home-to-home return that repeats this row",
    "motivo:vecinos": "motive and destination type imputed from the nearest timed trips",
    "motivo:tipo_destino": "a 'Regresar a Casa' that did not reach the home zone, recoded from its destination type",
    "origen:casa": "origin set to the home zone: the previous trip was a return home that reached it",
    "hora:+12h": "start hour read on a 12-hour clock: 12 hours added",
    "hora:-10h": "start hour read with an extra leading 1: 10 hours removed",
    "hora:+10h": "start hour read with a missing leading 1: 10 hours added",
    "hora:-10h+12h": "start hour read with an extra leading 1 on a 12-hour-clock entry: 2 hours added",
}
ISSUE_CODES = {
    "regreso_en_casa": "a 'Regresar a Casa' made while already at home: not a trip; the model build leaves it out",
    "regreso_duplicado": "a home-to-home 'Regresar a Casa' repeating the untimed return before it, whose "
                         "time and motive it carried: not a trip; the model build leaves it out",
    "hora_invertida": "starts before the previous trip could have arrived, beyond the tolerance and not "
                      "overnight, and no unique repair exists",
    "origen_discontinuo": "does not start in the zone the previous trip ended in; either side could be wrong",
    "regreso_sin_llegar": "a 'Regresar a Casa' whose destination zone is not the household's and whose "
                          "destination type gives nothing to recode from",
    "tipo_destino_dudoso": "a 'Regresar a Casa' that reached the home zone but reports a destination type "
                           "that is not a home; the motive is kept",
    "hora_nocturna": "starts by 06:00 after a trip that started at or after 18:00: read as overnight and "
                     "left alone; a night shift or a mistyped hour, nothing says which",
    "hora_anterior": "starts before the previous trip's start but within the tolerance of its arrival, so "
                     "not repaired: minute noise",
    "hora_repetida": "starts at the same minute as the previous trip, whose legs fit within the tolerance "
                     "(beyond it the row is hora_invertida instead)",
    "hora_traslapada": "starts after the previous trip's start but before its reported arrival, by no more "
                       "than the tolerance: the leg minutes overrun the next start; rounding, not an hour error",
    "inicio_fuera_de_casa": "the day's first trip does not leave from 'Su casa'",
    "inicio_zona_ajena": "the day's first trip leaves from 'Su casa' but not from the household's zone; "
                         "the two disagree and nothing says which is right",
    "fin_fuera_de_casa": "the day's last trip is not a 'Regresar a Casa' that reaches the household's zone",
    "actividad_en_casa": "an activity motive with destination type 'Su casa': work from home, or a return "
                         "home mislabelled the other way round",
    "motivo_guarderia": "a 'Guardería' motive, which the model maps to school; most are adults escorting a child",
}
NON_TRIP_ISSUES = ("regreso_en_casa", "regreso_duplicado")   # rows kept in trips that are not trips
# The five hora_* codes are mutually exclusive: a row carries at most one of them.

# Nearest-neighbour imputation of the lost questionnaire block (motive,
# destination type, start time). A target's distance to a donor is the sum of
# these weights over the categorical and boolean features that differ plus the
# weights times the absolute differences on the numeric ones; the k nearest
# donors vote. Decided 2026-09-03 from the diagnosis behind
# _impute_untimed_trips(): the weights are hand-set, and held-out accuracy
# moves by about a point when any one feature is dropped.
_IMPUTE_NEIGHBOURS = 30
_IMPUTE_WEIGHTS = {
    # categorical: main mode; sex; occupation (blank is its own level); the person's
    # own earlier motive at the same destination zone, "none" if never visited
    "modo": 1.5, "sexo": 0.5, "ocupacion": 1.0, "motivo_propio": 2.0,
    # boolean: same origin and destination zone; leaves from the home zone; the next
    # row is a return home; first trip of the day; a work or school trip earlier in the day
    "intrazonal": 1.0, "desde_casa": 1.0, "siguiente_regreso": 1.0, "primero": 1.0, "trabajo_previo": 1.0,
    # numeric: log(1 + leg minutes); the next trip's start in hours; age in decades
    "log_min": 1.0, "hora_siguiente": 1.0, "edad": 1.0,
}
_IMPUTE_CATEGORICAL = ("modo", "sexo", "ocupacion", "motivo_propio")
_IMPUTE_BOOLEAN = ("intrazonal", "desde_casa", "siguiente_regreso", "primero", "trabajo_previo")
_IMPUTE_NUMERIC = ("log_min", "hora_siguiente", "edad")


def _person_ids(trips: pd.DataFrame) -> np.ndarray:
    """An integer label per row, constant within a person; rows in index order."""
    return pd.factorize(trips.index.droplevel("folio_viaje"))[0]


def _add_code(flags: np.ndarray, mask, code: str) -> None:
    """Append ``code`` to the ';'-joined flags of the rows ``mask`` selects, in place."""
    mask = np.asarray(mask, dtype=bool)
    flags[mask] = np.where(flags[mask] == "", code, flags[mask] + ";" + code)


def has_code(flags: pd.Series, code: str) -> pd.Series:
    """True where a ';'-joined flag column (``ajustes`` or ``problemas``) carries ``code``."""
    return flags.astype(str).map(lambda s: code in s.split(";"))


def non_trips(trips: pd.DataFrame) -> pd.Series:
    """True for the rows ``load_eod`` marked as not being trips (``NON_TRIP_ISSUES``).

    All False on a table without the ``problemas`` column, such as the survey
    as shipped: there the returns home made from home are still unmarked.
    """
    if ISSUE_FLAG not in trips.columns:
        return pd.Series(False, index=trips.index)
    flagged = pd.Series(False, index=trips.index)
    for code in NON_TRIP_ISSUES:
        flagged |= has_code(trips[ISSUE_FLAG], code)
    return flagged


def _start_minutes(trips: pd.DataFrame) -> np.ndarray:
    """Start time in minutes from midnight per row; NaN where hour or minute is missing."""
    return (trips.hora_inicio_h.astype(float) * 60 + trips.hora_inicio_m.astype(float)).to_numpy()


def _shift(values: np.ndarray, pid: np.ndarray, by: int):
    """``values`` shifted by ``by`` rows within each person; None/NaN across persons."""
    out = pd.Series(values).groupby(pid).shift(by)
    return out.to_numpy()


def _duplicate_returns(trips: pd.DataFrame, home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The motive-less returns home whose next row repeats them from home to home, and those rows.

    A trip with no motive that comes from elsewhere into the home zone is a
    return home. When the row after it is a timed 'Regresar a Casa' that
    starts and ends in the home zone, that row is the return's questionnaire
    block attached to a duplicate rather than a trip: the person was already
    home. 38 such pairs in the shipped survey — the last row of every
    multi-trip untimed block that ends at home (37) and one timed trip with no
    motive; 20 duplicates repeat the return's mode and leg minutes exactly and
    the others are five-minute car rows. Returns two masks over ``trips``: the
    return, and the duplicate that follows it.
    """
    pid = _person_ids(trips)
    start = _start_minutes(trips)
    origin, dest = trips.origen.astype(str).to_numpy(), trips.destino.astype(str).to_numpy()
    motive = trips.motivo_viaje.astype(object).to_numpy()
    next_is_home_return = (
        (_shift(motive, pid, -1) == HOME_MOTIVE)
        & (_shift(origin, pid, -1) == home) & (_shift(dest, pid, -1) == home)
        & ~np.isnan(_shift(start, pid, -1).astype(float))
    )
    target = pd.isna(motive) & (origin != home) & (dest == home) & next_is_home_return
    duplicate = np.r_[False, target[:-1]]
    return target, duplicate


def _trip_features(trips: pd.DataFrame, home: np.ndarray, hab: pd.DataFrame | None,
                   legs: pd.DataFrame | None) -> pd.DataFrame:
    """One row per trip: the features the nearest-neighbour imputation compares, plus its arithmetic.

    Carries ``start``, ``next_start``, ``prev_arrival`` and ``legmin`` in minutes,
    ``activity`` (a timed trip with a motive other than 'Regresar a Casa') and
    ``motivo``/``tipo`` as strings, alongside the ``_IMPUTE_WEIGHTS`` features.
    The person features come from ``hab`` and are absent when it is None.
    """
    pid = _person_ids(trips)
    start = _start_minutes(trips)
    legmin = _leg_minutes(trips, legs)
    origin, dest = trips.origen.astype(str).to_numpy(), trips.destino.astype(str).to_numpy()
    motive = trips.motivo_viaje.astype(object).to_numpy()
    activity = pd.notna(motive) & (motive != HOME_MOTIVE)
    mandatory = activity & pd.Series(motive).isin(["Trabajar", "Estudiar"]).to_numpy()
    seen_mandatory = pd.Series(mandatory.astype(int)).groupby(pid).cumsum().to_numpy() - mandatory
    person_key = pd.MultiIndex.from_arrays(
        [trips.index.get_level_values(0), trips.index.get_level_values(1), dest])
    own = pd.Series(np.where(activity, motive, None), index=person_key, dtype=object)
    own = own.groupby(level=[0, 1, 2], sort=False).ffill().groupby(level=[0, 1, 2], sort=False).shift(1)
    next_start = _shift(start, pid, -1).astype(float)
    f = pd.DataFrame({
        "modo": trips.modo_principal.astype(str).to_numpy(),
        "motivo_propio": pd.Series(own.to_numpy(), dtype=object).fillna("none").astype(str).to_numpy(),
        "intrazonal": origin == dest,
        "desde_casa": origin == home,
        "siguiente_regreso": _shift(motive, pid, -1) == HOME_MOTIVE,
        "primero": np.r_[True, pid[1:] != pid[:-1]],
        "trabajo_previo": seen_mandatory > 0,
        "log_min": np.log1p(legmin),
        "hora_siguiente": next_start / 60,
        "start": start,
        "next_start": next_start,
        "prev_arrival": _shift(start + legmin, pid, 1).astype(float),
        "legmin": legmin,
        "activity": activity,
        "motivo": pd.Series(motive, dtype=object).astype(str).to_numpy(),
        "tipo": trips.tipo_lugar_destino.astype(object).astype(str).to_numpy(),
    }, index=trips.index)
    if hab is not None:
        person = hab.reindex(trips.index.droplevel("folio_viaje"))
        f["sexo"] = person.sexo_nacimiento.astype(str).to_numpy()
        f["ocupacion"] = person.ocupacion.astype(object).fillna("NA").astype(str).to_numpy()
        f["edad"] = person.edad.to_numpy(dtype=float) / 10
    return f


def _nearest_donors(targets: pd.DataFrame, donors: pd.DataFrame, k: int = _IMPUTE_NEIGHBOURS) -> pd.DataFrame:
    """The k nearest donors' vote for every target row: motive, type, activity duration, start.

    Distance is the weighted mismatch count over the categorical and boolean
    features plus the weighted absolute differences over the numeric ones
    (``_IMPUTE_WEIGHTS``); a numeric feature the target lacks costs nothing.
    The motive is the most common among the k nearest, ties going to the
    closer set; the type is the most common among the donors that voted for
    it, and ``duration`` and ``start`` are their medians, in minutes.
    """
    categorical = [c for c in _IMPUTE_CATEGORICAL if c in donors.columns and c in targets.columns]
    boolean = list(_IMPUTE_BOOLEAN)
    numeric = [c for c in _IMPUTE_NUMERIC if c in donors.columns and c in targets.columns]
    col = {c: donors[c].to_numpy() for c in categorical + boolean + numeric}
    d_motive, d_type = donors.motivo.to_numpy(), donors.tipo.to_numpy()
    d_duration = np.clip(donors.next_start.to_numpy() - donors.start.to_numpy() - donors.legmin.to_numpy(), 0, None)
    d_start = donors.start.to_numpy()
    k = min(k, len(donors))
    rows = []
    for _, t in targets.iterrows():
        d = np.zeros(len(donors))
        for c in categorical:
            d += _IMPUTE_WEIGHTS[c] * (col[c] != str(t[c]))
        for c in boolean:
            d += _IMPUTE_WEIGHTS[c] * (col[c] != bool(t[c]))
        for c in numeric:
            if not np.isnan(t[c]):
                d += _IMPUTE_WEIGHTS[c] * np.abs(col[c] - t[c])
        near = np.argpartition(d, k - 1)[:k]
        tally = pd.DataFrame({"m": d_motive[near], "d": d[near]}).groupby("m").d.agg(["size", "sum"])
        motive = tally.sort_values(["size", "sum"], ascending=[False, True]).index[0]
        vote = near[d_motive[near] == motive]
        rows.append((motive, pd.Series(d_type[vote]).mode().iloc[0],
                     float(np.median(d_duration[vote])), float(np.median(d_start[vote]))))
    return pd.DataFrame(rows, columns=["motivo", "tipo", "duration", "start"], index=targets.index)


def _impute_untimed_trips(
    trips: pd.DataFrame, home: np.ndarray, hab: pd.DataFrame | None, legs: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fill the start time, motive and destination type of the trips that lost their questionnaire block.

    325 trips report neither hour nor minute; they also lack motivo_viaje and
    tipo_lugar_destino but carry zones, a mode and legs with minutes, so they
    are real trips whose questionnaire block was not captured (71% in April
    2023, the last month of fieldwork). None is a person's last trip, and 281
    persons hold them in 284 blocks. Seven more trips are timed but have no
    motive. Nothing is dropped; the lost answers are imputed in two steps and
    every imputed row is marked in the ``ajustes`` column.

    First, the 38 motive-less returns home — a row from elsewhere into the
    home zone, 37 of them untimed — whose next row is a timed 'Regresar a
    Casa' from home to home take that row's motive and, where missing, its
    time (``motivo:duplicado``, ``hora:duplicado``): it is the return's own
    questionnaire block attached to a duplicate, and the duplicate is marked
    ``regreso_duplicado`` (see ``_duplicate_returns``).

    Second, every remaining row without a motive takes the vote of its 30
    nearest timed activity trips (``_nearest_donors``, features in
    ``_IMPUTE_WEIGHTS``): the most common motive and, among those donors, the
    most common destination type (``motivo:vecinos``). A row without a start
    time is then placed back from the next trip: its start is the next trip's
    start less its own leg minutes less the donors' median activity duration,
    kept no earlier than the previous trip's arrival and no later than the
    next trip's start less the leg minutes (``hora:vecinos``); a row with no
    timed successor takes the donors' median start instead. Blocks are filled
    back to front, so every row has a timed successor when its turn comes.
    Every untimed row is an activity or a duplicated return, so the donors are
    activity trips and the vote never yields 'Regresar a Casa'. The
    neighbouring times are read as shipped, so a mistyped neighbour gives a
    mistyped imputation, which the typo search that follows cannot edit: two
    imputed rows end up marked ``hora_invertida``.

    Scored on 1,500 timed trips of the same shape — from home, followed by a
    return, second tour of the day, on foot, by car or by bus, return between
    17:00 and 22:00 — with the person's own trips withheld: the vote names the
    motive 55% of the time against 25% for the most common motive, 87% where
    the person had visited the zone earlier in the day and 36% elsewhere, and
    the destination type 49% of the time; the start lands a median of 10
    minutes (mean 40) from the true one, against 66 (mean 94) for the midpoint
    of the window between the neighbouring trips. On the survey 288 rows get a
    time and 294 a motive this way; the modal answer is 'Compras (comida)',
    which the vote over-produces (48% against 24% in the held-out set), so the
    imputed motives are plausible per row and skewed as a set.

    Returns the trips and a dict of masks over them: ``time_from_duplicate``,
    ``motive_from_duplicate``, ``duplicate``, ``time_from_neighbours``,
    ``motive_from_neighbours``.
    """
    trips = trips.copy()
    n = len(trips)
    masks = {key: np.zeros(n, dtype=bool) for key in (
        "time_from_duplicate", "motive_from_duplicate", "duplicate", "time_from_neighbours", "motive_from_neighbours")}
    target, duplicate = _duplicate_returns(trips, home)
    if target.any():
        untimed = np.isnan(_start_minutes(trips))
        pos = np.flatnonzero(target)
        for col in ("motivo_viaje", "tipo_lugar_destino"):
            trips.iloc[pos, trips.columns.get_loc(col)] = trips.iloc[pos + 1][col].to_numpy()
        timed_pos = pos[untimed[pos]]
        for col in ("hora_inicio_h", "hora_inicio_m"):
            trips.iloc[timed_pos, trips.columns.get_loc(col)] = trips.iloc[timed_pos + 1][col].to_numpy()
        masks["time_from_duplicate"] |= target & untimed
        masks["motive_from_duplicate"] |= target
        masks["duplicate"] |= duplicate

    for _ in range(20):   # blocks are filled back to front, one row of each per pass
        f = _trip_features(trips, home, hab, legs)
        untimed = np.isnan(f.start.to_numpy())
        no_motive = trips.motivo_viaje.isna().to_numpy()
        pending = untimed | no_motive
        if not pending.any():
            break
        next_untimed = np.isnan(f.next_start.to_numpy()) & (_shift(untimed, _person_ids(trips), -1) == True)  # noqa: E712
        ready = pending & ~next_untimed
        if not ready.any():
            ready = pending
        donors = f[f.activity & ~np.isnan(f.start.to_numpy()) & ~np.isnan(f.next_start.to_numpy())]
        if donors.empty:
            raise ValueError("no timed activity trips to impute from")
        vote = _nearest_donors(f[ready], donors)
        need_motive = no_motive[ready]
        idx = vote.index[need_motive]
        trips.loc[idx, "motivo_viaje"] = vote.motivo[need_motive].to_numpy()
        trips.loc[idx, "tipo_lugar_destino"] = vote.tipo[need_motive].to_numpy()
        masks["motive_from_neighbours"][np.flatnonzero(ready)[need_motive]] = True
        need_time = untimed[ready]
        if need_time.any():
            t = f[ready][need_time]
            v = vote[need_time]
            latest = t.next_start.to_numpy() - t.legmin.to_numpy()
            start = np.where(np.isnan(latest), v.start.to_numpy(), latest - v.duration.to_numpy())
            start = np.fmax(start, t.prev_arrival.to_numpy())          # no earlier than the previous arrival
            start = np.where(np.isnan(latest), start, np.fmin(start, latest))
            start = np.clip(np.round(start), 0, 24 * 60 - 1).astype(int)
            trips.loc[t.index, "hora_inicio_h"] = pd.array(start // 60, dtype="Int64")
            trips.loc[t.index, "hora_inicio_m"] = pd.array(start % 60, dtype="Int64")
            masks["time_from_neighbours"][np.flatnonzero(ready)[need_time]] = True
    else:
        raise ValueError("untimed trips left after 20 passes")
    return trips, masks


def _recode_returns_by_destination(trips: pd.DataFrame, home: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Give a 'Regresar a Casa' that did not reach the home zone its destination type's motive.

    See ``_MOTIVO_POR_TIPO_LUGAR_DESTINO``. The zone is the arbiter: where it
    says the trip did not go home, the motive is wrong and any activity beats
    it. Some of the 84 labels are probably stale too — 24 repeat the previous
    trip's type and 18 stay in its zone — but each is a trip that ended away
    from home. Returns the trips and the mask of the rows recoded.
    """
    kind = trips.tipo_lugar_destino
    away = ((trips.motivo_viaje == HOME_MOTIVE) & kind.notna() & (kind != HOME_PLACE)
            & (trips.destino.astype(str).to_numpy() != home))
    if away.any():
        trips = trips.copy()
        trips["motivo_viaje"] = trips.motivo_viaje.mask(
            away, kind.map(_MOTIVO_POR_TIPO_LUGAR_DESTINO).fillna("Otros (especifique)")
        )
    return trips, away.to_numpy()


def _home_to_home_returns(trips: pd.DataFrame, skip: np.ndarray | None = None) -> np.ndarray:
    """Mark the 'return home' trips made by a person who is already at home.

    Walk each chain with an at-home flag: it starts as whether the first trip
    left from 'Su casa', and after each trip it is whether that trip was a
    'Regresar a Casa'. A return home while the flag is set is not a trip; the
    flag stays set, so a run of such rows is marked whole. Rows in ``skip``
    (the duplicates of ``_duplicate_returns``) are stepped over as if absent.
    491 rows in the shipped survey once the untimed trips are imputed: 158
    first trips coded 'Regresar a Casa' from 'Su casa' and 333 second and
    later members of consecutive returns. 314 start and end in the home zone
    and their leg minutes usually differ from the return they follow, so they
    are not literal duplicates — a within-zone errand coded as 'return home'
    is the likeliest reading. They are kept and marked ``regreso_en_casa``;
    the model build leaves them out, and without them and the 38 duplicates
    the survey's 178 zone-continuity breaks fall to 156. Returns the mask of
    the rows marked.
    """
    to_home = (trips.motivo_viaje == HOME_MOTIVE).to_numpy()
    from_home = (trips.tipo_lugar_origen == HOME_PLACE).to_numpy()
    skip = np.zeros(len(trips), dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    pid = _person_ids(trips)
    marked = np.zeros(len(trips), dtype=bool)
    at_home = False
    for i in range(len(trips)):
        if i == 0 or pid[i] != pid[i - 1]:
            at_home = from_home[i]
        if skip[i]:
            continue
        if to_home[i] and at_home:
            marked[i] = True
        else:
            at_home = to_home[i]
    return marked


def _start_from_home_after_return(trips: pd.DataFrame, home: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
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
    trip that went elsewhere) are left and marked ``origen_discontinuo``, since
    either side could be wrong. On the chain without its non-trips 124 of the
    156 breaks are repaired and 32 left. Returns the trips and the mask of
    origins repaired.
    """
    prev_return = (trips.motivo_viaje == HOME_MOTIVE).groupby(level=PERSON).shift(1)
    prev_dest = trips.destino.astype(str).groupby(level=PERSON).shift(1)
    fix = (prev_return.fillna(False).astype(bool).to_numpy()
           & (prev_dest.to_numpy() == home) & (trips.origen.astype(str).to_numpy() != home))
    if fix.any():
        trips = trips.copy()
        trips.loc[fix, "origen"] = home[fix]
    return trips, fix


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


def _search_edits(hours: np.ndarray, mins: np.ndarray, travel: np.ndarray, locked: np.ndarray | None = None):
    """The unique fewest-cost combination of edits that makes one chain feasible, or None.

    Feasible means every trip starts no earlier than the previous trip's
    arrival — its start plus its travel minutes — less the tolerance. Depth-first
    over the trips, pruned by the running cost and by the chain so far; an
    overnight wrap is accepted only between two unedited trips, so the search
    cannot manufacture a night shift. A ``locked`` trip (an imputed start time,
    which cannot carry a typo) is never edited. The cheapest reading must be unique.
    """
    tol, cap = _START_TIME_TOLERANCE, _START_TIME_MAX_COST
    locked = np.zeros(len(hours), dtype=bool) if locked is None else locked
    options = [[("", 0.0, h)] + ([] if lock else [(name, c, h + d) for name, c, lo, hi, d in _START_TIME_EDITS if lo <= h <= hi])
               for h, lock in zip(hours, locked)]
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


def _repair_start_times(
    trips: pd.DataFrame, legs: pd.DataFrame | None = None, locked: np.ndarray | None = None
) -> tuple[pd.DataFrame, np.ndarray, int]:
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
    ``_START_TIME_MAX_COST``. A chain with no such reading is left as it is,
    and the trips in it that still start before the previous arrival are
    marked ``hora_invertida``. Every edited row is marked in the ``ajustes``
    column with the edit applied (``hora:+12h`` and so on), so a consumer can
    treat it as uncertain; ``locked`` rows (imputed start times) are never edited.

    On the survey, with the untimed trips imputed and the non-trips set
    aside, 2,319 chains are infeasible; 1,552 get a unique reading and 2,032
    rows change (+12h 662, an extra leading 1 431, both 787, a missing
    leading 1 152), none of them to an hour before 05:00. The strict 12-hour
    rule this replaced moved 579 rows; the search moves 536 of them
    identically, reads 5 differently because the arrival constraint rules out
    the 12-hour reading, and leaves 38 in chains it cannot resolve as a whole.
    885 trips in 767 people still start before the previous trip could have
    arrived by more than the tolerance and are marked ``hora_invertida``;
    tasha.chain_report counts them again on the built table. Household 8,
    person 3 is the worked example: 07:24, 07:37,
    19:00, 16:30, 18:02, 18:00, with 30-minute drives, becomes feasible by
    reading the 19:00 as 09:00 and the closing 18:00 as 20:00 — the two-minute
    step at the end is a 32-minute contradiction once the drive is counted.
    Returns the trips, the edit name per row ("" where none) and the number of
    chains repaired.
    """
    hours = trips.hora_inicio_h.to_numpy(dtype=int)
    mins = trips.hora_inicio_m.to_numpy(dtype=int)
    travel = _leg_minutes(trips, legs)
    locked = np.zeros(len(trips), dtype=bool) if locked is None else np.asarray(locked, dtype=bool)
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
        edits = _search_edits(hours[a:b], mins[a:b], travel[a:b], locked[a:b])
        if edits is None:
            continue
        chains += 1
        for j, name in enumerate(edits):
            if name:
                new_hours[a + j] += delta[name]
                flag[a + j] = name
    trips = trips.copy()
    trips["hora_inicio_h"] = pd.array(new_hours, dtype="Int64")
    return trips, flag, chains


def _remaining_issues(trips: pd.DataFrame, home: np.ndarray, legs: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """What is still wrong with each trip of a chain after the rules, one mask per ``ISSUE_CODES`` entry it can produce.

    Runs over the chain without its non-trips, so "first", "last" and
    "previous" are read over the trips the model build sees. Every kind of
    error the chain diagnosis (``reports/trip_chains.qmd``) found and the
    rules do not repair is a code here, so the trips carrying at least one
    error are exactly the trips with a non-empty ``problemas``; nothing is
    repaired, since none of these has a repair that does not invent data. On
    the shipped survey, after the rules: 885 ``hora_invertida``, 93
    ``hora_nocturna``, 25 ``hora_anterior``, 101 ``hora_repetida`` and 569
    ``hora_traslapada`` (one time code per row at most: the mild codes are
    what ``hora_invertida`` does not cover; the overlaps within the tolerance
    spike at 5, 10 and 15 minutes, the rounding of the reported leg minutes,
    and 221 of them follow a leg of more than an hour), 32
    ``origen_discontinuo``, 163 ``regreso_sin_llegar``,
    16 ``tipo_destino_dudoso``; 906 days start from somewhere other than 'Su
    casa' (``inicio_fuera_de_casa``), 639 from 'Su casa' in a zone that is
    not the household's (``inicio_zona_ajena``) and 514 do not end with a
    return to the home zone (``fin_fuera_de_casa``) — second homes, nights
    spent elsewhere and geocoding slips all look alike here; 787 activity
    trips end at 'Su casa' (``actividad_en_casa``), 705 of them in the home
    zone, the mirror image of the returns ``_recode_returns_by_destination``
    recodes; and 217 trips carry the 'Guardería' motive
    (``motivo_guarderia``), most made by adults escorting a child, which the
    model reads as a school trip.
    """
    pid = _person_ids(trips)
    first = np.r_[True, pid[1:] != pid[:-1]]
    last = np.r_[pid[1:] != pid[:-1], True]
    start = _start_minutes(trips)
    travel = _leg_minutes(trips, legs)
    prev_start = _shift(start, pid, 1).astype(float)
    prev_arrival = _shift(start + travel, pid, 1).astype(float)
    origin, dest = trips.origen.astype(str).to_numpy(), trips.destino.astype(str).to_numpy()
    prev_dest = _shift(dest, pid, 1)
    has_prev = pd.notna(prev_dest)
    is_return = (trips.motivo_viaje == HOME_MOTIVE).to_numpy()
    kind = trips.tipo_lugar_destino
    origin_kind = trips.tipo_lugar_origen.astype(str).to_numpy()
    from_home = origin_kind == HOME_PLACE
    with np.errstate(invalid="ignore"):
        wrap = has_prev & (prev_start >= 18 * 60) & (start <= 6 * 60)
        inverted = has_prev & (start < prev_arrival - _START_TIME_TOLERANCE) & ~wrap
        earlier = has_prev & (start < prev_start) & ~inverted & ~wrap
        same_minute = has_prev & (start == prev_start) & ~inverted
        overlapped = has_prev & (start > prev_start) & (start < prev_arrival) & ~inverted
    return {
        "hora_invertida": inverted,
        "hora_nocturna": wrap,
        "hora_anterior": earlier,
        "hora_repetida": same_minute,
        "hora_traslapada": overlapped,
        "origen_discontinuo": has_prev & (origin != prev_dest.astype(str)),
        "regreso_sin_llegar": is_return & (dest != home),
        "tipo_destino_dudoso": is_return & (dest == home) & kind.notna().to_numpy() & (kind != HOME_PLACE).to_numpy(),
        "inicio_fuera_de_casa": first & ~from_home,
        "inicio_zona_ajena": first & from_home & (origin != home),
        "fin_fuera_de_casa": last & ~(is_return & (dest == home)),
        "actividad_en_casa": ~is_return & (kind == HOME_PLACE).to_numpy(),
        "motivo_guarderia": (trips.motivo_viaje == "Guardería").to_numpy(),
    }


def _home_zone(trips: pd.DataFrame, viv: pd.DataFrame) -> np.ndarray:
    """The household's zone id on every trip row."""
    return viv.ageb.astype(str).reindex(trips.index.get_level_values("folio_vivienda")).to_numpy()


def clean_trip_chains(
    trips: pd.DataFrame, viv: pd.DataFrame, legs: pd.DataFrame | None = None, hab: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict]:
    """Apply the chain rules to a trip table in row (folio_viaje) order; drop nothing, flag everything.

    In order: impute the start time, motive and destination type of the trips
    that lost their questionnaire block, from the duplicate return that
    follows or from the nearest timed trips; recode the 'Regresar a Casa'
    trips that did not reach the household's zone from their destination
    type; mark the returns home made while already at home as non-trips; make
    trip that follows a return home start in the home zone; repair mistyped
    start hours with the fewest edits that let every trip start after the
    previous one arrived; then mark every defect left — start times, anchors,
    zones and purposes (``_remaining_issues``) — so that the trips carrying
    an error are exactly the trips with a non-empty ``problemas``. The order
    matters: the recode reads the imputed motives, the at-home walk reads the
    recoded ones, and the last two rules and the marking run over the chain
    without its non-trips. Each rule's evidence is in its docstring. ``viv`` supplies the household zone (``ageb``); ``hab`` the
    person features of the imputation (age, sex, occupation; skipped when
    None); ``legs`` the travel minutes when the table no longer carries the
    traslado columns.

    Returns the trips — every row kept, ``folio_viaje`` untouched — with two
    columns added: ``ajustes`` names what changed on the row and ``problemas``
    what is still wrong with it, ';'-joined codes from ``FIX_CODES`` and
    ``ISSUE_CODES``, "" where nothing — and a dict of counts: ``untimed_trips``,
    ``untimed_persons``, ``times_from_duplicate``, ``times_from_neighbours``,
    ``motives_from_duplicate``, ``motives_from_neighbours``,
    ``recoded_returns``, ``home_to_home``, ``duplicate_returns``,
    ``origins_repaired``, ``start_times_edited``, ``chains_repaired``, and
    one ``left_<code>`` entry per issue left.
    """
    trips = trips.sort_index()
    n = len(trips)
    fixes = np.full(n, "", dtype=object)
    issues = np.full(n, "", dtype=object)
    home = _home_zone(trips, viv)
    untimed = (trips.hora_inicio_h.isna() | trips.hora_inicio_m.isna()).to_numpy()

    trips, imputed = _impute_untimed_trips(trips, home, hab, legs)
    for key, code in (("time_from_duplicate", "hora:duplicado"), ("motive_from_duplicate", "motivo:duplicado"),
                      ("time_from_neighbours", "hora:vecinos"), ("motive_from_neighbours", "motivo:vecinos")):
        _add_code(fixes, imputed[key], code)
    _add_code(issues, imputed["duplicate"], "regreso_duplicado")

    trips, recoded = _recode_returns_by_destination(trips, home)
    _add_code(fixes, recoded, "motivo:tipo_destino")
    at_home = _home_to_home_returns(trips, skip=imputed["duplicate"])
    _add_code(issues, at_home, "regreso_en_casa")

    is_trip = ~(at_home | imputed["duplicate"])
    where = np.flatnonzero(is_trip)
    chain = trips[is_trip]
    chain, repaired = _start_from_home_after_return(chain, home[is_trip])
    _add_code(fixes, _expand(repaired, where, n), "origen:casa")
    locked = (imputed["time_from_duplicate"] | imputed["time_from_neighbours"])[is_trip]
    chain, edit, chains = _repair_start_times(chain, legs, locked)
    for name in {e for e in edit if e}:
        _add_code(fixes, _expand(edit == name, where, n), f"hora:{name}")
    trips = trips.copy()
    trips.loc[chain.index, "origen"] = chain.origen.to_numpy()
    trips.loc[chain.index, "hora_inicio_h"] = chain.hora_inicio_h.to_numpy()
    left = _remaining_issues(chain, home[is_trip], legs)
    for code, mask in left.items():
        _add_code(issues, _expand(mask, where, n), code)
    trips[FIX_FLAG] = fixes
    trips[ISSUE_FLAG] = issues

    counts = {
        "untimed_trips": int(untimed.sum()),
        "untimed_persons": int(trips.index[untimed].droplevel("folio_viaje").nunique()),
        "times_from_duplicate": int(imputed["time_from_duplicate"].sum()),
        "times_from_neighbours": int(imputed["time_from_neighbours"].sum()),
        "motives_from_duplicate": int(imputed["motive_from_duplicate"].sum()),
        "motives_from_neighbours": int(imputed["motive_from_neighbours"].sum()),
        "recoded_returns": int(recoded.sum()),
        "home_to_home": int(at_home.sum()),
        "duplicate_returns": int(imputed["duplicate"].sum()),
        "origins_repaired": int(repaired.sum()),
        "start_times_edited": int((edit != "").sum()),
        "chains_repaired": chains,
        **{f"left_{code}": int(mask.sum()) for code, mask in left.items()},
    }
    return trips, counts


def _expand(mask: np.ndarray, where: np.ndarray, n: int) -> np.ndarray:
    """A mask over a sub-table lifted to the full table: ``where`` holds the sub-table's row positions."""
    full = np.zeros(n, dtype=bool)
    full[where[np.asarray(mask, dtype=bool)]] = True
    return full


# --------------------------------------------------------------- repeated diaries


def _as_text(df: pd.DataFrame) -> pd.Series:
    """The rows of ``df`` joined with '|', missing values as empty strings, categoricals as their labels."""
    cols = [df[c].astype(object).where(df[c].notna(), "").astype(str) for c in df.columns]
    out = cols[0]
    for c in cols[1:]:
        out = out + "|" + c
    return out


def _diary_signatures(trips: pd.DataFrame, legs: pd.DataFrame | None) -> pd.DataFrame:
    """One row per trip: its exact signature, whether a leg is not on foot, and its start minute."""
    sig = _as_text(trips[["origen", "destino", "motivo_viaje", "tipo_lugar_destino", "n_traslados"]])
    mode_cols = [f"traslado{i}_medio" for i in range(1, 6)]
    if "traslado1_medio" in trips.columns:
        leg_cols = [f"traslado{i}_{k}" for i in range(1, 6) for k in ("medio", "min", "pago")]
        sig = sig + "|" + _as_text(trips[leg_cols])
        modes = trips[mode_cols]
        moving = ((modes.astype(object) != "A PIE") & modes.notna()).any(axis=1).to_numpy()
    else:
        if legs is None:
            raise ValueError("trips has no traslado columns; pass legs")
        leg_text = _as_text(legs[["traslado_medio", "traslado_min", "traslado_pago"]])
        by_trip = leg_text.groupby(level=trips.index.names, sort=False).agg("|".join)
        sig = sig + "|" + by_trip.reindex(trips.index).fillna("")
        moving = (
            (legs.traslado_medio.astype(object) != "A PIE")
            .groupby(level=trips.index.names, sort=False).any()
            .reindex(trips.index).fillna(False).to_numpy()
        )
    start = trips.hora_inicio_h.astype(float) * 60 + trips.hora_inicio_m.astype(float)
    return pd.DataFrame({"sig": sig, "moving": moving, "start": start.to_numpy()}, index=trips.index)


def flag_repeated_diaries(
    trips: pd.DataFrame, hab: pd.DataFrame, viv: pd.DataFrame, legs: pd.DataFrame | None = None
) -> pd.Series:
    """True for every person whose diary repeats in another household of the same block.

    A diary is the person's trips in row order, each read as its origin and
    destination zone, motive, destination place type and every leg's mode,
    minutes and fare; start times are compared separately. A person is flagged
    when a person in another household of the same AGEB and interview date has
    the same diary with every start time within ten minutes of theirs. Diaries
    of one trip, or made only of walks, are left out: they repeat by chance.

    Evidence (reports/duplicate_diaries.qmd): of 37,878 diaries of two or more
    trips with a leg not on foot, some 2,360 repeat in another household, all in
    the same AGEB and 99% on the same date. For two thirds of them every start
    time is one to ten minutes from the twin's, the differences spread evenly
    over that range and stopping at ten; within a household, where shared trips
    are real, 69% of paired trips agree to the minute. The twin is a different
    person in 87% of the pairs. The pattern is a diary copied onto another
    record and nudged; which record is the copy cannot be told, so both are
    flagged. Nothing is dropped: the flagged persons' trips are 3.5% of the
    weighted total and the expansion factors reconcile only on the file as
    shipped. Pairs whose times agree exactly are flagged too, although some of
    them may be genuine shared travel across two dwellings.

    ``trips`` may carry the ``traslado*`` columns or, after the legs unpivot,
    be paired with ``legs``. ``viv`` supplies the AGEB and ``hab`` the
    interview date. Returns a boolean Series over ``hab.index``.
    """
    d = _diary_signatures(trips.sort_index(), legs)
    person = d.groupby(level=PERSON, sort=False)
    diary = pd.DataFrame({
        "sig": person.sig.agg(" ## ".join),
        "n": person.size(),
        "moving": person.moving.any(),
        "starts": person.start.agg(list),
    })
    diary["ageb"] = viv.ageb.reindex(diary.index.get_level_values("folio_vivienda")).to_numpy()
    diary["fecha"] = hab.fecha.reindex(diary.index).to_numpy()
    diary = diary[(diary.n >= 2) & diary.moving]
    flagged = pd.Series(False, index=hab.index, name=DIARY_FLAG)
    for _, group in diary.groupby(["ageb", "fecha", "sig"], sort=False, observed=True):
        if group.index.get_level_values("folio_vivienda").nunique() < 2:
            continue
        keys = list(group.index)
        starts = [np.asarray(s, dtype=float) for s in group.starts]
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                if a[0] == b[0]:
                    continue
                offsets = np.abs(starts[i] - starts[j])
                if not np.isnan(offsets).any() and offsets.max() <= _DIARY_TWIN_MAX_OFFSET:
                    flagged.loc[a] = True
                    break
    return flagged


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

    By default the trip chains are cleaned (:func:`clean_trip_chains`) and
    nothing is dropped: the 325 trips with no start time (and no motive) are
    imputed — 37 from the home-to-home return that duplicates them, the rest
    from their nearest timed trips — as are the 7 timed trips with no motive;
    84 mislabelled 'Regresar a Casa' trips take their destination type's
    motive; 491 returns home made from home and the 38 duplicates are kept
    and marked as non-trips; 124 trips that follow a return home start in the
    home zone; and 2,032 mistyped start hours in 1,552 chains are repaired.
    ``trips`` gains two columns: ``ajustes`` names
    what changed on each row and ``problemas`` what is still wrong with it
    (';'-joined codes, see ``FIX_CODES`` and ``ISSUE_CODES``; ``non_trips()``
    picks out the rows the model build leaves out). ``folio_viaje`` and
    ``hab.viajes_contados`` are left as shipped. Pass ``clean_chains=False``
    for the survey as shipped — the expansion factors reconcile to the
    published totals on either, since no row leaves.

    Either way ``hab`` carries a boolean ``diario_repetido`` column
    (:func:`flag_repeated_diaries`): True for the persons whose whole diary is
    also the diary of a person in another household of the same AGEB and
    interview date, with every start time within ten minutes. They are copies
    with nudged times, not shared travel, and are flagged rather than dropped.

    ``legs`` carries the fare as reported (``traslado_pago``): nine bus legs
    have none, seventeen walking legs have one, and 145 legs of 500 pesos or
    more are a day's fuel or a month's parking rather than a fare. Nothing in
    the package reads it.

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

    # Flag the diaries that repeat in another household, on the survey as shipped:
    # the chain rules below edit start times and motives, and the flag reads both.
    df_hab[DIARY_FLAG] = flag_repeated_diaries(df_trips, df_hab, df_viv)
    log.info("repeated diaries flagged: %d persons", int(df_hab[DIARY_FLAG].sum()))

    if clean_chains:
        df_trips, counts = clean_trip_chains(df_trips, df_viv, hab=df_hab)
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
