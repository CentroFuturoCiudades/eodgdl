"""The trip-chain rules of the EOD survey: impute, repair, mark, and the code vocabulary they write.

``clean_trip_chains`` applies the rules to a trip table in chain order and
``mark_issues`` recomputes the ``problemas`` column on its own; ``load_eod``
(:mod:`eodgdl.eod`) calls the first, :mod:`eodgdl.review` the second. Every
change a rule makes is a code from ``FIX_CODES`` in the row's ``ajustes`` and
every defect left a code from ``ISSUE_CODES`` in its ``problemas``; each
rule's evidence is its docstring and ``reports/trip_chains.qmd`` the
diagnosis behind it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERSON = ["folio_vivienda", "folio_habitante"]

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
WORK_MOTIVE = "Trabajar"
MANDATORY_MOTIVES = (WORK_MOTIVE, "Estudiar")
OTHER = "Otros (especifique)"  # the survey's 'other' level, for motives and place types alike
# 'Guardería' names both attending daycare and taking a child there; the survey
# interviews ages 6 and over, so from ESCORT_FROM_AGE on the traveller is the
# escort (_daycare_escorts; the model build reads the age the same way).
DAYCARE = "Guardería"
ESCORT = "Llevar o recoger a alguien"
ESCORT_FROM_AGE = 12
# Place types where a day can have started — somebody spends the night there —
# so a first trip that names one as where the day started is not read as
# starting at home (_first_origins_at_home, origen:tipo_copia). Decided
# 2026-09-26 in the fourth review of the chains.
_NIGHT_PLACES = (HOME_PLACE, "Otra vivienda", "Hospital, clínica, consultorio, laboratorio clínico", OTHER)

# Start-time repair. Each entry is one way a start hour can have been mistyped,
# with the cost the search minimises: (name, cost, lowest hour it applies to,
# highest hour, hours added). The costs rank the mechanisms by how common they
# are — the 12-hour clock is the dominant one — and break ties between readings
# that would otherwise be equally cheap. Decided 2026-09-03 from the diagnosis
# behind _repair_start_times().
_START_TIME_EDITS = (
    ("+12h", 1.0, 0, 11, 12),  # 12-hour clock without AM/PM
    ("-10h", 1.2, 15, 19, -10),  # an extra leading 1, only where it yields a morning hour
    ("+10h", 1.5, 0, 9, 10),  # a missing leading 1
    ("-10h+12h", 2.0, 10, 19, 2),  # an extra leading 1 on a 12-hour-clock entry
)
# The second menu, tried only on a chain the first cannot read: a morning start
# typed on the afternoon side of the 12-hour clock (21:00 for 09:00), from 17:00
# on only, so no reading lands before 05:00. Decided 2026-09-26 in the fourth
# review of the chains (see _repair_start_times).
_START_TIME_EDITS_2 = _START_TIME_EDITS + (("-12h", 1.1, 17, 23, -12),)
_START_TIME_TOLERANCE = 15  # minutes: an inversion this small is minute noise, not an hour error
_START_TIME_MAX_COST = 6.0  # give up rather than rewrite a day
_MIN_MANDATORY_STAY = 30  # minutes: a reading of the hours may leave no less at work or school
# A return home that starts before _NIGHT_SHIFT_RETURN after a trip to work that
# started at 18:00 or later is the next morning's: the end of a night shift
# (_next_day). Decided 2026-09-26 in the fourth review of the chains.
_NIGHT_SHIFT_RETURN = 10 * 60
_DAY = 24 * 60  # minutes: a day read past midnight ends before its first start comes round again (_day_wraps)
# A start that falls before the previous trip's arrival by this much or less,
# where the typo search reads no hour, moves to the arrival (_start_at_arrival).
_SLIDE_LIMIT = 60  # minutes
# A day whose first trip, to anything but work, starts before _EARLY_START and is
# followed by a stay of _EARLY_LONG_STAY or more at that place is a mistyped hour
# the order of the times cannot show (see _repair_start_times). Decided
# 2026-09-26 in the third review of the chains.
_EARLY_START = 5 * 60  # minutes from midnight
_EARLY_LONG_STAY = 8 * 60  # minutes
# A return home that starts _LONG_ERRAND or more after the arrival at a short
# errand — shopping, the doctor, an office errand, a payment, an escort — in a
# chain whose times hold is a morning return typed on the afternoon side of the
# 12-hour clock: the search reads such a chain with the -12h edit alone
# (_LONG_ERRAND_EDITS; see _repair_start_times). A visit or an outing can last
# all day and is left, and so is 'Otros', which names day trips out of town as
# well as errands. Decided 2026-09-26 in a review of the long errands.
_SHORT_ERRANDS = (
    "Compras (comida)",
    "Compras (bienes, productos y servicios)",
    "Al médico o atención de salud",
    "Hacer un trámite",
    "Pagar algún servicio o al banco",
    ESCORT,
)
_LONG_ERRAND = 12 * 60  # minutes
_LONG_ERRAND_EDITS = tuple(edit for edit in _START_TIME_EDITS_2 if edit[0] == "-12h")
# 23:01 in minutes: the start the survey heaps its late returns on (see
# _remaining_issues, hora_2301). Found 2026-09-25 in a review of the chains.
_SENTINEL_START = 23 * 60 + 1
# The capture's start for a return that repeats the trip before it the other way:
# that trip's start plus 1h01 or 5h01 (see _remaining_issues, hora_1h01 and
# hora_5h01). Found 2026-09-26 in a review of the heaped returns.
_DEFAULT_RETURN_GAPS = {"hora_1h01": 60 + 1, "hora_5h01": 5 * 60 + 1}

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
# implies, and drop only the 38 home-to-home rows that duplicate an imputed
# return: every change is written to the row's FIX_FLAG
# column and everything left wrong to its ISSUE_FLAG column, so a consumer can
# keep, filter or weight the rows as it sees fit (eodgdl.tasha.build leaves out
# the rows marked as not being trips; tasha.chain_report counts the rest).

FIX_FLAG = "ajustes"  # column load_eod adds to trips: what it changed on the row, ';'-joined, "" if nothing
ISSUE_FLAG = "problemas"  # column load_eod adds to trips: what is still wrong with the row, ';'-joined, "" if nothing

# The vocabulary of the two columns. Each code names the field it touches and
# where the value came from (fixes) or the defect left (issues); the docstring
# of the rule that writes it is the evidence.
FIX_CODES = {
    "hora:duplicado": "start time taken from the home-to-home return that repeats this untimed return; "
    "that duplicate row is dropped",
    "hora:vecinos": "start time imputed from the nearest timed trips: the next trip's start less this "
    "trip's leg minutes and the donors' median activity duration",
    "motivo:duplicado": "motive and destination type taken from the home-to-home return that repeats this row; "
    "that duplicate row is dropped",
    "motivo:vecinos": "motive and destination type imputed from the nearest timed trips",
    "motivo:tipo_destino": "a 'Regresar a Casa' that did not reach the home zone, recoded from its destination type",
    "destino:copia": "destination set to the home zone: a 'Regresar a Casa' was recorded ending in its own origin's "
    "AGEB, away from home",
    "motivo:casa": "an activity trip recorded arriving at 'Su casa' in the home zone from another zone: recoded "
    "'Regresar a Casa'",
    "motivo:lugar": "a 'Regresar a Casa' that ends away from home, not made from home: the motive and place type the "
    "person's own trips give that place, or their day-start answer where the day started there",
    "motivo:guarderia": "a 'Guardería' trip made from age 12: recoded 'Llevar o recoger a alguien', the escort",
    "destino:casa": "destination set to the home zone: the day's last 'Regresar a Casa' ends in an AGEB that no other "
    "trip of the day touches, the home geocoded in another AGEB",
    "motivo:otros": "a 'Regresar a Casa' that ends away from home with nothing to say where it went: motive and place "
    "type 'Otros (especifique)', the O the model build gave it already",
    "origen:casa": "origin set to the home zone: the previous trip was a return home that reached it",
    "origen:anterior": "origin set to the previous trip's destination: the trip was recorded starting elsewhere",
    "origen:copia": "origin set to the home zone: the day's first trip was recorded leaving its own destination's AGEB",
    "origen:respuesta": "origin set to the home zone: the day's first trip leaves another zone, but the person "
    "answered that the day started at home",
    "origen:tipo_copia": "origin set to the home zone: the day's first trip was recorded with its destination's place "
    "type as where the day started",
    "origen:rotado": "origin set to the home zone: the day's first trip was recorded from a place to home and the next "
    "trip leaves from home, so the first went from home to that place (destino:rotado) and the next leaves from there "
    "(origen:anterior)",
    "destino:rotado": "destination set to the place the day's first trip was recorded leaving (origen:rotado)",
    "tipo_destino:rotado": "destination type set to the person's answer to where the day started, on a rotated first "
    "trip (origen:rotado) recorded arriving at 'Su casa': the answer named the place it went to",
    "tipo_origen:copia": "the person's answer to where the day started set to 'Su casa', with origen:copia, "
    "origen:tipo_copia or origen:rotado",
    "tipo_origen:regreso": "the person's answer to where the day started, 'Su casa' on a day whose first trip returns "
    "home from another AGEB, set to the place type trips to that AGEB report",
    "hora:+12h": "start hour read on a 12-hour clock: 12 hours added",
    "hora:-12h": "start hour read on a 12-hour clock the other way: 12 hours removed, a morning start typed as an "
    "evening one",
    "hora:-10h": "start hour read with an extra leading 1: 10 hours removed",
    "hora:+10h": "start hour read with a missing leading 1: 10 hours added",
    "hora:-10h+12h": "start hour read with an extra leading 1 on a 12-hour-clock entry: 2 hours added",
    "hora:llegada": "start moved to the minute the previous trip arrives: it started before, by no more than the "
    "tolerance (the rounding of the leg minutes), or by up to an hour where no reading of the hours fits, and "
    "the model cannot schedule an overlap",
    # read before the rules from eodgdl/revisions/leg_minutes.csv.gz, a review round's correction of the survey's minutes
    "minutos:revision": "travel minutes corrected by a review round before the rules read them "
    "(eodgdl/revisions/leg_minutes.csv.gz), spread over the trip's legs in proportion",
    # set by hand on a review sheet (eodgdl.review): the sheet's note column is the evidence
    "hora:revision": "start time set by hand on a review sheet",
    "motivo:revision": "motive set by hand on a review sheet",
    "tipo_destino:revision": "destination type set by hand on a review sheet",
    "tipo_origen:revision": "origin type set by hand on a review sheet",
    "origen:revision": "origin zone set by hand on a review sheet",
    "destino:revision": "destination zone set by hand on a review sheet",
    "modo:revision": "main mode set by hand on a review sheet",
    "fila:revision": "a row the chain rules dropped (a duplicate return), restored by hand on a review sheet",
}
ISSUE_CODES = {
    "regreso_en_casa": "a 'Regresar a Casa' made while already at home: not a trip; the model build leaves it out",
    "regreso_dia_siguiente": "the next morning's return from a night shift, the last row of a day that opens with "
    "the return from that job: not a trip of the survey day; the model build leaves it out",
    "hora_invertida": "starts before the previous trip could have arrived, beyond the tolerance and not "
    "overnight, and no reading of the hours or move to the arrival repairs it; a return read overnight is not "
    "overnight when the day goes to work or school again after it, or would last 24 hours",
    "origen_discontinuo": "does not start in the zone the previous trip ended in: a hand edit left it so",
    "regreso_sin_llegar": "a 'Regresar a Casa' whose destination zone is not the household's: the rules read every "
    "one (a motive, or home), so a hand edit left it so",
    "tipo_destino_dudoso": "a 'Regresar a Casa' that reached the home zone but reports a destination type "
    "that is not a home; the motive is kept",
    "hora_nocturna": "starts by 06:00 after a trip that started at or after 18:00, is a return home by 06:00 "
    "after a trip that started at or after noon, or a return home before 10:00 after a trip to work that "
    "started at or after 18:00, and no trip to work or school follows it and the day ends within 24 hours of "
    "its first start: read as overnight and left alone; a night or second shift or a mistyped hour, "
    "nothing says which. The day passes midnight here: the model build counts this trip's hours and the next "
    "ones' past 2400 (days_past_midnight)",
    "hora_anterior": "starts before the previous trip's start but within the tolerance of its arrival, so "
    "not repaired: minute noise, which the model cannot schedule",
    "hora_repetida": "starts at the same minute as the previous trip, whose legs fit within the tolerance "
    "(beyond it the row is hora_invertida instead); the model cannot schedule it",
    "hora_traslapada": "starts after the previous trip's start but before its reported arrival, by no more "
    "than the tolerance: the leg minutes overrun the next start; rounding, not an hour error, but an "
    "overlap the model cannot schedule",
    "inicio_fuera_de_casa": "the day's first trip does not leave from 'Su casa'",
    "inicio_zona_ajena": "the day's first trip leaves from 'Su casa' but not from the household's zone; the rules move "
    "such a trip home, or read it as rotated where it goes to the home zone itself and the next trip leaves from "
    "home, so a hand edit, or a next trip that leaves from elsewhere, left it so",
    "fin_fuera_de_casa": "the day's last trip is not a 'Regresar a Casa' that reaches the household's zone",
    "actividad_en_casa": "an activity motive with destination type 'Su casa': work from home, or a return "
    "home mislabelled the other way round",
    "motivo_guarderia": "a 'Guardería' motive, a child under 12 attending daycare: the model reads it as school",
    "hora_2301": "starts at 23:01 as the survey reports it, the minute it heaps the day's late returns on: most came "
    "then or later, perhaps after midnight, and a few are the capture's own start for a return (hora_1h01 or "
    "hora_5h01 beside it); the value is kept",
    "hora_1h01": "starts 1h01 after the trip before it, which it repeats the other way (the zones reversed, the same "
    "main mode and minutes), both starts as the survey reports them: a start the capture filled in, most of them on "
    "the evening walks to the shops of April 2023; when the return came is unknown, and the value is kept",
    "hora_5h01": "starts 5h01 after the trip before it, which it repeats the other way (the zones reversed, the same "
    "main mode and minutes), both starts as the survey reports them: the capture's default start for a return, left "
    "as it was; when the return came is unknown, and the value is kept",
    "hora_madrugada": "the day's first trip, to anything but work, starts before 05:00: an early market trip or "
    "escort, or a 12-hour-clock entry the rules could not read (they read the ones a stay of 8 hours or more "
    "follows)",
}
NON_TRIP_ISSUES = ("regreso_en_casa", "regreso_dia_siguiente")  # rows kept in trips that are not trips
# The codes that say how a start relates to the previous trip's; a row carries at
# most one of them. hora_2301, hora_1h01, hora_5h01 and hora_madrugada are about the
# reported value itself and may sit beside one.
TIME_ORDER_ISSUES = (
    "hora_invertida",
    "hora_nocturna",
    "hora_anterior",
    "hora_repetida",
    "hora_traslapada",
)
# ISSUE_CODES in two halves, for the hand review. A breaking issue leaves the
# chain inconsistent — a trip starts before the one it follows or before the
# previous one arrives, a trip end sits in two places, a return home never
# reaches home, home is named where the home zone is not — and is resolved by
# hand before the chain is modelled. TASHA schedules each activity between one
# trip's arrival and the next trip's start and cannot take an overlap, even one
# of a few minutes of rounding (decided 2026-09-26), so every time-order code
# but the overnight one is breaking; the rules move the starts that overlap by
# the tolerance or less to the arrival themselves (_start_at_arrival), so of
# those codes only a hand edit leaves one. Since the fourth review of the
# chains (2026-09-26) the rules also close every origin break, so
# origen_discontinuo too comes only from a hand edit, and nearly every day that
# says it started at home in another zone; since the fifth (the same day) they
# also read every return that stops short of home and the rotated first trips,
# so what the rules leave for the hand is hora_invertida. A tolerated issue leaves a
# consistent chain: a day that starts or ends away from home, an overnight trip
# read as such, a motive and a place type that disagree while the zones agree,
# a return from home the model build leaves out, an early start.
BREAKING_ISSUES = (
    "hora_invertida",
    "hora_anterior",
    "hora_repetida",
    "hora_traslapada",
    "origen_discontinuo",
    "regreso_sin_llegar",
    "inicio_zona_ajena",
)
TOLERATED_ISSUES = tuple(c for c in ISSUE_CODES if c not in BREAKING_ISSUES)
ISSUE_GROUPS = {"breaking": BREAKING_ISSUES, "tolerated": TOLERATED_ISSUES}

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
    "modo": 1.5,
    "sexo": 0.5,
    "ocupacion": 1.0,
    "motivo_propio": 2.0,
    # boolean: same origin and destination zone; leaves from the home zone; the next
    # row is a return home; first trip of the day; a work or school trip earlier in the day
    "intrazonal": 1.0,
    "desde_casa": 1.0,
    "siguiente_regreso": 1.0,
    "primero": 1.0,
    "trabajo_previo": 1.0,
    # numeric: log(1 + leg minutes); the next trip's start in hours; age in decades
    "log_min": 1.0,
    "hora_siguiente": 1.0,
    "edad": 1.0,
}
_IMPUTE_CATEGORICAL = ("modo", "sexo", "ocupacion", "motivo_propio")
_IMPUTE_BOOLEAN = (
    "intrazonal",
    "desde_casa",
    "siguiente_regreso",
    "primero",
    "trabajo_previo",
)
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
    return flags.astype(str).map(lambda s: code in s.split(";")).astype(bool)


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
    return (
        trips.hora_inicio_h.astype(float) * 60 + trips.hora_inicio_m.astype(float)
    ).to_numpy()


def _shift(values: np.ndarray, pid: np.ndarray, by: int):
    """``values`` shifted by ``by`` rows within each person; None/NaN across persons."""
    out = pd.Series(values).groupby(pid).shift(by)
    return out.to_numpy()


def _spread(minutes, total: int) -> list[int]:
    """``total`` whole minutes over legs in proportion to their ``minutes``, one at least each (largest remainder)."""
    m = np.asarray(minutes, dtype=float)
    extra = total - len(m)  # every leg keeps a minute; the rest goes by share
    share = m / m.sum() * extra
    out = np.floor(share).astype(int)
    out[np.argsort(-(share - out), kind="stable")[: extra - out.sum()]] += 1
    return (out + 1).tolist()


def _correct_minutes(
    trips: pd.DataFrame, legs: pd.DataFrame | None, minutes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray]:
    """The travel minutes a review round corrected, put in before the rules read them.

    ``minutes`` is the table ``eodgdl/revisions/leg_minutes.csv.gz`` holds
    (``household, person, trip, before, after``, see its README): a trip's
    reported minutes summed over its legs, and the minutes it takes instead,
    spread over its legs in proportion to what each reported, a minute each at
    least (:func:`_spread`). They are corrections of the survey's own values,
    like the three leg modes of ``eodgdl.eod._VIAJES_MEDIO_FIXES``, and come
    before the rules because the rules read the minutes: a trip's arrival, which
    the typo search and the move to the arrival test every start against, is its
    start plus its minutes. Until 2026-09-26 (the fifth review of the chains)
    they were hand decisions applied after the rules, so the rules read the
    reported minutes: household 7425, person 3, reports 120 minutes for each of
    four trips to work and back, at 06:05, 12:32, 14:10 and 18:08, and the search
    read the last two as 16:10 and 20:08 to make room for them, while with round
    9's 60 minutes the survey's hours hold. On the survey 371 trips (432 legs)
    are corrected.

    The minutes are corrected in the trips' ``traslado{k}_min`` columns where the
    table carries them, else in a copy of ``legs``, which is returned: a caller
    that passes ``legs`` gets the corrected legs back. A trip the table lacks, or
    whose minutes are not ``before``, raises: the correction was made against
    other minutes. Returns the trips, the legs and the mask of the trips corrected.
    """
    trips = trips.copy()
    keys = [(int(h), int(p), int(t)) for h, p, t in zip(minutes.household, minutes.person, minutes.trip)]
    pos = trips.index.get_indexer(keys)
    bad = [f"household {k[0]}, person {k[1]}, trip {k[2]}: no such trip" for k, p in zip(keys, pos) if p < 0]
    cols = [c for c in trips.columns if c.startswith("traslado") and c.endswith("_min")]
    if not cols and legs is not None:
        legs = legs.copy()
    mask = np.zeros(len(trips), dtype=bool)
    for key, p, before, after in zip(keys, pos, minutes.before.astype(int), minutes.after.astype(int)):
        if p < 0:
            continue
        if cols:
            row = np.array([np.nan if pd.isna(v) else float(v) for v in trips.iloc[p][cols]])
            have = ~np.isnan(row)
            own, where = row[have], [trips.columns.get_loc(c) for c, h in zip(cols, have) if h]
        elif legs is not None:
            own = legs.loc[key, "traslado_min"].to_numpy(dtype=float)
        else:
            raise ValueError("the trips carry no travel minutes to correct: pass legs")
        if int(own.sum()) != before:
            bad.append(f"household {key[0]}, person {key[1]}, trip {key[2]}: made against {before} minutes, "
                       f"the table holds {int(own.sum())}")
            continue
        spread = _spread(own, after)
        if cols:
            for column, value in zip(where, spread):
                trips.iat[p, column] = value
        else:
            legs.loc[[key + (n,) for n in legs.loc[key].index], "traslado_min"] = spread
        mask[p] = True
    if bad:
        raise ValueError("travel minutes that cannot be corrected:\n  " + "\n  ".join(bad))
    return trips, legs, mask


def _duplicate_returns(
    trips: pd.DataFrame, home: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
    origin, dest = (
        trips.origen.astype(str).to_numpy(),
        trips.destino.astype(str).to_numpy(),
    )
    motive = trips.motivo_viaje.astype(object).to_numpy()
    next_is_home_return = (
        (_shift(motive, pid, -1) == HOME_MOTIVE)
        & (_shift(origin, pid, -1) == home)
        & (_shift(dest, pid, -1) == home)
        & ~np.isnan(_shift(start, pid, -1).astype(float))
    )
    target = pd.isna(motive) & (origin != home) & (dest == home) & next_is_home_return
    duplicate = np.r_[False, target[:-1]]
    return target, duplicate


def _trip_features(
    trips: pd.DataFrame,
    home: np.ndarray,
    hab: pd.DataFrame | None,
    legs: pd.DataFrame | None,
) -> pd.DataFrame:
    """One row per trip: the features the nearest-neighbour imputation compares, plus its arithmetic.

    Carries ``start``, ``next_start``, ``prev_arrival`` and ``legmin`` in minutes,
    ``activity`` (a timed trip with a motive other than 'Regresar a Casa') and
    ``motivo``/``tipo`` as strings, alongside the ``_IMPUTE_WEIGHTS`` features.
    The person features come from ``hab`` and are absent when it is None.
    """
    pid = _person_ids(trips)
    start = _start_minutes(trips)
    legmin = _leg_minutes(trips, legs)
    origin, dest = (
        trips.origen.astype(str).to_numpy(),
        trips.destino.astype(str).to_numpy(),
    )
    motive = trips.motivo_viaje.astype(object).to_numpy()
    activity = pd.notna(motive) & (motive != HOME_MOTIVE)
    mandatory = activity & pd.Series(motive).isin(["Trabajar", "Estudiar"]).to_numpy()
    seen_mandatory = (
        pd.Series(mandatory.astype(int)).groupby(pid).cumsum().to_numpy() - mandatory
    )
    person_key = pd.MultiIndex.from_arrays(
        [trips.index.get_level_values(0), trips.index.get_level_values(1), dest]
    )
    own = pd.Series(np.where(activity, motive, None), index=person_key, dtype=object)
    own = (
        own.groupby(level=[0, 1, 2], sort=False)
        .ffill()
        .groupby(level=[0, 1, 2], sort=False)
        .shift(1)
    )
    next_start = _shift(start, pid, -1).astype(float)
    f = pd.DataFrame(
        {
            "modo": trips.modo_principal.astype(str).to_numpy(),
            "motivo_propio": pd.Series(own.to_numpy(), dtype=object)
            .fillna("none")
            .astype(str)
            .to_numpy(),
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
        },
        index=trips.index,
    )
    if hab is not None:
        person = hab.reindex(trips.index.droplevel("folio_viaje"))
        f["sexo"] = person.sexo_nacimiento.astype(str).to_numpy()
        f["ocupacion"] = (
            person.ocupacion.astype(object).fillna("NA").astype(str).to_numpy()
        )
        f["edad"] = person.edad.to_numpy(dtype=float) / 10
    return f


def _donor_mask(f: pd.DataFrame) -> np.ndarray:
    """The rows of a ``_trip_features`` table that may donate: timed activity trips whose times hold up.

    A donor needs a timed successor, since its activity duration — the next
    start less its own start and leg minutes — is what the vote's median
    places an imputed start with. It must also be consistent on the face of
    the shipped times: a duration of at least zero, and a start no earlier
    than the previous trip's arrival less ``_START_TIME_TOLERANCE``. The
    imputation runs before the typo search, so a mistyped hour on the donor
    or on the trip after it would otherwise enter the vote as shipped: on
    the survey 2,407 of the 78,658 candidate activity trips have a negative
    duration (median 164 minutes, the 12-hour-clock and leading-1 typos the
    search later repairs on 1,210 of them and marks on the rest), and the
    filter leaves them out along with the trips that start before their
    predecessor could have arrived.
    """
    start, next_start, prev_arrival = (
        f.start.to_numpy(),
        f.next_start.to_numpy(),
        f.prev_arrival.to_numpy(),
    )
    with np.errstate(invalid="ignore"):
        duration_ok = (next_start - start - f.legmin.to_numpy()) >= 0
        start_ok = np.isnan(prev_arrival) | (
            start >= prev_arrival - _START_TIME_TOLERANCE
        )
    return (
        f.activity.to_numpy()
        & ~np.isnan(start)
        & ~np.isnan(next_start)
        & duration_ok
        & start_ok
    )


def _nearest_donors(
    targets: pd.DataFrame, donors: pd.DataFrame, k: int = _IMPUTE_NEIGHBOURS
) -> pd.DataFrame:
    """The k nearest donors' vote for every target row: motive, type, activity duration, start.

    Distance is the weighted mismatch count over the categorical and boolean
    features plus the weighted absolute differences over the numeric ones
    (``_IMPUTE_WEIGHTS``); a numeric feature the target lacks costs nothing.
    The motive is the most common among the k nearest, ties going to the
    closer set; the type is the most common among the donors that voted for
    it, and ``duration`` and ``start`` are their medians, in minutes.
    """
    categorical = [
        c for c in _IMPUTE_CATEGORICAL if c in donors.columns and c in targets.columns
    ]
    boolean = list(_IMPUTE_BOOLEAN)
    numeric = [
        c for c in _IMPUTE_NUMERIC if c in donors.columns and c in targets.columns
    ]
    col = {c: donors[c].to_numpy() for c in categorical + boolean + numeric}
    d_motive, d_type = donors.motivo.to_numpy(), donors.tipo.to_numpy()
    d_duration = np.clip(
        donors.next_start.to_numpy()
        - donors.start.to_numpy()
        - donors.legmin.to_numpy(),
        0,
        None,
    )
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
        tally = (
            pd.DataFrame({"m": d_motive[near], "d": d[near]})
            .groupby("m")
            .d.agg(["size", "sum"])
        )
        motive = tally.sort_values(["size", "sum"], ascending=[False, True]).index[0]
        vote = near[d_motive[near] == motive]
        rows.append(
            (
                motive,
                pd.Series(d_type[vote]).mode().iloc[0],
                float(np.median(d_duration[vote])),
                float(np.median(d_start[vote])),
            )
        )
    return pd.DataFrame(
        rows, columns=["motivo", "tipo", "duration", "start"], index=targets.index
    )


def _impute_untimed_trips(
    trips: pd.DataFrame,
    home: np.ndarray,
    hab: pd.DataFrame | None,
    legs: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fill the start time, motive and destination type of the trips that lost their questionnaire block.

    325 trips report neither hour nor minute; they also lack motivo_viaje and
    tipo_lugar_destino but carry zones, a mode and legs with minutes, so they
    are real trips whose questionnaire block was not captured (71% in April
    2023, the last month of fieldwork). None is a person's last trip, and 281
    persons hold them in 284 blocks. Seven more trips are timed but have no
    motive. The lost answers are imputed in two steps and every imputed row
    is marked in the ``ajustes`` column; the 38 duplicate rows of the first
    step are the only rows ``clean_trip_chains`` drops.

    First, the 38 motive-less returns home — a row from elsewhere into the
    home zone, 37 of them untimed — whose next row is a timed 'Regresar a
    Casa' from home to home take that row's motive and, where missing, its
    time (``motivo:duplicado``, ``hora:duplicado``): it is the return's own
    questionnaire block attached to a duplicate, and once every rule has run
    the duplicate is dropped (see ``_duplicate_returns``): its answers now
    sit on the return, and the person was already home.

    Second, every remaining row without a motive takes the vote of its 30
    nearest timed activity trips whose own times hold up (``_donor_mask``,
    ``_nearest_donors``, features in ``_IMPUTE_WEIGHTS``): the most common
    motive and, among those donors, the
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
    masks = {
        key: np.zeros(n, dtype=bool)
        for key in (
            "time_from_duplicate",
            "motive_from_duplicate",
            "duplicate",
            "time_from_neighbours",
            "motive_from_neighbours",
        )
    }
    target, duplicate = _duplicate_returns(trips, home)
    if target.any():
        untimed = np.isnan(_start_minutes(trips))
        pos = np.flatnonzero(target)
        for col in ("motivo_viaje", "tipo_lugar_destino"):
            trips.iloc[pos, trips.columns.get_loc(col)] = trips.iloc[pos + 1][
                col
            ].to_numpy()
        timed_pos = pos[untimed[pos]]
        for col in ("hora_inicio_h", "hora_inicio_m"):
            trips.iloc[timed_pos, trips.columns.get_loc(col)] = trips.iloc[
                timed_pos + 1
            ][col].to_numpy()
        masks["time_from_duplicate"] |= target & untimed
        masks["motive_from_duplicate"] |= target
        masks["duplicate"] |= duplicate

    for _ in range(20):  # blocks are filled back to front, one row of each per pass
        f = _trip_features(trips, home, hab, legs)
        untimed = np.isnan(f.start.to_numpy())
        no_motive = trips.motivo_viaje.isna().to_numpy()
        pending = untimed | no_motive
        if not pending.any():
            break
        next_untimed = np.isnan(f.next_start.to_numpy()) & (
            _shift(untimed, _person_ids(trips), -1) == True
        )
        ready = pending & ~next_untimed
        if not ready.any():
            ready = pending
        donors = f[_donor_mask(f)]
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
            start = np.where(
                np.isnan(latest), v.start.to_numpy(), latest - v.duration.to_numpy()
            )
            start = np.fmax(
                start, t.prev_arrival.to_numpy()
            )  # no earlier than the previous arrival
            start = np.where(np.isnan(latest), start, np.fmin(start, latest))
            start = np.clip(np.round(start), 0, 24 * 60 - 1).astype(int)
            trips.loc[t.index, "hora_inicio_h"] = pd.array(start // 60, dtype="Int64")
            trips.loc[t.index, "hora_inicio_m"] = pd.array(start % 60, dtype="Int64")
            masks["time_from_neighbours"][np.flatnonzero(ready)[need_time]] = True
    if trips.motivo_viaje.isna().any() or np.isnan(_start_minutes(trips)).any():
        raise ValueError("untimed trips left after 20 passes")
    return trips, masks


def _recode_returns_by_destination(
    trips: pd.DataFrame, home: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Give a 'Regresar a Casa' that did not reach the home zone its destination type's motive.

    See ``_MOTIVO_POR_TIPO_LUGAR_DESTINO``. The zone is the arbiter: where it
    says the trip did not go home, the motive is wrong and any activity beats
    it. Some of the 84 labels are probably stale too — 24 repeat the previous
    trip's type and 18 stay in its zone — but each is a trip that ended away
    from home. Returns the trips and the mask of the rows recoded.
    """
    kind = trips.tipo_lugar_destino
    away = (
        (trips.motivo_viaje == HOME_MOTIVE)
        & kind.notna()
        & (kind != HOME_PLACE)
        & (trips.destino.astype(str).to_numpy() != home)
    )
    if away.any():
        trips = trips.copy()
        trips["motivo_viaje"] = trips.motivo_viaje.mask(
            away, kind.map(_MOTIVO_POR_TIPO_LUGAR_DESTINO).fillna(OTHER)
        )
    return trips, away.to_numpy()


def _returns_copying_origin(trips: pd.DataFrame, home: np.ndarray) -> np.ndarray:
    """The 'Regresar a Casa' trips recorded ending in their own origin's AGEB, away from home: they went home.

    The destination is a copy of the origin, the mirror of the day's first
    trips recorded leaving their own destination (``_first_origins_at_home``):
    the motive and the place type ('Su casa') both say home, and only the
    copied AGEB says otherwise. Runs after ``_recode_returns_by_destination``,
    which reads a return typed as a place by its type first. On the survey 69
    returns are so recorded; the first hand pass moved every one of the 67 it
    reviewed home ("esa AGEB funciona como Su casa dentro de la cadena") and
    read none as an activity there. Made a rule on 2026-09-26 (the fourth
    review of the chains). Returns the mask of the returns whose destination
    becomes the home zone.
    """
    dest = trips.destino.astype(str).to_numpy()
    return (trips.motivo_viaje == HOME_MOTIVE).to_numpy() & (dest == trips.origen.astype(str).to_numpy()) & (dest != home)


def _arrivals_home(trips: pd.DataFrame, home: np.ndarray) -> np.ndarray:
    """The activity trips recorded arriving at 'Su casa' in the home zone from another zone: returns home.

    The zone and the place type both say the trip went home, and only the
    motive, the activity's, says otherwise — the return home typed with the
    motive of the place it came from, the mirror of the returns
    ``_recode_returns_by_destination`` recodes. Such trips nearly always follow
    an activity at their origin and precede a trip that leaves home, and where
    the next row repeats the return from home to home with the same mode and
    minutes, the at-home walk then marks that row a non-trip. On the survey
    231 trips after the day's first are so recorded; the hand passes had
    recoded 229 of the 270 including first trips (the first pass 95, the sixth
    134) and kept most of the rest as first trips recorded place to home whose
    place was really the day's first destination (the fifth pass's rotated
    records), which is why a first trip is left alone. Made a rule on
    2026-09-26 (the fourth review of the chains). Returns the mask of the trips
    recoded 'Regresar a Casa'.
    """
    motive = trips.motivo_viaje
    pid = _person_ids(trips)
    later = np.r_[False, pid[1:] == pid[:-1]]
    return (
        later
        & (motive.notna() & (motive != HOME_MOTIVE)).to_numpy()
        & (trips.tipo_lugar_destino == HOME_PLACE).to_numpy()
        & (trips.destino.astype(str).to_numpy() == home)
        & (trips.origen.astype(str).to_numpy() != home)
    )


def _daycare_escorts(trips: pd.DataFrame, hab: pd.DataFrame | None) -> np.ndarray:
    """The 'Guardería' trips made from ``ESCORT_FROM_AGE`` on: the traveller takes or fetches a child.

    The survey uses the one motive both for the child who attends daycare and
    for whoever takes the child there or brings it back; it interviews ages 6
    and over, so a traveller of 12 or more is the escort. 163 of the survey's
    'Guardería' trips are made from that age; the hand passes recoded every
    one as 'Llevar o recoger a alguien' (the first pass 118, the ninth round 46)
    and put the children's back (the seventh pass, 23), and the model build
    reads the age the same way. Made a rule on 2026-09-26 (the fourth review
    of the chains). Without ``hab`` (no ages) nothing is recoded. Returns the
    mask of the trips recoded.
    """
    if hab is None:
        return np.zeros(len(trips), dtype=bool)
    age = hab.edad.reindex(trips.index.droplevel("folio_viaje")).to_numpy(dtype=float)
    return (trips.motivo_viaje == DAYCARE).to_numpy() & (age >= ESCORT_FROM_AGE)


def _returns_elsewhere(trips: pd.DataFrame, home: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The motive and place type of a 'Regresar a Casa' that went somewhere else than home.

    A return that ends away from the home zone, in another AGEB than the one it
    left, and that no rule reads as reaching home (its type is 'Su casa', so
    ``_recode_returns_by_destination`` has nothing to go by; its destination is
    no copy of its origin, ``_returns_copying_origin``) went to a place. What
    the person does there is what their own trips to that AGEB do — the most
    common motive among them and that motive's most common place type — or,
    where no other trip goes there and the day started there, what the
    day-start answer names (its motive through ``_MOTIVO_POR_TIPO_LUGAR_DESTINO``,
    and the answer as the type). ``trips`` is the chain without the returns
    made from home (``_home_to_home_returns``), which are no trips at all. On
    the survey 67 returns are read so and 29 are left with nothing to go by,
    which ``_short_returns`` reads once the origins are continuous. The first hand pass recoded 87 such returns, 56
    of them to 'Trabajar' — 20 of those for students, homemakers or children,
    a default rather than a reading — and moved the others home. Made a rule on
    2026-09-26 (the fourth review of the chains). Returns the mask of the
    returns read and, over all rows, the motive and type they take (None
    elsewhere).
    """
    n = len(trips)
    pid = _person_ids(trips)
    dest, origin = trips.destino.astype(str).to_numpy(), trips.origen.astype(str).to_numpy()
    motive = trips.motivo_viaje.astype(object).to_numpy()
    kind = trips.tipo_lugar_destino.astype(object).to_numpy()
    answer = trips.tipo_lugar_origen.astype(object).to_numpy()
    target = (motive == HOME_MOTIVE) & (dest != home) & (dest != origin)
    new_motive = np.full(n, None, dtype=object)
    new_kind = np.full(n, None, dtype=object)
    if not target.any():
        return target, new_motive, new_kind
    first = pd.Series(np.arange(n)).groupby(pid).transform("min").to_numpy()
    visits = pd.DataFrame({"pid": pid, "dest": dest, "motive": motive, "kind": kind})
    visits = visits[pd.notna(visits.motive) & (visits.motive != HOME_MOTIVE)]
    by_place = {key: g for key, g in visits.groupby(["pid", "dest"], sort=False)}
    for i in np.flatnonzero(target):
        own = by_place.get((pid[i], dest[i]))
        if own is not None:
            top = own.motive.astype(str).value_counts()
            new_motive[i] = top.index[0]
            new_kind[i] = own[own.motive == top.index[0]].kind.astype(str).value_counts().index[0]
        elif origin[first[i]] == dest[i] and pd.notna(answer[i]) and answer[i] != HOME_PLACE:
            new_motive[i] = _MOTIVO_POR_TIPO_LUGAR_DESTINO.get(answer[i], OTHER)
            new_kind[i] = answer[i]
    return target & pd.notna(new_motive), new_motive, new_kind


def _next_morning_returns(trips: pd.DataFrame, home: np.ndarray) -> np.ndarray:
    """The last row of a night-shift day: the next morning's return from the job, outside the survey day.

    A night worker's diary opens with the morning return from the job — the
    day starts at work, and the person's answer names the workplace — and
    records the evening departure for it. When the day's last row is a return
    home from that workplace right after the trip to it that starts at 18:00
    or later, that row is the next morning's return: a repeat of the first one
    a day on, its hour usually an evening one (19:00–20:59, the morning's
    typed on the wrong side of the clock). The survey day ends at work. On
    the survey 75 days end so; passes 2 and 3 dropped the last row of every
    such day they reviewed (62), the convention a rule now applies to all.
    ``trips`` is the chain without the returns made from home; the row is kept
    and marked ``regreso_dia_siguiente``, and the model build leaves it out.
    Made a rule on 2026-09-26 (the fourth review of the chains). Returns the
    mask of the rows marked.
    """
    n = len(trips)
    pid = _person_ids(trips)
    start = _start_minutes(trips)
    origin, dest = trips.origen.astype(str).to_numpy(), trips.destino.astype(str).to_numpy()
    motive = trips.motivo_viaje.astype(object).to_numpy()
    last = np.r_[pid[1:] != pid[:-1], True]
    first = pd.Series(np.arange(n)).groupby(pid).transform("min").to_numpy()
    out = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(last):
        j, f = i - 1, first[i]
        if f >= j or motive[i] != HOME_MOTIVE or origin[i] == home[i]:
            continue
        if motive[j] == WORK_MOTIVE and dest[j] == origin[i] and start[j] >= 18 * 60 \
                and motive[f] == HOME_MOTIVE and origin[f] == origin[i]:
            out[i] = True
    return out


def _home_to_home_returns(
    trips: pd.DataFrame, home: np.ndarray, skip: np.ndarray | None = None
) -> np.ndarray:
    """Mark the 'return home' trips made by a person who is already at home.

    Walk each chain with an at-home flag: it starts as whether the first trip
    leaves from 'Su casa' and from the household's zone (``home``), and after
    each trip it is whether that trip was a 'Regresar a Casa' that reached the
    household's zone. A return home while the flag is set is not a trip; the
    flag stays set, so a run of such rows is marked whole. Rows in ``skip``
    (the duplicates of ``_duplicate_returns``) are stepped over as if absent.

    The origin type is one answer per person — the survey asks where the day
    started and repeats the answer on every row, so it is constant within all
    52,758 persons with trips — which is why the first row needs the zone as
    well: a first trip that leaves another zone was made away from home,
    whatever the answer says. Until 2026-09-25 the type alone started the
    flag, and 75 such first trips were marked; they are trips (66 start after
    noon, 30 by students: most are the return of a day whose trip out was not
    recorded) and now carry ``inicio_zona_ajena`` instead. For the same
    reason a return that stops short of the home zone does not put the person
    at home: until 2026-09-26 it did, and the return that followed it from
    that other zone — the person's actual way home, or a trip on from there —
    was marked; those 14 rows are trips now.

    556 rows in the shipped survey once the untimed trips are imputed and the
    motives recoded: 83 first trips coded 'Regresar a Casa' from home and 473
    second and later members of consecutive returns. 467 start and end in the
    home zone; many are the return recorded twice, the activity-motive row
    that ``_arrivals_home`` reads as the return and a 'Regresar a Casa' from
    home to home with the same mode and minutes after it (402 rows before that
    rule, 2026-09-26), and in the rest the leg minutes usually differ from
    the return they follow, so a within-zone errand coded as 'return home' is
    the likeliest reading. They are kept and marked ``regreso_en_casa``; the
    model build leaves them out. Returns the mask of the rows marked.
    """
    home = np.asarray(home, dtype=object)
    to_home = (trips.motivo_viaje == HOME_MOTIVE).to_numpy()
    reached = to_home & (trips.destino.astype(str).to_numpy() == home)
    from_home = (trips.tipo_lugar_origen == HOME_PLACE).to_numpy() & (
        trips.origen.astype(str).to_numpy() == home
    )
    skip = (
        np.zeros(len(trips), dtype=bool)
        if skip is None
        else np.asarray(skip, dtype=bool)
    )
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
            at_home = reached[i]
    return marked


def _mark_non_trips(trips: pd.DataFrame, home: np.ndarray, skip: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The rows of a chain table that are not trips, one mask per ``NON_TRIP_ISSUES`` code.

    The returns home made while already at home (``_home_to_home_returns``,
    ``regreso_en_casa``) and, over the chain without them, the next morning's
    return that closes a night-shift day (``_next_morning_returns``,
    ``regreso_dia_siguiente``). ``clean_trip_chains`` and ``mark_issues`` both
    mark through here, so a table the hand decisions changed is read the way
    the rules read theirs. Rows in ``skip`` are stepped over as if absent.
    """
    n = len(trips)
    skip = np.zeros(n, dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    at_home = _home_to_home_returns(trips, home, skip=skip)
    rest = np.flatnonzero(~(at_home | skip))
    next_morning = _expand(_next_morning_returns(trips.iloc[rest], home[rest]), rest, n)
    return {"regreso_en_casa": at_home, "regreso_dia_siguiente": next_morning}


def _start_where_previous_ended(
    trips: pd.DataFrame, home: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Make every trip start where the previous one ended: at home after a return that reached it, elsewhere at
    the previous destination.

    A trip whose origin is not where the previous trip ended breaks the chain.
    The breaks are wrong zones, not wrong sequence: of the 176 people with one,
    82 admit no continuous ordering at all and only one of the other 94 has an
    ordering the times agree with. In 147 of the 178 breaks the previous trip
    is a 'Regresar a Casa' that reached the household's zone, and in 95 of
    those the offending origin is a stop visited earlier in the day, as if
    carried over from it. The person was at home, so the origin becomes the
    home zone (``origen:casa``) — the reading a home-based activity model makes
    anyway. On the chain without its non-trips 147 breaks are of that kind.
    The other 31 — a return that stopped short of home, a trip recorded
    leaving home after one that went elsewhere, most of them a return home
    that was never recorded — were left marked ``origen_discontinuo`` until
    2026-09-26, since either side could be wrong; the first hand pass set the
    origin of every one of them to the previous trip's destination, and the
    fourth review of the chains made that a rule (``origen:anterior``): the
    row order is the chain, so the person set out from where the last trip
    left them. The origin takes the previous trip's destination AGEB and the
    zone the survey coded for it (``zona_origen`` from that trip's
    ``zona_destino``), so the AGEB and the zone stay one place. Returns the
    trips and the masks of the origins set at home and elsewhere.
    """
    prev_return = (trips.motivo_viaje == HOME_MOTIVE).groupby(level=PERSON).shift(1)
    prev_dest = trips.destino.astype(str).groupby(level=PERSON).shift(1)
    zoned = {"zona_origen", "zona_destino"} <= set(trips.columns)
    broken = prev_dest.notna().to_numpy() & (trips.origen.astype(str).to_numpy() != prev_dest.to_numpy())
    at_home = broken & prev_return.fillna(False).astype(bool).to_numpy() & (prev_dest.to_numpy() == home)
    if broken.any():
        trips = trips.copy()
        trips.loc[broken, "origen"] = prev_dest.to_numpy()[broken]
        if zoned:
            prev_zone = trips.zona_destino.astype(str).groupby(level=PERSON).shift(1)
            trips.loc[broken, "zona_origen"] = prev_zone.to_numpy()[broken]
    return trips, at_home, broken & ~at_home


def _first_origins_at_home(trips: pd.DataFrame, home: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The day's first trips that started at home, though recorded leaving another zone; three readings.

    **Copied origins** (``origen:copia``). A first trip that leaves the AGEB it
    goes to — an activity, not a return home — was recorded with its
    destination copied into its origin. On the chain without its non-trips 955
    first trips do, and for 361 of them the person's answer to where the day
    started names the destination's type as well. The survey's own evidence
    says the day started at home: the first hand pass moved every one of them
    home and set the answer to 'Su casa' on the 418 that gave another, and the
    fifth pass, weighing the first pass's first-trip readings against the
    trips' own minutes, found that the minutes of such copied origins fit a
    trip from home 87–92% of the time. Made a rule on 2026-09-26 (the third
    review of the chains), when the 1,373 hand decisions that applied it one
    by one were folded into it.

    **The answer says home** (``origen:respuesta``). A first trip whose person
    answered that the day started at 'Su casa' but that leaves another zone
    (``inicio_zona_ajena`` until then): the answer, put to the person once and
    repeated on every row, wins over the one AGEB. 86 first trips (and 16 that
    go to the home zone itself, below); the first hand pass had moved all 102
    home.

    **A copied place type** (``origen:tipo_copia``). A first trip whose person
    names the trip's own destination type as where the day started — a shop,
    a school, a factory, an office — copied the type as the first reading
    copied the AGEB. 151 first trips (and 13 that go to the home zone itself);
    the first hand pass moved 148 of the 160 it reviewed home, the fifth pass
    found the minutes of such trips fit a
    trip from home as often as the copied origins', and the median distance
    between the recorded origin and home is 1.4 km. A place where somebody
    spends the night (``_NIGHT_PLACES``: another dwelling, a hospital,
    'Otros') is left: the first pass moved 3 of the 20 other dwellings. The
    answer becomes 'Su casa' with it (``tipo_origen:copia``).

    The last two readings were made rules on 2026-09-26, in the fourth review
    of the chains. They leave a first trip that goes to the home zone itself:
    moved home it would go from home to home. It is the rotated record the
    fifth hand pass read — the first trip recorded from a place to home, the
    next from home onwards, so the place is where the first trip went —
    which ``_rotated_first_trips`` reads, or a day that started at that place.
    A first trip
    that is a return home is left alone too: from its own destination's AGEB
    it is the end of a night shift with a home geocoded one AGEB off, and from
    another AGEB a day that started away (``_first_return_answers``).
    ``trips`` is a chain table without its non-trips, sorted in chain order;
    returns the three masks of its rows to move home.
    """
    pid = _person_ids(trips)
    first = np.r_[True, pid[1:] != pid[:-1]][: len(trips)]
    origin, dest = trips.origen.astype(str).to_numpy(), trips.destino.astype(str).to_numpy()
    answer = trips.tipo_lugar_origen.astype(object).to_numpy()
    away = first & (origin != home) & (trips.motivo_viaje != HOME_MOTIVE).to_numpy()
    copied = away & (origin == dest)
    # a first trip that goes to the home zone would go from home to home: the rotated record, left to the review
    other = away & ~copied & (dest != home)
    answered = other & (answer == HOME_PLACE)
    kind = other & (answer == trips.tipo_lugar_destino.astype(object).to_numpy()) & ~np.isin(
        answer.astype(str), list(_NIGHT_PLACES)
    )
    return copied, answered, kind


def _first_return_answers(trips: pd.DataFrame, home: np.ndarray, hab: pd.DataFrame | None) -> dict:
    """The day-start answer of a person whose day opens with a return home from another AGEB: where they were.

    Such a day did not start at home, whatever the answer says: 75 persons
    answered 'Su casa' (``inicio_zona_ajena`` until 2026-09-26; most are the
    return of a day whose trip out was not recorded, or of a night shift). The
    answer becomes the place type the person's own trips to that AGEB report,
    else their household's, else the one that 60% or more of at least five
    trips there by people of the same occupation report (a student coming
    home from an AGEB whose student trips go 91% to a school is coming from
    school), else 'Otros (especifique)'. Passes 2 and 3 did the same by hand
    for the 75 first returns they reviewed; the fourth review of the chains
    made it a rule. ``trips`` is a chain table without its non-trips, sorted
    in chain order; ``hab`` gives the occupation (skipped when None). Returns
    person -> the answer.
    """
    pid = _person_ids(trips)
    first = np.r_[True, pid[1:] != pid[:-1]]
    origin, dest = trips.origen.astype(str).to_numpy(), trips.destino.astype(str).to_numpy()
    target = first & (trips.motivo_viaje == HOME_MOTIVE).to_numpy() & (origin != home) & (
        trips.tipo_lugar_origen == HOME_PLACE
    ).to_numpy()
    if not target.any():
        return {}
    kind = trips.tipo_lugar_destino.astype(object)
    places = (trips.motivo_viaje != HOME_MOTIVE).to_numpy() & kind.notna().to_numpy() & (kind != HOME_PLACE).to_numpy()
    persons = trips.index.droplevel("folio_viaje")
    household = trips.index.get_level_values("folio_vivienda").to_numpy()
    occupation = (
        hab.ocupacion.astype(object).reindex(persons).to_numpy() if hab is not None else np.full(len(trips), None)
    )
    out = {}
    for i in np.flatnonzero(target):
        there = places & (dest == origin[i])
        for trips_there in (there & (persons == persons[i]), there & (household == household[i])):
            if trips_there.any():
                out[persons[i]] = kind[trips_there].astype(str).value_counts().index[0]
                break
        else:
            answer = OTHER
            if hab is not None and pd.notna(occupation[i]):
                same = there & (occupation == occupation[i])
                if same.sum() >= 5:
                    share = kind[same].astype(str).value_counts(normalize=True)
                    if share.iloc[0] >= 0.6:
                        answer = share.index[0]
            out[persons[i]] = answer
    return out


def _rotated_first_trips(trips: pd.DataFrame, home: np.ndarray) -> np.ndarray:
    """The day's first trips recorded from a place to home whose next trip leaves from home: they went to the place.

    A first trip, an activity, recorded from another zone to the home zone,
    followed by a trip that leaves from home, is the record rotated: the first
    trip went from home to the place it names as its origin, and the next trip
    left from there. It holds where the person answered that the day started at
    'Su casa', or where the next trip is a 'Regresar a Casa' from home to home,
    a return that makes sense only from the place — such as household 7733,
    person 2: to work at 08:00 recorded from the office to home, then home from
    home at 17:00. ``_first_origins_at_home`` leaves these first trips alone,
    since moved home they would go from home to home. On the survey 47 days are
    so recorded: the fifth hand pass rotated 22 of them one by one and round 14
    of the hand decisions 16 more (the days answering 'Su casa', until then
    ``inicio_zona_ajena``); 8 more had been read by neither, and one the fourth
    pass had read as a return from work. A day whose next trip
    leaves from somewhere else, or goes back to the place, is not rotated: the
    record says something else, a return from a night shift or a day that
    started at the place, and the hand reads it. Where the trip was recorded
    arriving at 'Su casa' and the person's answer names a place, the place is
    of that type (``tipo_destino:rotado``): the fifth pass had set it so on
    10 of its rotations, and 20 of the 47 trips are so recorded. Made a rule on
    2026-09-26, in the fifth review of the chains. ``trips`` is a chain table
    without its non-trips, sorted in chain order; returns the mask of the first
    trips rotated, whose origin becomes home, destination the place and next
    trip's origin the place.
    """
    pid = _person_ids(trips)
    first = np.r_[True, pid[1:] != pid[:-1]]
    origin, dest = trips.origen.astype(str).to_numpy(), trips.destino.astype(str).to_numpy()
    motive = trips.motivo_viaje.astype(object).to_numpy()
    next_origin, next_dest, next_motive = (_shift(v, pid, -1) for v in (origin, dest, motive))
    has_next = pd.notna(next_origin)
    answer = trips.tipo_lugar_origen.astype(object).to_numpy()
    back_home = has_next & (next_motive == HOME_MOTIVE) & (next_dest == home)
    return (
        first
        & (motive != HOME_MOTIVE)
        & (origin != home)
        & (dest == home)
        & (~has_next | (next_origin == home))
        & ((answer == HOME_PLACE) | back_home)
    )


def _short_returns(trips: pd.DataFrame, home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The returns home that end away from home with nothing to say where they went: home, or a place nothing names.

    A 'Regresar a Casa' still away from the home zone once the rules have read
    its destination type (``_recode_returns_by_destination``), its copied origin
    (``_returns_copying_origin``), the person's own trips and the day-start
    answer (``_returns_elsewhere``) went somewhere nothing names. The day's last
    trip to an AGEB that no other trip of the day touches, as an origin or a
    destination, is the return home with the home geocoded in another AGEB: its
    destination becomes the home zone (``destino:casa``). Any other went to a
    place the day leaves from or comes back to, and takes 'Otros (especifique)'
    as its motive and place type (``motivo:otros``), the O the model build gave
    it already. On the survey 29 returns are left so (``regreso_sin_llegar``
    until then), 8 of them last trips to an AGEB nothing else touches; round 14
    of the hand decisions read them this way one by one, where the first hand
    pass had recoded most of them 'Trabajar' or 'Estudiar', for students,
    children and homemakers alike, and moved the last trips home. Made a rule on
    2026-09-26, in the fifth review of the chains. ``trips`` is a chain table
    without its non-trips, sorted in chain order, its origins continuous;
    returns the masks of the returns that go home and of those that go to 'Otros'.
    """
    n = len(trips)
    pid = _person_ids(trips)
    origin, dest = trips.origen.astype(str).to_numpy(), trips.destino.astype(str).to_numpy()
    target = (trips.motivo_viaje == HOME_MOTIVE).to_numpy() & (dest != home)
    first = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    bounds = dict(zip(pid[first], zip(first, np.r_[first[1:], n])))
    to_home, to_other = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i in np.flatnonzero(target):
        a, b = bounds[pid[i]]
        touched = set(origin[a:b]) | set(np.delete(dest[a:b], i - a))
        if i == b - 1 and dest[i] not in touched:
            to_home[i] = True
        else:
            to_other[i] = True
    return to_home, to_other


def _start_at_arrival(
    trips: pd.DataFrame, legs: pd.DataFrame | None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Move a start that falls before the previous trip's arrival, by an hour at most, to that arrival.

    A trip that starts before the previous one could have arrived — its start
    plus its leg minutes — by ``_START_TIME_TOLERANCE`` or less is rounding
    of the reported minutes (the overlaps spike at 5, 10 and 15 minutes), not
    an hour typed wrong, and the typo search leaves it (``hora_anterior``,
    ``hora_repetida``, ``hora_traslapada`` until 2026-09-26). TASHA
    schedules each activity between one trip's arrival and the next start
    and cannot take an overlap, and a stay of no minutes is better than one
    (user, 2026-09-26), so the start becomes the arrival: 548 of the 696 such
    starts had been moved so by hand, one by one, and the rule does it for
    all of them.

    A contradiction of more than the tolerance and up to ``_SLIDE_LIMIT``
    (an hour) that the typo search found no reading of the hours for is
    minutes, not an hour: a 15-minute errand whose return was recorded before
    it, a drop-off whose leg minutes overrun the next start. The first hand
    pass slid such starts to the arrival one by one ("La salida se recorre…"),
    and since the fourth review of the chains (2026-09-26) the rule does too,
    except after a trip to work or school, where a stay of no minutes would
    make the day implausible and the hand reads the hours instead. On the
    survey 687 starts move, 600 of them by the tolerance or less and 86 by up
    to an hour (and an imputed start with the trip before); 816 until the
    fifth review of the chains (2026-09-26), when the travel minutes a review
    corrected came before the rules and their overlaps went with them.

    A start the move of the trip before pushes into overlap moves with it; a
    larger contradiction (``hora_invertida``) and an overnight wrap
    (``hora_nocturna``, :func:`_day_wraps`) are left. ``trips`` is a chain
    table without its non-trips, sorted in chain order, after the typo search.
    Returns the trips and the mask of starts moved.
    """
    start = _start_minutes(trips)
    travel = _leg_minutes(trips, legs)
    wraps = _day_wraps(trips, start)
    mandatory = trips.motivo_viaje.isin(MANDATORY_MOTIVES).to_numpy()
    pid = _person_ids(trips)
    moved = np.zeros(len(trips), dtype=bool)
    new = start.copy()
    for j in range(1, len(trips)):
        if pid[j] != pid[j - 1]:
            continue
        if wraps[j]:
            continue  # overnight: the next calendar day's
        arrival = new[j - 1] + travel[j - 1]
        gap = start[j - 1] + travel[j - 1] - start[j]  # how far short of the recorded arrival
        if start[j] < arrival and (gap <= _START_TIME_TOLERANCE or (gap <= _SLIDE_LIMIT and not mandatory[j - 1])):
            new[j], moved[j] = arrival, True
    if moved.any():
        trips = trips.copy()
        minutes = new.astype(int)
        trips["hora_inicio_h"] = pd.array(minutes // 60 % 24, dtype="Int64")
        trips["hora_inicio_m"] = pd.array(minutes % 60, dtype="Int64")
    return trips, moved


def _leg_minutes(trips: pd.DataFrame, legs: pd.DataFrame | None) -> np.ndarray:
    """Travel minutes per trip, from ``legs`` or from the traslado*_min columns; zeros if neither."""
    if legs is not None:
        return (
            legs.groupby(level=[0, 1, 2])
            .traslado_min.sum()
            .reindex(trips.index)
            .fillna(0)
            .to_numpy(dtype=float)
        )
    cols = [c for c in trips.columns if c.startswith("traslado") and c.endswith("_min")]
    if cols:
        return trips[cols].sum(axis=1, min_count=1).fillna(0).to_numpy(dtype=float)
    return np.zeros(len(trips))


def _overnight(prev: float, this: float, night_return: bool = False) -> bool:
    """The typo search's overnight window: by 06:00 after a start at or after 18:00, or, for a return home after a
    trip to work (``night_return``), before ``_NIGHT_SHIFT_RETURN``."""
    return prev >= 18 * 60 and (this <= 6 * 60 or (night_return and this < _NIGHT_SHIFT_RETURN))


def _next_day(
    start: np.ndarray, prev_start: np.ndarray, is_return: np.ndarray, after_work: np.ndarray | None = None
) -> np.ndarray:
    """The starts read as the next calendar day's: by 06:00 after a trip that started at or after 18:00,
    a return home by 06:00 after one that started at or after noon, or a return home before 10:00 after a
    trip to work (``after_work``) that started at or after 18:00.

    The second clause is the second shift the typo search cannot read otherwise: its own
    window (``_overnight``) stays at 18:00, where a 03:00 after 14:00 is likelier a mistyped
    15:00. The third is the night shift. A return home from work recorded between 06:01 and
    09:59 after a departure at 18:00 or later is the next morning's — 20:00 to 08:00 and 19:00
    to 07:00 are the factories' shifts — but until 2026-09-26 the window closed at 06:00, and
    the typo search read 34 such returns as the same evening's, a median of 39 minutes after
    the arrival at work, while the first hand pass read 56 more as a day shift by moving both
    hours by twelve. Since the fourth review of the chains both readings of the hours are
    the night shift's. A reading stands only where the day holds together after it
    (:func:`_day_wraps`), which ``_remaining_issues`` marks ``hora_nocturna`` and the model
    build codes past 2400 (:func:`days_past_midnight`), so both read one definition.
    """
    with np.errstate(invalid="ignore"):
        overnight = (prev_start >= 18 * 60) | ((prev_start >= 12 * 60) & is_return)
        out = ~np.isnan(prev_start) & overnight & (start <= 6 * 60)
        if after_work is not None:
            night = np.asarray(after_work, dtype=bool) & is_return & (prev_start >= 18 * 60)
            out |= ~np.isnan(prev_start) & night & (start < _NIGHT_SHIFT_RETURN)
        return out


def _day_wraps(trips: pd.DataFrame, start: np.ndarray | None = None) -> np.ndarray:
    """The trips where a day passes midnight: ``_next_day``'s reading, where the day holds together after it.

    Read overnight, a start puts the rest of the day on the next calendar day,
    so the reading stands only where no trip to work or school comes after the
    trip that passes midnight — a night shift ends the day, and a day that goes
    to work again after one is two shifts in one diary — and where the day's
    last start then comes less than 24 hours after its first. Until 2026-09-26
    (the fifth review of the chains) the reading stood regardless: 15 days went
    to work again after the night, 11 of them running past 24 hours, such as
    household 2191, person 2 (to work at 18:00, home at
    08:50, to work at 10:13, home at 19:00), and ``eodgdl.review.screens`` took
    them for night shifts. Where the reading does not stand the trip starts
    before the previous one arrived (``hora_invertida``) unless the typo search
    reads the hours otherwise, which holds its readings to the same two
    conditions (:func:`_search_edits`): household 8992, person 1, to work at
    21:00 and home at 02:00, then to work at 16:00 and home at 19:00, is the
    split shift from 09:00 and 14:00 that the eighth hand round read. ``trips``
    is a chain table sorted in chain order and without its non-trips; ``start``
    its starts in minutes, where the caller has them.
    """
    pid = _person_ids(trips)
    start = _start_minutes(trips) if start is None else np.asarray(start, dtype=float)
    prev_start = _shift(start, pid, 1).astype(float)
    is_return = (trips.motivo_viaje == HOME_MOTIVE).to_numpy()
    wraps = _next_day(start, prev_start, is_return, _after_work(trips, pid))
    if not wraps.any():
        return wraps
    mandatory = trips.motivo_viaje.isin(MANDATORY_MOTIVES).to_numpy()
    # the trips to work or school after each row within its person: a cumulative sum run backwards
    later = pd.Series(mandatory[::-1].astype(int)).groupby(pid[::-1]).cumsum().to_numpy()[::-1] - mandatory
    wraps &= later == 0
    days = pd.Series(wraps.astype(int)).groupby(pid).cumsum().to_numpy()
    absolute = pd.Series(start + _DAY * days).groupby(pid)
    span = (absolute.transform("max") - absolute.transform("first")).to_numpy()
    return wraps & (span < _DAY)


def days_past_midnight(trips: pd.DataFrame) -> pd.Series:
    """How many midnights a person's day has passed when each trip starts: 0, then 1 from a start read as the next day's on.

    ``trips`` is a chain table sorted in chain order and without the rows that
    are not trips, the chain the model build reads and ``problemas`` is marked
    on; the trip where the day passes midnight is the one marked
    ``hora_nocturna`` (:func:`_day_wraps`), and every trip after it is on the
    next day too. The model's ``StartTime`` adds 2400 per midnight, so a diary
    that runs into the next morning keeps counting its hours (a 06:00 return
    after a 21:00 trip to work is 3000). A day passes midnight at most once and
    its last start comes less than 24 hours after its first, since
    ``_day_wraps`` reads no other day overnight (``tests/test_tasha.py`` holds
    the build to both).
    """
    pid = _person_ids(trips)
    wraps = pd.Series(_day_wraps(trips).astype(int), index=trips.index)
    return wraps.groupby(pid).cumsum().rename("days_past_midnight")


def _after_work(trips: pd.DataFrame, pid: np.ndarray | None = None) -> np.ndarray:
    """True where the previous trip of the same person went to work."""
    pid = _person_ids(trips) if pid is None else pid
    return _shift((trips.motivo_viaje == WORK_MOTIVE).to_numpy(), pid, 1) == True  # noqa: E712 (object array)


def _chain_needs_repair(
    start: np.ndarray, travel: np.ndarray, night: np.ndarray | None = None, mandatory: np.ndarray | None = None
) -> bool:
    """Does some trip start before the previous one could have arrived, beyond the tolerance and not overnight?

    ``night`` marks the returns home after a trip to work, whose overnight window is wider (``_overnight``).
    An overnight wrap counts only where the day holds together after it, as :func:`_day_wraps` reads it: no
    trip to work or school (``mandatory``) after the wrap, and the day's last start less than 24 hours after
    its first.
    """
    tol = _START_TIME_TOLERANCE
    night = np.zeros(len(start), dtype=bool) if night is None else night
    mandatory = np.zeros(len(start), dtype=bool) if mandatory is None else mandatory
    offset = 0
    for i in range(1, len(start)):
        wrapped = offset
        if start[i] < start[i - 1] + travel[i - 1] - tol:
            if not _overnight(start[i - 1], start[i], night[i]):
                return True
            offset += _DAY
        if (wrapped and mandatory[i]) or start[i] + offset - start[0] >= _DAY:
            return True
    return False


def _early_long_stay(first_start: float, first_travel: float, next_start: float) -> bool:
    """Does the day's first trip start before ``_EARLY_START`` and a stay of ``_EARLY_LONG_STAY`` or more follow it?"""
    return first_start < _EARLY_START and next_start - (first_start + first_travel) >= _EARLY_LONG_STAY


def _errands_before_returns(trips: pd.DataFrame, pid: np.ndarray) -> np.ndarray:
    """True where a trip goes to a short errand (``_SHORT_ERRANDS``) and the person's next trip returns home."""
    motive = trips.motivo_viaje.astype(object).to_numpy()
    return np.isin(motive, _SHORT_ERRANDS) & (_shift(motive, pid, -1) == HOME_MOTIVE)


def _long_errand(start: np.ndarray, travel: np.ndarray, errand: np.ndarray) -> bool:
    """Does a chain's return home start ``_LONG_ERRAND`` or more after the arrival at the errand before it?

    ``errand`` marks the errands a return home follows (:func:`_errands_before_returns`).
    """
    return bool((errand[:-1] & (start[1:] - (start[:-1] + travel[:-1]) >= _LONG_ERRAND)).any())


def _search_edits(
    hours: np.ndarray,
    mins: np.ndarray,
    travel: np.ndarray,
    locked: np.ndarray | None = None,
    early: bool = False,
    menu: tuple = _START_TIME_EDITS,
    night: np.ndarray | None = None,
    mandatory: np.ndarray | None = None,
    errand: np.ndarray | None = None,
):
    """The unique fewest-cost combination of edits from ``menu`` that makes one chain feasible, or None.

    Feasible means every trip starts no earlier than the previous trip's
    arrival — its start plus its travel minutes — less the tolerance, and,
    with ``early`` (the chain's first trip goes to anything but work), that
    the first trip does not start before ``_EARLY_START`` with a stay of
    ``_EARLY_LONG_STAY`` or more after it; where ``errand`` marks the errands
    a return home follows, that no such return starts ``_LONG_ERRAND`` or
    more after the arrival at its errand. An edit must also leave
    ``_MIN_MANDATORY_STAY`` or more at work or school (``mandatory`` marks the
    trips there): a reading of the hours that puts the next start at the
    arrival at work is no reading. Depth-first over the trips, pruned by the
    running cost and by the chain so far; an overnight wrap (``_overnight``,
    the wider window for the returns home after a trip to work that ``night``
    marks) is accepted only between two unedited trips, so the search cannot
    manufacture a night shift, and only where the day holds together after
    it, as :func:`_day_wraps` reads it: no trip to work or school after the
    wrap, and the day's last start less than 24 hours after its first. A
    ``locked`` trip (an imputed start time, which cannot carry a typo, or one
    an earlier pass set) is never edited. The cheapest reading must be unique.
    """
    tol, cap = _START_TIME_TOLERANCE, _START_TIME_MAX_COST
    n = len(hours)
    locked = np.zeros(n, dtype=bool) if locked is None else locked
    night = np.zeros(n, dtype=bool) if night is None else night
    mandatory = np.zeros(n, dtype=bool) if mandatory is None else mandatory
    errand = np.zeros(n, dtype=bool) if errand is None else errand
    options = [
        [("", 0.0, h)] + ([] if lock else [(name, c, h + d) for name, c, lo, hi, d in menu if lo <= h <= hi])
        for h, lock in zip(hours, locked)
    ]
    best = [cap + 1e-9]
    found: list[tuple[str, ...]] = []
    chosen: list[str] = [""] * n

    def walk(i, cost, prev_arrival, prev_start, prev_edited, first, offset):
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
            shift = offset  # minutes past the day's own clock: a day each midnight passed
            if prev_arrival is not None:
                wrap = not edited and not prev_edited and _overnight(prev_start, start, night[i])
                if not (start >= prev_arrival - tol or wrap):
                    continue
                if (edited or prev_edited) and mandatory[i - 1] and not wrap and start < prev_arrival + _MIN_MANDATORY_STAY:
                    continue
                if offset and mandatory[i]:
                    continue  # to work or school again after the day passed midnight: no night shift
                if errand[i - 1] and start + (_DAY if wrap else 0) - prev_arrival >= _LONG_ERRAND:
                    continue  # still half a day at the errand
                shift += _DAY if wrap else 0
                if start + shift - first >= _DAY:
                    continue  # a day of 24 hours
            if early and i == 1 and _early_long_stay(prev_start, prev_arrival - prev_start, start):
                continue
            chosen[i] = name
            walk(i + 1, cost + c, start + travel[i], start, edited, start if first is None else first, shift)

    walk(0, 0.0, None, None, False, None, 0)
    return found[0] if len(found) == 1 else None


def _repair_start_times(
    trips: pd.DataFrame,
    legs: pd.DataFrame | None = None,
    locked: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    """Repair mistyped start hours with the fewest edits that make a chain monotone.

    The start times are the noisy field, and the noise has a structure: hours
    entered on a 12-hour clock, and an extra or missing leading 1 (19:00 for
    9:00, 8:00 for 18:00), sometimes both on one field. A chain is feasible
    when every trip starts no earlier than the previous trip's arrival — its
    start plus its leg minutes — less ``_START_TIME_TOLERANCE``; an overnight
    wrap between a trip at or after 18:00 and one by 06:00, or a return home
    from work before 10:00 (``_next_day``), is not a violation. For every
    infeasible chain the search tries the menu in ``_START_TIME_EDITS`` on
    each trip and keeps the cheapest combination that makes the chain
    feasible, provided it is unique, costs at most ``_START_TIME_MAX_COST``
    and leaves ``_MIN_MANDATORY_STAY`` or more at work or school. A chain
    with no such reading is searched again with the second menu,
    ``_START_TIME_EDITS_2``, which adds a morning start typed as an evening
    one (``hora:-12h``, from 17:00 on); tried first, it would change 241
    readings the first menu already makes, and none for the better. A chain
    with no reading is left as it is, and the trips in it that still start
    before the previous arrival are marked ``hora_invertida`` unless
    ``_start_at_arrival`` closes them. Every edited row is marked in the
    ``ajustes`` column with the edit applied (``hora:+12h`` and so on), so a
    consumer can treat it as uncertain; ``locked`` rows (imputed start times)
    are never edited.

    The second menu, the stay at work and the cost cap (4 until then) were
    added on 2026-09-26, in the fourth review of the chains, from the first
    hand pass's readings of the chains the search left: 550 of its 1,026
    start edits read an evening start as the morning's ("Corrección AM/PM"),
    most on a day's first trip to work recorded at 20:00–22:59 and followed
    by a return in the afternoon. The stay at work rules out readings such as
    household 15423, person 1's: to work at 07:30, 35 minutes, home at 17:50
    and to the shops at 18:00, where an extra leading 1 read off the 17:50
    closed a 25-minute overlap by leaving no minutes at work; 137 stays at
    work shorter than half an hour sat next to such an edit.

    A day whose order of times holds can still carry the typo. Since
    2026-09-26 the search also runs on a chain whose first trip, to anything
    but work, starts before ``_EARLY_START`` with a stay of
    ``_EARLY_LONG_STAY`` or more after it, and then counts that start as
    infeasible too. 161 days of the survey start between 01:00 and 04:59
    with a trip to school, shops, sport, a visit or the doctor, and a stay of
    eight hours or more follows 111 of them (median 15 hours): 33 school days
    start at 01:xx and 27 of them come home between 18:00 and 20:59, as 1,202
    of the 1,319 school days that start at 13:xx do, the afternoon shift
    typed on a 12-hour clock. No code marked them, since the order of their
    times holds. The short early trips — a market at 04:30, an escort — are
    left, and marked ``hora_madrugada``.

    A return home typed on the afternoon side of the 12-hour clock leaves the
    order intact too, and half a day at the errand before it. Since
    2026-09-26 (a review of the long errands) the search also runs on a
    chain whose times hold but whose return home starts ``_LONG_ERRAND`` (12
    hours) or more after the arrival at a short errand (``_SHORT_ERRANDS``:
    shopping, the doctor, an office errand, a payment, an escort), with the
    -12h edit alone (``_LONG_ERRAND_EDITS``), and keeps the unique reading
    that leaves no such stay. On the survey's hours the stays before a return
    from those errands thin out to eleven hours — 98, 62, 35 and 17 from
    eight hours to eleven — and jump at twelve, to 115; 62 of the 234 such
    returns 12 hours or more after the errand start at 23:xx, where 0.9% of
    all returns do. Of the chains read, a household member who made the same
    errand came home exactly 12 hours before the person in 12 — household
    9541, person 1, to the doctor at 07:00 with a companion who was home at
    11:00, is home at 23:00 — and left 12 hours later in 2, the errand's own
    hour typed wrong, which the -12h reads with the right stay at the wrong
    end of the day; the first menu would read nearly every one so, a doctor
    at 21:00. 13 came home at the same hour as a companion, mostly days at a
    hospital, where one slip typed twice and a whole day there look alike.
    Visits and outings are left, and 'Otros', which names day trips out of
    town as well as errands: the stays before a return from a visit or an
    outing thin out with no jump (129, 138, 53, 24, 16, 10 from eight hours
    to thirteen for visits), and of the ones of eight to twelve hours whose
    trip a companion made too, 100 of 115 visits and 182 of 187 outings end
    with the companion home at the same hour. Nor does a 12-hour slip explain
    a stay under twelve hours. 123 chains are read this way, every return to
    between 05:00 and 11:59 (106 of them from 09:00), leaving a median stay
    of 88 minutes; the long stays in chains the first menus read are theirs
    (household 2585, person 1, whose last return the first menu read as
    23:00, keeps 13 hours at the shops).

    On the survey, with the untimed trips imputed and the non-trips set
    aside, 2,046 chains get a reading — 377 of them with the second menu's
    -12h, 123 with the -12h alone for a long errand, 110 for their early
    start alone and 3 once the move to the arrival cleared the way
    (:func:`_read_start_times`) — and 2,602 rows change (+12h 713, -12h 515,
    an extra leading 1 392, both 826, a missing leading 1 156), none of them
    to an hour before 05:00. 351 trips in 246 people still
    start before the previous trip could have arrived by more than the
    tolerance, not overnight, and are marked ``hora_invertida`` (870 in 756
    until the fourth review of the chains, 355 in 251 until the fifth, which
    read no day overnight that goes to work again after the night,
    :func:`_day_wraps`); the hand decisions read them
    (``scripts/revisions/chains_15.py``) and tasha.chain_report counts them
    again on the built table. Household 8,
    person 3 is the worked example: 07:24, 07:37,
    19:00, 16:30, 18:02, 18:00, with 30-minute drives, becomes feasible by
    reading the 19:00 as 09:00 and the closing 18:00 as 20:00 — the two-minute
    step at the end is a 32-minute contradiction once the drive is counted.
    Returns the trips, the edit name per row ("" where none), and the counts
    ``chains_repaired``, ``early_starts_read`` and ``long_errands_read`` (the
    chains read for their early start or their long errand alone).
    """
    hours = trips.hora_inicio_h.to_numpy(dtype=int)
    mins = trips.hora_inicio_m.to_numpy(dtype=int)
    travel = _leg_minutes(trips, legs)
    locked = (
        np.zeros(len(trips), dtype=bool)
        if locked is None
        else np.asarray(locked, dtype=bool)
    )
    pid = _person_ids(trips)
    first = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    last = np.r_[first[1:], len(trips)]
    activity = ~trips.motivo_viaje.isin([WORK_MOTIVE, HOME_MOTIVE]).to_numpy()
    night = (trips.motivo_viaje == HOME_MOTIVE).to_numpy() & _after_work(trips, pid)
    mandatory = trips.motivo_viaje.isin(MANDATORY_MOTIVES).to_numpy()
    errand = _errands_before_returns(trips, pid)
    delta = {name: d for name, _, _, _, d in _START_TIME_EDITS_2}
    new_hours = hours.copy()
    flag = np.full(len(trips), "", dtype=object)
    counts = dict.fromkeys(("chains_repaired", "early_starts_read", "long_errands_read"), 0)
    for a, b in zip(first, last):
        start = hours[a:b] * 60 + mins[a:b]
        early = bool(activity[a])
        infeasible = _chain_needs_repair(start, travel[a:b], night[a:b], mandatory[a:b])
        early_stay = early and b - a > 1 and _early_long_stay(start[0], travel[a], start[1])
        # a long errand is read only in a chain nothing else sends to the search, and with the -12h alone
        long_errand = not (infeasible or early_stay) and _long_errand(start, travel[a:b], errand[a:b])
        if not (infeasible or early_stay or long_errand):
            continue
        menus = (_LONG_ERRAND_EDITS,) if long_errand else (_START_TIME_EDITS, _START_TIME_EDITS_2)
        for menu in menus:
            edits = _search_edits(hours[a:b], mins[a:b], travel[a:b], locked[a:b], early, menu, night[a:b],
                                  mandatory[a:b], errand[a:b] if long_errand else None)
            if edits is not None:
                break
        if edits is None:
            continue
        counts["chains_repaired"] += 1
        counts["early_starts_read"] += early_stay and not infeasible
        counts["long_errands_read"] += long_errand
        for j, name in enumerate(edits):
            if name:
                new_hours[a + j] += delta[name]
                flag[a + j] = name
    trips = trips.copy()
    trips["hora_inicio_h"] = pd.array(new_hours, dtype="Int64")
    return trips, flag, counts


def _read_start_times(
    trips: pd.DataFrame, legs: pd.DataFrame | None, locked: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, int]]:
    """The start times of a chain table read: the typo search, the move to the arrival, and the search once more.

    The typo search (:func:`_repair_start_times`) reads a chain only if one
    reading makes all of it feasible, so a contradiction of minutes that
    :func:`_start_at_arrival` closes afterwards, 15 to 60 minutes after a trip
    that is not to work or school, stood in the way of reading an hour typed
    wrong elsewhere in the same chain. Once the starts have moved to the
    arrival, the chains still infeasible are searched again, the starts moved
    and the ones already read locked, and whatever that reading leaves within
    reach of the arrival moves to it. Until 2026-09-26 (the fifth review of the
    chains) the rules stopped after the first move, and running them twice read
    more: household 834, person 1, an escort at 04:30 between work at 08:00
    and a return at 18:00, took hora_invertida because a visit's return at
    20:10 came 28 minutes before the visit's 38-minute ride could arrive; the
    return moves to 20:38 and the escort reads as 16:30, as the hand decisions
    had read it. ``locked`` marks the starts no pass may edit (the imputed
    ones). Returns the trips, the edit name per row ("" where none), the mask
    of starts moved to the arrival, and the counts ``chains_repaired``,
    ``early_starts_read``, ``long_errands_read`` and
    ``chains_read_after_the_slide``.
    """
    trips, edit, first = _repair_start_times(trips, legs, locked)
    trips, slid = _start_at_arrival(trips, legs)
    trips, again, second = _repair_start_times(trips, legs, locked | slid | (edit != ""))
    if second["chains_repaired"]:
        trips, slid_again = _start_at_arrival(trips, legs)
        slid = slid | slid_again
    edit = np.where(again != "", again, edit)
    counts = {key: int(first[key] + second[key]) for key in first}
    counts["chains_read_after_the_slide"] = int(second["chains_repaired"])
    return trips, edit, slid, counts


def _remaining_issues(
    trips: pd.DataFrame,
    home: np.ndarray,
    legs: pd.DataFrame | None,
    time_edited: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """What is still wrong with each trip of a chain after the rules, one mask per ``ISSUE_CODES`` entry it can produce.

    Runs over the chain without its non-trips, so "first", "last" and
    "previous" are read over the trips the model build sees. Every kind of
    error the chain diagnosis (``reports/trip_chains.qmd``) found and the
    rules do not repair is a code here, so the trips carrying at least one
    error are exactly the trips with a non-empty ``problemas``; nothing is
    repaired, since none of these has a repair that does not invent data. On
    the shipped survey, after the rules: 351 ``hora_invertida`` and 178
    ``hora_nocturna`` (90 an evening trip followed by one in the small hours,
    8 a return home by 06:00 after an afternoon start, the second shift
    this marking reads as overnight while the typo search keeps its 18:00
    window, and 80 a return home before 10:00 after an evening trip to work,
    the night shift; a day that goes to work or school again after the night,
    or would run 24 hours, is not read overnight, :func:`_day_wraps`).
    ``hora_anterior``, ``hora_repetida`` and
    ``hora_traslapada`` are what ``hora_invertida`` does not cover, one time
    code per row at most: overlaps within the tolerance, spiking at 5, 10 and
    15 minutes, the rounding of the reported leg minutes. Until 2026-09-26
    the rules left 696 of them (26, 101 and 569); ``_start_at_arrival`` now
    closes them all, so only a hand edit leaves one. Since the fourth review
    of the chains (2026-09-26) the rules also close every origin break, so
    ``origen_discontinuo`` is 0 after them too (31 until then), and since the
    fifth, the same day, every first trip from 'Su casa' in another zone
    (``inicio_zona_ajena``, 177 until the fourth, then the 16 rotated records
    ``_rotated_first_trips`` reads) and every return left away from home
    (``regreso_sin_llegar``, 29 until then, ``_short_returns``). Then 16
    ``tipo_destino_dudoso``; 381 days start from somewhere other than 'Su
    casa' (``inicio_fuera_de_casa``; 906 until the third review: the first
    trips that leave their own destination's AGEB, answer 'Su casa' or copy
    the destination's place type start at home, ``_first_origins_at_home``,
    the rotated records do too, and a day that opens with a return from
    another AGEB answers where it started, ``_first_return_answers``) and 494
    do not end with a return to the home zone (``fin_fuera_de_casa``) —
    second homes, nights spent elsewhere, night shifts and geocoding slips
    all look alike here; 536 activity trips end at 'Su casa'
    (``actividad_en_casa``), 450 of them in the home zone, most
    leaving it too (a tour recorded without its place; the ones that arrive
    from another zone are returns, ``_arrivals_home``); and 30 trips carry the
    'Guardería' motive (``motivo_guarderia``), all made by children under 12
    (the escorts are recoded, ``_daycare_escorts``), which the model reads as
    a school trip.

    ``hora_2301`` marks a start of 23:01 that is the survey's own value
    (``time_edited`` holds the rows whose start a rule or a hand edit set):
    106 shipped trips start at 23:01 against 4 at 23:02, 10 at 23:03 and none
    at 23:04, 105 of them the person's last trip and a return home, while the
    whole hour from 00:00 holds 10 starts against 785 from 23:00 and 182 from
    01:00. The capture heaps the day's late returns on that minute, so the
    return came at 23:01 or later, likely after midnight; nothing says when,
    and the value is kept. 94 of them are on the chain (9 are non-trips and 3
    moved by a rule; 2 more reach 23:01 by the 12-hour-clock repair of an
    11:01, and are not the survey's own). The share of returns on it grows
    with the hour of the trip they return from — 0.6% after a trip at 19:xx,
    11% after one at 22:xx — as late returns capped there would; 20 of the
    94 are the capture's own start for a return instead, 5h01 after a trip
    at 18:00 or 1h01 after one at 22:00, and carry that code too (below).

    ``hora_1h01`` and ``hora_5h01`` mark a trip that repeats the one before
    it the other way — the zones reversed, the same main mode, the same
    minutes, as the capture's return writes it — and starts exactly 1h01 or
    5h01 after it (``_DEFAULT_RETURN_GAPS``), both starts the survey's own.
    Found 2026-09-26 in a review of the heaped returns: as shipped, 747 and
    662 such pairs start that far apart against 24 to 44 at the neighbouring
    minutes, pairs that are no such copies show no heap there (62 and 71),
    and the other whole hours plus a minute show none either. The 5h01 is the
    capture's default start for a return, left as it was: 7.1% of the
    returns of the first fieldwork week, 4.3% of the second and 0.6% after,
    in diaries otherwise ordinary; a work tour with such a return spends
    4h11 at work (median) against 7h40 for the rest. The 1h01 is April
    2023's: 715 of the 747 are evening walks to the shops, leaving at 17:xx
    or 18:xx at an unrounded minute, by homemakers and retirees (one in five
    of April's), whose trips per person did not rise with them, so the trips
    look real and their times filled in; the same walks from January to March
    come back after a median 14 minutes. When such a return came is unknown,
    so the stay before it is the capture's, not the person's; the value is
    kept. After the rules 718 and 613 are on the chain.

    ``hora_madrugada`` marks a day's first trip, to anything but work, that
    starts before 05:00 (``_EARLY_START``): 59 after the rules, 39 of them at
    04:xx — a market, an escort, a gym — where the typo search found no
    long stay to read the hour by, or no unique reading.
    """
    time_edited = (
        np.zeros(len(trips), dtype=bool) if time_edited is None else np.asarray(time_edited, dtype=bool)
    )
    pid = _person_ids(trips)
    first = np.r_[True, pid[1:] != pid[:-1]]
    last = np.r_[pid[1:] != pid[:-1], True]
    start = _start_minutes(trips)
    travel = _leg_minutes(trips, legs)
    prev_start = _shift(start, pid, 1).astype(float)
    prev_arrival = _shift(start + travel, pid, 1).astype(float)
    origin, dest = (
        trips.origen.astype(str).to_numpy(),
        trips.destino.astype(str).to_numpy(),
    )
    prev_dest = _shift(dest, pid, 1)
    has_prev = pd.notna(prev_dest)
    is_return = (trips.motivo_viaje == HOME_MOTIVE).to_numpy()
    kind = trips.tipo_lugar_destino
    origin_kind = trips.tipo_lugar_origen.astype(str).to_numpy()
    from_home = origin_kind == HOME_PLACE
    wrap = has_prev & _day_wraps(trips, start)  # overnight: the next calendar day's
    with np.errstate(invalid="ignore"):
        inverted = has_prev & (start < prev_arrival - _START_TIME_TOLERANCE) & ~wrap
        earlier = has_prev & (start < prev_start) & ~inverted & ~wrap
        same_minute = has_prev & (start == prev_start) & ~inverted
        overlapped = (
            has_prev & (start > prev_start) & (start < prev_arrival) & ~inverted
        )
    # a trip that repeats the one before it the other way, both starts the survey's own, at the capture's gap
    mode = trips.modo_principal.astype(str).to_numpy() if "modo_principal" in trips else np.zeros(len(trips))
    repeats = (has_prev & (origin == prev_dest.astype(str)) & (dest == _shift(origin, pid, 1).astype(str))
               & (travel == _shift(travel, pid, 1)) & (mode == _shift(mode, pid, 1)))
    as_reported = ~time_edited & (_shift(time_edited, pid, 1) == False)  # noqa: E712 (object array)
    return {
        "hora_invertida": inverted,
        "hora_nocturna": wrap,
        "hora_anterior": earlier,
        "hora_repetida": same_minute,
        "hora_traslapada": overlapped,
        "origen_discontinuo": has_prev & (origin != prev_dest.astype(str)),
        "regreso_sin_llegar": is_return & (dest != home),
        "tipo_destino_dudoso": is_return
        & (dest == home)
        & kind.notna().to_numpy()
        & (kind != HOME_PLACE).to_numpy(),
        "inicio_fuera_de_casa": first & ~from_home,
        "inicio_zona_ajena": first & from_home & (origin != home),
        "fin_fuera_de_casa": last & ~(is_return & (dest == home)),
        "actividad_en_casa": ~is_return & (kind == HOME_PLACE).to_numpy(),
        "motivo_guarderia": (trips.motivo_viaje == "Guardería").to_numpy(),
        "hora_2301": (start == _SENTINEL_START) & ~time_edited,
        **{code: repeats & as_reported & (start - prev_start == gap) for code, gap in _DEFAULT_RETURN_GAPS.items()},
        "hora_madrugada": first & ~is_return & (trips.motivo_viaje != "Trabajar").to_numpy() & (start < _EARLY_START),
    }


def _home_zone(trips: pd.DataFrame, viv: pd.DataFrame) -> np.ndarray:
    """The household's zone id on every trip row."""
    return (
        viv.ageb.astype(str)
        .reindex(trips.index.get_level_values("folio_vivienda"))
        .to_numpy()
    )


def mark_issues(
    trips: pd.DataFrame, viv: pd.DataFrame, legs: pd.DataFrame | None = None
) -> pd.Series:
    """The ``problemas`` column of a trip table in chain order, recomputed from its current values.

    The marking stage of :func:`clean_trip_chains` on its own: the rows that
    are not trips (``_mark_non_trips``) and, over the chain without them,
    every defect ``_remaining_issues`` knows. On the table ``load_eod``
    returns it reproduces the ``problemas`` column exactly; on that table
    with hand edits applied (:mod:`eodgdl.review`) it says what the edits
    left. ``trips`` must be sorted by its index, the chain order.
    """
    n = len(trips)
    issues = np.full(n, "", dtype=object)
    home = _home_zone(trips, viv)
    not_trips = np.zeros(n, dtype=bool)
    for code, mask in _mark_non_trips(trips, home).items():
        _add_code(issues, mask, code)
        not_trips |= mask
    where = np.flatnonzero(~not_trips)
    time_edited = (
        trips[FIX_FLAG].astype(str).str.contains(r"(?:^|;)hora:").to_numpy()
        if FIX_FLAG in trips
        else np.zeros(n, dtype=bool)
    )
    left = _remaining_issues(trips[~not_trips], home[~not_trips], legs, time_edited[~not_trips])
    for code, mask in left.items():
        _add_code(issues, _expand(mask, where, n), code)
    return pd.Series(issues, index=trips.index, name=ISSUE_FLAG, dtype=str)


def clean_trip_chains(
    trips: pd.DataFrame,
    viv: pd.DataFrame,
    legs: pd.DataFrame | None = None,
    hab: pd.DataFrame | None = None,
    minutes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply the chain rules to a trip table in row (folio_viaje) order; flag everything, drop only the duplicate returns.

    First the travel minutes a review round corrected (``minutes``, the table
    ``eodgdl/revisions/leg_minutes.csv.gz`` holds; ``load_eod`` passes it)
    replace the reported ones, since the rules read the minutes
    (``_correct_minutes``, ``minutos:revision``). Then five stages, in order:

    1. **What the questionnaire lost.** Impute the start time, motive and
       destination type of the trips that lost their questionnaire block, from
       the duplicate return that follows or from the nearest timed trips
       (``_impute_untimed_trips``).
    2. **Motives and destinations.** A 'Regresar a Casa' that did not reach
       the household's zone takes its destination type's motive
       (``_recode_returns_by_destination``) or, recorded ending in its own
       origin's AGEB, goes home (``_returns_copying_origin``); an activity
       trip recorded arriving at 'Su casa' in the home zone is a return
       (``_arrivals_home``); a daycare trip from age 12 is an escort
       (``_daycare_escorts``).
    3. **The rows that are not trips, and where the other returns went.** A
       return home made while already at home is marked a non-trip
       (``_home_to_home_returns``); any other return that ends away from home
       takes the motive of the place it went to (``_returns_elsewhere``); the
       next morning's return that closes a night-shift day is marked a
       non-trip too (``_next_morning_returns``).
    4. **Where the trips start and end.** Over the chain without its
       non-trips, every trip starts where the previous one ended
       (``_start_where_previous_ended``); a day's first trip recorded leaving
       another zone starts at home where its origin copies its destination's
       AGEB, the answer says home, or the answer copies the destination's
       place type, and the answer becomes 'Su casa' (``_first_origins_at_home``);
       a first trip recorded from a place to home whose next trip leaves from
       home went from home to the place (``_rotated_first_trips``); a day that
       opens with a return from elsewhere takes an answer that is not 'Su casa'
       (``_first_return_answers``); a return still away from home ends at home
       if it is the day's last trip to an AGEB nothing else touches, else at a
       place nothing names, 'Otros' (``_short_returns``).
    5. **Start times.** Repair mistyped start hours with the fewest edits
       that let every trip start after the previous one arrived, and read a
       day that starts in the small hours with a long stay after, or a return
       home half a day after a short errand (``_repair_start_times``); move a
       start that falls before the previous arrival by the tolerance, or by
       up to an hour where no reading of the hours fits, to that arrival
       (``_start_at_arrival``); search once more where that move cleared the
       way (``_read_start_times``). A day read past midnight must hold
       together after it (``_day_wraps``).

    Then every defect left is marked — start times, anchors, zones and
    purposes (``_remaining_issues``) — so that the trips carrying an error
    are exactly the trips with a non-empty ``problemas``. The order matters:
    the recodes read the imputed motives, the at-home walk reads the recoded
    ones, and the origins, the times and the marking run over the chain
    without its non-trips. Each rule's evidence is in its docstring.
    ``viv`` supplies the household zone (``ageb``) and its survey zone
    (``centralidad``, for a trip end moved home; skipped when absent);
    ``hab`` the person features of the imputation (age, sex, occupation) and
    the ages and occupations of the daycare and day-start rules (skipped when
    None); ``legs`` the travel minutes when the table no longer carries the
    traslado columns. The ``minutes`` corrections are written into the
    returned trips' traslado columns where the table carries them; given
    ``legs`` instead, the rules read a corrected copy and the caller's legs
    stay as they were (``load_eod`` passes the traslado columns, so its
    ``legs`` carry the corrections).

    Returns the trips — every row kept except the home-to-home returns that
    duplicate an imputed return, ``folio_viaje`` untouched, so it has a gap
    where a duplicate sat — with two columns added: ``ajustes`` names what
    changed on the row and ``problemas`` what is still wrong with it,
    ';'-joined codes from ``FIX_CODES`` and ``ISSUE_CODES``, "" where nothing
    — and a dict of counts: ``leg_minutes_corrected``, ``untimed_trips``,
    ``untimed_persons``, ``times_from_duplicate``, ``times_from_neighbours``,
    ``motives_from_duplicate``, ``motives_from_neighbours``,
    ``recoded_returns``, ``returns_copying_origin``, ``arrivals_home``,
    ``daycare_escorts``, ``home_to_home``, ``returns_elsewhere``,
    ``next_morning_returns``, ``duplicate_returns`` (the rows dropped),
    ``origins_repaired`` (at home after a return), ``origins_continued``
    (elsewhere), ``first_origins_copied``, ``first_origins_answered``,
    ``first_origins_typed``, ``first_trips_rotated``, ``answers_set_home``
    (persons), ``first_return_answers`` (persons), ``short_returns_home``,
    ``short_returns_other``, ``start_times_edited``, ``chains_repaired``,
    ``early_starts_read``, ``long_errands_read``, ``chains_read_after_the_slide`` (chains),
    ``starts_at_arrival``, and one ``left_<code>`` entry per issue left.
    """
    trips = trips.sort_index()
    n = len(trips)
    fixes = np.full(n, "", dtype=object)
    issues = np.full(n, "", dtype=object)
    home = _home_zone(trips, viv)
    home_zone = (
        viv.centralidad.astype(str).reindex(trips.index.get_level_values("folio_vivienda")).to_numpy()
        if "centralidad" in viv
        else None
    )
    untimed = (trips.hora_inicio_h.isna() | trips.hora_inicio_m.isna()).to_numpy()

    # the travel minutes a review corrected, before any rule reads them
    corrected = np.zeros(n, dtype=bool)
    if minutes is not None and len(minutes):
        trips, legs, corrected = _correct_minutes(trips, legs, minutes)
    _add_code(fixes, corrected, "minutos:revision")

    # 1. what the questionnaire lost
    trips, imputed = _impute_untimed_trips(trips, home, hab, legs)
    for key, code in (
        ("time_from_duplicate", "hora:duplicado"),
        ("motive_from_duplicate", "motivo:duplicado"),
        ("time_from_neighbours", "hora:vecinos"),
        ("motive_from_neighbours", "motivo:vecinos"),
    ):
        _add_code(fixes, imputed[key], code)
    duplicate = imputed["duplicate"]  # dropped at the end; every rule steps over them

    # 2. motives and destinations
    trips, recoded = _recode_returns_by_destination(trips, home)
    _add_code(fixes, recoded, "motivo:tipo_destino")
    trips = trips.copy()
    went_home = _returns_copying_origin(trips, home) & ~duplicate
    trips.loc[went_home, "destino"] = home[went_home]
    if home_zone is not None and "zona_destino" in trips:
        trips.loc[went_home, "zona_destino"] = home_zone[went_home]
    _add_code(fixes, went_home, "destino:copia")
    arrived = _arrivals_home(trips, home) & ~duplicate
    trips.loc[arrived, "motivo_viaje"] = HOME_MOTIVE
    _add_code(fixes, arrived, "motivo:casa")
    escorts = _daycare_escorts(trips, hab)
    trips.loc[escorts, "motivo_viaje"] = ESCORT
    _add_code(fixes, escorts, "motivo:guarderia")

    # 3. the rows that are not trips, and where the other returns went
    at_home = _home_to_home_returns(trips, home, skip=duplicate)
    rest = np.flatnonzero(~(at_home | duplicate))
    elsewhere, motive, kind = _returns_elsewhere(trips.iloc[rest], home[rest])
    went = _expand(elsewhere, rest, n)
    trips.loc[went, "motivo_viaje"] = motive[elsewhere]
    trips.loc[went, "tipo_lugar_destino"] = kind[elsewhere]
    _add_code(fixes, went, "motivo:lugar")
    not_trips = _mark_non_trips(trips, home, skip=duplicate)
    for code, mask in not_trips.items():
        _add_code(issues, mask, code)
    is_trip = ~(not_trips["regreso_en_casa"] | not_trips["regreso_dia_siguiente"] | duplicate)
    where = np.flatnonzero(is_trip)
    chain = trips[is_trip]
    chain_home = home[is_trip]

    # 4. where the trips start and end
    chain, after_return, continued = _start_where_previous_ended(chain, chain_home)
    _add_code(fixes, _expand(after_return, where, n), "origen:casa")
    _add_code(fixes, _expand(continued, where, n), "origen:anterior")
    zoned = home_zone is not None and {"zona_origen", "zona_destino"} <= set(chain.columns)
    chain_zone = home_zone[is_trip] if home_zone is not None else None
    copied, answered_home, typed = _first_origins_at_home(chain, chain_home)
    to_home = copied | answered_home | typed
    chain = chain.copy()
    if to_home.any():
        chain.loc[to_home, "origen"] = chain_home[to_home]
        if zoned:
            chain.loc[to_home, "zona_origen"] = chain_zone[to_home]
    # a first trip recorded from a place to home, the next trip from home onwards: home to the place, and on from there
    rotated = _rotated_first_trips(chain, chain_home)
    onwards, retyped = np.zeros(len(chain), dtype=bool), np.zeros(len(chain), dtype=bool)
    if rotated.any():
        pos = np.flatnonzero(rotated)
        cpid = _person_ids(chain)
        after = pos + 1 < len(chain)
        after[after] = cpid[pos[after] + 1] == cpid[pos[after]]
        onwards[pos[after] + 1] = True
        column = chain.columns.get_loc
        place = chain.origen.astype(str).to_numpy()[pos]
        chain.iloc[pos, column("origen")] = chain_home[pos]
        chain.iloc[pos, column("destino")] = place
        chain.iloc[pos[after] + 1, column("origen")] = place[after]
        # recorded arriving at 'Su casa', the trip goes to the place the day-start answer names
        answer = chain.tipo_lugar_origen.astype(object).to_numpy()[pos]
        named = (chain.tipo_lugar_destino.astype(object).to_numpy()[pos] == HOME_PLACE) & pd.notna(answer) & (
            answer != HOME_PLACE)
        chain.iloc[pos[named], column("tipo_lugar_destino")] = answer[named]
        retyped[pos[named]] = True
        if zoned:
            place_zone = chain.zona_origen.astype(object).to_numpy()[pos]
            chain.iloc[pos, column("zona_origen")] = chain_zone[pos]
            chain.iloc[pos, column("zona_destino")] = place_zone
            chain.iloc[pos[after] + 1, column("zona_origen")] = place_zone[after]
    # the person's answer to where the day started goes with it, on every row the person has
    persons = trips.index.droplevel("folio_viaje")
    moved = persons.isin(set(chain.index[copied | typed | rotated].droplevel("folio_viaje")))
    answered = moved & (trips.tipo_lugar_origen != HOME_PLACE).to_numpy()
    trips.loc[answered, "tipo_lugar_origen"] = HOME_PLACE
    _add_code(fixes, _expand(copied, where, n), "origen:copia")
    _add_code(fixes, _expand(answered_home, where, n), "origen:respuesta")
    _add_code(fixes, _expand(typed, where, n), "origen:tipo_copia")
    _add_code(fixes, _expand(rotated, where, n), "origen:rotado")
    _add_code(fixes, _expand(rotated, where, n), "destino:rotado")
    _add_code(fixes, _expand(retyped, where, n), "tipo_destino:rotado")
    _add_code(fixes, _expand(onwards & ~continued, where, n), "origen:anterior")
    _add_code(fixes, answered, "tipo_origen:copia")
    given = _first_return_answers(chain, chain_home, hab)
    returned = persons.isin(list(given))
    if returned.any():
        trips.loc[returned, "tipo_lugar_origen"] = [given[p] for p in persons[returned]]
    _add_code(fixes, returned, "tipo_origen:regreso")
    chain["tipo_lugar_origen"] = trips.tipo_lugar_origen[is_trip].to_numpy()
    last_home, other = _short_returns(chain, chain_home)
    chain.loc[last_home, "destino"] = chain_home[last_home]
    if zoned:
        chain.loc[last_home, "zona_destino"] = chain_zone[last_home]
    chain.loc[other, ["motivo_viaje", "tipo_lugar_destino"]] = OTHER
    _add_code(fixes, _expand(last_home, where, n), "destino:casa")
    _add_code(fixes, _expand(other, where, n), "motivo:otros")

    # 5. start times
    locked = (imputed["time_from_duplicate"] | imputed["time_from_neighbours"])[is_trip]
    chain, edit, slid, time_counts = _read_start_times(chain, legs, locked)
    for name in {e for e in edit if e}:
        _add_code(fixes, _expand(edit == name, where, n), f"hora:{name}")
    _add_code(fixes, _expand(slid, where, n), "hora:llegada")
    for column in ("origen", "zona_origen", "destino", "zona_destino", "motivo_viaje", "tipo_lugar_destino",
                   "hora_inicio_h", "hora_inicio_m"):
        if column in chain:
            trips.loc[chain.index, column] = chain[column].to_numpy()

    left = _remaining_issues(chain, chain_home, legs, locked | (edit != "") | slid)
    for code, mask in left.items():
        _add_code(issues, _expand(mask, where, n), code)
    untimed_persons = int(trips.index[untimed].droplevel("folio_viaje").nunique())
    keep = ~duplicate  # the duplicates' answers now sit on the returns they repeat
    trips = trips[keep].copy()
    trips[FIX_FLAG] = fixes[keep]
    trips[ISSUE_FLAG] = issues[keep]

    counts = {
        "leg_minutes_corrected": int(corrected.sum()),
        "untimed_trips": int(untimed.sum()),
        "untimed_persons": untimed_persons,
        "times_from_duplicate": int(imputed["time_from_duplicate"].sum()),
        "times_from_neighbours": int(imputed["time_from_neighbours"].sum()),
        "motives_from_duplicate": int(imputed["motive_from_duplicate"].sum()),
        "motives_from_neighbours": int(imputed["motive_from_neighbours"].sum()),
        "recoded_returns": int(recoded.sum()),
        "returns_copying_origin": int(went_home.sum()),
        "arrivals_home": int(arrived.sum()),
        "daycare_escorts": int(escorts.sum()),
        "home_to_home": int(not_trips["regreso_en_casa"].sum()),
        "returns_elsewhere": int(went.sum()),
        "next_morning_returns": int(not_trips["regreso_dia_siguiente"].sum()),
        "duplicate_returns": int(duplicate.sum()),
        "origins_repaired": int(after_return.sum()),
        "origins_continued": int(continued.sum()),
        "first_origins_copied": int(copied.sum()),
        "first_origins_answered": int(answered_home.sum()),
        "first_origins_typed": int(typed.sum()),
        "first_trips_rotated": int(rotated.sum()),
        "answers_set_home": int(trips.index[answered[keep]].droplevel("folio_viaje").nunique()),
        "first_return_answers": len(given),
        "short_returns_home": int(last_home.sum()),
        "short_returns_other": int(other.sum()),
        "start_times_edited": int((edit != "").sum()),
        **time_counts,
        "starts_at_arrival": int(slid.sum()),
        **{f"left_{code}": int(mask.sum()) for code, mask in left.items()},
    }
    return trips, counts


def _expand(mask: np.ndarray, where: np.ndarray, n: int) -> np.ndarray:
    """A mask over a sub-table lifted to the full table: ``where`` holds the sub-table's row positions."""
    full = np.zeros(n, dtype=bool)
    full[where[np.asarray(mask, dtype=bool)]] = True
    return full
