"""Load and clean the IMEPLAN Guadalajara EOD 2023 origin-destination survey."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from eodgdl._resources import imeplan_rename_map
from eodgdl.chains import PERSON, clean_trip_chains
from eodgdl.data._catalog import HABITANTES_CSV, VIAJES_CSV, VIVIENDAS_CSV
from eodgdl.schemas import hab_schema, trips_schema, viv_schema

log = logging.getLogger(__name__)


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
DIARY_FLAG = (
    "diario_repetido"  # column load_eod adds to hab: True when the diary repeats
)
_DIARY_TWIN_MAX_OFFSET = (
    10  # minutes: the largest start-time difference a twin may show
)


class EODTables(NamedTuple):
    """The four linked levels of the cleaned EOD survey."""

    viv: pd.DataFrame  # dwellings, indexed by folio_vivienda
    hab: pd.DataFrame  # persons, indexed by (folio_vivienda, folio_habitante)
    trips: (
        pd.DataFrame
    )  # trips, indexed by (folio_vivienda, folio_habitante, folio_viaje)
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



# --------------------------------------------------------------- repeated diaries


def _as_text(df: pd.DataFrame) -> pd.Series:
    """The rows of ``df`` joined with '|', missing values as empty strings, categoricals as their labels."""
    cols = [
        df[c].astype(object).where(df[c].notna(), "").astype(str) for c in df.columns
    ]
    out = cols[0]
    for c in cols[1:]:
        out = out + "|" + c
    return out


def _diary_signatures(trips: pd.DataFrame, legs: pd.DataFrame | None) -> pd.DataFrame:
    """One row per trip: its exact signature, whether a leg is not on foot, and its start minute."""
    sig = _as_text(
        trips[
            ["origen", "destino", "motivo_viaje", "tipo_lugar_destino", "n_traslados"]
        ]
    )
    mode_cols = [f"traslado{i}_medio" for i in range(1, 6)]
    if "traslado1_medio" in trips.columns:
        leg_cols = [
            f"traslado{i}_{k}" for i in range(1, 6) for k in ("medio", "min", "pago")
        ]
        sig = sig + "|" + _as_text(trips[leg_cols])
        modes = trips[mode_cols]
        moving = (
            ((modes.astype(object) != "A PIE") & modes.notna()).any(axis=1).to_numpy()
        )
    else:
        if legs is None:
            raise ValueError("trips has no traslado columns; pass legs")
        leg_text = _as_text(legs[["traslado_medio", "traslado_min", "traslado_pago"]])
        by_trip = leg_text.groupby(level=trips.index.names, sort=False).agg("|".join)
        sig = sig + "|" + by_trip.reindex(trips.index).fillna("")
        moving = (
            (legs.traslado_medio.astype(object) != "A PIE")
            .groupby(level=trips.index.names, sort=False)
            .any()
            .reindex(trips.index)
            .fillna(False)
            .to_numpy()
        )
    start = trips.hora_inicio_h.astype(float) * 60 + trips.hora_inicio_m.astype(float)
    return pd.DataFrame(
        {"sig": sig, "moving": moving, "start": start.to_numpy()}, index=trips.index
    )


def flag_repeated_diaries(
    trips: pd.DataFrame,
    hab: pd.DataFrame,
    viv: pd.DataFrame,
    legs: pd.DataFrame | None = None,
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
    diary = pd.DataFrame(
        {
            "sig": person.sig.agg(" ## ".join),
            "n": person.size(),
            "moving": person.moving.any(),
            "starts": person.start.agg(list),
        }
    )
    diary["ageb"] = viv.ageb.reindex(
        diary.index.get_level_values("folio_vivienda")
    ).to_numpy()
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
                if (
                    not np.isnan(offsets).any()
                    and offsets.max() <= _DIARY_TWIN_MAX_OFFSET
                ):
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

    By default the trip chains are cleaned (:func:`eodgdl.chains.clean_trip_chains`): the
    325 trips with no start time (and no motive) are imputed — 37 from the
    home-to-home return that duplicates them, the rest from their nearest
    timed trips — as are the 7 timed trips with no motive; 84 mislabelled
    'Regresar a Casa' trips take their destination type's motive; 491
    returns home made from home are kept and marked as non-trips; 124 trips
    that follow a return home start in the home zone; and 2,032 mistyped
    start hours in 1,552 chains are repaired. One kind of row is dropped: the
    38 home-to-home returns that duplicate an imputed return, whose time and
    motive now sit on the return they repeat; ``hab.viajes_contados`` is
    reduced by one for those persons so it still counts the person's trip
    rows, and ``folio_viaje`` keeps its shipped numbering with a gap there.
    ``trips`` gains two columns: ``ajustes`` names what changed on each row
    and ``problemas`` what is still wrong with it (';'-joined codes, see
    ``FIX_CODES`` and ``ISSUE_CODES``; ``non_trips()`` picks out the rows the
    model build leaves out). Pass ``clean_chains=False`` for the survey as
    shipped, on which the expansion factors reconcile to the published
    totals; the cleaned table is short the 38 duplicates' weight.

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
        shipped = df_trips.index
        df_trips, counts = clean_trip_chains(df_trips, df_viv, hab=df_hab)
        # the dropped duplicates leave viajes_contados one too high for their persons
        lost = (
            shipped.difference(df_trips.index).droplevel("folio_viaje").value_counts()
        )
        df_hab.loc[lost.index, "viajes_contados"] -= lost.to_numpy()
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
