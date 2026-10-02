"""Load and clean the IMEPLAN Guadalajara EOD 2023 origin-destination survey."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from eodgdl._resources import imeplan_rename_map, leg_minutes
from eodgdl.chains import PERSON, clean_trip_chains
from eodgdl.data import HABITANTES_CSV, VIAJES_CSV, VIVIENDAS_CSV
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


# What load_eod() reads besides the survey files: its code and bundled configuration (the chain rules, the hand
# decisions and leg minutes, the schemas, the rename map). The workflow's eod rule lists them as its code.
LOADER_FILES = ("eod.py", "chains.py", "review.py", "schemas.py", "_resources.py", "imeplan_rename_map.json")


def loader_files() -> list[Path]:
    """The files of :data:`LOADER_FILES` and ``revisions/*.csv.gz``, as paths."""
    package = Path(__file__).parent
    return [package / name for name in LOADER_FILES] + sorted((package / "revisions").glob("*.csv.gz"))


def tables_digest(tables: EODTables) -> str:
    """One sha256 over what ``tables`` hold: each table's index names, columns and dtypes (a categorical's levels and
    their order included) and every value with its index. Two loads that give the same tables give the same digest,
    whatever code produced them."""
    digest = hashlib.sha256()
    for name, frame in zip(tables._fields, tables):
        schema = [name, [str(level) for level in frame.index.names], [(str(column), repr(dtype)) for column, dtype in frame.dtypes.items()]]
        digest.update(json.dumps(schema, ensure_ascii=False).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


class EODStages(NamedTuple):
    """The survey at the three points ``load_eod`` can stop at, from one read of the files."""

    shipped: EODTables  # load_eod(clean_chains=False): as shipped, typed and linked
    rules: EODTables  # load_eod(revise_chains=False): the chain rules applied
    revised: EODTables  # load_eod(): the hand revisions applied on top


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


def repeated_diary_pairs(
    trips: pd.DataFrame,
    hab: pd.DataFrame,
    viv: pd.DataFrame,
    legs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The pairs of persons whose diaries repeat across two households.

    The rule is :func:`flag_repeated_diaries`'s. One row per pair:
    ``folio_vivienda_a``, ``folio_habitante_a``, ``folio_vivienda_b``,
    ``folio_habitante_b`` (the first in the survey's order as ``a``) and
    ``max_offset``, the largest start-time difference in minutes. A person
    repeated in several households is in several pairs. Read the survey as
    shipped (``load_stages().shipped``): the chain rules edit the start times
    and motives the rule compares.
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
    pairs = []
    for _, group in diary.groupby(["ageb", "fecha", "sig"], sort=False, observed=True):
        if group.index.get_level_values("folio_vivienda").nunique() < 2:
            continue
        keys = sorted(group.index)
        starts = dict(
            zip(group.index, (np.asarray(s, dtype=float) for s in group.starts))
        )
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if a[0] == b[0]:
                    continue
                offsets = np.abs(starts[a] - starts[b])
                if (
                    not np.isnan(offsets).any()
                    and offsets.max() <= _DIARY_TWIN_MAX_OFFSET
                ):
                    pairs.append((*a, *b, float(offsets.max())))
    columns = [
        "folio_vivienda_a",
        "folio_habitante_a",
        "folio_vivienda_b",
        "folio_habitante_b",
        "max_offset",
    ]
    return pd.DataFrame(pairs, columns=columns)


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
    interview date. Returns a boolean Series over ``hab.index``; the pairs
    themselves are :func:`repeated_diary_pairs`.
    """
    pairs = repeated_diary_pairs(trips, hab, viv, legs)
    flagged = pd.Series(False, index=hab.index, name=DIARY_FLAG)
    for side in ("a", "b"):
        members = pd.MultiIndex.from_arrays(
            [pairs[f"folio_vivienda_{side}"], pairs[f"folio_habitante_{side}"]]
        )
        flagged[hab.index.isin(members)] = True
    return flagged


# ------------------------------------------------------------------- loading


def _resolve_csv(eod_path: Path | None, filename: str) -> Path:
    if eod_path is not None:
        return Path(eod_path) / filename
    from eodgdl.data import resolve

    return resolve(filename)


def _split_legs(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The trips without their ``traslado{1..5}_*`` columns, and those columns unpivoted into legs."""
    legs = (
        pd.concat(
            [
                trips[
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
    trips = trips.drop(columns=[c for c in trips.columns if c.startswith("traslado")])
    return trips, legs


def load_stages(eod_path: Path | None = None, *, verbose: bool = False, skip_stale: bool = False) -> EODStages:
    """The survey as shipped, after the chain rules and after the hand revisions, from one read of the files.

    The three tables ``load_eod`` returns with ``clean_chains=False``, with
    ``revise_chains=False`` and by default, for the price of the last: a
    caller that compares stages — a review sheet, the chain browser, a
    report — reads and cleans the survey once. The stages share ``viv``;
    treat them as read-only.

    A hand decision the rules' output no longer supports — a change to the
    rules moved the value it was made against — fails the load, as in
    ``load_eod``; with ``skip_stale`` it is set aside with a warning and
    ``revised`` holds the others, so the review tools can put the stale ones
    on a sheet (:func:`eodgdl.review.stale_decisions`).
    """
    return EODStages(*_load(eod_path, "revised", verbose, skip_stale))


def load_eod(
    eod_path: Path | None = None,
    *,
    verbose: bool = False,
    clean_chains: bool = True,
    revise_chains: bool = True,
) -> EODTables:
    """Load and clean the EOD survey at four linked levels.

    With no argument the three master CSVs are read from the clone's ``data/`` (or
    ``$EODGDL_DATA_DIR``); pass ``eod_path`` to read them from another directory.
    :func:`load_stages` returns all three stages below from one read.

    By default the trip chains are cleaned (:func:`eodgdl.chains.clean_trip_chains`, whose
    docstring and rules carry the evidence and the counts): the travel minutes a review
    corrected (``eodgdl/revisions/leg_minutes.csv.gz``) go in first, since the rules read
    them; the trips with no start time or motive are imputed; mislabelled returns home and
    activities that arrive home are recoded, daycare trips from age 12 become escorts; the
    returns home made from home, and the next-morning returns that close a night-shift day,
    are kept and marked as non-trips; every trip starts where the previous one ended, the
    first trips recorded leaving another zone start at home or, recorded rotated, go from
    home to the place they name; the returns left away from home go home or to a place
    nothing names; mistyped start hours are read by the fewest typo edits, a day read past
    midnight holds together after it, and starts that fall a little before the previous
    arrival move to it. One kind of row is dropped: the
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

    After the rules come the hand revisions (``revise_chains``, on by
    default, ignored without ``clean_chains``): the hand decisions in
    ``eodgdl/revisions/chains.csv.gz``, one per trip and field, each made
    against the rules' output and merged there from the review sheets
    (:mod:`eodgdl.review`; the day-start answer is one per person, trip 0),
    are applied in one pass. A row dropped by hand is
    gone (its legs too, ``viajes_contados`` kept in step), ``ajustes`` then
    says field by field whether a value that differs from the shipped one is
    the rules' (their code) or was set by hand (``<field>:revision``), and
    ``problemas`` is recomputed. Pass ``revise_chains=False`` for the rules'
    output alone. A decision whose ``before`` the rules no longer give — a
    rule changed under it — fails the load; ``load_stages(skip_stale=True)``
    sets such decisions aside so that ``eodgdl review export --stale`` can put
    them on a sheet to decide again.

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
    stop = "revised" if clean_chains and revise_chains else "rules" if clean_chains else "shipped"
    return _load(eod_path, stop, verbose)[-1]


def _load(eod_path: Path | None, stop: str, verbose: bool, skip_stale: bool = False) -> list[EODTables]:
    """The stages up to ``stop`` (``shipped``, ``rules`` or ``revised``), in that order, from one read."""
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

    stages = [EODTables(df_viv, df_hab.copy(), *_split_legs(df_trips))]
    if stop != "shipped":
        # the travel minutes a review corrected go in first: the rules read them (eodgdl/revisions/leg_minutes.csv.gz)
        cleaned, counts = clean_trip_chains(df_trips, df_viv, hab=df_hab, minutes=leg_minutes())
        # the dropped duplicates leave viajes_contados one too high for their persons
        lost = (
            df_trips.index.difference(cleaned.index).droplevel("folio_viaje").value_counts()
        )
        df_hab.loc[lost.index, "viajes_contados"] -= lost.to_numpy()
        log.info("trip chains cleaned: %s", counts)
        if verbose:
            print("Trip chains cleaned:", counts)
        stages.append(EODTables(df_viv, df_hab, *_split_legs(cleaned)))
    if stop == "revised":
        # imported here: eodgdl.review builds on EODTables
        from eodgdl.review import apply_revisions

        stages.append(apply_revisions(stages[1], stages[0], skip_stale=skip_stale))
        log.info("hand revisions applied: %d trips", len(stages[-1].trips))

    if verbose:
        print("DF shapes viv/hab/viaj/legs", *(df.shape for df in stages[-1]))
    return stages
