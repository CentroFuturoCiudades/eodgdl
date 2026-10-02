"""Lazy loaders for bundled package config: the IMEPLAN column rename map, the hand decisions, the leg minutes and
the TAZ ids.

This is *code config*, not data: ``rename_imeplan`` and ``load_eod`` need it to run, so it
lives inside the package, beside the code that reads it, not under ``data/``.
"""
from __future__ import annotations

import functools
import json
from importlib import resources


@functools.cache
def imeplan_rename_map() -> dict:
    """{raw IMEPLAN header: snake_case name} for each of the three tables.

    Keyed by "habitantes", "viajes", "viviendas". The categorical levels and their
    ordering live in schemas.py, not here.
    """
    text = (resources.files("eodgdl") / "imeplan_rename_map.json").read_text(encoding="utf-8")
    return json.loads(text)


@functools.cache
def chain_decisions() -> "pd.DataFrame":
    """The hand decisions over the trip chains that ``load_eod`` applies: ``eodgdl/revisions/chains.csv.gz``.

    One row per trip and field (``household, person, trip, field, before, after, note,
    source``), each made against the chain rules' output; a review round merges into it
    (:func:`eodgdl.review.merge_decisions`). See the README there. Keys are integers,
    every other column text; the frame is shared, so do not modify it.
    """
    import pandas as pd

    with (resources.files("eodgdl") / "revisions" / "chains.csv.gz").open("rb") as fh:
        decisions = pd.read_csv(fh, compression="gzip", dtype=str, keep_default_na=False)
    for key in ("household", "person", "trip"):
        decisions[key] = decisions[key].astype(int)
    return decisions


@functools.cache
def leg_minutes() -> "pd.DataFrame":
    """The travel minutes a review round corrected, read before the chain rules: ``eodgdl/revisions/leg_minutes.csv.gz``.

    One row per trip (``household, person, trip, before, after, note, source``): the trip's
    reported minutes summed over its legs, and the minutes it takes instead, which
    :func:`eodgdl.chains.clean_trip_chains` spreads over its legs in proportion before any rule
    reads them. See the README there. Keys and minutes are integers, the rest text; the frame
    is shared, so do not modify it.
    """
    import pandas as pd

    with (resources.files("eodgdl") / "revisions" / "leg_minutes.csv.gz").open("rb") as fh:
        table = pd.read_csv(fh, compression="gzip", dtype=str, keep_default_na=False)
    for key in ("household", "person", "trip", "before", "after"):
        table[key] = table[key].astype(int)
    return table


@functools.cache
def taz_ids() -> "pd.DataFrame":
    """The integer TAZ of every AGEB of the zone system and every access point: ``eodgdl/taz_ids.csv``.

    One row per code (``AGEB, TAZ, note, source``), every column text, as committed; a table we keep, ids never
    renumbered (the reweight README's "TAZ ids"). A code still waiting for its TAZ has an empty ``TAZ`` and says why
    in ``note``. :func:`eodgdl.reweight.zoning.check_taz_ids` holds it to the zone
    system; the frame is shared, so do not modify it.
    """
    import pandas as pd

    with (resources.files("eodgdl") / "taz_ids.csv").open("rb") as fh:
        return pd.read_csv(fh, dtype=str, keep_default_na=False)


def taz_of(codes) -> "pd.Series":
    """The integer TAZ of each code (an AGEB or access point, as text), indexed like ``codes``; raises naming every
    code :func:`taz_ids` lacks or leaves pending (empty)."""
    table = taz_ids()
    mapping = table[table.TAZ != ""].set_index("AGEB").TAZ.astype("int64")
    out = codes.map(mapping)
    if out.isna().any():
        missing = sorted(set(codes[out.isna()]))
        raise ValueError(f"codes with no TAZ in taz_ids.csv ({len(missing)}): {missing[:20]}")
    return out.astype("int64")
