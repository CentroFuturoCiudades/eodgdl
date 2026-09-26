"""Lazy loaders for bundled package config: the IMEPLAN column rename map and the hand decisions.

This is *code config*, not data: ``rename_imeplan`` and ``load_eod`` need it to run, so it
ships inside the package (not via the data mirror) and the package works offline.
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
