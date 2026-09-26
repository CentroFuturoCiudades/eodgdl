"""Lazy loaders for bundled package config: the IMEPLAN column rename map and the hand revisions.

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


def pass_number(name: str) -> int:
    """The n of a frozen pass's file name ``chains_<n>.csv.gz``."""
    return int(name.removeprefix("chains_").removesuffix(".csv.gz"))


def pass_files() -> list:
    """The frozen passes under ``eodgdl/revisions/``, in the order they apply."""
    root = resources.files("eodgdl") / "revisions"
    return sorted(
        (f for f in root.iterdir() if f.name.startswith("chains_") and f.name.endswith(".csv.gz")),
        key=lambda f: pass_number(f.name),
    )


@functools.cache
def chain_revisions() -> list[tuple[str, "pd.DataFrame"]]:
    """The frozen hand passes over the trip chains, in the order they apply: [(file name, edits)].

    Each file under ``eodgdl/revisions/`` (``chains_<n>.csv.gz``) holds the edits one review
    sheet gave (:func:`eodgdl.review.freeze_edits`); see the README there. Keys are integers,
    every other column text.
    """
    import pandas as pd

    out = []
    for f in pass_files():
        with f.open("rb") as fh:
            edits = pd.read_csv(fh, compression="gzip", dtype=str, keep_default_na=False)
        for key in ("household", "person", "trip"):
            edits[key] = edits[key].astype(int)
        out.append((f.name, edits))
    return out
