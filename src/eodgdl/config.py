"""The processing's run parameters: ``config/config.yaml`` in the clone (``$EODGDL_CONFIG`` names another file).

eodgdl reads the run's weight from it (:func:`weight`: the EOD sources of eodgdl.impute read the entry of their
``run_weights`` it names); the drivers read the rest (``python -m eodgdl.config NAME`` prints one parameter). Every
parameter must be there with its type, and nothing else: a typo fails the run instead of being ignored.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from eodgdl.data import REPO

PARAMETERS = {"weight": str, "bootstrap": int, "draws": int, "draw": int}


def path() -> Path:
    """The run's configuration file: ``$EODGDL_CONFIG``, else the clone's ``config/config.yaml``."""
    if value := os.environ.get("EODGDL_CONFIG"):
        return Path(value).expanduser().resolve()
    return REPO / "config" / "config.yaml"


def load() -> dict:
    """The run parameters, checked: each of ``PARAMETERS`` present with its type, no other key."""
    import yaml

    file = path()
    values = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    problems = [f"{name}: missing" for name in PARAMETERS if name not in values]
    problems += [f"{name}: not a run parameter (known: {', '.join(PARAMETERS)})" for name in values if name not in PARAMETERS]
    problems += [f"{name}: {values[name]!r} is not {kind.__name__}" for name, kind in PARAMETERS.items()
                 if name in values and type(values[name]) is not kind]
    if problems:
        raise ValueError(f"{file}: " + "; ".join(problems))
    return values


def weight() -> str:
    """The run's weight: the name every EOD source's ``run_weights`` resolves (``design``, ``tmg``)."""
    return load()["weight"]


if __name__ == "__main__":
    print(load()[sys.argv[1]])
