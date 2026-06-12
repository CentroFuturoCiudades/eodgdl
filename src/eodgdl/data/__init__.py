"""On-demand access to the IMEPLAN EOD 2023 data files.

Files are fetched from the repo's data mirror via Pooch (downloaded once, then cached),
unless ``$EODGDL_DATA_DIR`` points at a local copy (e.g. a clone's ``data/`` dir).
"""
from __future__ import annotations

import os
from pathlib import Path

from eodgdl.data._catalog import FILES, SURVEY_FILES, ZONE_FILES
from eodgdl.data._paths import get_data_dir, get_pooch_cache_dir
from eodgdl.data._registry import POOCH


def resolve(filename: str) -> Path:
    """Return a local path to ``filename`` from the EOD dataset.

    If ``$EODGDL_DATA_DIR`` is set and contains ``filename`` (e.g. a local clone's
    ``data/`` directory), that file is used directly. Otherwise the file is fetched from
    the mirror via Pooch (downloaded once, then cached under the platform cache dir).
    """
    if env := os.environ.get("EODGDL_DATA_DIR"):
        local = Path(env).expanduser() / filename
        if local.exists():
            return local
    return Path(POOCH.fetch(filename))


__all__ = [
    "POOCH",
    "resolve",
    "get_data_dir",
    "get_pooch_cache_dir",
    "FILES",
    "SURVEY_FILES",
    "ZONE_FILES",
]
