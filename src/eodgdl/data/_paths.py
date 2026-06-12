"""Platform-appropriate directory resolution via platformdirs."""
from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

_APP = "eodgdl"


def get_data_dir() -> Path:
    """Return the eodgdl user-data directory.

    Override with ``$EODGDL_DATA_DIR`` (e.g. point it at a local clone's ``data/`` dir
    to read files locally instead of fetching them).
    """
    if env := os.environ.get("EODGDL_DATA_DIR"):
        return Path(env).expanduser().resolve()
    return Path(user_data_dir(_APP))


def get_pooch_cache_dir() -> Path:
    """Return the directory where Pooch caches downloaded files.

    Override with ``$EODGDL_CACHE_DIR``.
    """
    if env := os.environ.get("EODGDL_CACHE_DIR"):
        return Path(env).expanduser().resolve()
    return Path(user_cache_dir(_APP))
