"""On-demand access to the IMEPLAN EOD 2023 data files and the fitted model bundles.

Files are fetched from the repo's mirror via Pooch (downloaded once, then cached): the input
data from its ``data/`` directory, the model bundles from its ``models/`` directory. A local
copy is read instead where one exists: ``$EODGDL_DATA_DIR`` for the data (e.g. a clone's
``data/`` dir), ``$EODGDL_MODELS_DIR`` for the models, else the ``models/`` dir beside it.
"""
from __future__ import annotations

import os
from pathlib import Path

from eodgdl.data._catalog import FILES, MODEL_FILES, SURVEY_FILES, ZONE_FILES
from eodgdl.data._paths import get_data_dir, get_models_dir, get_pooch_cache_dir
from eodgdl.data._registry import POOCH


def resolve(filename: str) -> Path:
    """Return a local path to ``filename`` from the EOD dataset or its model bundles.

    A model bundle (``MODEL_FILES``) is read from :func:`get_models_dir` (``$EODGDL_MODELS_DIR``,
    else the ``models/`` directory beside ``$EODGDL_DATA_DIR``: a local clone's) when it is
    there, any other file from ``$EODGDL_DATA_DIR`` (e.g. a local clone's ``data/``
    directory). Otherwise the file is fetched from the mirror via Pooch (downloaded once,
    then cached under the platform cache dir).
    """
    if filename in MODEL_FILES:
        local = get_models_dir()
    elif env := os.environ.get("EODGDL_DATA_DIR"):
        local = Path(env).expanduser()
    else:
        local = None
    if local is not None and (local / filename).exists():
        return local / filename
    return Path(POOCH.fetch(filename))


__all__ = [
    "POOCH",
    "resolve",
    "get_data_dir",
    "get_models_dir",
    "get_pooch_cache_dir",
    "FILES",
    "SURVEY_FILES",
    "ZONE_FILES",
    "MODEL_FILES",
]
