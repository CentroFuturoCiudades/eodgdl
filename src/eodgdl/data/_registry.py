"""Global Pooch registry for the IMEPLAN EOD 2023 data files and model bundles.

No network traffic occurs at import time — files are only downloaded on the first
``POOCH.fetch()`` call for each file.
"""
from __future__ import annotations

import os
from importlib import resources

import pooch

from eodgdl.data._catalog import MODEL_FILES
from eodgdl.data._paths import get_pooch_cache_dir

# The input data live in the ``data/`` directory of the eodgdl repo itself, the fitted model
# bundles in its ``models/`` directory. Pooch fetches them over plain HTTPS from
# raw.githubusercontent.com, pinned to a tag/commit for reproducibility (registry keys are
# bare filenames → ``base_url`` ends in the ``data/`` path, and each model file has a URL of
# its own under ``models/``). Override $EODGDL_BASE_URL (data) and $EODGDL_MODELS_URL
# (models) to point at a fork/mirror or a different ref (keep the trailing "/").
GH = "CentroFuturoCiudades/eodgdl"
REF = "v0.9.0"
_ROOT_URL = f"https://raw.githubusercontent.com/{GH}/{REF}/"
_BASE_URL = os.environ.get("EODGDL_BASE_URL", _ROOT_URL + "data/")
_MODELS_URL = os.environ.get("EODGDL_MODELS_URL", _ROOT_URL + "models/")

POOCH = pooch.create(
    path=get_pooch_cache_dir(),
    base_url=_BASE_URL,
    registry={},
    urls={name: _MODELS_URL + name for name in MODEL_FILES},
    env="EODGDL_CACHE_DIR",
)

# Show a tqdm progress bar by default on every fetch (the trips master is ~48 MB). Pooch
# only draws the bar when actually downloading, so cache hits stay silent. Callers can pass
# progressbar=False to opt out.
_pooch_fetch = POOCH.fetch


def _fetch_with_progress(fname, *args, progressbar=True, **kwargs):
    return _pooch_fetch(fname, *args, progressbar=progressbar, **kwargs)


POOCH.fetch = _fetch_with_progress

_reg = resources.files("eodgdl.data") / "registry.txt"
with resources.as_file(_reg) as _p:
    POOCH.load_registry(_p)
