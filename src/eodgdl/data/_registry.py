"""Global Pooch registry for the IMEPLAN EOD 2023 data files.

No network traffic occurs at import time — files are only downloaded on the first
``POOCH.fetch()`` call for each file.
"""
from __future__ import annotations

import os
from importlib import resources

import pooch

from eodgdl.data._paths import get_pooch_cache_dir

# The data files live in the ``data/`` directory of the eodgdl repo itself. Pooch fetches
# them over plain HTTPS from raw.githubusercontent.com, pinned to a tag/commit for
# reproducibility (registry keys are bare filenames → ``base_url`` ends in the ``data/``
# path). Override $EODGDL_BASE_URL to point at a fork/mirror or a different ref (keep the
# trailing "/").
GH = "CentroFuturoCiudades/eodgdl"
REF = "v0.1.0"
_BASE_URL = os.environ.get(
    "EODGDL_BASE_URL",
    f"https://raw.githubusercontent.com/{GH}/{REF}/data/",
)

POOCH = pooch.create(
    path=get_pooch_cache_dir(),
    base_url=_BASE_URL,
    registry={},
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
