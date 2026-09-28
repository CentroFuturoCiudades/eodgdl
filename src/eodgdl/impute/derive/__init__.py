"""Derive steps: deterministic functions of the imputed values, run per completion of a chain
(:mod:`eodgdl.impute.chain`), so a derived quantity's distribution is carried (e.g. an AMAI points total and the NSE
level cut from it, per draw).

A derive function is registered with :func:`register_derive` in a module listed in ``DERIVE_MODULES``; it takes
``(completions, base, config)`` -- the completions table (``row``, ``weight``, each step's value under its task
prefix, earlier derived columns), the scoring source's frame (``base.iloc[completions["row"]]`` are a completion's
covariates) and the step's configuration from the chain YAML -- and returns a frame of new columns aligned to
``completions``. A categorical column is aggregated into ``prob_<column>_<level>`` (levels from ``config["levels"]``,
else the values seen), a numeric one into ``<column>_media``."""

import importlib

DERIVE_MODULES = ()
_DERIVES = {}


def register_derive(name):
    def decorate(function):
        _DERIVES[name] = function
        return function
    return decorate


def get_derive(name):
    for module in DERIVE_MODULES:
        importlib.import_module(module)
    if name not in _DERIVES:
        raise ValueError(f"No derive step {name!r}; known: {sorted(_DERIVES)}")
    return _DERIVES[name]
