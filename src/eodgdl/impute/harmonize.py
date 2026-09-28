"""Value maps from a source's raw codes to a task's common levels. A code the map does not cover raises unless it is
listed in ``allowed_unmapped``: a new or renamed source category (a new ENOE quarter, an eodgdl revision) must fail
loudly instead of degrading the data into the fallback level."""

import pandas as pd

from .levels import MISSING_LABEL


def assert_mapping_covers(values, mapping, allowed_unmapped=(), name=None):
    """Raise if a non-missing value of ``values`` is neither a key of ``mapping`` nor in ``allowed_unmapped``."""
    observed = set(pd.Series(values).dropna().unique())
    unmapped = observed - set(mapping) - set(allowed_unmapped)
    if unmapped:
        raise ValueError(f"Unmapped categories in {name or getattr(values, 'name', 'series')}: {sorted(map(str, unmapped))}")


def harmonize(values, mapping, allowed_unmapped=(), fallback=MISSING_LABEL, name=None):
    """``values`` mapped through ``mapping``; the codes in ``allowed_unmapped`` become ``fallback``, missing values
    stay missing, and any other unmapped code raises (:func:`assert_mapping_covers`)."""
    values = pd.Series(values)
    assert_mapping_covers(values, mapping, allowed_unmapped, name=name)
    mapped = values.map(mapping).astype(object)
    mapped[mapped.isna() & values.notna()] = fallback  # after the check, only the allowed codes

    return mapped.astype("string")
