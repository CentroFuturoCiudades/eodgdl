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


# Variable definitions in YAML (impute/harmonization/<source>.yaml): every source mapped to the levels of common.yaml
HARMONIZATION_DIR = __import__("pathlib").Path(__file__).parent / "harmonization"


def load_harmonization(name):
    """``impute/harmonization/<name>.yaml`` parsed (``common`` holds the shared levels and cut points)."""
    import yaml

    with open(HARMONIZATION_DIR / f"{name}.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _codes(values):
    """A column of codes as plain Python values: integers where the column is numeric, else stripped strings."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").astype("Int64")
    return values.astype("string").str.strip()


def _map_variable(values, definition, missing_label, name):
    codes = _codes(values)
    if "floordiv" in definition:
        codes = codes // int(definition["floordiv"])
    mapping = definition["map"]
    allowed = list(definition.get("allowed_unmapped", []))
    otherwise = definition.get("otherwise")
    if otherwise:
        low, high = otherwise["domain"]
        allowed += [code for code in codes.dropna().unique() if low <= code <= high and code not in mapping]
    mapped = harmonize(codes, mapping, allowed_unmapped=allowed, fallback=missing_label, name=name).astype(object)
    if otherwise:
        other = codes.notna() & ~codes.isin(list(mapping)) & codes.isin(allowed) & ~codes.isin(definition.get("allowed_unmapped", []))
        mapped[other.to_numpy()] = otherwise["level"]
    return pd.Series(mapped, index=values.index).fillna(missing_label).astype("string")


def _number_variable(values, definition, name):
    if "map" in definition:
        codes = values.astype("string").str.strip()
        assert_mapping_covers(codes, definition["map"], name=name)
        numbers = codes.map(definition["map"])
    else:
        numbers = values
    numbers = pd.to_numeric(numbers, errors="coerce").astype("Int64")
    for code in definition.get("unspecified", []):
        numbers = numbers.mask(numbers == code)
    if "cap" in definition:
        numbers = numbers.clip(upper=definition["cap"])
    return numbers


def _bins_variable(numbers, cut, missing_label):
    import numpy as np

    bins = list(cut["bins"]) + [np.inf]
    return pd.cut(numbers, bins=bins, labels=cut["labels"], right=False).astype("string").fillna(missing_label)


def _rules_variable(frame, definition, missing_label, name):
    for column, domain in definition.get("domains", {}).items():
        codes = _codes(frame[column])
        outside = set(codes.dropna().unique()) - set(domain)
        if outside:
            raise ValueError(f"{name}: codes of {column} outside its declared domain: {sorted(map(str, outside))}")
    level = pd.Series(definition.get("default", missing_label), index=frame.index, dtype=object)
    for rule in definition["rules"]:
        match = pd.Series("any" not in rule, index=frame.index)      # no `any`: the `all` conditions alone decide
        for column, codes in rule.get("any", {}).items():
            match |= _codes(frame[column]).isin(codes).fillna(False)
        for column, codes in rule.get("all", {}).items():
            match &= _codes(frame[column]).isin(codes).fillna(False)
        level[match.to_numpy()] = rule["level"]
    return level.astype("string")


def apply_variables(frame, variables, common=None, missing_label=None):
    """The harmonized columns of ``frame`` under ``variables`` (a source's ``variables:`` section), in order; a
    ``from`` definition reads an earlier output, ``bins`` names a cut of ``common`` (common.yaml)."""
    common = common or load_harmonization("common")
    missing_label = missing_label or common["missing_label"]
    out = pd.DataFrame(index=frame.index)
    for name, definition in variables.items():
        if "rules" in definition:
            out[name] = _rules_variable(frame, definition, missing_label, name)
        elif "bins" in definition:
            out[name] = _bins_variable(out[definition["from"]], common[definition["bins"]], missing_label)
        elif "number" in definition:
            out[name] = _number_variable(frame[definition["column"]], definition["number"], name)
        else:
            out[name] = _map_variable(frame[definition["column"]], definition, missing_label, name)
    return out
