"""Value maps from a source's raw codes to a task's common levels. A code the map does not cover raises unless it is
listed in ``allowed_unmapped``: a new or renamed source category (a new ENOE quarter, an eodgdl revision) must fail
loudly instead of degrading the data into the fallback level."""

from pathlib import Path

import pandas as pd

from .levels import MISSING_LABEL
from .spec import read_yaml


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
HARMONIZATION_DIR = Path(__file__).parent / "harmonization"


def load_harmonization(name):
    """``impute/harmonization/<name>.yaml`` parsed (``common`` holds the shared levels and cut points)."""
    return read_yaml(HARMONIZATION_DIR / f"{name}.yaml")


def named_transform(name, common=None):
    """common.yaml's ``transforms[name]``: a map from a task's class slugs to the common levels."""
    transforms = (common or load_harmonization("common")).get("transforms", {})
    if name not in transforms:
        raise ValueError(f"No transform {name!r} in harmonization/common.yaml; known: {sorted(transforms)}")
    return {str(key): str(value) for key, value in transforms[name].items()}


def resolve_map(definition, common=None):
    """A variable's value map: its ``map``, or the named ``transform`` composed with the classes of the task named by
    ``labels_of`` (survey label -> class slug -> level), which must cover every class."""
    if "transform" not in definition:
        return definition["map"]
    from .spec import load_task

    transform = named_transform(definition["transform"], common)
    classes = load_task(definition["labels_of"]).classes
    if set(classes.values()) != set(transform):
        raise ValueError(f"transform {definition['transform']!r} maps {sorted(transform)}, not the classes of "
                         f"{definition['labels_of']!r} ({sorted(set(classes.values()))})")
    return {label: transform[slug] for label, slug in classes.items()}


def _labels_versions(table):
    """The classes of every task a table's variables read their labels from (``labels_of``), for the cache key."""
    from .spec import load_task

    tasks = sorted({definition["labels_of"] for definition in load_harmonization(table)["variables"].values() if "labels_of" in definition})
    return {task: load_task(task).classes for task in tasks}


def register_harmonization(name, table):
    """Register the feature builder ``name``: the variables of ``impute/harmonization/<table>.yaml``
    (:func:`apply_variables`), which replace the frame's columns of the same name (e.g. the survey's own
    ``escolaridad``). Its cache key covers both YAMLs and this module's code."""
    from .features import register_builder

    def build(frame, context, config, spec):
        return apply_variables(frame, load_harmonization(table)["variables"])

    def versions(context, config):
        labels = _labels_versions(table)
        return {"harmonization": load_harmonization(table), "common": load_harmonization("common"), **({"labels_of": labels} if labels else {})}

    register_builder(name, versions=versions, replaces=True)(build)


def _codes(values):
    """A column of codes as plain Python values: integers where the column is numeric, else stripped strings."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").astype("Int64")
    return values.astype("string").str.strip()


def _map_variable(values, definition, missing_label, name, common=None):
    codes = _codes(values)
    if "floordiv" in definition:
        codes = codes // int(definition["floordiv"])
    mapping = resolve_map(definition, common)
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
    if isinstance(values, pd.DataFrame):          # several columns summed (e.g. cars + vans + pickups)
        values = values.apply(lambda column: pd.to_numeric(column.astype("string").str.strip(), errors="coerce")).sum(axis=1, min_count=1)
    if "map" in definition:
        codes = values.astype("string").str.strip()
        assert_mapping_covers(codes, definition["map"], definition.get("allowed_unmapped", []), name=name)
        numbers = codes.map(definition["map"])
    else:
        numbers = values
    numbers = pd.to_numeric(numbers.astype("string").str.strip() if numbers.dtype == object or pd.api.types.is_string_dtype(numbers) else numbers, errors="coerce")
    if "divide" in definition:
        numbers = numbers / definition["divide"]
    else:
        numbers = numbers.astype("Int64") if (numbers.dropna() % 1 == 0).all() else numbers
    for code in definition.get("unspecified", []):
        numbers = numbers.mask(numbers == code)
    if "cap" in definition:
        numbers = numbers.clip(upper=definition["cap"])
    return numbers


def _bins_variable(numbers, cut, missing_label):
    """Cut a number: common.yaml's left-closed bins (the last open), or inline ``edges`` (``-inf`` / ``inf`` allowed)
    with ``right: true`` for right-closed intervals."""
    import numpy as np

    if "edges" in cut:
        edges = [float(edge) for edge in cut["edges"]]
        return pd.cut(numbers.astype("Float64"), bins=edges, labels=cut["labels"], right=cut.get("right", False)).astype("string").fillna(missing_label)
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
        if definition.get("unobserved"):                 # not asked in this source: missing (a chain fills it)
            out[name] = pd.Series(pd.NA, index=frame.index, dtype="Float64" if definition["unobserved"] == "number" else "string")
        elif "rules" in definition:
            out[name] = _rules_variable(frame, definition, missing_label, name)
        elif "bins" in definition:
            cut = definition["bins"] if isinstance(definition["bins"], dict) else common[definition["bins"]]
            numbers = out[definition["from"]] if definition.get("from") in out else frame[definition["from"]]
            out[name] = _bins_variable(numbers, cut, missing_label)
        elif "number" in definition:
            source = definition.get("columns", definition.get("column"))
            if isinstance(source, str):
                values = out[source] if source in out else frame[source]      # an earlier output, else a source column
            else:
                values = frame[list(source)]
            out[name] = _number_variable(values, definition["number"], name)
        else:
            out[name] = _map_variable(frame[definition["column"]], definition, missing_label, name, common)
    return out


def _reads_output(definition):
    """The name a definition reads as an earlier output when :func:`apply_variables` has computed one by then: a
    ``bins``'s ``from``, a ``number``'s single column; None where it reads the source's columns only."""
    if definition.get("unobserved") or "rules" in definition:
        return None
    if "bins" in definition:
        return definition["from"]
    if "number" in definition:
        source = definition.get("columns", definition.get("column"))
        return source if isinstance(source, str) else None
    return None


def needed_variables(variables, name):
    """The definitions of ``variables`` that :func:`apply_variables` reads to compute ``name``, in their order: ``name``
    and, recursively, the earlier outputs it reads (a variable defined after the one reading its name is not one: the
    source's column is read then), so that applying them alone gives ``name`` as applying them all does."""
    order = list(variables)
    needed, pending = set(), [name]
    while pending:
        current = pending.pop()
        needed.add(current)
        read = _reads_output(variables[current])
        if read in variables and order.index(read) < order.index(current) and read not in needed:
            pending.append(read)
    return {key: variables[key] for key in order if key in needed}
