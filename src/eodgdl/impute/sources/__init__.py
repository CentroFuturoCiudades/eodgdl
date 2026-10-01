"""Sources: registered builders that return a task's rows, with their contract (row keys, weight column, CV group
column) in the source's configuration, and the :class:`Context` a run shares (the survey tables, loaded once, and the
cache).

A source is registered with :func:`register_source` in a module listed in ``SOURCE_MODULES``; its configuration is
the entry of the same name in that module's YAML (``keys``, ``weight``, ``group`` and whatever the builder reads).

A source's ``weight`` names a column of its rows (``weight: ponderador``, the survey's design weight) or a column of a
weight file under the data directory, joined on a key (``weight: {file: EOD_peso_hogar_TMG.csv, column: peso,
key: folio_vivienda}``; ``key`` defaults to ``folio_vivienda``): every row must find its weight there, and the file's
sha256 is one of the source's data ``versions`` (:meth:`Source.data_versions`), so the feature cache, a bundle's
training data version and :func:`changed_versions` all move when the weight's values do."""

import functools
import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_MODULES = ("eodgdl.impute.sources.eod", "eodgdl.impute.sources.enoe", "eodgdl.impute.sources.enigh")
_SOURCES = {}


@dataclass
class SourceFrame:
    """A task's frame (:func:`eodgdl.impute.features.build_frame`): ``frame`` plus its row ``keys``, ``weight`` and CV
    ``group`` columns and the data ``versions`` it was built from."""

    frame: object
    keys: list
    weight: str
    group: str
    versions: dict


@dataclass
class Source:
    name: str
    build: object                  # build(context, config) -> DataFrame
    config: dict
    versions: object               # versions(context, config) -> dict, without building the frame
    schema_levels: object          # schema_levels(column) -> list
    module_file: str

    @property
    def keys(self):
        return list(self.config["keys"])

    @property
    def weight(self):
        """The weight column of the source's rows (a weight file's column joins them under its own name)."""
        weight = self.config["weight"]
        return weight if isinstance(weight, str) else weight["column"]

    def data_versions(self, context):
        """The data versions the source's rows depend on: its own (``versions``) and its weight file's sha256."""
        return {**self.versions(context, self.config), **weight_versions(self.config)}

    @property
    def group(self):
        return self.config["group"]


def register_source(name, config, versions, schema_levels):
    """Decorator registering ``build(context, config) -> DataFrame`` as source ``name``."""
    def decorate(function):
        module = importlib.import_module(function.__module__)
        _SOURCES[name] = Source(name, function, config, versions, schema_levels, module.__file__)
        return function
    return decorate


def get_source(name):
    for module in SOURCE_MODULES:
        importlib.import_module(module)
    if name not in _SOURCES:
        raise ValueError(f"No source {name!r}; known: {sorted(_SOURCES)}")
    return _SOURCES[name]


def changed_versions(name, recorded, context=None):
    """The data versions of source ``name`` that now differ from ``recorded`` (what a run recorded: a chain's
    ``provenance["versions"]``), the eodgdl version aside (it moves with every release, the data only when they do): an
    empty list when the run read the data the source reads now."""
    source = get_source(name)
    current = source.data_versions(context or Context())
    return sorted(key for key in current if key != "eodgdl" and recorded.get(key) != current[key])


def weight_file(config):
    """The weight file a source configuration names, ``{file, column, key}``, or None where ``weight`` is a column of
    the source's own rows."""
    weight = config["weight"]
    if isinstance(weight, str):
        return None
    unknown = set(weight) - {"file", "column", "key"}
    if unknown or not {"file", "column"} <= set(weight):
        raise ValueError(f"a weight file is {{file, column, key}} (key optional); got {weight}")
    return {"key": "folio_vivienda", **weight}


def weight_versions(config):
    """``{"weight": sha256}`` of the weight file a source configuration names, or nothing (a column of its rows: the
    survey files' versions already cover it)."""
    weight = weight_file(config)
    if weight is None:
        return {}
    from eodgdl.data import resolve

    return {"weight": file_digest(str(resolve(weight["file"])))}


def with_weight(frame, config):
    """``frame`` with the weight column of the weight file ``config`` names joined on its key (unchanged where the
    weight is a column of the rows). Fails where the file repeats a key, holds a weight that is not a positive number,
    leaves a row without a weight, or names a column the rows already hold."""
    import numpy as np
    import pandas as pd

    weight = weight_file(config)
    if weight is None:
        return frame
    from eodgdl.data import resolve

    path, key, column = resolve(weight["file"]), weight["key"], weight["column"]
    sidecar = path.with_suffix(".yaml")       # TMG's weight records its own sha256 (eodgdl.reweight.weight)
    if sidecar.exists():
        from ..spec import read_yaml

        recorded = read_yaml(sidecar).get("weight_sha256")
        if recorded is not None and recorded != file_digest(str(path)):
            raise ValueError(f"{path.name}: not the weight its sidecar {sidecar.name} records")
    table = pd.read_csv(path)
    missing = {key, column} - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: no column {sorted(missing)}")
    if column in frame.columns:
        raise ValueError(f"{path.name}: the weight column {column!r} is already a column of the source's rows")
    if table[key].duplicated().any():
        raise ValueError(f"{path.name}: {int(table[key].duplicated().sum())} repeated {key} values")
    values = pd.to_numeric(table[column], errors="coerce")
    if not (np.isfinite(values) & (values > 0)).all():
        raise ValueError(f"{path.name}: {int((~(np.isfinite(values) & (values > 0))).sum())} weights that are not positive numbers")
    table = pd.DataFrame({key: table[key].astype(frame[key].dtype), column: values.astype(float)})
    joined = frame.merge(table, on=key, how="left", validate="many_to_one")
    unweighted = joined[column].isna()
    if unweighted.any():
        examples = ", ".join(map(str, joined.loc[unweighted, key].drop_duplicates().head(5)))
        raise ValueError(f"{path.name}: no weight for {int(unweighted.sum())} rows (e.g. {key} {examples})")
    return joined


def module_config(module_file):
    """The YAML beside a source module (``<module>.yaml``), parsed."""
    from ..spec import read_yaml

    return read_yaml(Path(module_file).with_suffix(".yaml"))


@functools.cache
def file_digest(path):
    """sha256 of a file's bytes (cached per path for the process)."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def files_digest(paths):
    """One sha256 over several files (their names and bytes)."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(path.name.encode())
        digest.update(file_digest(str(path)).encode())
    return digest.hexdigest()


@dataclass
class Context:
    """What the sources and feature builders of one run share.

    ``tables`` are the survey tables (:class:`eodgdl.EODTables`), loaded with ``load_eod()`` on first use when not
    given. The on-disk feature cache is used only when the engine loaded the tables itself (``cache=True`` and no
    ``tables`` passed): tables handed in may differ from the files the cache keys describe. ``refresh`` rebuilds
    and rewrites every cached frame."""

    tables: object = None
    cache: bool = True
    refresh: bool = False
    cache_dir: Path | None = None
    _given_tables: bool = field(init=False, default=False)

    def __post_init__(self):
        self._given_tables = self.tables is not None

    @property
    def uses_cache(self):
        return self.cache and not self._given_tables

    def eod(self):
        if self.tables is None:
            import eodgdl

            self.tables = eodgdl.load_eod()
        return self.tables

    def cache_path(self):
        if self.cache_dir is not None:
            return Path(self.cache_dir)
        from eodgdl.data import get_pooch_cache_dir

        return get_pooch_cache_dir() / "impute"
