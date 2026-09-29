"""Sources: registered builders that return a task's rows, with their contract (row keys, weight column, CV group
column) in the source's configuration, and the :class:`Context` a run shares (the survey tables, loaded once, and the
cache).

A source is registered with :func:`register_source` in a module listed in ``SOURCE_MODULES``; its configuration is
the entry of the same name in that module's YAML (``keys``, ``weight``, ``group`` and whatever the builder reads)."""

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
        return self.config["weight"]

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
