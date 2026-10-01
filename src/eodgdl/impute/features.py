"""Feature builders and the on-disk feature cache.

A builder adds columns to a source's frame: ``build(frame, context, config, spec)`` returns a frame of new columns
aligned to ``frame``'s rows. It is registered with :func:`register_builder` in a module listed in
``BUILDER_MODULES``; its configuration is its registered default updated by the task's ``builders:`` entry.
Harmonizations (:func:`eodgdl.impute.harmonize.register_harmonization`) are builders too.

:func:`build_frame` assembles a task's frame: the source, then each builder in the task's order, then the target
slug. With a :class:`~eodgdl.impute.sources.Context` that uses the cache, the source frame and each builder's columns
are read from parquet under the cache directory when a file with the same key exists, else built and written. A key
covers the builder's module source, this module's, its configuration, the data versions it declares, the task's
classes when the builder reads them (``reads_classes``) and the keys of everything it was built on, so an edit to the
chain rules, a survey file, DENUE's release or the builder's code moves it, while tasks that share a source and its
builders (the AMAI tasks on ENIGH) share its cached columns; ``Context(refresh=True)`` rebuilds regardless."""

import functools
import importlib
from dataclasses import dataclass

import pandas as pd

from .spec import stable_hash

BUILDER_MODULES = ("eodgdl.impute.sources.eod", "eodgdl.impute.sources.enoe", "eodgdl.impute.sources.enigh", "eodgdl.impute.sources.census", "eodgdl.giro.features")
_BUILDERS = {}


@dataclass
class Builder:
    name: str
    build: object                  # build(frame, context, config, spec) -> DataFrame of new columns
    config: dict
    versions: object               # versions(context, config) -> dict
    module_file: str
    replaces: bool = False         # its columns may replace the frame's columns of the same name
    reads_classes: bool = False    # its columns depend on the task's classes (``spec.classes``), which key its cache


def register_builder(name, config=None, versions=None, replaces=False, reads_classes=False):
    """Decorator registering ``build(frame, context, config, spec)`` as feature builder ``name``; with ``replaces`` its
    columns replace any of the frame's columns of the same name (a harmonization over a source's raw columns); with
    ``reads_classes`` its columns depend on the task's classes (the only part of ``spec`` a builder may read), so they
    key its cache."""
    def decorate(function):
        module = importlib.import_module(function.__module__)
        _BUILDERS[name] = Builder(name, function, dict(config or {}), versions or (lambda context, config: {}), module.__file__, replaces, reads_classes)
        return function
    return decorate


def get_builder(name):
    for module in BUILDER_MODULES:
        importlib.import_module(module)
    if name not in _BUILDERS:
        raise ValueError(f"No feature builder {name!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[name]


def _normalize(frame):
    """Text and categorical columns as the pandas string dtype, so a frame read back from the cache equals the one
    built (categoricals are plain strings in a task frame; the level contract lives in the task spec)."""
    frame = frame.copy()
    categorical = frame.columns[frame.dtypes.eq("category")]
    frame[categorical] = frame[categorical].astype("string")
    for column in frame.columns[frame.dtypes.eq(object)]:
        values = frame[column].dropna()
        if values.map(lambda value: isinstance(value, str)).all():
            frame[column] = frame[column].astype("string")
    return frame


def _cached(context, kind, name, key, build):
    """``build()`` or its cached parquet ``<cache>/<kind>/<name>-<key>.parquet``."""
    if not context.uses_cache:
        return _normalize(build())
    path = context.cache_path() / kind / f"{name}-{key[:24]}.parquet"
    if path.exists() and not context.refresh:
        return pd.read_parquet(path)
    frame = _normalize(build())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return pd.read_parquet(path)   # the cached form, so a first run returns what later runs will


def _key(**parts):
    """A cache key: ``parts`` plus the code of this module (how every cached frame is normalized and assembled)."""
    from .sources import file_digest

    return stable_hash({**parts, "cache_code": file_digest(__file__)})


def _builder_columns(builder, frame, context, config, spec, keys):
    """The builder's columns for ``frame``'s rows, after the rows' ``keys`` (what the feature cache stores)."""
    columns = builder.build(frame, context, config, spec)
    assert len(columns) == len(frame), f"builder {builder.name} returned {len(columns)} rows for {len(frame)}"
    return pd.concat([frame[keys].reset_index(drop=True), columns.reset_index(drop=True)], axis=1)


def builder_config(spec, name, overrides=None):
    return {**get_builder(name).config, **spec.builder_config.get(name, {}), **((overrides or {}).get(name, {}))}


def build_frame(spec, context, overrides=None, role="train"):
    """The task's frame: the source's rows with every builder's columns and the target slug (``spec.target``, NA
    where unobserved) plus ``<prefix>_desconocido``. ``role="score"`` builds the rows the task imputes (its
    ``score_source``, where the target column may be absent: every row unknown). ``overrides`` ({builder: {key:
    value}}) changes a builder's configuration for this call. Returns a
    :class:`~eodgdl.impute.sources.SourceFrame`."""
    from .sources import SourceFrame, file_digest, get_source, with_weight

    source = get_source(spec.source if role == "train" else spec.score_source)
    versions = source.data_versions(context)
    key = _key(source=source.name, module=file_digest(source.module_file), config=source.config, versions=versions)
    frame = _cached(context, "sources", source.name, key, lambda: with_weight(source.build(context, source.config), source.config))
    keys = source.keys
    for name in spec.builders_for(source.name):
        builder = get_builder(name)
        config = builder_config(spec, name, overrides)
        key = _key(builder=name, module=file_digest(builder.module_file), config=config, versions=builder.versions(context, config), input=key,
                   **({"classes": spec.classes} if builder.reads_classes else {}))
        columns = _cached(context, "features", name, key, functools.partial(_builder_columns, builder, frame, context, config, spec, keys))
        assert columns[keys].astype(str).equals(frame[keys].astype(str)), f"cached {name} columns do not line up with the source rows"
        clashes = (set(columns.columns) - set(keys)) & set(frame.columns)
        assert builder.replaces or not clashes, f"builder {name} overwrites {sorted(clashes)}"
        frame = pd.concat([frame.drop(columns=sorted(clashes)), columns.drop(columns=keys)], axis=1)

    labels = frame[spec.target_column] if spec.target_column in frame.columns or role == "train" else pd.Series(pd.NA, index=frame.index, dtype="string")
    labels = labels.mask(labels.astype("string") == spec.missing_label)      # a harmonized target's missing label is unobserved
    unknown_labels = set(labels.dropna().unique()) - set(spec.classes)
    assert not unknown_labels, f"{spec.target_column} labels missing from the {spec.name} task's classes: {sorted(unknown_labels)}"
    frame[spec.target] = labels.map(spec.classes).astype("string")
    frame[spec.unknown_column] = frame[spec.target].isna()
    missing = [feature for feature in spec.features if feature not in frame.columns]
    assert not missing, f"{spec.name}: features no source column or builder provides: {missing}"

    return SourceFrame(frame, keys, source.weight, source.group, {**versions, "features": key})
