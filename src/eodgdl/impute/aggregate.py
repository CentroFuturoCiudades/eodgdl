"""Aggregates: per-draw features of one row level computed from another level's completed datasets, the step between
the levels of a pipeline (:mod:`eodgdl.impute.pipeline`).

An aggregate is registered with :func:`register_aggregate`; it takes ``(upstream, base, config, counted)`` -- the
upstream level's :class:`~eodgdl.impute.chain.ChainResult` (its completions and its source frame, ``base``), the
downstream chain's source frame, the ``given`` entry's configuration from the pipeline YAML and ``counted`` (False
switches the drawn values off, so the aggregate gives what the source reports: the pipeline's parity check) -- and
returns a frame of the downstream keys, ``completion`` and the new columns, one row per (row, completion). The chain
reads them after its feature cache (:func:`eodgdl.impute.chain.run_chain`'s ``given``), so a cached frame never holds
a per-draw value."""

import numpy as np
import pandas as pd

_AGGREGATES = {}


def register_aggregate(name):
    def decorate(function):
        _AGGREGATES[name] = function
        return function
    return decorate


def get_aggregate(name):
    if name not in _AGGREGATES:
        raise ValueError(f"No aggregate {name!r}; known: {sorted(_AGGREGATES)}")
    return _AGGREGATES[name]


def harmonized(frame, table, variable):
    """``variable`` of ``impute/harmonization/<table>.yaml`` computed on ``frame`` as the source computes it, with the
    earlier definitions it reads (:func:`~eodgdl.impute.harmonize.needed_variables`), so a per-draw value takes the
    source's own bins and levels."""
    from .harmonize import apply_variables, load_harmonization, needed_variables

    return apply_variables(frame, needed_variables(load_harmonization(table)["variables"], variable))[variable]


@register_aggregate("workers")
def workers(upstream, base, config, counted=True):
    """The dwelling's workers per completion: the source's reported count (``base``: its column) plus the upstream rows
    whose ``task`` was not answered, are drawn ``value`` and are aged ``min_age`` or more, grouped by ``key``.
    Columns: the count (``base``'s name) and its harmonized level (``harmonize``: ``{table, variable}``, named after
    the variable). A reported worker is in the source's count, so only drawn ones are added."""
    from .spec import load_task

    spec = load_task(config["task"])
    key, age = config["key"], config["min_age"]
    persons = upstream.base
    if spec.unknown_column not in persons.columns:
        raise ValueError(f"workers: the upstream source frame (its chain's first task's) has no {spec.unknown_column}: "
                         f"{config['task']} must be the first task of the chain it aggregates")
    drawable = persons[spec.unknown_column].astype(bool) & persons[age["column"]].ge(age["value"])
    eligible = persons.loc[drawable.to_numpy(), list(upstream.keys)]
    completions = upstream.completions
    drawn = completions[list(upstream.keys)].merge(eligible.assign(eligible=True), on=list(upstream.keys), how="left")["eligible"]
    drawn = drawn.fillna(False).to_numpy(bool) & completions[spec.prefix].astype(str).eq(config["value"]).to_numpy() & counted
    added = pd.DataFrame({key: completions[key].astype(str).to_numpy(), "completion": completions["completion"].to_numpy(), "added": drawn.astype(int)})
    added = added.groupby([key, "completion"])["added"].sum()

    completion_ids = np.sort(completions["completion"].unique())
    reported = base[[key, config["base"]]].assign(**{key: base[key].astype(str)})
    grid = reported.loc[reported.index.repeat(len(completion_ids))].reset_index(drop=True)
    grid["completion"] = np.tile(completion_ids, len(reported))
    extra = added.reindex(pd.MultiIndex.from_frame(grid[[key, "completion"]])).fillna(0).to_numpy(int)
    grid[config["base"]] = grid[config["base"]].astype(int).to_numpy() + extra
    unknown = set(added.index.get_level_values(key)) - set(reported[key])
    if unknown:
        raise ValueError(f"workers: {len(unknown)} upstream {key}s are not rows of the downstream source, e.g. {sorted(unknown)[:3]}")
    harmonize = config["harmonize"]
    grid[harmonize["variable"]] = harmonized(grid, harmonize["table"], harmonize["variable"]).astype(str).to_numpy(object)
    return grid
