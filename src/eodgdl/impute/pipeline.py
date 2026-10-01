"""Pipelines: the chains of several row levels drawn jointly, one completed dataset per draw.

A pipeline (one YAML per pipeline under ``impute/pipelines/``) lists its row **levels** in order, each one chain
(``chains/<chain>.yaml``) run with ``propagation: draws``, the pipeline's ``draws`` and ``uncertainty``. A level may be
**given** per-draw features aggregated from an earlier level's completed datasets (:mod:`eodgdl.impute.aggregate`):
completion d of the level reads completion d of the level it aggregates, so every level's draw d is one completed
dataset. A given feature enters after the feature cache, as a column of the chain's completions
(:func:`eodgdl.impute.chain.run_chain`'s ``given``), so no cached frame holds a per-draw value.

The pipeline ``tasha`` (what the TASHA build reads) runs the chain ``labour`` on the persons (stage 1 within the survey:
empleo, giro; stage 2 data fusion: informality), then counts each dwelling's workers aged 14+ with the drawn ones, then
runs the chain ``nse`` on the dwellings given that count (stage 1: educacion_jefe; stage 2: the ENIGH components; stage
3: the AMAI points, level and calibration). Each level keeps its chain's own seed, and no two levels may share one, so
each draws from a random stream of its own (two levels on one seed would read the same uniforms, row by row): with the
aggregate's drawn values switched off (``aggregate=False``) each level reproduces its chain's multiple imputations.

A run records each level's scoring source and data versions (its chain's ``provenance``): ``eodgdl.tasha.load_completed``
refuses to build from a run whose data changed since (:func:`eodgdl.impute.sources.changed_versions`), as it refuses one
with the aggregates switched off.
"""

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path

from .aggregate import get_aggregate
from .chain import load_chain, parse_chain, retrain_chain, run_chain, step_spec, write_chain
from .sources import get_source
from .spec import load_task, read_yaml, stable_hash

PIPELINES_DIR = Path(__file__).parent / "pipelines"


@dataclass(frozen=True)
class Level:
    name: str
    chain: str
    given: dict = field(default_factory=dict)    # column -> the aggregate's configuration (its ``aggregate`` and ``from``)


@dataclass
class PipelineSpec:
    name: str
    draws: int
    uncertainty: str | None
    levels: list
    raw: dict = field(repr=False)

    def chain(self, level, draws=None, uncertainty=...):
        """The chain a level runs: its YAML with ``propagation: draws``, the pipeline's draws (or ``draws``) and
        uncertainty (or ``uncertainty``)."""
        level = self.level(level) if isinstance(level, str) else level
        uncertainty = self.uncertainty if uncertainty is ... else uncertainty
        raw = {**load_chain(level.chain).raw, "propagation": "draws", "draws": int(self.draws if draws is None else draws)}
        if uncertainty:
            raw["uncertainty"] = uncertainty
        else:
            raw.pop("uncertainty", None)
        return parse_chain(raw)

    def level(self, name):
        found = [level for level in self.levels if level.name == name]
        if not found:
            raise KeyError(f"pipeline {self.name!r} has no level {name!r}; its levels: {[level.name for level in self.levels]}")
        return found[0]

    def column_level(self, column):
        """The level whose completions hold ``column``: a task's output prefix, a derive step's column or a given one."""
        from .chain import DeriveStep

        for level in self.levels:
            chain = load_chain(level.chain)
            columns = {load_task(task).prefix for task in chain.tasks} | set(level.given)
            columns |= {name for step in chain.steps if isinstance(step, DeriveStep) for name in step.config.get("levels", {})}
            if column in columns:
                return level
        raise KeyError(f"pipeline {self.name!r} has no output column {column!r}")

    @property
    def hash(self):
        """Over the YAML and every level's chain hash."""
        return stable_hash({"raw": self.raw, "chains": {level.name: load_chain(level.chain).hash for level in self.levels}})


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def parse_pipeline(raw):
    """A :class:`PipelineSpec` from a parsed pipeline YAML, checked: every level's chain exists, a given column is a
    feature some task of the level reads and is aggregated from an earlier level, no two levels share a task or a seed,
    and the draws are a number of imputations."""
    name = raw["pipeline"]
    uncertainty = raw.get("uncertainty")
    _check(uncertainty in (None, "bootstrap"), f"{name}: uncertainty {uncertainty!r} is not 'bootstrap'")
    draws = int(raw.get("draws", 50))
    _check(draws >= 1, f"{name}: draws {draws} is not a number of imputations")
    levels, seen, tasks, seeds = [], [], set(), {}
    for level_name, entry in raw["levels"].items():
        chain = load_chain(entry["chain"])
        clash = tasks & set(chain.tasks)
        _check(not clash, f"{name}: tasks {sorted(clash)} appear in two levels")
        tasks |= set(chain.tasks)
        _check(chain.seed not in seeds, f"{name}: {level_name} and {seeds.get(chain.seed)} draw with the same seed {chain.seed}, so their draws "
                                        "would read the same uniforms row by row: give one of their chains a seed of its own")
        seeds[chain.seed] = level_name
        features = {feature for task in chain.tasks for feature in load_task(task).features}
        given = dict(entry.get("given") or {})
        for column, config in given.items():
            _check(column in features, f"{name}: {level_name} is given {column}, which no task of the chain {chain.name} reads")
            _check(config.get("from") in seen, f"{name}: {level_name}.{column} aggregates {config.get('from')!r}, which is not an earlier level")
            get_aggregate(config["aggregate"])
        levels.append(Level(level_name, chain.name, given))
        seen.append(level_name)
    _check(levels, f"{name}: no levels")
    return PipelineSpec(name, draws, uncertainty, levels, raw)


@functools.cache
def load_pipeline(name):
    """The pipeline spec ``impute/pipelines/<name>.yaml``."""
    path = PIPELINES_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"No pipeline {name!r}; known: {sorted(p.stem for p in PIPELINES_DIR.glob('*.yaml'))}")
    return parse_pipeline(read_yaml(path))


def is_pipeline(name):
    """Whether ``name`` is a pipeline (``impute/pipelines/<name>.yaml``)."""
    return (PIPELINES_DIR / f"{name}.yaml").exists()


@dataclass
class PipelineResult:
    """``levels``: each level's :class:`~eodgdl.impute.chain.ChainResult` (completion d of every level is completed
    dataset d); ``provenance``: the pipeline's settings and every level's chain provenance."""

    pipeline: PipelineSpec
    levels: dict
    provenance: dict


def given_frame(level, results, aggregate=True):
    """The function of the level's source frame that :func:`~eodgdl.impute.chain.run_chain` calls for its ``given``
    columns: every aggregate of the level over the earlier levels' ``results``, merged on the keys and completion. A
    column an aggregate gives besides the declared ones (the workers' count beside its bins) is carried along, but not
    one a task of the level reads: it would replace the source's value unannounced."""
    chain = load_chain(level.chain)
    keys = get_source(step_spec(chain, chain.tasks[0], load=load_task).score_source).keys
    read = {feature for task in chain.tasks for feature in load_task(task).features}

    def build(base):
        frames = []
        for column, config in level.given.items():
            frame = get_aggregate(config["aggregate"])(results[config["from"]], base, config, counted=aggregate)
            _check(column in frame.columns, f"{level.name}: the aggregate {config['aggregate']} gives no column {column}")
            frames.append(frame)
        merged = frames[0]
        for frame in frames[1:]:       # on the keys and completion (the columns they share)
            merged = merged.merge(frame, on=[column for column in frame.columns if column in merged.columns], how="inner", validate="one_to_one")
        undeclared = (set(merged.columns) - set(level.given) - {*keys, "completion"}) & read
        _check(not undeclared, f"{level.name}: the aggregates also give {sorted(undeclared)}, which tasks of the chain {chain.name} read: declare it under given")
        return merged
    return build


def run_pipeline(pipeline, context=None, retrained=None, draws=None, aggregate=True, uncertainty=..., bundles=None, bootstrap=None,
                 specs=None, derive_functions=None):
    """Run every level of ``pipeline`` in order (:func:`~eodgdl.impute.chain.run_chain`), each given its aggregates of
    the earlier levels' draws. ``draws`` and ``uncertainty`` override the pipeline's; ``aggregate=False`` switches the
    aggregates' drawn values off (each level then reproduces its chain run alone); ``bundles``, ``bootstrap`` (with
    uncertainty bootstrap only), ``specs`` and ``derive_functions`` are passed to every level's chain. Returns a
    :class:`PipelineResult`: each level's provenance is its chain's (:func:`~eodgdl.impute.chain.run_chain`: settings,
    bundles, data versions, ``given_columns``) with ``given``, the pipeline's aggregates of the level."""
    from .sources import Context

    pipeline = load_pipeline(pipeline) if isinstance(pipeline, str) else pipeline
    context = context or Context()
    results, provenance = {}, {}
    for level in pipeline.levels:
        chain = pipeline.chain(level, draws, uncertainty)
        boot = {task: models for task, models in (bootstrap or {}).items() if task in chain.tasks} or None
        given = given_frame(level, results, aggregate) if level.given else None
        results[level.name] = run_chain(chain, context=context, retrained=retrained, bundles={task: bundle for task, bundle in (bundles or {}).items() if task in chain.tasks} or None,
                                        bootstrap=boot, specs=specs, derive_functions=derive_functions, given=given)
        provenance[level.name] = {"chain": chain.name, **results[level.name].provenance, "given": dict(level.given)}
    settings = {"pipeline": pipeline.name, "pipeline_hash": pipeline.hash, "draws": int(pipeline.draws if draws is None else draws),
                "uncertainty": pipeline.uncertainty if uncertainty is ... else uncertainty, "aggregate": bool(aggregate), "levels": provenance}
    return PipelineResult(pipeline, results, settings)


def write_pipeline(result, out):
    """Write a pipeline run: ``<out>/<level>/`` as :func:`~eodgdl.impute.chain.write_chain` writes a chain run
    (``scores.parquet``, ``completions.parquet``: one row per row and draw, ``provenance.json``) and
    ``<out>/provenance.json``, the pipeline's."""
    out = Path(out)
    for name, level in result.levels.items():
        write_chain(level, out / name)
    (out / "provenance.json").write_text(json.dumps(result.provenance, indent=2, default=str), encoding="utf-8")
    return out


def retrain_pipeline(pipeline, out, context=None, n_jobs=-1, progress=True, bootstrap=0):
    """Retrain every task of every level's chain (:func:`~eodgdl.impute.chain.retrain_chain`) into ``out/<task>``."""
    pipeline = load_pipeline(pipeline) if isinstance(pipeline, str) else pipeline
    written = {}
    for level in pipeline.levels:
        written.update(retrain_chain(level.chain, out, context=context, n_jobs=n_jobs, progress=progress, bootstrap=bootstrap))
    return written
