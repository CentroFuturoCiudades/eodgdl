"""Chains of imputation tasks, run in parallel or in sequence.

A chain (one YAML per chain under ``impute/chains/``) lists steps in order: tasks, and derive steps (deterministic
functions of the imputed values, :mod:`eodgdl.impute.derive`). A task step may say which earlier tasks it ``uses``: the
downstream task was trained with the upstream target as a feature (observed in its training source), and it is scored
conditioning on the upstream imputation.

``mode: parallel`` scores every task from observed covariates alone (no ``uses``). ``mode: sequential`` carries the
upstream imputations along with ``propagation``:

- ``enumerate``: exact. Every row is expanded over the classes of each step it needs later (weights P(class | x, the
  scenario so far)); P(y | x) = Σ_s P(s | x) P(y | x, s). A categorical downstream feature takes the class (or its
  ``transform``); a numeric one takes a number per class (a numeric ``transform``, else the upstream's target score).
- ``draws``: ``draws`` seeded multiple imputations: y₁ ~ P(y₁ | x), then y₂ ~ P(y₂ | x, y₁), ...; each step's marginal
  is the mean over draws of its conditional probabilities (not the share of draws), and derive steps run per draw.
- ``expected``: the upstream's expected score (its task declares ``target.scores``) is plugged in as a numeric feature;
  the downstream outputs are flagged as plug-in.

Every mode builds **completions**: one row per (row, scenario) or (row, draw) with a weight (the weights of a row sum
to one) and the value of each step. A step is scored once over the unique (row, upstream values) combinations, in one
batched call. A parallel chain with derive steps forms its completions the same way from the independent marginals.

A downstream task is trained on the upstream target as observed in its training source, not on imputed values, so
retraining an upstream task does not invalidate it; what must agree is the downstream feature's declared levels and
the values the upstream fills it with (checked on every run). ``provenance`` records the bundles each run used.
"""

import functools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .arms import set_class_probabilities
from .bundle import as_v2, load_bundle
from .features import build_frame
from .run import expected_score_column, output_columns, score_frame, task_levels
from .sources import Context, get_source
from .spec import load_task, read_yaml, stable_hash

CHAINS_DIR = Path(__file__).parent / "chains"
MODES = ("parallel", "sequential")
PROPAGATIONS = ("enumerate", "draws", "expected")


@dataclass(frozen=True)
class Uses:
    task: str
    feature: str            # the downstream feature the upstream value fills (``as``)
    transform: tuple = ()   # ((upstream slug, feature level), ...): a many-to-one relabelling; empty = identity


@dataclass
class TaskStep:
    task: str
    uses: list = field(default_factory=list)


@dataclass
class DeriveStep:
    name: str
    config: dict = field(default_factory=dict)


@dataclass
class ChainSpec:
    name: str
    mode: str
    propagation: str
    draws: int
    seed: int
    steps: list
    raw: dict = field(repr=False)

    @property
    def tasks(self):
        return [step.task for step in self.steps if isinstance(step, TaskStep)]

    @property
    def hash(self):
        return stable_hash(self.raw)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def parse_chain(raw, load=load_task):
    """A :class:`ChainSpec` from a parsed chain YAML, checked against its tasks' specs (``load(name)``)."""
    name = raw["chain"]
    mode = raw.get("mode", "sequential")
    propagation = raw.get("propagation", "enumerate")
    _check(mode in MODES, f"{name}: mode {mode!r} not in {MODES}")
    _check(propagation in PROPAGATIONS, f"{name}: propagation {propagation!r} not in {PROPAGATIONS}")
    steps, seen = [], []
    for entry in raw["steps"]:
        if isinstance(entry, str):
            steps.append(TaskStep(entry))
        elif "derive" in entry:
            steps.append(DeriveStep(entry["derive"], {key: value for key, value in entry.items() if key != "derive"}))
        else:
            [(task, options)] = entry.items()
            uses = []
            for upstream, how in ((options or {}).get("uses") or {}).items():
                how = how or {}
                uses.append(Uses(upstream, how.get("as", upstream), tuple((str(k), v if isinstance(v, (int, float)) else str(v)) for k, v in (how.get("transform") or {}).items())))
            steps.append(TaskStep(task, uses))
        if isinstance(steps[-1], TaskStep):
            step = steps[-1]
            _check(step.task not in seen, f"{name}: task {step.task!r} appears twice")
            spec = load(step.task)
            for use in step.uses:
                _check(mode == "sequential", f"{name}: {step.task} uses {use.task}, but a parallel chain scores every task from observed covariates")
                _check(use.task in seen, f"{name}: {step.task} uses {use.task}, which is not an earlier step")
                _check(use.feature in spec.features, f"{name}: {step.task} has no feature {use.feature!r} for {use.task}")
                upstream = load(use.task)
                if propagation == "expected":
                    _check(bool(upstream.scores), f"{name}: expected propagation needs target scores on {use.task}")
                    _check(use.feature in spec.numeric, f"{name}: {step.task}.{use.feature} must be numeric to take {use.task}'s expected score")
                    _check(not use.transform, f"{name}: a transform does not apply to an expected score")
                else:
                    if use.transform:
                        _check({k for k, _ in use.transform} == set(upstream.class_slugs), f"{name}: the transform of {use.task} must map every class slug")
                    if use.feature in spec.numeric:   # a class's number: the transform's, else the upstream's target score
                        numbers = [v for _, v in use.transform] if use.transform else list(upstream.scores.values())
                        _check(numbers and all(isinstance(v, (int, float)) for v in numbers),
                               f"{name}: {step.task}.{use.feature} is numeric: {use.task}'s classes need numbers (a numeric transform or target scores)")
            seen.append(step.task)
    _check(any(isinstance(step, TaskStep) for step in steps), f"{name}: no task steps")
    return ChainSpec(name, mode, propagation, int(raw.get("draws", 50)), int(raw.get("seed", 42)), steps, raw)


@functools.cache
def load_chain(name):
    """The chain spec ``impute/chains/<name>.yaml``."""
    path = CHAINS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"No chain {name!r}; known: {sorted(p.stem for p in CHAINS_DIR.glob('*.yaml'))}")
    return parse_chain(read_yaml(path))


def is_chain(name):
    """Whether ``name`` is a chain (``impute/chains/<name>.yaml``) rather than a task."""
    return (CHAINS_DIR / f"{name}.yaml").exists()


@dataclass
class ChainResult:
    """``frame``: the source's keys, every task's outputs (marginal probabilities; per-scenario conditionals
    ``prob_<p>_<class>_given_<feature>_<level>``, every row at every level, for an enumerated step with one upstream) and every derived output
    (``prob_<column>_<level>`` or ``<column>_media``). ``completions``: the long table (keys, ``completion``,
    ``weight``, each step's value, derived values). ``provenance``: the bundles and settings used."""

    chain: ChainSpec
    frame: pd.DataFrame
    completions: pd.DataFrame
    provenance: dict


def _level_map(use, upstream=None, numeric=False):
    """The value each upstream class gives the downstream feature: the transform's, the upstream's score for a numeric
    feature without one, else the class itself (None)."""
    if use.transform:
        return dict(use.transform)
    if numeric:
        return dict(upstream.scores)
    return None


def _sample(probabilities, classes, rng):
    cumulative = probabilities.cumsum(axis=1)
    u = rng.random(len(probabilities))[:, None] * cumulative[:, -1:]
    return np.asarray(classes, dtype=object)[np.minimum((cumulative < u).sum(axis=1), len(classes) - 1)]


def _aggregate(values, weights, rows, n_rows, levels=None):
    """Per-row weighted distribution of a categorical column (``prob`` frame) or weighted mean of a numeric one."""
    if pd.api.types.is_numeric_dtype(values):
        totals = np.bincount(rows, weights=weights * values.to_numpy(float), minlength=n_rows)
        return totals / np.bincount(rows, weights=weights, minlength=n_rows)
    levels = levels or sorted(values.dropna().astype(str).unique())
    out = np.zeros((n_rows, len(levels)))
    index = {level: i for i, level in enumerate(levels)}
    codes = values.astype(str).map(index)
    assert codes.notna().all(), f"derived values outside the declared levels: {sorted(set(values.astype(str)) - set(levels))}"
    np.add.at(out, (rows, codes.to_numpy(int)), weights)
    return pd.DataFrame(out, columns=levels)


def _needed(chain):
    """The tasks whose values later steps read: the upstream of a ``uses``, and every task before a derive step."""
    needed = set()
    for index, step in enumerate(chain.steps):
        if isinstance(step, TaskStep):
            needed |= {use.task for use in step.uses}
        elif index > 0:
            needed |= {s.task for s in chain.steps[:index] if isinstance(s, TaskStep)}
    return needed


def _check_uses(chain, specs):
    """Every level an upstream fills a downstream categorical feature with is one the downstream task declares."""
    if chain.propagation == "expected":
        return
    for step in chain.steps:
        if not isinstance(step, TaskStep):
            continue
        declared = task_levels(specs[step.task])
        for use in step.uses:
            if use.feature in specs[step.task].numeric:
                continue
            values = set(dict(use.transform).values()) if use.transform else set(specs[use.task].class_slugs)
            _check(values <= set(declared[use.feature]), f"{chain.name}: {use.task} fills {step.task}.{use.feature} with levels it does not declare: {sorted(values - set(declared[use.feature]))}")


def _conditionals(chain, step, spec, frame, bundle, completions, specs, level_subsets):
    """P(y | x, the upstream values of each completion), one row per completion, each distinct (row, upstream values)
    scored once: ``(conditional, first, conditionals)``, with the task's outputs on each row's first combination (the
    columns that do not depend on the upstream) and, for an enumerated step with one upstream, the conditionals of
    every row at every level of the feature (``{prob_<p>_<class>_given_<feature>_<level>: values}``)."""
    n_rows = len(frame)
    filled = pd.DataFrame({"row": completions["row"].to_numpy()})
    for use in step.uses:
        upstream = specs[use.task]
        level_map = _level_map(use, upstream, use.feature in spec.numeric) if chain.propagation != "expected" else None
        values = completions[upstream.prefix]
        filled[use.feature] = values.map(level_map).to_numpy() if level_map else values.to_numpy()
    combos = filled.drop_duplicates().reset_index(drop=True)
    by_level = chain.propagation == "enumerate" and len(step.uses) == 1
    if by_level:
        # every row at every level of the feature, so the conditionals P(y | x, level) are complete
        use = step.uses[0]
        level_map = _level_map(use, specs[use.task], use.feature in spec.numeric)
        levels = sorted(set(level_map.values())) if level_map else list(specs[use.task].class_slugs)
        grid = pd.DataFrame({"row": np.repeat(np.arange(n_rows), len(levels)), use.feature: np.tile(np.asarray(levels, dtype=object), n_rows)})
        combos = pd.concat([combos, grid]).drop_duplicates().reset_index(drop=True)
    expanded = frame.iloc[combos["row"].to_numpy()].reset_index(drop=True)
    for use in step.uses:
        expanded[use.feature] = combos[use.feature].to_numpy()
    scored = score_frame(spec, expanded, bundle, level_subsets=level_subsets)
    position = filled.merge(combos.reset_index(), on=list(filled.columns), how="left")["index"].to_numpy()
    firsts = combos.drop_duplicates("row")
    first = scored.iloc[firsts.index].set_index(firsts["row"].to_numpy()).sort_index()
    first.index = frame.index

    conditionals = {}
    if by_level:
        feature = step.uses[0].feature
        table = pd.concat([combos, scored[spec.probability_columns]], axis=1)
        for level, group in table.groupby(feature):
            for slug, column in zip(spec.class_slugs, spec.probability_columns):
                values = np.full(n_rows, np.nan)
                values[group["row"].to_numpy()] = group[column].to_numpy()
                conditionals[f"prob_{spec.prefix}_{slug}_given_{feature}_{level}"] = values
    return scored[spec.probability_columns].to_numpy()[position], first, conditionals


def _marginal_outputs(chain, step, spec, first, conditional, completions):
    """The task's output columns with its marginal P(y | x) = Σ weight × P(y | x, completion) over each row's
    completions (an observed target stays one-hot), the arg-max, confidence and expected score read from it, and
    ``<prefix>_condicionado_en``."""
    marginal = np.zeros((len(first), len(spec.class_slugs)))
    np.add.at(marginal, completions["row"].to_numpy(), completions["weight"].to_numpy()[:, None] * conditional)
    out = first[output_columns(spec, [])].copy()
    unknown = out[f"{spec.prefix}_fue_imputado"].to_numpy(bool)
    marginal[~unknown] = out.loc[~unknown, spec.probability_columns].to_numpy()
    set_class_probabilities(out, spec.prefix, spec.class_slugs, marginal[unknown], unknown)
    if spec.scores:
        out[expected_score_column(spec)] = marginal @ np.array([spec.scores[slug] for slug in spec.class_slugs])
    out[f"{spec.prefix}_condicionado_en"] = "+".join(f"{use.task}:{chain.propagation}" + (" (plug-in)" if chain.propagation == "expected" else "") for use in step.uses)
    return out


def _advance(completions, spec, conditional, propagation, rng):
    """The completions carrying the task's value: expanded over its classes (``enumerate``, weights times the
    conditional, zero weights dropped), one class drawn per completion (``draws``), or the expected score
    (``expected``)."""
    classes = spec.class_slugs
    if propagation == "enumerate":
        weights = completions["weight"].to_numpy()
        completions = completions.iloc[np.repeat(np.arange(len(completions)), len(classes))].reset_index(drop=True)
        completions["weight"] = (weights[:, None] * conditional).ravel()
        completions[spec.prefix] = np.tile(np.asarray(classes, dtype=object), len(weights))
        completions = completions[completions["weight"] > 0].reset_index(drop=True)
        completions["completion"] = completions.groupby("row").cumcount()
    elif propagation == "draws":
        completions[spec.prefix] = _sample(conditional, classes, rng)
    else:
        completions[spec.prefix] = conditional @ np.array([spec.scores[slug] for slug in classes]) if spec.scores else pd.NA
    return completions


def _derive(step, index, chain, completions, base, derive_functions):
    """Run a derive step (``index``: its position in the chain) on every completion (its columns added to
    ``completions`` in place) and aggregate each new column per row: ``{prob_<column>_<level> or <column>_media:
    values}``. The step's ``seed`` is a stream of its own, spawned from the chain's seed by its position, apart from
    the draws'."""
    from .derive import get_derive

    function = (derive_functions or {}).get(step.name) or get_derive(step.name)
    seed = np.random.SeedSequence(chain.seed, spawn_key=(index,))
    values = function(completions, base, {**step.config, "propagation": chain.propagation, "seed": seed})
    derived = {}
    for column in values.columns:
        completions[column] = values[column].to_numpy()
        aggregated = _aggregate(completions[column], completions["weight"].to_numpy(), completions["row"].to_numpy(), len(base), step.config.get("levels", {}).get(column))
        if isinstance(aggregated, pd.DataFrame):
            for level in aggregated.columns:
                derived[f"prob_{column}_{level}"] = aggregated[level].to_numpy()
        else:
            derived[f"{column}_media"] = aggregated
    return derived


def run_chain(chain, context=None, tables=None, bundles=None, retrained=None, derive_functions=None, specs=None, upstream_outputs=None, options=None):
    """Score every task of ``chain`` on its common scoring source and propagate the imputations as the chain says.
    ``bundles`` ({task: bundle}) overrides the tasks' bundles, which are otherwise read by
    :func:`~eodgdl.impute.bundle.load_bundle` (from ``<retrained>/<task>/`` when a retrain wrote one there, else the
    data file); ``derive_functions`` ({name: fn}) overrides the derive registry and ``specs`` ({task: TaskSpec}) the
    task files. For sensitivity runs, ``upstream_outputs`` ({task: frame with the source's keys and the task's output
    columns}) replaces a first-level task's scoring (e.g. a retrain scenario), and ``options`` ({task:
    {"level_subsets": {...}, "auxiliary": False}}) changes how a task scores: extra level subsets, or no auxiliary
    models (training shares instead). Returns a :class:`ChainResult`."""
    chain = load_chain(chain) if isinstance(chain, str) else chain
    context = context or Context(tables=tables)
    specs = {task: (specs or {}).get(task) or load_task(task) for task in chain.tasks}
    sources = {spec.score_source for spec in specs.values()}
    _check(len(sources) == 1, f"{chain.name}: the tasks score different sources {sorted(sources)}")
    keys = get_source(sources.pop()).keys
    bundles = {task: as_v2(bundles[task], spec) if task in (bundles or {}) else load_bundle(spec, retrained=retrained) for task, spec in specs.items()}
    frames = {task: build_frame(spec, context, role="score").frame for task, spec in specs.items()}
    for task in chain.tasks:
        _check(frames[task][keys].astype(str).equals(frames[chain.tasks[0]][keys].astype(str)), f"{chain.name}: {task}'s rows differ from {chain.tasks[0]}'s")
    _check_uses(chain, specs)
    provenance = {"chain_hash": chain.hash, "mode": chain.mode, "propagation": chain.propagation, "draws": chain.draws, "seed": chain.seed,
                  "bundles": {task: {key: bundle["metadata"].get(key) for key in ("spec_hash", "scoring_hash", "sklearn_version", "eodgdl_version")}
                              for task, bundle in bundles.items()}}
    base = frames[chain.tasks[0]]
    n_rows = len(base)

    rng = np.random.default_rng(chain.seed)
    per_row = chain.draws if chain.propagation == "draws" else 1
    completions = pd.DataFrame({"row": np.repeat(np.arange(n_rows), per_row), "completion": np.tile(np.arange(per_row), n_rows),
                                "weight": np.full(n_rows * per_row, 1.0 / per_row)})
    needed = _needed(chain)
    outputs, derived = {}, {}
    for index, step in enumerate(chain.steps):
        if isinstance(step, DeriveStep):
            derived.update(_derive(step, index, chain, completions, base, derive_functions))
            continue

        spec, frame, bundle = specs[step.task], frames[step.task], bundles[step.task]
        option = (options or {}).get(step.task, {})
        if option.get("auxiliary") is False:
            bundle = {**bundle, "auxiliary": {}}
        subsets = {**spec.level_subsets, **option.get("level_subsets", {})} or None
        rows = completions["row"].to_numpy()
        conditionals = {}
        if step.task in (upstream_outputs or {}):
            _check(not step.uses, f"{chain.name}: only a task that uses no other can take given outputs")
            first = frame[keys].merge(upstream_outputs[step.task], on=keys, how="left", validate="one_to_one")
            _check(first[spec.probability_columns].notna().all().all(), f"{chain.name}: the given {step.task} outputs miss rows")
            first.index = frame.index
            conditional = first[spec.probability_columns].to_numpy()[rows]
        elif not step.uses:
            first = score_frame(spec, frame, bundle, level_subsets=subsets)
            conditional = first[spec.probability_columns].to_numpy()[rows]
        else:
            conditional, first, conditionals = _conditionals(chain, step, spec, frame, bundle, completions, specs, subsets)
        outputs[step.task] = {**_marginal_outputs(chain, step, spec, first, conditional, completions).to_dict("series"), **conditionals}
        if step.task in needed:
            completions = _advance(completions, spec, conditional, chain.propagation, rng)

    frame = base[keys].copy()
    for task in chain.tasks:
        for column, values in outputs[task].items():
            frame[column] = np.asarray(values)
    for column, values in derived.items():
        frame[column] = values
    long = pd.concat([base[keys].iloc[completions["row"].to_numpy()].reset_index(drop=True), completions.drop(columns="row")], axis=1)

    return ChainResult(chain, frame, long, provenance)


def write_chain(result, out):
    """Write a chain run: ``scores.parquet`` (keys and every output), ``completions.parquet`` (the long table: for
    ``draws``, the multiple imputations with a ``completion`` index) and ``provenance.json``."""
    import json

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result.frame.to_parquet(out / "scores.parquet", index=False)
    result.completions.to_parquet(out / "completions.parquet", index=False)
    (out / "provenance.json").write_text(json.dumps(result.provenance, indent=2, default=str), encoding="utf-8")
    return out


def retrain_chain(chain, out, context=None, n_jobs=-1, progress=True):
    """Retrain every task of ``chain`` in order (:func:`eodgdl.impute.run.retrain`), each written to ``out/<task>``
    (where ``retrained=out`` finds it). Returns ``{task: (bundle path, sha256)}``."""
    from .run import retrain, write_retrain

    chain = load_chain(chain) if isinstance(chain, str) else chain
    context = context or Context()
    return {task: write_retrain(retrain(task, context=context, n_jobs=n_jobs, progress=progress), Path(out) / task) for task in chain.tasks}
