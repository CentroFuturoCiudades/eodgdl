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
  is the mean over draws of its conditional probabilities (not the share of draws), and derive steps run per draw. Every
  task's value is drawn, so each draw is a completed dataset (``completions``). With ``uncertainty: bootstrap`` draw d
  is scored with the d-th (modulo B) of each task's bootstrap bundles (:func:`eodgdl.impute.run.bootstrap_bundles`): a
  draw of the models, then of their predictions, so the imputations are proper and carry the models' uncertainty.
- ``expected``: the upstream's expected score is plugged in as a numeric feature (a derive step reads the expected
  scores too, so every task whose value a later step reads declares ``target.scores``); the downstream outputs are
  flagged as plug-in.

Every mode builds **completions**: one row per (row, scenario) or (row, draw) with a weight (the weights of a row sum
to one) and the value of each step. A step is scored once over the unique (row, upstream values) combinations, in one
batched call. A parallel chain with derive steps forms its completions the same way from the independent marginals.

A downstream task is trained on the upstream target as observed in its training source, not on imputed values, so
retraining an upstream task does not invalidate it; what must agree is the downstream feature's declared levels and
the values the upstream fills it with (checked on every run). ``provenance`` records the bundles each run used.

A draws chain may also be ``given`` per-completion features from outside it (a pipeline's aggregate of another row
level's draws, :mod:`eodgdl.impute.pipeline`): they enter after the feature cache, as columns of the completions, and
every step whose task reads one is scored on it per completion, as on an upstream value.
"""

import functools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .arms import set_class_probabilities
from .bundle import as_v2, load_bootstrap, load_bundle
from .features import build_frame
from .run import expected_score_column, output_columns, score_frame, task_levels
from .sources import Context, get_source
from .spec import load_task, parse_task, read_yaml, stable_hash

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
    score_source: str | None = None   # the chain scores the task on these rows instead of the task's own score_source


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
    uncertainty: str | None = None   # "bootstrap": each draw scored with a bootstrap refit of every task's models

    @property
    def tasks(self):
        return [step.task for step in self.steps if isinstance(step, TaskStep)]

    @property
    def hash(self):
        """Over the YAML and every resolved transform (a named one lives in harmonization/common.yaml)."""
        transforms = {step.task: [[use.task, list(use.transform)] for use in step.uses] for step in self.steps if isinstance(step, TaskStep) and step.uses}
        return stable_hash({"raw": self.raw, "transforms": transforms}) if transforms else stable_hash(self.raw)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def parse_chain(raw, load=None):
    """A :class:`ChainSpec` from a parsed chain YAML, checked against its tasks' specs (``load(name)``, default
    :func:`~eodgdl.impute.spec.load_task`)."""
    load = load or load_task
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
                transform = how.get("transform") or {}
                if isinstance(transform, str):        # a named map of harmonization/common.yaml (defined once)
                    from .harmonize import named_transform

                    transform = named_transform(transform)
                uses.append(Uses(upstream, how.get("as", upstream), tuple((str(k), v if isinstance(v, (int, float)) else str(v)) for k, v in transform.items())))
            steps.append(TaskStep(task, uses, (options or {}).get("score_source")))
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
    uncertainty = raw.get("uncertainty")
    _check(uncertainty in (None, "bootstrap"), f"{name}: uncertainty {uncertainty!r} is not 'bootstrap'")
    _check(uncertainty is None or propagation == "draws", f"{name}: bootstrap uncertainty draws a model per imputation, so it needs propagation: draws")
    chain = ChainSpec(name, mode, propagation, int(raw.get("draws", 50)), int(raw.get("seed", 42)), steps, raw, uncertainty)
    if propagation == "expected":   # every value a later step reads is an expected score
        needed = _needed(chain)
        unscored = [task for task in chain.tasks if task in needed and not load(task).scores]
        _check(not unscored, f"{name}: expected propagation carries the expected score of {unscored}, which declare no target scores")
    return chain


def step_spec(chain, task, load=None):
    """The spec a chain scores ``task`` with: the task's own, on the step's ``score_source`` where it names one."""
    spec = (load or load_task)(task)
    step = next(step for step in chain.steps if isinstance(step, TaskStep) and step.task == task)
    return parse_task({**spec.raw, "score_source": step.score_source}) if step.score_source else spec


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
    base: pd.DataFrame = field(default=None, repr=False)   # the scoring source's frame (the first task's), row by row
    keys: list = field(default=None)                        # the scoring source's keys


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
    assert codes.notna().all(), "derived values outside the declared levels (_check_levels stops the chain before)"
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


def _joined_over_scenarios(values, rows, n_rows):
    """Per row (``rows``: the row of each value), the value its scenarios share, else the union of their "+"-joined
    names in order of appearance (e.g. every feature marginalized in some scenario)."""
    distinct = pd.DataFrame({"row": rows, "value": values}).drop_duplicates()
    several = distinct["row"].duplicated(keep=False)
    joined = distinct[~several].set_index("row")["value"]
    if several.any():
        union = distinct[several].groupby("row")["value"].agg(lambda group: "+".join(dict.fromkeys(name for value in group for name in value.split("+") if name)))
        joined = pd.concat([joined, union])
    return pd.array(joined.reindex(range(n_rows)).to_numpy(), dtype="string")


def _conditionals(chain, step, spec, frame, bundle, completions, specs, level_subsets, given=()):
    """P(y | x, the upstream values of each completion, and its ``given`` columns), one row per completion, each
    distinct (row, upstream and given values) scored once: ``(conditional, first, conditionals)``, with the task's outputs on each row's first combination (the
    columns that do not depend on the upstream; the arms used and the marginalized features joined over the
    combinations the row's completions take) and, for an enumerated step with one upstream, the conditionals of every
    row at every level of the feature (``{prob_<p>_<class>_given_<feature>_<level>: values}``)."""
    n_rows = len(frame)
    filled = pd.DataFrame({"row": completions["row"].to_numpy()})
    for use in step.uses:
        upstream = specs[use.task]
        level_map = _level_map(use, upstream, use.feature in spec.numeric) if chain.propagation != "expected" else None
        values = completions[upstream.prefix]
        filled[use.feature] = values.map(level_map).to_numpy() if level_map else values.to_numpy()
    for column in given:
        filled[column] = completions[column].to_numpy()
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
    for column in given:
        expanded[column] = combos[column].to_numpy()
    scored = score_frame(spec, expanded, bundle, level_subsets=level_subsets)
    position = filled.merge(combos.reset_index(), on=list(filled.columns), how="left")["index"].to_numpy()
    firsts = combos.drop_duplicates("row")
    first = scored.iloc[firsts.index].set_index(firsts["row"].to_numpy()).sort_index()
    first.index = frame.index
    used = np.unique(position)                      # the combinations some completion takes (not the grid's other levels)
    for column in (f"{spec.prefix}_model_used", f"{spec.prefix}_marginalized_features"):
        first[column] = _joined_over_scenarios(scored[column].to_numpy()[used], combos["row"].to_numpy()[used], n_rows)

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
        completions[spec.prefix] = conditional @ np.array([spec.scores[slug] for slug in classes])   # parse_chain: every needed task has scores
    return completions


def _check_levels(chain, step, column, completions, base, keys, levels):
    """Raise where a categorical derived value is missing or outside the step's declared ``levels`` (any value but a
    missing one when it declares none), naming the step, the column and the rows (e.g. no NSE level for a dwelling
    whose points miss an observed answer)."""
    values = completions[column].astype("string")
    outside = (values.isna() | (~values.isin([str(level) for level in levels]) if levels else values.isna())).to_numpy(bool)
    if outside.any():
        rows = np.unique(completions["row"].to_numpy()[outside])
        found = ", ".join(sorted({"missing" if pd.isna(value) else str(value) for value in values[outside]}))
        examples = "; ".join(" ".join(f"{key}={value}" for key, value in record.items()) for record in base[keys].iloc[rows[:5]].to_dict("records"))
        raise ValueError(f"{chain.name}: the derive step {step.name!r} gives {column} values outside its levels ({found}) in {int(outside.sum())} "
                         f"completions ({len(rows)} {'row' if len(rows) == 1 else 'rows'}, e.g. {examples})")


def _derive(step, index, chain, completions, base, keys, derive_functions):
    """Run a derive step (``index``: its position in the chain) on every completion (its columns added to
    ``completions`` in place) and aggregate each new column per row: ``{prob_<column>_<level> or <column>_media:
    values}``; a categorical value outside its levels stops the chain (:func:`_check_levels`). The step's ``seed`` is a
    stream of its own, spawned from the chain's seed by its position, apart from the draws'."""
    from .derive import get_derive

    function = (derive_functions or {}).get(step.name) or get_derive(step.name)
    seed = np.random.SeedSequence(chain.seed, spawn_key=(index,))
    values = function(completions, base, {**step.config, "propagation": chain.propagation, "seed": seed})
    derived = {}
    for column in values.columns:
        completions[column] = values[column].to_numpy()
        levels = step.config.get("levels", {}).get(column)
        if not pd.api.types.is_numeric_dtype(completions[column]):
            _check_levels(chain, step, column, completions, base, keys, levels)
        aggregated = _aggregate(completions[column], completions["weight"].to_numpy(), completions["row"].to_numpy(), len(base), levels)
        if isinstance(aggregated, pd.DataFrame):
            for level in aggregated.columns:
                derived[f"prob_{column}_{level}"] = aggregated[level].to_numpy()
        else:
            derived[f"{column}_media"] = aggregated
    return derived


def _bootstrap_conditional(chain, step, spec, frame, models, completions, specs, level_subsets, option, given=()):
    """P(y | x, the completion's upstream values) as the completion's bootstrap bundle gives it: draw d scored with the
    d-th (modulo B) of ``models``, each distinct (row, upstream values) once per model."""
    which = completions["completion"].to_numpy() % len(models)
    conditional = np.empty((len(completions), len(spec.class_slugs)))
    for m in np.unique(which):
        bundle = models[m] if option.get("auxiliary") is not False else {**models[m], "auxiliary": {}}
        rows = which == m
        if step.uses or given:
            conditional[rows] = _conditionals(chain, step, spec, frame, bundle, completions[rows].reset_index(drop=True), specs, level_subsets, given)[0]
        else:
            scored = score_frame(spec, frame, bundle, level_subsets=level_subsets)
            conditional[rows] = scored[spec.probability_columns].to_numpy()[completions["row"].to_numpy()[rows]]
    return conditional


def _given_columns(chain, given, base, keys, specs, completions):
    """``given`` (a frame of the source's keys, ``completion`` and feature columns, or a function of ``base`` returning
    one) aligned to ``completions``: ``{column: values}``, one value per completion. Every completion must have one, a
    column must not shadow a task's output, and a categorical feature's values must be levels its tasks declare."""
    if callable(given):
        given = given(base)
    _check(chain.propagation == "draws", f"{chain.name}: given features vary by completion, so they need propagation: draws")
    columns = [column for column in given.columns if column not in (*keys, "completion")]
    clash = set(columns) & ({spec.prefix for spec in specs.values()} | set(completions.columns))
    _check(not clash, f"{chain.name}: given columns {sorted(clash)} clash with the chain's own")
    index = base[keys].astype(str).reset_index(drop=True).reset_index(names="row")
    given = given.assign(**{key: given[key].astype(str) for key in keys}).merge(index, on=keys, how="inner")
    aligned = completions[["row", "completion"]].merge(given[["row", "completion", *columns]], on=["row", "completion"], how="left", validate="one_to_one")
    missing = aligned[columns].isna().any(axis=1)
    _check(not missing.any(), f"{chain.name}: no given value for {int(missing.sum())} completions")
    for task, spec in specs.items():
        declared = task_levels(spec)
        for column in columns:
            if column in spec.features and column not in spec.numeric:
                outside = set(aligned[column].astype(str)) - set(declared[column])
                _check(not outside, f"{chain.name}: given {column} takes levels {task} does not declare: {sorted(outside)}")
    return {column: aligned[column].to_numpy() for column in columns}


def run_chain(chain, context=None, tables=None, bundles=None, retrained=None, derive_functions=None, specs=None, upstream_outputs=None, options=None,
              bootstrap=None, given=None):
    """Score every task of ``chain`` on its common scoring source and propagate the imputations as the chain says.
    ``bundles`` ({task: bundle}) overrides the tasks' bundles, which are otherwise read by
    :func:`~eodgdl.impute.bundle.load_bundle` (from ``<retrained>/<task>/`` when a retrain wrote one there, else the
    model file); ``derive_functions`` ({name: fn}) overrides the derive registry and ``specs`` ({task: TaskSpec}) the
    task files. For sensitivity runs, ``upstream_outputs`` ({task: frame with the source's keys and the task's output
    columns}) replaces a first-level task's scoring (e.g. a retrain scenario), and ``options`` ({task:
    {"level_subsets": {...}, "auxiliary": False}}) changes how a task scores: extra level subsets, or no auxiliary
    models (training shares instead). ``bootstrap`` ({task: [bundles]}) gives the bootstrap bundles a chain with
    ``uncertainty: bootstrap`` scores its draws with (default: :func:`~eodgdl.impute.bundle.load_bootstrap` under
    ``retrained``); the marginals, arms and flags are the bundles', the draws the bootstrap models'. ``given`` (draws
    only: a frame of the source's keys, ``completion`` and feature columns, or a function of the source frame returning
    one) gives every completion values from outside the chain (:func:`_given_columns`): they become columns of the
    completions, each step whose task reads one is scored on them, a derive step finds them among the completions, and
    each is aggregated per row like a derived value. Returns a :class:`ChainResult`."""
    chain = load_chain(chain) if isinstance(chain, str) else chain
    context = context or Context(tables=tables)
    specs = {task: (specs or {}).get(task) or step_spec(chain, task) for task in chain.tasks}
    sources = {spec.score_source for spec in specs.values()}
    _check(len(sources) == 1, f"{chain.name}: the tasks score different sources {sorted(sources)}")
    keys = get_source(sources.pop()).keys
    bundles = {task: as_v2(bundles[task], spec) if task in (bundles or {}) else load_bundle(spec, retrained=retrained) for task, spec in specs.items()}
    frames = {task: build_frame(spec, context, role="score").frame for task, spec in specs.items()}
    for task in chain.tasks:
        _check(frames[task][keys].astype(str).equals(frames[chain.tasks[0]][keys].astype(str)), f"{chain.name}: {task}'s rows differ from {chain.tasks[0]}'s")
    _check_uses(chain, specs)
    if chain.uncertainty == "bootstrap" and bootstrap is None:
        bootstrap = {task: load_bootstrap(spec, retrained) for task, spec in specs.items()}
    if bootstrap:
        _check(chain.propagation == "draws", f"{chain.name}: bootstrap models are drawn per imputation, so they need propagation: draws")
        missing = [task for task in chain.tasks if not bootstrap.get(task)]
        _check(not missing, f"{chain.name}: no bootstrap bundles for {missing}: run `eodgdl impute retrain <task> --bootstrap B` and point --retrained at its root")
        _check(len({len(bootstrap[task]) for task in chain.tasks}) == 1, f"{chain.name}: the tasks' bootstrap sets differ in size")
        bootstrap = {task: [as_v2(model, specs[task]) for model in bootstrap[task]] for task in chain.tasks}
    provenance = {"chain_hash": chain.hash, "mode": chain.mode, "propagation": chain.propagation, "draws": chain.draws, "seed": chain.seed,
                  "bundles": {task: {key: bundle["metadata"].get(key) for key in ("spec_hash", "scoring_hash", "sklearn_version", "eodgdl_version")}
                              for task, bundle in bundles.items()},
                  **({"uncertainty": "bootstrap", "bootstrap": {task: len(models) for task, models in bootstrap.items()}} if bootstrap else {})}
    base = frames[chain.tasks[0]]
    n_rows = len(base)

    rng = np.random.default_rng(chain.seed)
    per_row = chain.draws if chain.propagation == "draws" else 1
    completions = pd.DataFrame({"row": np.repeat(np.arange(n_rows), per_row), "completion": np.tile(np.arange(per_row), n_rows),
                                "weight": np.full(n_rows * per_row, 1.0 / per_row)})
    needed = _needed(chain)
    outputs, derived = {}, {}
    given_columns = []
    if given is not None:
        for column, values in _given_columns(chain, given, base, keys, specs, completions).items():
            completions[column] = values
            given_columns.append(column)
            readers = [spec for spec in specs.values() if column in spec.features and column not in spec.numeric]
            levels = list(task_levels(readers[0])[column]) if readers else None
            aggregated = _aggregate(completions[column], completions["weight"].to_numpy(), completions["row"].to_numpy(), n_rows, levels)
            if isinstance(aggregated, pd.DataFrame):
                derived.update({f"prob_{column}_{level}": aggregated[level].to_numpy() for level in aggregated.columns})
            else:
                derived[f"{column}_media"] = aggregated
        provenance["given"] = given_columns
    for index, step in enumerate(chain.steps):
        if isinstance(step, DeriveStep):
            derived.update(_derive(step, index, chain, completions, base, keys, derive_functions))
            continue

        spec, frame, bundle = specs[step.task], frames[step.task], bundles[step.task]
        option = (options or {}).get(step.task, {})
        if option.get("auxiliary") is False:
            bundle = {**bundle, "auxiliary": {}}
        subsets = {**spec.level_subsets, **option.get("level_subsets", {})} or None
        step_given = [column for column in given_columns if column in spec.features]
        rows = completions["row"].to_numpy()
        conditionals = {}
        if step.task in (upstream_outputs or {}):
            _check(not step.uses and not step_given, f"{chain.name}: only a task that uses no other value can take given outputs")
            first = frame[keys].merge(upstream_outputs[step.task], on=keys, how="left", validate="one_to_one")
            _check(first[spec.probability_columns].notna().all().all(), f"{chain.name}: the given {step.task} outputs miss rows")
            first.index = frame.index
            conditional = first[spec.probability_columns].to_numpy()[rows]
        elif not step.uses and not step_given:
            first = score_frame(spec, frame, bundle, level_subsets=subsets)
            conditional = first[spec.probability_columns].to_numpy()[rows]
        else:
            conditional, first, conditionals = _conditionals(chain, step, spec, frame, bundle, completions, specs, subsets, step_given)
        if bootstrap and step.task not in (upstream_outputs or {}):
            conditional = _bootstrap_conditional(chain, step, spec, frame, bootstrap[step.task], completions, specs, subsets, option, step_given)
        outputs[step.task] = {**_marginal_outputs(chain, step, spec, first, conditional, completions).to_dict("series"), **conditionals}
        if step.task in needed or chain.propagation == "draws":      # a draw is a completed dataset: every value drawn
            completions = _advance(completions, spec, conditional, chain.propagation, rng)

    frame = base[keys].copy()
    for task in chain.tasks:
        for column, values in outputs[task].items():
            frame[column] = np.asarray(values)
    for column, values in derived.items():
        frame[column] = values
    long = pd.concat([base[keys].iloc[completions["row"].to_numpy()].reset_index(drop=True), completions.drop(columns="row")], axis=1)

    return ChainResult(chain, frame, long, provenance, base, list(keys))


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


def retrain_chain(chain, out, context=None, n_jobs=-1, progress=True, bootstrap=0):
    """Retrain every task of ``chain`` in order (:func:`eodgdl.impute.run.retrain`), each written to ``out/<task>``
    (where ``retrained=out`` finds it), with ``bootstrap`` bootstrap bundles each when asked
    (:func:`eodgdl.impute.run.bootstrap_bundles`). Returns ``{task: (bundle path, sha256)}``."""
    from .run import bootstrap_bundles, retrain, write_bootstrap, write_retrain

    chain = load_chain(chain) if isinstance(chain, str) else chain
    context = context or Context()
    written = {}
    for task in chain.tasks:
        result = retrain(task, context=context, n_jobs=n_jobs, progress=progress)
        written[task] = write_retrain(result, Path(out) / task)
        if bootstrap:
            write_bootstrap(result.spec, bootstrap_bundles(result.spec, result.bundle, bootstrap, context=context, n_jobs=n_jobs), Path(out) / task)
    return written
