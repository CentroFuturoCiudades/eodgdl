"""Task specifications: one YAML per task under ``impute/tasks/``, parsed and checked into a :class:`TaskSpec`.

A task names its source, target classes, features (builders and named groups), declared levels, feature arms,
auxiliary level models, candidate grid and evaluation settings. Hashes pin what a bundle was trained with: the whole
spec (:attr:`TaskSpec.hash`), the part that decides how rows are scored (:meth:`TaskSpec.scoring_hash`) and the
category-level contract (:meth:`TaskSpec.level_contract_hash`)."""

import functools
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .levels import MISSING_LABEL, identify_missing

TASKS_DIR = Path(__file__).parent / "tasks"


def stable_hash(value):
    """sha256 of a JSON-serializable value, independent of dict order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


@functools.cache
def read_yaml(path):
    """A YAML file parsed (once per path and process: callers must not mutate it)."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    features: tuple
    requires: tuple = ()


@dataclass
class TaskSpec:
    name: str
    source: str                    # the training rows
    score_source: str              # the rows the task imputes (default: the training source)
    target_column: str
    classes: dict                  # label -> slug, in output order
    scores: dict                   # slug -> ordinal score (optional: an expected score is written, and `expected` propagation allowed)
    prefix: str
    missing_label: str
    missing_values: dict           # feature -> answers that count as unobserved
    builders: object               # [builder, ...], or {source: [builder, ...]} when the sources differ
    groups: dict
    numeric: list
    declared_levels: dict
    extra_levels: dict             # feature -> levels appended to its schema levels (e.g. a builder's own)
    schema_columns: dict
    arms: list                     # [ArmSpec], in dispatch order
    auxiliary: dict                # feature -> {"exclude": [...]}
    selection: dict
    evaluation: dict
    builder_config: dict
    bundle_file: str | None
    level_subsets: dict            # feature -> levels an unsupported value is averaged over at scoring time
    raw: dict = field(repr=False)

    @property
    def hash(self):
        return stable_hash(self.raw)

    @property
    def class_slugs(self):
        return list(self.classes.values())

    @property
    def target(self):
        """The column holding the target's slug (NA where unobserved)."""
        return self.prefix

    @property
    def unknown_column(self):
        return f"{self.prefix}_desconocido"

    @property
    def probability_columns(self):
        return [f"prob_{self.prefix}_{slug}" for slug in self.class_slugs]

    @property
    def bundle_name(self):
        """File name of the task's bundle: its data file, else ``<task>.joblib``."""
        return self.bundle_file or f"{self.name}.joblib"

    def builders_for(self, source):
        """The feature builders run on ``source``'s frame."""
        if isinstance(self.builders, dict):
            return list(self.builders.get(source, []))
        return list(self.builders)

    @property
    def all_builders(self):
        return [name for names in self.builders.values() for name in names] if isinstance(self.builders, dict) else list(self.builders)

    def expand(self, names):
        """Feature names with group names replaced by their members, in order, without repeats."""
        out = []
        for name in names:
            for feature in self.groups.get(name, [name]):
                if feature not in out:
                    out.append(feature)
        return out

    def arm(self, name):
        return next(arm for arm in self.arms if arm.name == name)

    @property
    def features(self):
        """Every feature any arm uses, in order."""
        return self.expand([feature for arm in self.arms for feature in arm.features])

    def auxiliary_predictors(self, feature, arm):
        """Predictors of the auxiliary model for ``feature`` in ``arm``: the arm's features minus ``exclude``."""
        exclude = set(self.expand(self.auxiliary[feature].get("exclude", [])))
        return [column for column in self.arm(arm).features if column not in exclude]

    def is_missing(self, frame, feature):
        """Rows whose value of ``feature`` is unobserved: NA, blank, the missing label or a listed answer."""
        missing = identify_missing(frame[feature], self.missing_label)
        values = self.missing_values.get(feature)
        if values:
            missing |= frame[feature].astype("string").isin(values).fillna(False)
        return missing

    def category_levels(self, schema_levels):
        """Declared levels of every categorical feature, the missing label last. ``schema_levels(column)`` gives the
        source's levels for a feature not declared here (under ``schema_columns`` or its own name), with the task's
        ``levels.extra`` for it appended."""
        levels = {}
        for feature in self.features:
            if feature in self.numeric:
                continue
            values = self.declared_levels.get(feature)
            if values is None:
                values = list(schema_levels(self.schema_columns.get(feature, feature))) + list(self.extra_levels.get(feature, []))
            levels[feature] = [str(value) for value in values] + [self.missing_label]
        return levels

    def level_contract(self, category_levels):
        return {"missing_label": self.missing_label, "numeric": sorted(self.numeric), "levels": category_levels}

    def level_contract_hash(self, category_levels):
        return stable_hash(self.level_contract(category_levels))

    def scoring_hash(self):
        """Hash of what decides how a row is scored by a fitted bundle: classes, features per arm and their
        dispatch, auxiliary predictors, numeric features, missing label and missing answers."""
        return stable_hash({
            "classes": self.classes, "prefix": self.prefix, "numeric": sorted(self.numeric), "missing_label": self.missing_label,
            "missing_values": self.missing_values,
            "arms": [{"name": arm.name, "features": list(arm.features), "requires": list(arm.requires)} for arm in self.arms],
            "auxiliary": {feature: {arm.name: self.auxiliary_predictors(feature, arm.name) for arm in self.arms} for feature in self.auxiliary},
            **({"level_subsets": self.level_subsets} if self.level_subsets else {}),
        })


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def parse_task(raw):
    """A :class:`TaskSpec` from a parsed task YAML, validated."""
    name = raw["task"]
    target = raw["target"]
    classes = {str(label): str(slug) for label, slug in target["classes"].items()}
    _check(len(set(classes.values())) == len(classes), f"{name}: class slugs repeat")
    _check(not target.get("scores") or set(map(str, target["scores"])) == set(classes.values()), f"{name}: target scores must cover every class slug")
    features = raw.get("features", {})
    groups = {group: list(members) for group, members in features.get("groups", {}).items()}
    levels = raw.get("levels", {})

    spec = TaskSpec(
        name=name, source=raw["source"], score_source=raw.get("score_source", raw["source"]), target_column=target["column"], classes=classes,
        scores={str(slug): float(value) for slug, value in target.get("scores", {}).items()},
        prefix=raw.get("outputs", {}).get("prefix", name), missing_label=raw.get("missing_label", MISSING_LABEL),
        missing_values={feature: list(values) for feature, values in raw.get("missing_values", {}).items()},
        builders=({source: list(names) for source, names in features["builders"].items()} if isinstance(features.get("builders"), dict) else list(features.get("builders", []))), groups=groups, numeric=list(features.get("numeric", [])),
        declared_levels={feature: list(values) for feature, values in levels.get("declared", {}).items()},
        extra_levels={feature: list(values) for feature, values in levels.get("extra", {}).items()},
        schema_columns=dict(levels.get("schema_columns", {})), arms=[], auxiliary=dict(raw.get("auxiliary", {})),
        selection=dict(raw.get("selection", {})), evaluation=dict(raw.get("evaluation", {})),
        builder_config=dict(raw.get("builders", {})), bundle_file=raw.get("bundle"),
        level_subsets={feature: list(levels) for feature, levels in raw.get("scoring", {}).get("level_subsets", {}).items()}, raw=raw,
    )
    for arm_name, arm in raw["arms"].items():
        exclude = set(spec.expand(arm.get("exclude", [])))
        arm_features = tuple(feature for feature in spec.expand(arm["features"]) if feature not in exclude)
        requires = tuple(arm.get("requires", []))
        _check(set(requires) <= set(arm_features), f"{name}: arm {arm_name} requires features it does not use: {sorted(set(requires) - set(arm_features))}")
        spec.arms.append(ArmSpec(arm_name, arm_features, requires))
    _check(spec.arms, f"{name}: no arms")
    _check(not spec.arms[-1].requires, f"{name}: the last arm must require nothing, so every row to impute has an arm")
    _check(set(spec.numeric) <= set(spec.features), f"{name}: numeric features no arm uses: {sorted(set(spec.numeric) - set(spec.features))}")
    for feature in spec.auxiliary:
        _check(feature in spec.features and feature not in spec.numeric, f"{name}: auxiliary model for {feature!r}, not a categorical feature of the task")
    for feature in spec.level_subsets:
        _check(feature in spec.features and feature not in spec.numeric, f"{name}: level subset for {feature!r}, not a categorical feature of the task")
    for feature in spec.declared_levels:
        _check(feature in spec.features, f"{name}: levels declared for {feature!r}, which no arm uses")
    for feature in spec.extra_levels:
        _check(feature in spec.features and feature not in spec.numeric and feature not in spec.declared_levels,
               f"{name}: extra levels for {feature!r}, not a categorical feature whose levels come from the schema")
    for builder in spec.builder_config:
        _check(builder in spec.all_builders, f"{name}: configuration for builder {builder!r}, which the task does not use")
    if isinstance(spec.builders, dict):
        _check(set(spec.builders) <= {spec.source, spec.score_source}, f"{name}: builders for sources the task neither trains nor scores on")

    return spec


@functools.cache
def load_task(name):
    """The task spec ``impute/tasks/<name>.yaml``."""
    path = TASKS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"No task {name!r}; known: {sorted(p.stem for p in TASKS_DIR.glob('*.yaml'))}")
    return parse_task(read_yaml(path))
