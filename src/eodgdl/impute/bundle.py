"""Fitted task bundles: one joblib per task with its fitted arms, auxiliary models, category levels and the metadata
that proves what they were trained on, and a retrain's bootstrap refits of them (:func:`load_bootstrap`).

A bundle (format ``eodgdl.impute.bundle/2``) is a dict: ``task``, ``arms`` ({arm: {"model", "features",
"requires"}}), ``auxiliary`` ({arm: {feature: model}}), ``classes``, ``category_levels`` and ``metadata`` (spec
hash, scoring hash, level-contract hash, data and feature versions, sklearn and eodgdl versions, the selection and
test metrics, upstream bundle hashes). :func:`check_bundle` refuses a bundle whose scikit-learn version, level
contract or scoring definition differs from the running code and task spec; :func:`as_v2` refuses any other layout
(the giro notebook's, ``model_with_education`` ... keys, was converted on load until 2026-09-29).

:func:`load_bundle` finds a task's bundle the same way for every mode: an explicit file, else the one a retrain wrote
under ``<retrained>/<task>/``, else the installed model file (the repo's ``models/``). A chain draws with a retrain's
bootstrap refits only where they come from the bundle it loads (:func:`check_bootstrap`)."""

import hashlib
from pathlib import Path

from .arms import Arm

FORMAT = "eodgdl.impute.bundle/2"


class BundleMismatch(ValueError):
    """The bundle was fitted under a different scikit-learn, level contract or task definition."""


def make_bundle(spec, arms, auxiliary, category_levels, metadata):
    """A v2 bundle for ``spec``: ``arms`` {arm: fitted pipeline}, ``auxiliary`` {arm: {feature: fitted model}}."""
    import sklearn

    import eodgdl

    return {
        "format": FORMAT,
        "task": spec.name,
        "arms": {arm.name: {"model": arms[arm.name], "features": list(arm.features), "requires": list(arm.requires)} for arm in spec.arms},
        "auxiliary": {arm: dict(models) for arm, models in auxiliary.items()},
        "classes": dict(spec.classes),
        "category_levels": category_levels,
        "metadata": {
            "spec_hash": spec.hash, "scoring_hash": spec.scoring_hash(), "level_contract_hash": spec.level_contract_hash(category_levels),
            "sklearn_version": sklearn.__version__, "eodgdl_version": eodgdl.__version__, **metadata,
        },
    }


def as_v2(bundle, spec):
    """``bundle`` if it is in the v2 format, else :class:`BundleMismatch` (e.g. a giro bundle from before 0.3.0: retrain
    it with ``eodgdl impute retrain giro``)."""
    if isinstance(bundle, dict) and bundle.get("format") == FORMAT:
        return bundle
    raise BundleMismatch(f"Not a bundle of task {spec.name!r} in the {FORMAT} format")


def check_bundle(bundle, spec, category_levels):
    """Raise :class:`BundleMismatch` unless ``bundle`` can score ``spec``'s rows under the running code: same
    scikit-learn version, same category levels (declared plus the source's), same classes, arms and auxiliary
    predictors as the task spec."""
    import sklearn

    metadata = bundle["metadata"]
    problems = []
    if bundle["task"] != spec.name:
        problems.append(f"bundle of task {bundle['task']!r}")
    if metadata.get("sklearn_version") != sklearn.__version__:
        problems.append(f"fitted with scikit-learn {metadata.get('sklearn_version')}, running {sklearn.__version__}")
    if bundle["category_levels"] != category_levels:
        changed = sorted(feature for feature in set(bundle["category_levels"]) | set(category_levels)
                         if bundle["category_levels"].get(feature) != category_levels.get(feature))
        problems.append(f"category levels differ from the task's for {changed}")
    if bundle["classes"] != spec.classes:
        problems.append(f"classes {bundle['classes']} differ from the task's {spec.classes}")
    if [name for name in bundle["arms"]] != [arm.name for arm in spec.arms]:
        problems.append(f"arms {list(bundle['arms'])} differ from the task's {[arm.name for arm in spec.arms]}")
    else:
        for arm in spec.arms:
            if bundle["arms"][arm.name]["features"] != list(arm.features):
                problems.append(f"arm {arm.name}: features differ from the task's")
            if list(bundle["arms"][arm.name]["model"].named_steps["prepare"].kw_args["features"]) != list(arm.features):
                problems.append(f"arm {arm.name}: the fitted pipeline selects other features")
            for feature, model in bundle["auxiliary"].get(arm.name, {}).items():
                if feature not in spec.auxiliary:
                    problems.append(f"arm {arm.name}: auxiliary model for {feature!r}, which the task does not declare")
                elif list(model.named_steps["prepare"].kw_args["features"]) != spec.auxiliary_predictors(feature, arm.name):
                    problems.append(f"arm {arm.name}: the auxiliary {feature} model uses other predictors")
    for name, expected in (("scoring_hash", spec.scoring_hash()), ("level_contract_hash", spec.level_contract_hash(category_levels))):
        if name in metadata and metadata[name] != expected:
            problems.append(f"{name} differs from the task's")
    if problems:
        raise BundleMismatch(f"Bundle cannot score task {spec.name!r}: " + "; ".join(problems))
    metadata.setdefault("scoring_hash", spec.scoring_hash())
    metadata.setdefault("level_contract_hash", spec.level_contract_hash(category_levels))

    return bundle


def bundle_path(spec, path=None, retrained=None):
    """Where the task's bundle is read from: ``path``; else ``<retrained>/<task>/<bundle file>`` when a retrain wrote
    it there; else the task's model file under ``models/`` (:func:`eodgdl.data.resolve`)."""
    if path is not None:
        return Path(path)
    if retrained is not None and (Path(retrained) / spec.name / spec.bundle_name).exists():
        return Path(retrained) / spec.name / spec.bundle_name
    if spec.bundle_file is None:
        raise ValueError(f"Task {spec.name!r} names no bundle file")
    from eodgdl.data import resolve

    return Path(resolve(spec.bundle_file))


def load_bundle(spec, path=None, retrained=None):
    """The task's bundle (:func:`bundle_path`), refused unless it is a v2 bundle (:func:`as_v2`)."""
    import joblib

    return as_v2(joblib.load(bundle_path(spec, path, retrained)), spec)


def bootstrap_paths(spec, retrained):
    """The files of the task's bootstrap bundles under ``<retrained>/<task>/bootstrap/``, in order; none without
    ``retrained``."""
    if retrained is None:
        return []
    return sorted((Path(retrained) / spec.name / "bootstrap").glob(f"{Path(spec.bundle_name).stem}_*.joblib"))


def load_bootstrap(spec, retrained):
    """The task's bootstrap bundles under ``<retrained>/<task>/bootstrap/`` (:func:`eodgdl.impute.run.write_bootstrap`),
    in order, as v2 bundles; an empty list where there are none."""
    import joblib

    return [as_v2(joblib.load(path), spec) for path in bootstrap_paths(spec, retrained)]


def check_bootstrap(bundle, models):
    """``models`` (a task's bootstrap bundles) unless one was refitted from another bundle than ``bundle``: its task
    spec, its selected configurations (``metadata["selected"]``, which :func:`eodgdl.impute.run.bootstrap_bundles`
    copies) or its training weight (:func:`training_weight`) differ, e.g. a set an earlier retrain left beside a newer
    bundle. Raises :class:`BundleMismatch`."""
    metadata = bundle["metadata"]
    for key, what in (("spec_hash", "task spec"), ("selected", "selected configurations"), ("training_weight", "training weight")):
        value = (lambda m: training_weight(m)) if key == "training_weight" else (lambda m: m.get(key))
        stale = [m for m, model in enumerate(models) if value(model["metadata"]) != value(metadata)]
        if stale:
            raise BundleMismatch(f"The bootstrap bundles {stale[:5]} of task {bundle['task']!r} were refitted from another bundle (other {what}): "
                                 f"rerun `eodgdl impute retrain {bundle['task']} --bootstrap B`")
    return models


def training_weight(metadata):
    """The weight a bundle was trained on: the sha256 of the weight file its training source joined
    (``metadata["data_versions"]["training"]["weight"]``, :func:`eodgdl.impute.sources.with_weight`), else ``"design"``,
    a column of the source's own rows (the survey's ``ponderador``, ENOE's and ENIGH's factors: every bundle up to
    v0.9.0)."""
    return ((metadata.get("data_versions") or {}).get("training") or {}).get("weight", "design")


def bundle_identity(bundle, spec, path=None, bootstrap=()):
    """What a run records of a bundle (a chain's ``provenance["bundles"][task]``): its spec and scoring hashes, the
    scikit-learn and eodgdl versions it was fitted under, its training source and the data ``versions`` it was trained
    on (the weight file's sha256 among them), the sha256 of its file (``path``; None for a bundle handed in) and one
    sha256 over its bootstrap bundles' files (``bootstrap``), if any."""
    from .sources import file_digest, files_digest

    metadata = bundle["metadata"]
    identity = {key: metadata.get(key) for key in ("spec_hash", "scoring_hash", "sklearn_version", "eodgdl_version")}
    identity.update({"training_source": spec.source, "versions": (metadata.get("data_versions") or {}).get("training"),
                     "sha256": file_digest(str(path)) if path is not None else None})
    if bootstrap:
        identity["bootstrap_sha256"] = files_digest(bootstrap)
    return identity


def weight_conflicts(levels):
    """Where a run mixes weights, as readable strings (none when sound): ``levels`` maps a name to what a chain run
    records (its ``provenance``: ``bundles`` and the scoring source's ``versions``). Every bundle trained on the EOD (a
    training source ``eod.*``) must have been trained on the weight its level's scoring source reads now
    (``versions["weight"]``, else ``"design"``), and so on one weight across the levels; ENOE's and ENIGH's tasks train
    on their surveys' own weights and are exempt. A recorded bundle without ``training_source`` (a run before the
    weight could change) is reported, as nothing says which weight it was trained on."""
    problems, weights = [], {}
    for name, entry in levels.items():
        source_weight = (entry.get("versions") or {}).get("weight", "design")
        weights.setdefault(source_weight, []).append(f"{name}'s source")
        for task, identity in (entry.get("bundles") or {}).items():
            if "training_source" not in identity:
                problems.append(f"{name}: no training source recorded for {task}'s bundle")
                continue
            if not identity["training_source"].startswith("eod."):
                continue
            trained = (identity.get("versions") or {}).get("weight", "design")
            weights.setdefault(trained, []).append(f"{name}.{task}")
            if trained != source_weight:
                problems.append(f"{name}: {task}'s bundle was trained on weight {trained[:12]}, its source reads {source_weight[:12]}")
    if len(weights) > 1:
        problems.append("more than one weight: " + "; ".join(f"{weight[:12]}: {', '.join(names)}" for weight, names in weights.items()))
    return problems


def bundle_arms(bundle, auxiliary=True):
    """The bundle's arms, in dispatch order, with their auxiliary models (none with ``auxiliary=False``: unsupported
    levels are then averaged over the training shares)."""
    return [Arm(name, entry["model"], entry["features"], requires=entry["requires"], auxiliary=bundle["auxiliary"].get(name, {}) if auxiliary else {})
            for name, entry in bundle["arms"].items()]


def save_bundle(bundle, path):
    """Write ``bundle`` with joblib; returns its sha256 (for the manifests and downstream bundles)."""
    import joblib

    joblib.dump(bundle, path)
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
