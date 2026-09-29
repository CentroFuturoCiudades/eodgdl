"""Fitted task bundles: one joblib per task with its fitted arms, auxiliary models, category levels and the metadata
that proves what they were trained on.

A bundle (format ``eodgdl.impute.bundle/2``) is a dict: ``task``, ``arms`` ({arm: {"model", "features",
"requires"}}), ``auxiliary`` ({arm: {feature: model}}), ``classes``, ``category_levels`` and ``metadata`` (spec
hash, scoring hash, level-contract hash, data and feature versions, sklearn and eodgdl versions, the selection and
test metrics, upstream bundle hashes). :func:`check_bundle` refuses a bundle whose scikit-learn version, level
contract or scoring definition differs from the running code and task spec. The giro bundle fitted before the
engine (``model_with_education`` ... keys) is converted on load (:func:`from_legacy_giro`).

:func:`load_bundle` finds a task's bundle the same way for every mode: an explicit file, else the one a retrain wrote
under ``<retrained>/<task>/``, else the installed data file."""

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


def from_legacy_giro(legacy, spec):
    """The giro bundle as fitted by the retired giro notebook (keys ``model_with_education``,
    ``features_with_education``, ``destination_models``, ...) as a v2 bundle. Its arm features and category levels
    are checked against ``spec`` by :func:`check_bundle`; the hashes it lacks are filled from ``spec`` only after
    those checks pass, and ``metadata["converted_from"]`` says so."""
    arms, auxiliary = {}, {}
    for arm in spec.arms:
        arms[arm.name] = {"model": legacy[f"model_{arm.name}"], "features": list(legacy[f"features_{arm.name}"]), "requires": list(arm.requires)}
        destination = (legacy.get("destination_models") or {}).get(arm.name)
        if destination is not None:
            auxiliary[arm.name] = {"destino_trabajo": destination}
    metadata = dict(legacy.get("metadata", {}))
    metadata["converted_from"] = "legacy giro bundle"
    return {"format": FORMAT, "task": spec.name, "arms": arms, "auxiliary": auxiliary,
            "classes": {label: slug for slug, label in legacy["giro_labels"].items()}, "category_levels": legacy["category_levels"], "metadata": metadata}


def as_v2(bundle, spec):
    if isinstance(bundle, dict) and bundle.get("format") == FORMAT:
        return bundle
    if isinstance(bundle, dict) and "model_with_education" in bundle and spec.name == "giro":
        return from_legacy_giro(bundle, spec)
    raise BundleMismatch(f"Not a bundle of task {spec.name!r} in a known format")


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
    it there; else the task's data file (:func:`eodgdl.data.resolve`: ``$EODGDL_DATA_DIR`` or the data mirror)."""
    if path is not None:
        return Path(path)
    if retrained is not None and (Path(retrained) / spec.name / spec.bundle_name).exists():
        return Path(retrained) / spec.name / spec.bundle_name
    if spec.bundle_file is None:
        raise ValueError(f"Task {spec.name!r} names no bundle file")
    from eodgdl.data import resolve

    return Path(resolve(spec.bundle_file))


def load_bundle(spec, path=None, retrained=None):
    """The task's bundle (:func:`bundle_path`) as a v2 bundle (a legacy giro bundle converted, :func:`as_v2`)."""
    import joblib

    return as_v2(joblib.load(bundle_path(spec, path, retrained)), spec)


def bundle_arms(bundle, auxiliary=True):
    """The bundle's arms, in dispatch order, with their auxiliary models (none with ``auxiliary=False``: unsupported
    levels are then averaged over the training shares)."""
    return [Arm(name, entry["model"], entry["features"], requires=entry["requires"], auxiliary=bundle["auxiliary"].get(name, {}) if auxiliary else {})
            for name, entry in bundle["arms"].items()]


def save_bundle(bundle, path):
    """Write ``bundle`` with joblib; returns its sha256 (for ``registry.txt`` and downstream bundles)."""
    import joblib

    joblib.dump(bundle, path)
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def legacy_view(bundle):
    """A v2 giro bundle with the keys ``eodgdl.giro.load_model`` has always returned (``model_with_education``,
    ``features_with_education``, ``destination_models``, ``category_levels``, ``giro_classes``, ``giro_labels``,
    ``metadata``), sharing the fitted objects."""
    view = {}
    for arm, entry in bundle["arms"].items():
        view[f"model_{arm}"] = entry["model"]
        view[f"features_{arm}"] = entry["features"]
    view["category_levels"] = bundle["category_levels"]
    view["giro_classes"] = list(bundle["classes"].values())
    view["giro_labels"] = {slug: label for label, slug in bundle["classes"].items()}
    view["destination_models"] = {arm: models["destino_trabajo"] for arm, models in bundle["auxiliary"].items() if "destino_trabajo" in models} or None
    view["metadata"] = bundle["metadata"]
    return view
