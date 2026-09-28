"""Running a task: score its rows with a fitted bundle."""

import pandas as pd

from .arms import Arm, assign_arms, impute_with_arms
from .bundle import as_v2, check_bundle, load_bundle
from .evaluate import validate_probability_rows
from .features import build_frame
from .sources import Context, get_source
from .spec import load_task


def task_levels(spec):
    """The task's category-level contract: its declared levels, else its source's."""
    return spec.category_levels(get_source(spec.source).schema_levels)


def score_frame(spec, frame, bundle, level_subsets=None):
    """Score ``frame`` (a task frame: :func:`~eodgdl.impute.features.build_frame`) with a checked v2 ``bundle``:
    every row whose target is unobserved goes to the first arm whose required covariates it has; returns the frame
    with the task's output columns (:func:`~eodgdl.impute.arms.impute_with_arms`)."""
    levels = task_levels(spec)
    bundle = check_bundle(as_v2(bundle, spec), spec, levels)
    arms = [Arm(name, entry["model"], entry["features"], requires=entry["requires"], auxiliary=bundle["auxiliary"].get(name, {}))
            for name, entry in bundle["arms"].items()]
    assigned = assign_arms(frame, arms, spec.is_missing)
    assigned[~frame[spec.unknown_column].astype(bool).to_numpy()] = pd.NA
    scored = impute_with_arms(frame, spec.target, spec.class_slugs, arms, assigned, spec.prefix, spec.numeric, levels,
                              level_subsets=level_subsets, missing_label=spec.missing_label)
    validate_probability_rows(scored, spec.probability_columns)

    return scored


def score_task(task, tables=None, bundle=None, path=None, context=None):
    """Load (or take) the task's bundle and score its source's rows. ``tables`` are the survey tables (loaded, and
    the features cached, when omitted); ``bundle`` a fitted bundle (v2 or a legacy giro dict), else read from
    ``path`` or the task's data file."""
    spec = load_task(task) if isinstance(task, str) else task
    context = context or Context(tables=tables)
    bundle = bundle if bundle is not None else load_bundle(spec, path)
    frame = build_frame(spec, context).frame

    return score_frame(spec, frame, bundle)
