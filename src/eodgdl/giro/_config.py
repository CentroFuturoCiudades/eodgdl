"""Constants of the giro imputation model, read from its task spec (``eodgdl/impute/tasks/giro.yaml``), the
``eod.workers`` source configuration and the eodgdl schemas."""

import functools

from eodgdl.impute.sources.eod import load_config as _source_config
from eodgdl.impute.sources.eod import schema_levels as _schema_levels
from eodgdl.impute.spec import load_task

TASK = load_task("giro")


def load_config():
    """The giro task spec as parsed from its YAML."""
    return TASK.raw


GIRO_SLUGS = dict(TASK.classes)  # label -> slug
GIRO_LABELS = {slug: label for label, slug in GIRO_SLUGS.items()}  # slug -> label
GIRO_CLASSES = TASK.class_slugs
NO_ESPECIFICADO = TASK.missing_label
_WORKERS = _source_config()["eod.workers"]
EMPLOYED_CATEGORIES = list(_WORKERS["employed_categories"])
KEYS = list(_WORKERS["keys"])
WORK_TRIP_PURPOSE = _source_config()["eod.work_trip"]["purpose"]

DESTINATION_FEATURES = list(TASK.groups["destination"])
MOBILITY_FEATURES = list(TASK.groups["mobility"])
SECTOR_FEATURES = list(TASK.arm("with_education").features)
ROBUST_SECTOR_FEATURES = list(TASK.arm("without_education").features)
SHIFT_PROFILE_FEATURES = list(TASK.evaluation["shift"]["profile"])
NUMERIC_FEATURES = list(TASK.numeric)

_DESTINATION = TASK.builder_config["giro.destination"]
DENUE_STATE_CODE = _DESTINATION["state_code"]
DENUE_RELEASE = str(_DESTINATION["denue_release"])
DESTINATION_AMBITO_LEVELS = list(TASK.declared_levels["destino_ambito"])
DENUE_SCIAN2 = {str(code): str(giro) for code, giro in _DESTINATION["denue_scian2"].items()}


@functools.cache
def build_category_levels():
    """Every level a categorical feature can take, plus the missing label. This is the contract the encoders are
    built with (``categories=...``, ``handle_unknown="error"``): a level absent from the training data still gets its
    own column instead of a silent all-zero block, and a value outside the list raises."""
    return TASK.category_levels(_schema_levels)
