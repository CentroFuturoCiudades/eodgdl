"""Constants of the giro imputation model, read from ``config.yaml`` and the eodgdl schemas."""

import functools
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@functools.cache
def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


_CONFIG = load_config()
GIRO_SLUGS = {str(label): str(slug) for label, slug in _CONFIG["giro_levels"].items()}  # label -> slug
GIRO_LABELS = {slug: label for label, slug in GIRO_SLUGS.items()}  # slug -> label
GIRO_CLASSES = list(GIRO_SLUGS.values())
NO_ESPECIFICADO = _CONFIG["missing_label"]
EMPLOYED_CATEGORIES = list(_CONFIG["employed_categories"])
WORK_TRIP_PURPOSE = _CONFIG["work_trip_purpose"]
MISSING_EDUCATION_LEVELS = list(_CONFIG["missing_education_levels"])
DWELLING_COLUMNS = list(_CONFIG["dwelling_columns"])
KEYS = ["folio_vivienda", "folio_habitante"]

_FEATURES = _CONFIG["features"]
DESTINATION_SHARE_FEATURES = [f"dest_share_{giro}" for giro in GIRO_CLASSES]
DESTINATION_FEATURES = list(_FEATURES["destination"]) + DESTINATION_SHARE_FEATURES
MOBILITY_FEATURES = list(_FEATURES["mobility"])
SECTOR_FEATURES = list(_FEATURES["person"]) + DESTINATION_FEATURES + MOBILITY_FEATURES
ROBUST_SECTOR_FEATURES = [f for f in SECTOR_FEATURES if f != "escolaridad"]
SHIFT_PROFILE_FEATURES = list(_FEATURES["shift_profile"])
NUMERIC_FEATURES = list(_FEATURES["numeric"]) + DESTINATION_SHARE_FEATURES

_DESTINATION = _CONFIG["destination"]
DENUE_STATE_CODE = _DESTINATION["state_code"]
DENUE_RELEASE = str(_DESTINATION["denue_release"])
DESTINATION_AMBITO_LEVELS = list(_DESTINATION["ambito_levels"])
SMALL_ESTABLISHMENT_LEVELS = tuple(_DESTINATION["small_establishment_levels"])
DENUE_SCIAN2 = {str(code): str(giro) for code, giro in _DESTINATION["denue_scian2"].items()}

# Raw survey column behind each renamed feature (for the eodgdl schema levels).
_SCHEMA_SOURCE = {"destino_trabajo": "tipo_lugar_destino", "modo_trabajo": "modo_principal"}


def _eodgdl_levels(column):
    """Category labels eodgdl guarantees for a raw OD column (its pandera schema)."""
    from eodgdl import schemas

    schema = next(candidate for candidate in (schemas.hab_schema, schemas.viv_schema, schemas.trips_schema) if column in candidate.columns)

    return [str(level) for level in schema.columns[column].dtype.type.categories]


@functools.cache
def build_category_levels():
    """Every level a categorical feature can take, plus the missing label. This is the contract the encoders are
    built with (``categories=...``, ``handle_unknown="error"``): a level absent from the training data still gets its
    own column instead of a silent all-zero block, and a value outside the list raises."""
    levels = {"destino_ambito": DESTINATION_AMBITO_LEVELS, "weekend_dest_trabajar": ["Sí", "No"]}
    for column in SECTOR_FEATURES:
        if column in NUMERIC_FEATURES or column in levels:
            continue
        levels[column] = _eodgdl_levels(_SCHEMA_SOURCE.get(column, column))

    return {feature: values + [NO_ESPECIFICADO] for feature, values in levels.items()}
