"""Sources and feature builders on the EOD survey (``eodgdl.load_eod``): ``eod.workers`` and ``eod.work_trip``.
Configuration: ``eod.yaml`` beside this module."""

import functools
from pathlib import Path

import pandas as pd
import yaml

from ..features import register_builder
from . import files_digest, register_source

CONFIG_PATH = Path(__file__).with_suffix(".yaml")


@functools.cache
def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def survey_versions():
    """What ``load_eod()``'s tables depend on: the eodgdl version, the loader's code and bundled config (chain rules,
    hand decisions, schemas, rename map) and the three survey files' sha256."""
    from importlib import resources

    import eodgdl
    from eodgdl.data import SURVEY_FILES, resolve

    package = Path(str(resources.files("eodgdl")))
    loader = [package / name for name in ("eod.py", "chains.py", "review.py", "schemas.py", "_resources.py", "imeplan_rename_map.json")]
    loader += sorted((package / "revisions").glob("*.csv.gz"))
    return {
        "eodgdl": eodgdl.__version__,
        "eod_loader": files_digest(loader),
        "survey": files_digest([resolve(name) for name in SURVEY_FILES]),
    }


@functools.cache
def schema_levels(column):
    """Category labels the eodgdl schemas guarantee for a survey column."""
    from eodgdl import schemas

    schema = next(candidate for candidate in (schemas.hab_schema, schemas.viv_schema, schemas.trips_schema) if column in candidate.columns)
    return [str(level) for level in schema.columns[column].dtype.type.categories]


@register_source("eod.workers", config=load_config()["eod.workers"], versions=lambda context, config: survey_versions(), schema_levels=schema_levels)
def workers(context, config):
    """Persons who worked last week (``trabajo_semana_pasada`` in the employed categories), with the dwelling's
    columns attached; categoricals as plain strings."""
    from . import SourceFrame

    tables = context.eod()
    frame = tables.hab.reset_index()
    dwelling_columns = [column for column in config["dwelling_columns"] if column not in frame.columns]
    frame = frame.merge(tables.viv[dwelling_columns], left_on="folio_vivienda", right_index=True, how="left", validate="many_to_one")
    frame = frame[frame["trabajo_semana_pasada"].isin(config["employed_categories"])].copy()
    categorical = frame.columns[frame.dtypes.eq("category")]
    frame[categorical] = frame[categorical].astype("string")

    return SourceFrame(frame.reset_index(drop=True), list(config["keys"]), config["weight"], config["group"], survey_versions(), schema_levels)


def work_trip_destination(trips, purpose, keys=("folio_vivienda", "folio_habitante")):
    """Most frequent destination type, destination code / zone and main mode of each person's trips with motive
    ``purpose``; persons without such a trip are absent."""
    keys = list(keys)
    work_trips = trips.reset_index()
    work_trips = work_trips[work_trips["motivo_viaje"] == purpose]
    mode = lambda values: values.astype(str).value_counts().index[0]
    destination = work_trips.groupby(keys).agg(
        destino_trabajo=("tipo_lugar_destino", mode),
        destino_cvegeo=("destino", mode),
        destino_zona=("zona_destino", mode),
        modo_trabajo=("modo_principal", mode),
    )

    return destination.reset_index()


@register_builder("eod.work_trip", config=load_config()["eod.work_trip"], versions=lambda context, config: survey_versions())
def work_trip(frame, context, config, spec):
    """``destino_trabajo``, ``destino_cvegeo``, ``destino_zona`` and ``modo_trabajo`` of each row's work trips (NA
    without a work trip on the survey day)."""
    keys = ["folio_vivienda", "folio_habitante"]
    destination = work_trip_destination(context.eod().trips, config["purpose"], keys)
    return frame[keys].merge(destination, on=keys, how="left", validate="one_to_one").drop(columns=keys)


def _harmonization_versions(context, config):
    from ..harmonize import load_harmonization

    return {**survey_versions(), "harmonization": load_harmonization("eod"), "common": load_harmonization("common")}


@register_builder("harmonize.eod", versions=_harmonization_versions, replaces=True)
def harmonize_eod(frame, context, config, spec):
    """The harmonized variables of ``impute/harmonization/eod.yaml``; they replace the survey's own columns of the same
    name (``ocupacion``, ``escolaridad``, ``municipio``, ``estado_civil``, ``parentesco``) in the task frame."""
    from ..harmonize import apply_variables, load_harmonization

    return apply_variables(frame, load_harmonization("eod")["variables"])
