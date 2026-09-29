"""Sources and feature builders on the EOD survey (``eodgdl.load_eod``): the sources ``eod.workers`` and
``eod.dwellings``, the builder ``eod.work_trip`` and the harmonizations ``harmonize.eod`` and
``harmonize.eod_viviendas``. Configuration: ``eod.yaml`` beside this module."""

import functools
from pathlib import Path

import pandas as pd

from ..features import register_builder
from ..harmonize import load_harmonization, register_harmonization
from . import files_digest, module_config, register_source


def load_config():
    return module_config(__file__)


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


def _survey_versions(context, config):
    return survey_versions()


@functools.cache
def schema_levels(column):
    """Category labels the eodgdl schemas guarantee for a survey column."""
    from eodgdl import schemas

    schema = next(candidate for candidate in (schemas.hab_schema, schemas.viv_schema, schemas.trips_schema) if column in candidate.columns)
    return [str(level) for level in schema.columns[column].dtype.type.categories]


def _employed(hab, categories):
    """Rows of ``hab`` whose ``trabajo_semana_pasada`` is one of ``categories``, each of which must be a level of the
    survey's schema (a label split or misspelt in the configuration fails here instead of silently matching nothing)."""
    unknown = set(categories) - set(schema_levels("trabajo_semana_pasada"))
    if unknown:
        raise ValueError(f"employed categories that are not levels of trabajo_semana_pasada: {sorted(unknown)}")
    return hab["trabajo_semana_pasada"].isin(categories)


def _as_strings(frame):
    categorical = frame.columns[frame.dtypes.eq("category")]
    frame[categorical] = frame[categorical].astype("string")
    return frame


@register_source("eod.workers", config=load_config()["eod.workers"], versions=_survey_versions, schema_levels=schema_levels)
def workers(context, config):
    """Persons who worked last week (``trabajo_semana_pasada`` in the employed categories), with the dwelling's
    columns attached and ``mes_entrevista``, the month of the interview ("1" to "4"); categoricals as plain strings."""
    tables = context.eod()
    frame = tables.hab.reset_index()
    dwelling_columns = [column for column in config["dwelling_columns"] if column not in frame.columns]
    frame = frame.merge(tables.viv[dwelling_columns], left_on="folio_vivienda", right_index=True, how="left", validate="many_to_one")
    frame = frame[_employed(frame, config["employed_categories"])].copy()
    frame["mes_entrevista"] = frame["fecha"].dt.month.astype("string")        # the dates are UTC midnights: no shift

    return _as_strings(frame).reset_index(drop=True)


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


@register_builder("eod.work_trip", config=load_config()["eod.work_trip"], versions=_survey_versions)
def work_trip(frame, context, config, spec):
    """``destino_trabajo``, ``destino_cvegeo``, ``destino_zona`` and ``modo_trabajo`` of each row's work trips (NA
    without a work trip on the survey day)."""
    keys = ["folio_vivienda", "folio_habitante"]
    destination = work_trip_destination(context.eod().trips, config["purpose"], keys)
    return frame[keys].merge(destination, on=keys, how="left", validate="one_to_one").drop(columns=keys)


# the workers' variables on the common levels (they replace the survey's own ocupacion, escolaridad, municipio,
# estado_civil, parentesco in the task frame), and the dwellings' NSE variables
register_harmonization("harmonize.eod", "eod")
register_harmonization("harmonize.eod_viviendas", "eod_viviendas")


def dwelling_levels(column):
    """Levels of a dwelling-frame column: the eodgdl schema's for a survey column, else the harmonized ones
    (common.yaml: the head's renamed or derived columns, the NSE variables)."""
    from eodgdl import schemas

    if any(column in schema.columns for schema in (schemas.hab_schema, schemas.viv_schema, schemas.trips_schema)):
        return schema_levels(column)
    return list(load_harmonization("common")["levels"][column])


@register_source("eod.dwellings", config=load_config()["eod.dwellings"], versions=_survey_versions, schema_levels=dwelling_levels)
def dwellings(context, config):
    """One row per dwelling (``viv``) with its head's columns (``head_columns``, ``weekend_*``; ``jefe_fuente`` says
    whether the head was reported or is the oldest member) and ``trabajadores_14_n``, the members aged 14+ who
    worked last week; categoricals as plain strings."""
    tables = context.eod()
    hab = tables.hab.reset_index()
    frame = tables.viv.reset_index()
    reported = hab[hab["parentesco"] == config["head_relationship"]].drop_duplicates("folio_vivienda")
    others = hab[~hab["folio_vivienda"].isin(reported["folio_vivienda"])]
    oldest = others.loc[others.groupby("folio_vivienda")["edad"].idxmax().dropna()]
    heads = pd.concat([reported.assign(jefe_fuente="observado"), oldest.assign(jefe_fuente="mayor_edad")])
    head_columns = dict(config["head_columns"])
    head_columns.update({column: column for column in hab.columns if column.startswith(tuple(config["head_prefixes"]))})
    heads = heads[["folio_vivienda", "jefe_fuente", *head_columns]].rename(columns=head_columns)
    frame = frame.merge(heads, on="folio_vivienda", how="left", validate="one_to_one")
    working = hab["edad"].ge(config["worker_min_age"]) & _employed(hab, config["employed_categories"])
    workers = working.groupby(hab["folio_vivienda"]).sum().rename("trabajadores_14_n")
    frame = frame.merge(workers, left_on="folio_vivienda", right_index=True, how="left", validate="one_to_one")
    frame["trabajadores_14_n"] = frame["trabajadores_14_n"].fillna(0).astype("Int64")

    return _as_strings(frame)
