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


def _persons(context, config):
    """Every interviewed person aged ``min_age`` or more with the dwelling's columns and ``mes_entrevista`` (as
    :func:`workers`), and ``situacion_laboral``: the employment answer on the task's levels (``labour_status``), NA
    where unanswered."""
    tables = context.eod()
    frame = tables.hab.reset_index()
    dwelling_columns = [column for column in config["dwelling_columns"] if column not in frame.columns]
    frame = frame.merge(tables.viv[dwelling_columns], left_on="folio_vivienda", right_index=True, how="left", validate="many_to_one")
    frame = frame[frame["edad"].ge(config["min_age"])].copy()
    status = {str(label): slug for slug, labels in config["labour_status"].items() for label in labels}
    unknown = set(status) - set(schema_levels("trabajo_semana_pasada"))
    if unknown:
        raise ValueError(f"labour_status labels that are not levels of trabajo_semana_pasada: {sorted(unknown)}")
    unmapped = set(frame["trabajo_semana_pasada"].dropna().astype(str)) - set(status)
    if unmapped:
        raise ValueError(f"trabajo_semana_pasada answers missing from labour_status: {sorted(unmapped)}")
    frame["situacion_laboral"] = frame["trabajo_semana_pasada"].astype("string").map(status).astype("string")
    frame["mes_entrevista"] = frame["fecha"].dt.month.astype("string")
    return frame


@register_source("eod.persons", config=load_config()["eod.persons"], versions=_survey_versions, schema_levels=schema_levels)
def persons(context, config):
    """Persons aged ``min_age``+ (:func:`_persons`): the employment task's training rows."""
    return _as_strings(_persons(context, config)).reset_index(drop=True)


@register_source("eod.labour", config=load_config()["eod.labour"], versions=_survey_versions, schema_levels=schema_levels)
def labour(context, config):
    """Persons aged ``min_age``+ who worked last week or did not answer (:func:`_persons`): the rows the chain
    ``labour`` completes, a worker's giro and informality read only where the person works."""
    frame = _persons(context, config)
    frame = frame[frame["situacion_laboral"].isna() | frame["situacion_laboral"].eq(config["worker_status"])]
    return _as_strings(frame).reset_index(drop=True)


@register_builder("eod.person_context", config=load_config()["eod.person_context"], versions=_survey_versions)
def person_context(frame, context, config, spec):
    """The person's survey day and household: ``n_viajes`` (trips), ``viaje_trabajo`` / ``viaje_estudio`` ("Sí" if a
    trip had that motive), and the other members aged ``min_age``+ who answered: ``hogar_otros_trabajan``,
    ``hogar_otros_estudian``, ``hogar_otros_hogar``; and ``hogar_menores`` (members under ``child_age``)."""
    tables = context.eod()
    keys = ["folio_vivienda", "folio_habitante"]
    motive = tables.trips["motivo_viaje"].astype("string")
    by_person = pd.DataFrame({"n_viajes": motive.groupby(level=keys).size(),
                              "viaje_trabajo": motive.eq(config["work_motive"]).groupby(level=keys).any(),
                              "viaje_estudio": motive.eq(config["school_motive"]).groupby(level=keys).any()}).reset_index()
    rows = frame[keys].merge(by_person, on=keys, how="left", validate="one_to_one")
    hab = tables.hab.reset_index()
    answer = hab["trabajo_semana_pasada"].astype("string")
    adults = hab["edad"].ge(config["min_age"])
    roles = pd.DataFrame({"folio_vivienda": hab["folio_vivienda"], "folio_habitante": hab["folio_habitante"],
                          **{column: (adults & answer.isin(labels)).astype(int) for column, labels in config["household_roles"].items()},
                          "hogar_menores": hab["edad"].lt(config["child_age"]).astype(int)})
    totals = roles.groupby("folio_vivienda").sum(numeric_only=True).drop(columns="folio_habitante")
    own = rows[keys].merge(roles, on=keys, how="left", validate="one_to_one")
    household = totals.reindex(rows["folio_vivienda"]).reset_index(drop=True)
    columns = pd.DataFrame({
        "n_viajes": rows["n_viajes"].fillna(0).astype(float),
        "viaje_trabajo": rows["viaje_trabajo"].fillna(False).map({True: "Sí", False: "No"}),
        "viaje_estudio": rows["viaje_estudio"].fillna(False).map({True: "Sí", False: "No"}),
    })
    for column in [*config["household_roles"], "hogar_menores"]:
        columns[column] = (household[column] - own[column].fillna(0)).astype(float).to_numpy()   # the others
    return columns


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
    """``destino_trabajo``, ``destino_cvegeo``, ``destino_zona`` and ``modo_trabajo`` of each row's work trips. Without a
    work trip on the survey day they are NA, but for ``destino_trabajo`` and ``modo_trabajo`` when the configuration
    names a ``no_trip`` level: not having made one is then a state the model reads, not a missing value."""
    keys = ["folio_vivienda", "folio_habitante"]
    destination = work_trip_destination(context.eod().trips, config["purpose"], keys)
    columns = frame[keys].merge(destination, on=keys, how="left", validate="one_to_one").drop(columns=keys)
    if config.get("no_trip"):
        for column in ("destino_trabajo", "modo_trabajo"):
            columns[column] = columns[column].astype("string").fillna(config["no_trip"])
    return columns


@register_builder("eod.household_education", config=load_config()["eod.household_education"], versions=_survey_versions)
def household_education(frame, context, config, spec):
    """``hogar_escolaridad_max``: the highest education (``years``: the level's years of schooling) among the other
    members of the row's dwelling aged ``min_age`` or more whose education is known; NA without one. The row's own
    person is ``folio_habitante`` (a person row) or ``jefe_folio_habitante`` (a dwelling row: the head)."""
    hab = context.eod().hab.reset_index()
    years = hab["escolaridad"].astype("string").map({str(level): float(value) for level, value in config["years"].items()})
    members = hab.loc[hab["edad"].ge(config["min_age"]) & years.notna(), ["folio_vivienda", "folio_habitante"]].assign(years=years)
    person = "folio_habitante" if "folio_habitante" in frame.columns else "jefe_folio_habitante"
    rows = frame[["folio_vivienda", person]].reset_index(drop=True).rename(columns={person: "self"}).reset_index(names="row")
    pairs = rows.merge(members, on="folio_vivienda")
    others = pairs[pairs["folio_habitante"].astype("string") != pairs["self"].astype("string")]
    best = others.groupby("row")["years"].max()
    return pd.DataFrame({"hogar_escolaridad_max": best.reindex(rows["row"]).to_numpy(dtype=float)})


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
    whether the head was reported or is the oldest member; ``jefe_folio_habitante`` is the head's person number) and
    ``trabajadores_14_n``, the members aged 14+ who worked last week; categoricals as plain strings."""
    tables = context.eod()
    hab = tables.hab.reset_index()
    frame = tables.viv.reset_index()
    reported = hab[hab["parentesco"] == config["head_relationship"]].drop_duplicates("folio_vivienda")
    others = hab[~hab["folio_vivienda"].isin(reported["folio_vivienda"])]
    oldest = others.loc[others.groupby("folio_vivienda")["edad"].idxmax().dropna()]
    heads = pd.concat([reported.assign(jefe_fuente="observado"), oldest.assign(jefe_fuente="mayor_edad")])
    head_columns = dict(config["head_columns"])
    head_columns.update({column: column for column in hab.columns if column.startswith(tuple(config["head_prefixes"]))})
    heads = heads[["folio_vivienda", "jefe_fuente", "folio_habitante", *head_columns]].rename(columns={**head_columns, "folio_habitante": "jefe_folio_habitante"})
    frame = frame.merge(heads, on="folio_vivienda", how="left", validate="one_to_one")
    working = hab["edad"].ge(config["worker_min_age"]) & _employed(hab, config["employed_categories"])
    workers = working.groupby(hab["folio_vivienda"]).sum().rename("trabajadores_14_n")
    frame = frame.merge(workers, left_on="folio_vivienda", right_index=True, how="left", validate="one_to_one")
    frame["trabajadores_14_n"] = frame["trabajadores_14_n"].fillna(0).astype("Int64")

    return _as_strings(frame)
