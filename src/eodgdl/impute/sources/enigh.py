"""The ENIGH source (``enigh.households``, through mxcensus's public loaders) and its harmonization builder
(``harmonize.enigh``). Configuration: ``enigh.yaml`` beside this module; value maps:
``impute/harmonization/enigh.yaml``."""

import pandas as pd

from ..harmonize import register_harmonization
from . import module_config, register_source
from .enoe import harmonized_levels


def load_config():
    return module_config(__file__)


def enigh_versions(context, config):
    from importlib.metadata import version

    return {"mxcensus": version("mxcensus"), "period": str(config["period"])}


@register_source("enigh.households", config=load_config()["enigh.households"], versions=enigh_versions, schema_levels=harmonized_levels)
def households(context, config):
    """ENIGH households with their dwelling and questionnaire columns, ``trabajadores_14_n``, ``entidad`` (from
    ``ubica_geo``) and the weight ``factor`` as a number."""
    import mxcensus

    period, keys = str(config["period"]), list(config["keys"])
    frame = mxcensus.load_enigh_hogares(period=period, labels=False).reset_index()[keys + list(config["household_columns"])]
    questionnaire = mxcensus.load_enigh(table="hogares", period=period, labels=False).reset_index()[keys + list(config["questionnaire_columns"])]
    dwellings = mxcensus.load_enigh_viviendas(period=period, labels=False).reset_index()[["folioviv"] + list(config["dwelling_columns"])]
    persons = mxcensus.load_enigh_personas(period=period, labels=False).reset_index()
    worked = (pd.to_numeric(persons["edad"], errors="coerce") >= config["worker_min_age"]) & persons["trabajo_mp"].astype(str).str.strip().eq(config["worked_code"])
    workers = worked.groupby([persons["folioviv"], persons["foliohog"]]).sum().rename("trabajadores_14_n").reset_index()
    frame = frame.merge(questionnaire, on=keys, how="left", validate="one_to_one")
    frame = frame.merge(dwellings, on="folioviv", how="left", validate="many_to_one")
    frame = frame.merge(workers, on=keys, how="left", validate="one_to_one")
    frame["trabajadores_14_n"] = frame["trabajadores_14_n"].fillna(0).astype("Int64")
    frame["factor"] = pd.to_numeric(frame["factor"], errors="raise").astype("float64")
    frame["entidad"] = frame["ubica_geo"].astype(str).str[:2]
    assert not frame.duplicated(keys).any(), "ENIGH households must be unique"

    return frame


# ENIGH's codes on the EOD's labels
register_harmonization("harmonize.enigh", "enigh")
