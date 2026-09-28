"""The ENOE source (``enoe.workers``: employed persons of pooled quarters, through mxcensus's public loaders) and
its harmonization builder (``harmonize.enoe``). Configuration: ``enoe.yaml`` beside this module; value maps:
``impute/harmonization/enoe.yaml``."""

import functools
from pathlib import Path

import pandas as pd
import yaml

from ..features import register_builder
from ..harmonize import apply_variables, load_harmonization
from . import register_source

CONFIG_PATH = Path(__file__).with_suffix(".yaml")


@functools.cache
def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def harmonized_levels(column):
    """Levels of a harmonized variable (``impute/harmonization/common.yaml``)."""
    return list(load_harmonization("common")["levels"][column])


def enoe_versions(context, config):
    from importlib.metadata import version

    return {"mxcensus": version("mxcensus"), "periods": list(config["periods"]), "state": config["state_code"]}


def _dwelling_size(period, config):
    """Persons per dwelling (all ages, all its households) on the SDEM roster of habitual and new residents."""
    import mxcensus

    keys = config["key_parts"]["dwelling"]
    sdem = mxcensus.load_enoe(table="sdem", period=period, ent=config["state_code"], labels=False)
    sdem = sdem[sdem["c_res"].isin(config["resident_codes"])]
    return sdem.groupby(keys).size().rename("dwelling_size").reset_index()


def _quarter(period, config):
    import mxcensus

    keys = config["key_parts"]
    person_keys = keys["dwelling"] + keys["household"] + keys["person"]
    persons = mxcensus.load_enoe_persons(period=period, ent=config["state_code"], canonical_filter=False, labels=False)  # raw INEGI codes
    age = pd.to_numeric(persons["eda"], errors="coerce")
    employed = ((pd.to_numeric(persons["r_def"], errors="coerce") == 0) & persons["c_res"].isin(config["resident_codes"])
                & age.between(config["min_age"], config["max_age"]) & (persons["clase2"] == "1"))
    persons = persons[employed.fillna(False).astype(bool)].copy()
    assert not persons.duplicated(person_keys).any(), f"ENOE {period}: the configured keys do not identify persons"
    persons = persons.merge(_dwelling_size(period, config), on=keys["dwelling"], how="left", validate="many_to_one")
    assert persons["dwelling_size"].notna().all(), f"ENOE {period}: an employed person without a dwelling on the SDEM roster"
    columns = config["columns"]
    output = person_keys + columns["design"] + columns["weights"] + columns["attributes"] + columns["workplace"] + columns["components"]
    persons = persons.rename(columns=config["renames"])[output].copy()
    for column in output:
        if column not in columns["weights"]:
            persons[column] = pd.to_numeric(persons[column], errors="coerce").astype("Int64")
    for column in columns["weights"]:
        persons[column] = pd.to_numeric(persons[column], errors="coerce").astype("float64")
    assert persons[columns["weights"]].notna().all().all(), f"ENOE {period}: survey weights with missing values"
    return persons.reset_index(drop=True)


@register_source("enoe.workers", config=load_config()["enoe.workers"], versions=enoe_versions, schema_levels=harmonized_levels)
def workers(context, config):
    """Employed ENOE workers of the pooled quarters, with ``period``, the weight divided by the number of quarters
    and ``hogar``, the cross-quarter household key."""
    from . import SourceFrame

    periods = list(config["periods"])
    frame = pd.concat([_quarter(period, config).assign(period=period) for period in periods], ignore_index=True)
    frame["survey_weight"] = frame["survey_weight"] / len(periods)
    frame = frame[["period"] + [column for column in frame.columns if column != "period"]]
    assert not frame.duplicated(config["keys"]).any(), "ENOE person rows must be unique within a quarter"
    keys = config["key_parts"]
    group_keys = [key for key in keys["dwelling"] + keys["household"] if key not in keys["panel_visit"]]
    frame["hogar"] = frame[group_keys].astype("string").agg("_".join, axis=1)

    return SourceFrame(frame, list(config["keys"]), config["weight"], config["group"], enoe_versions(context, config), harmonized_levels)


@register_builder("harmonize.enoe", versions=lambda context, config: {"harmonization": load_harmonization("enoe"), "common": load_harmonization("common")})
def harmonize_enoe(frame, context, config, spec):
    """The harmonized variables of ``impute/harmonization/enoe.yaml``."""
    return apply_variables(frame, load_harmonization("enoe")["variables"])
