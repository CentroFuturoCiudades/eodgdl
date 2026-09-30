"""The builder ``census.home``: the census, DENUE and AMAI profile of each row's dwelling AGEB (who lives around the
worker or household). Configuration: ``census.yaml`` beside this module."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..features import register_builder
from . import file_digest, files_digest, module_config
from .eod import survey_versions


def load_config():
    return module_config(__file__)


def _versions(context, config):
    from importlib import resources
    from importlib.metadata import version

    from eodgdl.data import resolve

    package = Path(str(resources.files("eodgdl")))
    return {**survey_versions(), "mxcensus": version("mxcensus"), "zoning": files_digest([package / "reweight" / "zoning.py", package / "taz.py"]),
            "amai": file_digest(str(resolve(_amai_config(config)["file"]))), "chain": file_digest(str(package / "impute" / "chains" / f"{config['amai']['chain']}.yaml"))}


def _amai_config(config):
    from ..chain import load_chain

    chain = load_chain(config["amai"]["chain"])
    return next(step.config for step in chain.steps if getattr(step, "name", None) == config["amai"]["step"])


def census_units(tables, state):
    """``(urban, rural, rural_ageb)``: the census AGEB frame by 13-character CVEGEO, the locality frame by 9-character
    CVEGEO, and each rural locality's rural AGEB (:func:`eodgdl.reweight.zoning.assign_units` places them)."""
    import mxcensus

    from eodgdl.reweight.zoning import assign_units

    _, _, localities, agebs = mxcensus.load_census(state=state)
    urban = agebs.reset_index()
    urban.index = (urban["ENTIDAD"].astype(int).map("{:02d}".format) + urban["MUN"].astype(int).map("{:03d}".format)
                   + urban["LOC"].astype(int).map("{:04d}".format) + urban["AGEB"].astype(str))
    rural = localities.reset_index()
    rural.index = (rural["ENTIDAD"].astype(int).map("{:02d}".format) + rural["MUN"].astype(int).map("{:03d}".format)
                   + rural["LOC"].astype(int).map("{:04d}".format))
    units = assign_units(tables.viv, tables.trips, state=state)
    rural_ageb = units.loc[units["unit"] == "locality", "rural_ageb"].dropna()
    return urban, rural.loc[rural.index.intersection(rural_ageb.index)], rural_ageb


def unit_ratio(urban, rural, rural_ageb, numerator, denominator):
    """sum(numerator) / denominator per unit: an urban AGEB's own counts, a rural AGEB's summed over its localities
    whose numerator and denominator are all published (INEGI suppresses the smallest); missing where the denominator
    is zero or unpublished."""
    def ratio(frame, by=None):
        counts = frame[[*numerator, denominator]].astype(float)
        published = counts.notna().all(axis=1)
        top, bottom = counts[list(numerator)].sum(axis=1).where(published), counts[denominator].where(published)
        if by is not None:
            top, bottom = top.groupby(by).sum(min_count=1), bottom.groupby(by).sum(min_count=1)
        return top / bottom.where(bottom > 0)

    return pd.concat([ratio(urban), ratio(rural, rural_ageb.reindex(rural.index))])


def _establishments(units, config, state):
    """DENUE establishments per unit (an urban AGEB when DENUE's key is one, else the rural AGEB's 9 characters)."""
    import mxcensus

    denue = mxcensus.load_denue(state=state, release=str(config["release"]))
    key = (denue["cve_ent"].astype(str).str.zfill(2) + denue["cve_mun"].astype(str).str.zfill(3)
           + denue["cve_loc"].astype(str).str.zfill(4) + denue["ageb"].astype(str).str.zfill(4))
    unit = key.where(key.isin(units), key.str[:5] + key.str[9:])
    return unit.value_counts()


def _amai(codes, config):
    """AMAI's mean level and level shares per urban AGEB."""
    from ..derive.amai import nse_ageb_shares

    step = _amai_config(config)
    shares = nse_ageb_shares(step)
    levels = list(step["levels"]["nse_calibrado"])
    table = pd.DataFrame({config["amai"]["index"]: shares[levels].to_numpy() @ np.arange(len(levels), dtype=float)}, index=shares.index)
    for column, members in config["amai"]["shares"].items():
        table[column] = shares[list(members)].sum(axis=1)
    return table.reindex(codes)


@register_builder("census.home", config=load_config()["census.home"], versions=_versions)
def home(frame, context, config, spec):
    """The ratios of ``ratios``, the log population, DENUE establishments per adult and AMAI's NSE profile of each row's
    dwelling unit (``ageb``); missing where the unit has no census row or its counts are suppressed (AMAI: urban
    AGEBs only)."""
    urban, rural, rural_ageb = census_units(context.eod(), config["state"])
    codes = frame[config["ageb"]].astype("string").astype(object)
    columns = {column: unit_ratio(urban, rural, rural_ageb, list(r["numerator"]), r["denominator"]).clip(upper=r.get("cap")).reindex(codes).to_numpy()
               for column, r in config["ratios"].items()}
    population = pd.concat([urban["POBTOT"].astype(float), rural["POBTOT"].astype(float).groupby(rural_ageb.reindex(rural.index)).sum(min_count=1)])
    columns[config["log_population"]] = np.log1p(population.reindex(codes)).to_numpy()
    established = config["establishments"]
    per_unit = _establishments(set(urban.index), established, config["state"])
    adults = pd.concat([urban[established["denominator"]].astype(float),
                        rural[established["denominator"]].astype(float).groupby(rural_ageb.reindex(rural.index)).sum(min_count=1)])
    adults = adults.where(adults > 0).reindex(codes)
    columns[established["column"]] = (per_unit.reindex(codes).fillna(0).to_numpy() / adults).to_numpy()
    result = pd.DataFrame(columns)
    amai = _amai(codes, config)
    for column in amai.columns:
        result[column] = amai[column].to_numpy()
    return result.astype(float)
