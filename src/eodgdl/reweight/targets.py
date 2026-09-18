"""The 2020 census as the tool's constraint targets, by zone, municipality or region.

The census values come from INEGI through ``mxcensus`` (``load_census(state=14)``: the
AGEB and locality frames of the Censo de Población y Vivienda 2020). IMEPLAN's AGEB table
is used only as the **crosswalk**: which AGEBs and rural localities make up each survey
zone, and hence the survey universe. The two agree wherever both hold a value
(``reconcile()`` checks it); ``mxcensus`` keeps suppressed cells blank where IMEPLAN's
table wrote 0.

Every target is read from spec.yaml (``constraints``): an expression over census columns,
summed by geography first, or a constant with its provenance. A 2023 set multiplies each
AGEB's columns by the CONAPO 2023/2020 ratio of its municipality for the band group the
target names (``conapo``), before summing.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from eodgdl.data._catalog import AGEBS_ZONA_PARQUET, ZONIFICACION_PARQUET
from eodgdl.reweight._spec import (
    GEOGRAPHY_COLUMN,
    TABLES,
    census_columns,
    constraint_file,
    constraints,
    is_scaled,
    load_spec,
)
from eodgdl.reweight.records import zone_index
from eodgdl.taz import load_imeplan_agebs, load_taz, load_zm_muns

SEX = {"HOMBRES": "M", "MUJERES": "F"}
AGEB_KEY = ["ENTIDAD", "MUN", "LOC", "AGEB"]
LOC_KEY = ["ENTIDAD", "MUN", "LOC"]
GEOGRAPHY_IDS = ["Zone", "Municipality", "Region"]


def _local(data_dir, filename):
    return Path(data_dir) / filename if data_dir is not None else None


@functools.cache
def _census(state=14):
    """mxcensus.load_census for one state, loaded once per process."""
    try:
        import mxcensus
    except ImportError:
        raise ImportError(
            "eodgdl.reweight builds its targets from the census through mxcensus; "
            "install it with `uv sync --extra reweight`"
        ) from None
    return mxcensus.load_census(state=state)


# ---------------------------------------------------------------- the crosswalk


def crosswalk(data_dir=None):
    """IMEPLAN's AGEB table reduced to the survey universe and the census key of each row.

    One row per AGEB or rural locality with a 2023 survey zone (``EOD2023 != 'NA'``), with
    the INEGI key parsed from ``CVEGEO`` (13 characters for an AGEB, 9 for a locality;
    ``AGEB`` is NA on a locality row), the Zone / Municipality / Region ids the records use,
    and IMEPLAN's own ``POBTOT`` for reconciliation. Rows without a usable key (the access
    points, one schematic rural AGEB) carry no population and are dropped.
    """
    zones = load_spec()["zones"]
    taz = load_taz(_local(data_dir, ZONIFICACION_PARQUET))
    agebs = load_imeplan_agebs(taz, _local(data_dir, AGEBS_ZONA_PARQUET))
    zcol = zones["census_zone_column"]
    u = agebs.loc[(agebs[zcol].astype(str) != "NA") & agebs.CVEGEO.notna()]
    code = u.CVEGEO.astype(str)
    cw = pd.DataFrame({
        "ENTIDAD": pd.to_numeric(code.str[:2], errors="coerce"),
        "MUN": pd.to_numeric(code.str[2:5], errors="coerce"),
        "LOC": pd.to_numeric(code.str[5:9], errors="coerce"),
        "AGEB": code.str[9:13].where(code.str.len() == 13),
        "CVEGEO": code,
        "POBTOT_IMEPLAN": u.POBTOT.astype(float),
    }, index=u.index)
    unkeyed = cw[LOC_KEY].isna().any(axis=1)
    if cw.loc[unkeyed, "POBTOT_IMEPLAN"].fillna(0).sum() > 0:
        raise ValueError(f"rows with population but no census key: {cw.index[unkeyed].tolist()}")
    cw = cw.loc[~unkeyed].astype({"ENTIDAD": int, "MUN": int, "LOC": int})
    unknown = set(cw.MUN) - set(load_zm_muns())
    if unknown:
        raise ValueError(f"census municipalities outside the study area: {sorted(unknown)}")
    zone = u.loc[cw.index, zcol].astype(str).map(zone_index())
    if zone.isna().any():
        raise ValueError(f"survey zones outside the schema's levels: {sorted(u.loc[zone.isna(), zcol].unique())}")
    cw["Zone"] = zone.astype(int)
    cw["Municipality"] = cw.MUN
    cw["Region"] = 1
    return cw


def census_universe(data_dir=None, state=14):
    """Every numeric census column of every AGEB and locality in the survey universe.

    Indexed like the crosswalk (IMEPLAN's row id), with the Zone / Municipality / Region
    ids appended. An AGEB row joins the census AGEB frame on (ENTIDAD, MUN, LOC, AGEB), a
    locality row the locality frame on (ENTIDAD, MUN, LOC); a row the census does not have
    must carry no population in IMEPLAN's table, else this raises. Suppressed cells stay
    NaN, so sums skip them.
    """
    cw = crosswalk(data_dir)
    _, _, df_loc, df_ageb = _census(state)
    is_ageb = cw.AGEB.notna()
    from_agebs = df_ageb.reindex(pd.MultiIndex.from_frame(cw.loc[is_ageb, AGEB_KEY]))
    from_locs = df_loc.reindex(pd.MultiIndex.from_frame(cw.loc[~is_ageb, LOC_KEY]))
    from_agebs.index, from_locs.index = cw.index[is_ageb], cw.index[~is_ageb]
    values = pd.concat([from_agebs, from_locs]).reindex(cw.index).select_dtypes("number")
    unmatched = values.isna().all(axis=1)
    populated = cw.POBTOT_IMEPLAN.fillna(0) > 0
    if (unmatched & populated).any():
        raise ValueError(f"populated rows the census does not have: {cw.loc[unmatched & populated, 'CVEGEO'].tolist()}")
    out = values.loc[~unmatched].copy()
    for col in GEOGRAPHY_IDS:
        out[col] = cw.loc[out.index, col]
    return out


def reconcile(universe, cw=None, data_dir=None):
    """Zones whose census population differs from IMEPLAN's table; empty when they agree."""
    cw = crosswalk(data_dir) if cw is None else cw
    ours = universe.groupby("Zone").POBTOT.sum()
    theirs = cw.groupby("Zone").POBTOT_IMEPLAN.sum()
    diff = (ours - theirs.reindex(ours.index)).fillna(0)
    return [f"zone {z}: census {ours[z]:,.0f} vs IMEPLAN {theirs[z]:,.0f}" for z in diff.index[diff.abs() > 0.5]]


def coverage(universe, state=14, columns=("POBTOT", "TVIVPARHAB", "POCUPADA")):
    """The survey universe as a share of each whole municipality, per census column.

    Indexed by Municipality: ``<col>_universe``, ``<col>_municipality``, ``<col>_share``.
    Peripheral municipalities are only partly inside the study area (Juanacatlán a third),
    so a whole-municipality figure must be scaled by the share before it can be a target.
    """
    _, df_mun, _, _ = _census(state)
    ids = sorted(universe.Municipality.unique())
    whole = df_mun.loc[[(state, m) for m in ids]]
    whole.index = ids
    inside = universe.groupby("Municipality")[list(columns)].sum().reindex(ids)
    out = pd.DataFrame(index=pd.Index(ids, name="Municipality"))
    for col in columns:
        out[f"{col}_universe"] = inside[col]
        out[f"{col}_municipality"] = whole[col].astype(float)
        out[f"{col}_share"] = inside[col] / whole[col]
    return out


# ---------------------------------------------------------------- CONAPO


class Conapo(NamedTuple):
    """CONAPO population by (Municipality, sex) and band group, in the base and target years."""

    base: pd.DataFrame
    target: pd.DataFrame

    def ratio(self, band, sex="T", municipality=None):
        """target / base for one band group; pooled over municipalities when none is given."""
        if municipality is None:
            b = self.base.xs(sex, level="sex")[band].sum()
            t = self.target.xs(sex, level="sex")[band].sum()
            return float(t / b)
        return float(self.target.loc[(municipality, sex), band] / self.base.loc[(municipality, sex), band])


def load_conapo(path=None):
    """CONAPO projections for the study area's municipalities as a :class:`Conapo`."""
    spec = load_spec()["conapo"]
    if path is None:
        from eodgdl.data import resolve

        path = resolve(spec["file"])
    c = pd.read_csv(path)
    c = c[c.ANO.isin([spec["base_year"], spec["target_year"]])].copy()
    c["Municipality"] = c.CLAVE - 14000
    c["sex"] = c.SEXO.map(SEX)
    bands = {band: c[cols].sum(axis=1) for band, cols in spec["bands"].items()}
    c = c.assign(**bands)
    both = c.assign(sex="T")
    frame = pd.concat([c, both]).groupby(["Municipality", "sex", "ANO"])[list(bands)].sum()
    return Conapo(
        base=frame.xs(spec["base_year"], level="ANO"),
        target=frame.xs(spec["target_year"], level="ANO"),
    )


# ---------------------------------------------------------------- the targets


def _row_ratio(universe, entry, conapo):
    """The scaling factor of every census row for one target: its municipality's CONAPO ratio."""
    sex = entry.get("sex", "T")
    band = entry["conapo"]
    per_mun = {m: conapo.ratio(band, sex, m) for m in universe.Municipality.unique()}
    return universe.Municipality.map(per_mun)


def _target(universe, entry, geography, conapo, shares):
    """The target of one constraint per geography id (a Series indexed by the id)."""
    column = GEOGRAPHY_COLUMN[geography]
    ids = sorted(universe[column].unique())
    scale = conapo is not None and "conapo" in entry
    if "census" in entry:
        expr = entry["census"]
        cols = census_columns(expr)
        missing = [c for c in cols if c not in universe.columns]
        if missing:
            raise KeyError(f"census columns not in the census frames: {missing}")
        sub = universe[cols]
        if scale:
            sub = sub.mul(_row_ratio(universe, entry, conapo), axis=0)
        agg = sub.groupby(universe[column]).sum(min_count=1)
        return agg.eval(expr).reindex(ids)
    values = pd.Series(entry["constant"], dtype=float)
    if set(values.index) != set(ids):
        raise ValueError(f"constant geography ids {sorted(values.index)} != census {ids}")
    if "universe_share" in entry:
        if geography != "municipality":
            raise ValueError("universe_share applies to municipality constants only")
        values = values * shares[f"{entry['universe_share']}_share"].reindex(values.index)
    if scale:
        sex = entry.get("sex", "T")
        if geography == "region":
            values = values * conapo.ratio(entry["conapo"], sex)
        else:
            values = values * pd.Series({m: conapo.ratio(entry["conapo"], sex, m) for m in ids})
    return values.reindex(ids)


def build_constraints(universe, year=None, conapo=None, shares=None):
    """{file name: frame} for one constraint set; `year` None or the base year = the census as is.

    `shares` is :func:`coverage`'s frame, needed by a constant with `universe_share`.
    """
    base_year = load_spec()["conapo"]["base_year"]
    if year is not None and year != base_year and conapo is None:
        raise ValueError(f"a {year} set needs the CONAPO projections")
    scaling = conapo if (year is not None and year != base_year) else None
    if shares is None and any(
        "universe_share" in e for t in TABLES for e in constraints(t).values()
    ):
        shares = coverage(universe)
    files = {}
    for table in TABLES:
        for target, entry in constraints(table).items():
            geography = entry["geography"]
            suffix = year if (year is not None and is_scaled(table, geography)) else None
            name = constraint_file(table, geography, suffix)
            series = _target(universe, entry, geography, scaling, shares)
            if name not in files:
                files[name] = pd.DataFrame({GEOGRAPHY_COLUMN[geography]: series.index.astype(int)})
            files[name][target] = series.to_numpy()
    return files
