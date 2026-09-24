"""The 2020 census as the tool's constraint targets, by zone, municipality or region.

The census values come from INEGI through ``mxcensus`` (``load_census(state=14)``: the
AGEB and locality frames of the Censo de Población y Vivienda 2020). The **crosswalk**,
which urban AGEBs and rural localities make up each survey zone and hence the survey
universe, is :mod:`eodgdl.reweight.zoning`'s: the centralidad polygons and the survey's own
coding, never IMEPLAN's AGEB table.

Every target is read from spec.yaml (``constraints``): an expression over census columns,
summed by geography first, or a constant with its provenance. A 2023 set multiplies each
AGEB's columns by the CONAPO 2023/2020 ratio of its municipality for the band group the
target names (``conapo``), before summing -- or, for a dwelling attribute that names a
registered-vehicle fleet (``vmrc``), by that fleet's 2023/2020 ratio in the municipality.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import NamedTuple

import pandas as pd

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
from eodgdl.reweight.zoning import assign_units, check_assignment
from eodgdl.taz import load_zm_muns

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


def crosswalk(viv, state=14):
    """The census units of the survey universe with their census key and geography ids.

    One row per urban AGEB or rural locality that :func:`zoning.assign_units` places in a
    zone, indexed by CVEGEO: the INEGI key (``AGEB`` is NA on a locality row), the rule
    that placed it, and the Zone / Municipality / Region ids the records use.
    """
    a = assign_units(viv, state)
    problems = check_assignment(a, viv)
    if problems:
        raise ValueError("the zone assignment disagrees with the survey: " + "; ".join(problems))
    a = a[a.zone.notna()]
    code = a.index.to_series()
    cw = pd.DataFrame({
        "ENTIDAD": code.str[:2].astype(int),
        "MUN": code.str[2:5].astype(int),
        "LOC": code.str[5:9].astype(int),
        "AGEB": code.str[9:13].where(a.unit == "ageb"),
        "unit": a.unit, "rule": a.rule.astype(str),
    }, index=a.index)
    unknown = set(cw.MUN) - set(load_zm_muns())
    if unknown:
        raise ValueError(f"census municipalities outside the study area: {sorted(unknown)}")
    zone = a.zone.map(zone_index())
    if zone.isna().any():
        raise ValueError(f"zones outside the schema's levels: {sorted(a.zone[zone.isna()].unique())}")
    cw["Zone"] = zone.astype(int)
    cw["Municipality"] = cw.MUN
    cw["Region"] = 1
    return cw


def census_universe(viv, state=14):
    """Every numeric census column of every AGEB and rural locality in the survey universe.

    Indexed like the crosswalk (CVEGEO), with the Zone / Municipality / Region ids
    appended. An AGEB row joins the census AGEB frame on (ENTIDAD, MUN, LOC, AGEB), a
    locality row the locality frame on (ENTIDAD, MUN, LOC); the crosswalk is built from
    those frames, so every row matches. Suppressed cells stay NaN, so sums skip them.
    """
    cw = crosswalk(viv, state)
    _, _, df_loc, df_ageb = _census(state)
    is_ageb = cw.AGEB.notna()
    from_agebs = df_ageb.reindex(pd.MultiIndex.from_frame(cw.loc[is_ageb, AGEB_KEY]))
    from_locs = df_loc.reindex(pd.MultiIndex.from_frame(cw.loc[~is_ageb, LOC_KEY]))
    from_agebs.index, from_locs.index = cw.index[is_ageb], cw.index[~is_ageb]
    out = pd.concat([from_agebs, from_locs]).reindex(cw.index).select_dtypes("number")
    unmatched = out.isna().all(axis=1)
    if unmatched.any():
        raise ValueError(f"crosswalk rows the census does not have: {cw.index[unmatched].tolist()}")
    for col in GEOGRAPHY_IDS:
        out[col] = cw[col]
    return out


def coverage(universe, state=14, columns=("POBTOT", "TVIVPARHAB", "POCUPADA")):
    """The survey universe as a share of each whole municipality, per census column.

    Indexed by Municipality: ``<col>_universe``, ``<col>_municipality``, ``<col>_share``.
    Peripheral municipalities are only partly inside the study area (Zapotlanejo three quarters),
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


class Vmrc(NamedTuple):
    """INEGI's registered private fleet by municipality and year (the ``vmrc`` spec block)."""

    frame: pd.DataFrame  # index (Municipality, ANIO); one column per fleet

    def ratio(self, column, municipality, base, target):
        """target-year / base-year fleet in one municipality."""
        f = self.frame[column]
        return float(f.loc[(municipality, target)] / f.loc[(municipality, base)])


def load_vmrc(path=None):
    """The registered fleets of the study area's municipalities as a :class:`Vmrc`."""
    spec = load_spec()["vmrc"]
    if path is None:
        from eodgdl.data import resolve

        path = resolve(spec["file"])
    v = pd.read_csv(path)
    return Vmrc(v.set_index(["Municipality", "ANIO"])[spec["columns"]].astype(float))


def load_rates(data_dir=None):
    """{rate name: target/base ratio} for every entry of the ``rates`` spec block."""
    spec = load_spec()
    base, target = spec["conapo"]["base_year"], spec["conapo"]["target_year"]
    out = {}
    for name, entry in (spec.get("rates") or {}).items():
        path = _local(data_dir, entry["file"])
        if path is None:
            from eodgdl.data import resolve

            path = resolve(entry["file"])
        r = pd.read_csv(path)
        r = r[r.domain == entry["domain"]].set_index("year").share
        out[name] = float(r.loc[target] / r.loc[base])
    return out


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


STATED_AGES = ["P_0A2", "P_3A5", "P_6A11", "P_12A14", "P_15A17", "P_18YMAS"]


def person_scale(universe):
    """Per census row: ``(POBTOT - POBCOL) / residents with a stated age``.

    Two corrections in one factor, applied to every person count of the row before
    summing (``row_scale: persons`` in ``spec.yaml``):

    * **collective quarters** -- the census person counts include residents of prisons,
      barracks, care homes and dormitories (``POBCOL``), whom a household survey never
      samples;
    * **residents without a stated age** -- INEGI counts the occupants of dwellings without
      occupant information in ``POBTOT`` and usually imputes their ages, but not in every
      AGEB: in zone 56 43 % of residents have no age, and so no band, sex or activity.

    ``counts * (POBTOT - POBCOL) / POBTOT`` restricts a row to private dwellings, and
    ``* POBTOT / stated`` spreads the unknown ages over the known bands in proportion; the
    product is this factor. The census publishes neither group's composition, so both are
    assumed to share the row's known mix. A suppressed ``POBCOL`` counts as 0; a row with
    no stated age keeps its (then missing) counts.
    """
    u = universe.astype({c: float for c in [*STATED_AGES, "POBTOT", "POBCOL"]})
    stated = u[STATED_AGES].sum(axis=1, min_count=1)
    private = u["POBTOT"] - u["POBCOL"].fillna(0.0)
    factor = private / stated.where(stated > 0)
    return factor.astype(float).fillna(1.0).clip(lower=0.0)


def dwelling_scale(universe):
    """Per census row: ``TVIVPARHAB / dwellings whose characteristics were captured``.

    The ``VPH_*`` counts are over ``VIVPARH_CV``, which INEGI defines as including the
    dwellings without occupant information -- whose characteristics it usually imputes, but
    not in every AGEB. Where it did not, those dwellings sit in ``VIVPARH_CV`` without any
    ``VPH_*`` attribute. They are estimated from the same signal as the missing ages: the
    row's residents without a stated age over its occupants per dwelling, capped at its
    dwellings without occupant information (``VIVPARH_CV - VIVPAR_HAB``). Dividing by what
    remains (``row_scale: dwellings``) spreads the uncaptured dwellings over the captured
    mix and lifts the count to all private inhabited dwellings; results are capped at the
    row's ``TVIVPARHAB``.
    """
    u = universe.astype({c: float for c in [*STATED_AGES, "POBTOT", "OCUPVIVPAR", "TVIVPARHAB",
                                            "VIVPARH_CV", "VIVPAR_HAB"]})
    stated = u[STATED_AGES].sum(axis=1, min_count=1)
    no_age = (u["POBTOT"] - stated).clip(lower=0.0)
    per_dwelling = u["OCUPVIVPAR"] / u["TVIVPARHAB"].where(u["TVIVPARHAB"] > 0)
    no_info = (u["VIVPARH_CV"] - u["VIVPAR_HAB"]).clip(lower=0.0).fillna(0.0)
    missing = (no_age / per_dwelling.where(per_dwelling > 0)).fillna(0.0).clip(upper=no_info)
    captured = u["VIVPARH_CV"] - missing
    factor = u["TVIVPARHAB"] / captured.where(captured > 0)
    return factor.fillna(1.0).clip(lower=1.0)


ROW_SCALES = {"persons": person_scale, "dwellings": dwelling_scale}


def _vmrc_ratio(universe, entry, vmrc):
    """The scaling factor of every census row for a `vmrc` target: its municipality's fleet ratio."""
    spec = load_spec()["conapo"]
    per_mun = {m: vmrc.ratio(entry["vmrc"], m, spec["base_year"], spec["target_year"])
               for m in universe.Municipality.unique()}
    return universe.Municipality.map(per_mun)


def _census_rows(universe, entry, conapo=None, vmrc=None, rates=None):
    """One census target's columns per census row, corrected and (for a scaled set) grown.

    The row correction (`row_scale`) and the 2023 growth (`conapo`, `vmrc`, `rate`) act on
    each census row before anything is summed; `conapo` None means the census as is.
    """
    if conapo is not None and "vmrc" in entry and vmrc is None:
        raise ValueError(f"a scaled set needs the VMRC fleets for {entry['vmrc']}")
    cols = census_columns(entry["census"])
    missing = [c for c in cols if c not in universe.columns]
    if missing:
        raise KeyError(f"census columns not in the census frames: {missing}")
    sub = universe[cols]
    if "row_scale" in entry:
        sub = sub.mul(ROW_SCALES[entry["row_scale"]](universe), axis=0)
        if entry["row_scale"] == "dwellings":
            sub = sub.clip(upper=universe["TVIVPARHAB"].astype(float), axis=0)
    if conapo is not None and "conapo" in entry:
        growth = _row_ratio(universe, entry, conapo)
        sub = sub.mul(growth, axis=0)
        if "rate" in entry:
            if rates is None or entry["rate"] not in rates:
                raise ValueError(f"a scaled set needs the rate {entry['rate']}")
            cap = universe["TVIVPARHAB"].astype(float) * growth
            sub = sub.mul(rates[entry["rate"]]).clip(upper=cap, axis=0)
    elif conapo is not None and "vmrc" in entry:
        sub = sub.mul(_vmrc_ratio(universe, entry, vmrc), axis=0)
    return sub


def row_targets(universe, entry, conapo=None, vmrc=None, rates=None):
    """One census target's value per census row: what the zone target sums.

    The expression is evaluated on each row with suppressed cells as 0, which is how the
    zone sum skips them, so these add up to the zone target wherever a zone has any value.
    """
    return _census_rows(universe, entry, conapo, vmrc, rates).fillna(0.0).eval(entry["census"])


def _target(universe, entry, geography, conapo, shares, vmrc=None, rates=None):
    """The target of one constraint per geography id (a Series indexed by the id)."""
    column = GEOGRAPHY_COLUMN[geography]
    ids = sorted(universe[column].unique())
    scale = conapo is not None and "conapo" in entry
    if "census" in entry:
        sub = _census_rows(universe, entry, conapo, vmrc, rates)
        agg = sub.groupby(universe[column]).sum(min_count=1)
        return agg.eval(entry["census"]).reindex(ids)
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


SAMPLED_TARGETS = {"Persons": "persons", "Dwellings": "households"}


def sampled_agebs(universe, viv, assignment, conapo, vmrc=None, rates=None):
    """The census targets of every AGEB the survey sampled, per year, as the tool sees them.

    One row per AGEB code in ``viv.ageb``: its municipality, the centralidad the survey coded
    there, the TAZ (zone x municipality), its sampled dwellings, and ``Persons`` and
    ``Dwellings`` for the base and target years -- the same census expression, row
    correction and growth as the zone targets, so an AGEB's values are its share of its
    zone's. A sampled urban AGEB is its own census row. A sampled rural AGEB is not a census
    unit: its values sum the localities inside it that the assignment places in the zone the
    survey coded (``localities``), which is what its dwellings were drawn from.
    """
    spec = load_spec()["conapo"]
    years = {spec["base_year"]: None, spec["target_year"]: conapo}
    per_row = {
        f"{name}_{year}": row_targets(universe, constraints(table)[name], scaling, vmrc, rates)
        for year, scaling in years.items() for name, table in SAMPLED_TARGETS.items()
    }
    rows = pd.DataFrame(per_row)
    code = viv.ageb.astype(str)
    survey = pd.DataFrame({"code": code, "centralidad": viv.centralidad.astype(str),
                           "municipio": viv.municipio.astype(str)})
    sampled = survey.groupby("code").agg(centralidad=("centralidad", "first"), municipio=("municipio", "first"),
                                         sampled_dwellings=("centralidad", "size"))
    locs = assignment[(assignment.unit == "locality") & assignment.zone.notna()]
    out = []
    for ageb, r in sampled.iterrows():
        if len(ageb) == 13:
            members = [ageb]
        else:
            members = list(locs.index[(locs.rural_ageb == ageb) & (locs.zone == r.centralidad)])
        values = rows.reindex(members).sum()
        out.append({"AGEB": ageb, "unit": "urban" if len(ageb) == 13 else "rural",
                    "Municipality": int(ageb[2:5]), "municipio": r.municipio,
                    "centralidad": r.centralidad, "Zone": zone_index()[r.centralidad],
                    "sampled_dwellings": int(r.sampled_dwellings),
                    "localities": len(members) if len(ageb) == 9 else 0, **values.to_dict()})
    out = pd.DataFrame(out)
    out.insert(out.columns.get_loc("Zone") + 1, "TAZ", out.Zone * 1000 + out.Municipality)
    names = load_zm_muns()
    if (out.municipio != out.Municipality.map(names)).any():
        raise ValueError("a sampled AGEB's key and the survey's municipio disagree")
    return out


def build_constraints(universe, year=None, conapo=None, shares=None, vmrc=None, rates=None):
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
            series = _target(universe, entry, geography, scaling, shares, vmrc, rates)
            if name not in files:
                files[name] = pd.DataFrame({GEOGRAPHY_COLUMN[geography]: series.index.astype(int)})
            files[name][target] = series.to_numpy()
    return files
