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
from eodgdl.reweight.zoning import assign_units, check_assignment, taz_map, zone_system_codes
from eodgdl.taz import load_zm_muns

SEX = {"HOMBRES": "M", "MUJERES": "F"}
AGEB_KEY = ["ENTIDAD", "MUN", "LOC", "AGEB"]
LOC_KEY = ["ENTIDAD", "MUN", "LOC"]
GEOGRAPHY_IDS = ["MTAZ", "Municipality", "Region"]


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


def crosswalk(viv, trips, state=14):
    """The census units of the survey universe with their census key and geography ids.

    One row per urban AGEB or rural locality that :func:`zoning.assign_units` places in a
    zone, indexed by CVEGEO: the INEGI key (``AGEB`` is NA on a locality row), the rule
    that placed it, and the MTAZ / Municipality / Region ids the records use. The survey's
    dwellings and trip ends decide the zones, so both are needed; a rural AGEB placed on
    its own (no census locality in it) has no census row and is left out.
    """
    a = assign_units(viv, trips, state)
    problems = check_assignment(a, viv, trips)
    if problems:
        raise ValueError("the zone assignment disagrees with the survey: " + "; ".join(problems))
    a = a[a.zone.notna() & a.unit.isin(["ageb", "locality"])]
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
    cw["MTAZ"] = zone.astype(int)
    cw["Municipality"] = cw.MUN
    cw["Region"] = 1
    return cw


def census_universe(viv, trips, state=14):
    """Every numeric census column of every AGEB and rural locality in the survey universe.

    Indexed like the crosswalk (CVEGEO), with the MTAZ / Municipality / Region ids
    appended. An AGEB row joins the census AGEB frame on (ENTIDAD, MUN, LOC, AGEB), a
    locality row the locality frame on (ENTIDAD, MUN, LOC); the crosswalk is built from
    those frames, so every row matches. Suppressed cells stay NaN, so sums skip them.
    """
    cw = crosswalk(viv, trips, state)
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
    assumed to share the row's known mix. A suppressed ``POBCOL`` counts as 0. A row with
    residents but no stated age has no mix to spread over; :func:`fill_unaged` gives it
    its zone's.
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


def fill_unaged(universe, counts):
    """Person counts for the census rows with residents but no stated age: their zone's mix.

    117 rows (2,525 residents) state no age at all -- mostly tiny units whose age cells
    INEGI suppressed, and AGEB 1412000013876 (179 residents, 10 sampled dwellings). The
    row correction cannot spread unknown ages over known ones there, so every person count
    of such a row (``counts``, already corrected) becomes its zone's count per
    private-dwelling resident, over the rows that do state ages, times the row's own
    private-dwelling residents (``POBTOT - POBCOL``). Every band, sex and activity count
    comes from the same zone mix, so they stay consistent with each other.
    """
    u = universe.astype({c: float for c in [*STATED_AGES, "POBTOT", "POBCOL"]})
    stated = u[STATED_AGES].sum(axis=1, min_count=1).fillna(0.0)
    private = (u["POBTOT"] - u["POBCOL"].fillna(0.0)).clip(lower=0.0)
    unaged = (stated == 0) & (private > 0)
    if not unaged.any():
        return counts
    zone, aged = universe["MTAZ"], ~unaged
    per_resident = counts[aged].fillna(0.0).groupby(zone[aged]).sum().div(
        private[aged].groupby(zone[aged]).sum(), axis=0)
    filled = counts.copy()
    filled.loc[unaged] = per_resident.reindex(zone[unaged]).to_numpy() * private[unaged].to_numpy()[:, None]
    return filled


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
        if entry["row_scale"] == "persons":
            sub = fill_unaged(universe, sub.astype(float))
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


def _grown_rows(universe, entry, conapo=None, vmrc=None, rates=None):
    """A census target's value per row, corrected and grown by its own source only."""
    return _census_rows(universe, entry, conapo, vmrc, rates).fillna(0.0).eval(entry["census"])


def row_targets(universe, table, name, conapo=None, vmrc=None, rates=None):
    """One census target's value per census row: what its zone target sums.

    The expression is evaluated on each row with suppressed cells as 0, which is how the
    zone sum skips them, so these add up to the zone target wherever a zone has any value.
    In a grown set (``conapo`` given) a ``sum_of`` target is the sum of its parts' rows, a
    ``grow_with`` target grows at its parts' combined rate on each row (its own CONAPO rate
    where they are empty), and a ``grow_within`` group is rescaled, row by row, to sum to
    its grown total -- so the targets that partition the same people in the census still
    partition them in 2023.
    """
    entry = constraints(table)[name]
    if conapo is not None and "sum_of" in entry:
        return sum(row_targets(universe, table, part, conapo, vmrc, rates) for part in entry["sum_of"])
    if conapo is not None and "grow_with" in entry:
        parts = [constraints(table)[p] for p in entry["grow_with"]]
        before = sum(_grown_rows(universe, p) for p in parts)
        after = sum(_grown_rows(universe, p, conapo, vmrc, rates) for p in parts)
        rate = (after / before.where(before > 0)).fillna(_row_ratio(universe, entry, conapo))
        return _grown_rows(universe, entry) * rate
    rows = _grown_rows(universe, entry, conapo, vmrc, rates)
    if conapo is not None and "grow_within" in entry:
        g = entry["grow_within"]
        group = sum(_grown_rows(universe, constraints(table)[p], conapo, vmrc, rates) for p in g["group"])
        total = sum(row_targets(universe, table, t, conapo, vmrc, rates) for t in g["total"])
        rows = rows * (total / group.where(group > 0)).fillna(1.0)
    return rows


def _target(universe, table, name, geography, conapo, shares, vmrc=None, rates=None):
    """The target of one constraint per geography id (a Series indexed by the id)."""
    entry = constraints(table)[name]
    column = GEOGRAPHY_COLUMN[geography]
    ids = sorted(universe[column].unique())
    scale = conapo is not None and "conapo" in entry
    if conapo is not None and any(k in entry for k in ("sum_of", "grow_with", "grow_within")):
        rows = row_targets(universe, table, name, conapo, vmrc, rates)
        return rows.groupby(universe[column]).sum().reindex(ids)
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


# target -> (table, column label in ZoneSystemAGEBs.csv)
AGEB_TARGETS = {
    "Persons": ("persons", "Persons6plus"),      # residents aged 6+, the tool's person target
    "Occupants": ("households", "Occupants"),    # residents of private dwellings, all ages
    "Dwellings": ("households", "Dwellings"),    # private inhabited dwellings
}


def zone_system_agebs(universe, viv, assignment, conapo, vmrc=None, rates=None):
    """The complete zone system by AGEB: every AGEB placed in a zone, with its census targets per year.

    One row per urban AGEB and rural AGEB the assignment (:func:`~eodgdl.reweight.zoning.assign_units`)
    places in a zone, sampled or not: its ``TAZ`` (:func:`~eodgdl.reweight.zoning.taz_map`, which fails on any
    disagreement with the zone system), its municipality, its MTAZ (``MTAZ_code`` and the integer ``MTAZ``), the
    ``MTAZMun`` cell (MTAZ x municipality; a cell with no sampled dwelling is not in ``ZoneSystem.csv``), its
    sampled dwellings (0 where the survey drew none), and three targets for the base and target years:
    ``Persons6plus`` (residents aged 6+, the tool's ``Persons``), ``Occupants`` (residents of private
    dwellings, all ages) and ``Dwellings`` -- the same census expression, row correction and growth as
    the MTAZ targets, so each MTAZ's rows sum to its targets. An urban AGEB is its own census row. A
    rural AGEB is not a census unit: its values sum the localities inside it that the assignment places
    (``localities`` counts them), all in one MTAZ; one the survey records that holds no census locality
    is a row of zeros.
    """
    spec = load_spec()["conapo"]
    years = {spec["base_year"]: None, spec["target_year"]: conapo}
    per_row = pd.DataFrame({
        f"{label}_{year}": row_targets(universe, table, name, scaling, vmrc, rates)
        for year, scaling in years.items() for name, (table, label) in AGEB_TARGETS.items()
    })
    ageb = zone_system_codes(assignment)
    placed = assignment.loc[ageb.index]
    is_loc = placed.unit == "locality"
    zones = placed.zone.astype(str).groupby(ageb).unique()
    split = zones[zones.str.len() > 1]
    if len(split):
        raise ValueError(f"rural AGEBs placed in more than one zone: {split.to_dict()}")
    out = pd.DataFrame({"AGEB": zones.index})
    out["TAZ"] = out.AGEB.map(taz_map(assignment))
    out["unit"] = out.AGEB.str.len().map({13: "urban", 9: "rural"})
    out["Municipality"] = out.AGEB.str[2:5].astype(int)
    out["municipio"] = out.Municipality.map(load_zm_muns())
    out["MTAZ_code"] = zones.str[0].to_numpy()
    out["MTAZ"] = out.MTAZ_code.map(zone_index())
    out["MTAZMun"] = out.MTAZ * 1000 + out.Municipality
    code = viv.ageb.astype(str)
    out["sampled_dwellings"] = out.AGEB.map(code.value_counts()).fillna(0).astype(int)
    out["localities"] = out.AGEB.map(is_loc.groupby(ageb).sum()).astype(int)
    values = per_row.reindex(placed.index).fillna(0.0).groupby(ageb).sum()
    out = out.join(values, on="AGEB")
    if out[["unit", "municipio", "MTAZ"]].isna().any().any():
        raise ValueError("an AGEB with no unit kind, municipality or MTAZ")
    if missing := sorted(set(code) - set(out.AGEB)):
        raise ValueError(f"sampled AGEBs placed in no zone: {missing}")
    sampled = out[out.sampled_dwellings > 0].set_index("AGEB")
    survey = viv.assign(code=code).groupby("code")
    if (sampled.MTAZ_code != survey.centralidad.first().astype(str).reindex(sampled.index)).any():
        raise ValueError("a sampled AGEB placed outside the zone the survey coded")
    if (sampled.municipio != survey.municipio.first().astype(str).reindex(sampled.index)).any():
        raise ValueError("a sampled AGEB's key and the survey's municipio disagree")
    return out.sort_values(["MTAZ", "Municipality", "AGEB"], ignore_index=True)


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
            if geography == "taz":   # the TAZ set's, from ZoneSystemAGEBs.csv (taz_constraints)
                continue
            suffix = year if (year is not None and is_scaled(table, geography)) else None
            name = constraint_file(table, geography, suffix)
            series = _target(universe, table, target, geography, scaling, shares, vmrc, rates)
            if name not in files:
                files[name] = pd.DataFrame({GEOGRAPHY_COLUMN[geography]: series.index.astype(int)})
            files[name][target] = series.to_numpy()
    return files


def taz_targets(agebs, values):
    """One target per TAZ of the zone system: the census total of each MTAZMun cell on its sampled TAZ.

    ``agebs`` is :func:`zone_system_agebs`' frame, ``values`` one of its target columns (an AGEB's census value).
    The working group's proposal (2026-10-08): within each MTAZ x municipality cell the TAZ with a sampled dwelling
    share the cell's total in proportion to their own values, and every other TAZ gets 0, so the cell, its MTAZ and
    its municipality keep their census totals. A cell with no sampled TAZ (26098, one unsampled Tlaquepaque AGEB)
    has nowhere to go inside itself: its total goes to its MTAZ's sampled TAZ the same way, so the MTAZ keeps its
    total and the municipalities trade it. Raises where a sampled TAZ would get 0, which the tool would read as
    "remove these households". Indexed by TAZ, ascending.
    """
    name, values = values, agebs[values].astype(float)
    sampled = agebs.sampled_dwellings > 0
    on_sampled = values.where(sampled, 0.0)
    cell, mtaz = agebs.MTAZMun, agebs.MTAZ
    out = (values * values.groupby(cell).transform("sum") / on_sampled.groupby(cell).transform("sum")).where(sampled, 0.0)
    unsampled_cell = ~sampled.groupby(cell).transform("any")
    orphans = values.where(unsampled_cell, 0.0).groupby(mtaz).transform("sum")
    out += (values * orphans / on_sampled.groupby(mtaz).transform("sum")).where(sampled, 0.0)
    if out.isna().any() or (out[sampled] <= 0).any():
        raise ValueError(f"sampled TAZ with no {name}: {agebs.TAZ[sampled & ~(out > 0)].tolist()}")
    if abs(out.sum() - values.sum()) > 1e-6 * max(values.sum(), 1.0):
        raise ValueError(f"{name}: the TAZ targets lose {values.sum() - out.sum():,.1f} (an MTAZ with no sampled TAZ)")
    return pd.Series(out.to_numpy(), index=pd.Index(agebs.TAZ.to_numpy(), name="TAZ")).sort_index()


def taz_constraints(agebs, year):
    """{file name: frame} of the TAZ set's taz targets for one year (`rows_of` in spec.yaml): one row per TAZ of
    the zone system (:func:`taz_targets` over :func:`zone_system_agebs`' frame, ``agebs``)."""
    files = {}
    for table in TABLES:
        for target, entry in constraints(table).items():
            if entry["geography"] != "taz":
                continue
            series = taz_targets(agebs, f"{AGEB_TARGETS[entry['rows_of']][1]}_{year}")
            name = constraint_file(table, "taz", year)
            if name not in files:
                files[name] = pd.DataFrame({"TAZ": series.index.astype("int64")})
            files[name][target] = series.to_numpy()
    return files
