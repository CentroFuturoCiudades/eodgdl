"""The zone of every census unit, from the centralidad polygons and the survey's own coding.

The survey was fielded on the centralidad polygons (``load_taz``) and codes every dwelling
with its centralidad; it never splits an AGEB (every sampled AGEB's dwellings carry one
zone). The census counts people in two kinds of unit that tile the nine municipalities:
urban AGEBs and rural localities. :func:`assign_units` gives each unit at most one zone:

* **a sampled urban AGEB** takes the zone the survey coded there (``survey``), even where
  most of its population lies in another polygon (``survey over polygon``): the survey
  interviewed there and weights its households by that zone;
* **an unsampled urban AGEB** goes whole to the polygon holding most of its population,
  counted over its census blocks (``polygon`` when all of it lies in one polygon,
  ``majority`` when it straddles); one with most of its population outside every polygon
  goes to the polygon holding the most of the rest, or to the nearest polygon
  (``nearest polygon``), so every urban AGEB is in a zone. Blocks never split an AGEB:
  they only choose;
* **a rural AGEB the survey records** goes whole to the zone it coded, with every locality
  of it inside some polygon -- the survey's frame -- wherever the polygons would put it, so
  its sampled dwellings expand into all of its in-frame population; its localities outside
  every polygon stay out of the targets (IMEPLAN planned a sample in almost none of them),
  with two exceptions the survey itself forces: when none lies inside a polygon, its
  locality nearest to the zone's polygon stands for it, and when those counted hold fewer
  dwellings than the survey sampled there, its other localities join, nearest to the zone's
  polygon first, until they can hold the sample;
* **any other rural locality** goes to the polygon containing it (``polygon``) or is
  ``outside``.

IMEPLAN's AGEB table plays no part. :func:`zone_shapes` merges the result into redrawn
zone polygons that follow AGEB edges, a proposal for the zoning the targets are built on.
"""
from __future__ import annotations

import functools

import geopandas as gpd
import pandas as pd

from eodgdl.taz import load_taz, load_zm_muns

STATE = 14
RULES = ["survey", "survey over polygon", "trip ends", "trip ends over polygon", "polygon", "majority",
         "nearest polygon", "outside"]

def _mxcensus():
    try:
        import mxcensus
        from mxcensus.data import POOCH
    except ImportError:
        raise ImportError(
            "eodgdl.reweight builds its zone assignment from the census through mxcensus; "
            "install it with `uv sync --extra reweight`"
        ) from None
    return mxcensus, POOCH


def _mgn(layer, crs, state=STATE):
    """One Marco Geoestadístico 2020 layer, the nine municipalities only, in `crs`."""
    _, pooch = _mxcensus()
    g = gpd.read_parquet(pooch.fetch(f"mg_{layer}_{state:02d}.parquet"))
    return g[g.CVE_MUN.astype(int).isin(load_zm_muns())].to_crs(crs)


def _cvegeo(frame, *parts):
    """The INEGI key of each row of a census frame: ENTIDAD, MUN, LOC and optionally AGEB, MZA."""
    code = (frame.ENTIDAD.astype(int).map("{:02d}".format) + frame.MUN.astype(int).map("{:03d}".format)
            + frame.LOC.astype(int).map("{:04d}".format))
    if "AGEB" in parts:
        code = code + frame.AGEB.astype(str)
    if "MZA" in parts:
        code = code + frame.MZA.astype(int).map("{:03d}".format)
    return code


def zone_polygons():
    """The centralidad polygons the survey was fielded on, indexed by code, access points dropped."""
    return load_taz(drop_ap=True)


@functools.cache
def _geography(state=STATE):
    """(urban AGEBs, rural localities, blocks), each with its census keys and a geometry.

    Urban AGEBs are the census AGEB frame; the two it has that the Marco Geoestadístico
    does not draw (El Salto's AGEB 1467 inside localities 0116 and 0134) take their
    locality's polygon. Rural localities are the census locality frame less the localities
    that have AGEBs, placed at their polygon's interior point or, without one, at their
    point. Blocks carry their census population and their polygon's interior point.
    """
    mxcensus, pooch = _mxcensus()
    crs = zone_polygons().crs
    muns = load_zm_muns()
    _, _, df_loc, df_ageb = mxcensus.load_census(state=state)

    agebs = df_ageb.reset_index()
    agebs = agebs[agebs.MUN.astype(int).isin(muns)].copy()
    agebs["CVEGEO"] = _cvegeo(agebs, "AGEB")
    ageb_shapes = _mgn("a", crs).set_index("CVEGEO").geometry
    loc_shapes = _mgn("l", crs).set_index("CVEGEO").geometry
    geometry = agebs.CVEGEO.map(ageb_shapes)
    undrawn = geometry.isna()
    geometry[undrawn] = agebs.loc[undrawn, "CVEGEO"].str[:9].map(loc_shapes)
    if geometry.isna().any():
        raise ValueError(f"urban AGEBs with no shape: {agebs.loc[geometry.isna(), 'CVEGEO'].tolist()}")
    agebs = gpd.GeoDataFrame(agebs, geometry=geometry.to_list(), crs=crs).set_index("CVEGEO")

    locs = df_loc.reset_index()
    locs = locs[locs.MUN.astype(int).isin(muns)].copy()
    locs["CVEGEO"] = _cvegeo(locs)
    locs = locs[~locs.CVEGEO.isin(set(agebs.index.str[:9]))]
    points = _mgn("lpr", crs)
    points = points.assign(k=points.CVE_ENT + points.CVE_MUN + points.CVE_LOC).drop_duplicates("k").set_index("k").geometry
    where = locs.CVEGEO.map(loc_shapes.representative_point()).fillna(locs.CVEGEO.map(points))
    if where.isna().any():
        raise ValueError(f"rural localities with no location: {locs.loc[where.isna(), 'CVEGEO'].tolist()}")
    locs = gpd.GeoDataFrame(locs, geometry=where.to_list(), crs=crs).set_index("CVEGEO")

    blocks = mxcensus.load_resargebub(pooch.fetch(f"resargebub_{state:02d}.parquet"))[-1].reset_index()
    blocks = blocks[blocks.MUN.astype(int).isin(muns)]
    pop = pd.Series(blocks.POBTOT.astype(float).to_numpy(), index=_cvegeo(blocks, "AGEB", "MZA"))
    mza = _mgn("m", crs)
    mza = gpd.GeoDataFrame({"ageb": mza.CVEGEO.str[:13], "POBTOT": mza.CVEGEO.map(pop).fillna(0.0).to_numpy()},
                           geometry=mza.representative_point().to_list(), crs=crs)
    return agebs, locs, mza


def survey_codes(viv, trips):
    """How the survey codes each AGEB: one row per (code, source, zone) with its count.

    ``source`` is ``dwellings`` (``viv.ageb`` / ``centralidad``) or ``trip ends`` (every
    trip's ``origen`` / ``zona_origen`` and ``destino`` / ``zona_destino``, the non-trips
    left out). The survey's zone is a function of the AGEB code, not of where a dwelling or
    trip end lies: each of the 1,694 codes it records carries exactly one zone across its
    dwellings, origins and destinations -- also the 157 urban AGEBs with trip ends that
    straddle two polygons and the rural AGEBs whose localities span several zones. Access
    points (``99999...``) are left out: they are no census unit.
    """
    from eodgdl.chains import non_trips

    t = trips[~non_trips(trips)]
    frames = [
        pd.DataFrame({"code": viv.ageb.astype(str), "zone": viv.centralidad.astype(str), "source": "dwellings"}),
        pd.DataFrame({"code": t.origen.astype(str), "zone": t.zona_origen.astype(str), "source": "trip ends"}),
        pd.DataFrame({"code": t.destino.astype(str), "zone": t.zona_destino.astype(str), "source": "trip ends"}),
    ]
    codes = pd.concat(frames)
    codes = codes[~codes.code.str.startswith("99999")]
    return codes.groupby(["code", "source", "zone"]).size().rename("n").reset_index()


def _dwelling_zones(codes):
    """The zone the survey coded for each AGEB's dwellings, and how many; raises on a split AGEB."""
    dw = codes[codes.source == "dwellings"]
    split = dw.groupby("code").zone.nunique()
    if (split > 1).any():
        raise ValueError(f"AGEBs the survey coded with more than one zone: {sorted(split.index[split > 1])}")
    return dw.set_index("code").zone, dw.set_index("code").n


def _trip_zones(codes):
    """Each AGEB's most frequent trip-end zone (ties by zone code) and its number of trip ends."""
    te = codes[codes.source == "trip ends"].sort_values(["code", "n", "zone"], ascending=[True, False, True])
    return te.drop_duplicates("code").set_index("code").zone, te.groupby("code").n.sum()


def assign_units(viv, trips, state=STATE):
    """One row per census unit of the nine municipalities (urban AGEB or rural locality).

    Indexed by CVEGEO: ``unit`` (``ageb`` | ``locality``), ``MUN``, ``POBTOT``,
    ``polygon_zone`` (the polygon holding most of the unit's population, NA when most of it
    is outside every polygon), ``polygon_share`` (that share; 1 or 0 for a locality),
    ``survey_zone`` and ``sampled_dwellings`` (urban AGEBs the survey sampled),
    ``trip_zone`` and ``trip_ends`` (the most frequent zone the survey coded for the
    AGEB's trip ends, and how many), ``rural_ageb`` (localities), and the result: ``zone``
    (NA when outside) and ``rule``.

    An urban AGEB takes, in order, the zone the survey coded for its dwellings, the zone it
    coded for its trip ends, or the polygon holding most of its population. A rural AGEB the
    survey records (dwellings or trip ends) takes the zone it coded, with its localities
    inside any polygon (or, if none is, its locality nearest to that zone's polygon, and for
    a sampled one as many more of the nearest as its sample needs); its other localities are
    ``outside``. A rural locality the survey does not record goes with the polygon containing
    it. The survey's zone is a function of the AGEB code, so every AGEB it records,
    urban or rural, has its code, and census population, in the zone it coded.
    """
    agebs, locs, blocks = _geography(state)
    polys = zone_polygons()
    codes = survey_codes(viv, trips)
    at = gpd.sjoin(blocks, polys[["geometry"]].reset_index(names="polygon"), how="left", predicate="within")
    by_polygon = at.fillna({"polygon": ""}).groupby(["ageb", "polygon"]).POBTOT.sum().unstack(fill_value=0.0)
    by_polygon = by_polygon.reindex(agebs.index, fill_value=0.0)
    # An AGEB with no populated block (or no block drawn) chooses by area instead.
    empty = by_polygon.sum(axis=1) == 0
    if empty.any():
        pieces = gpd.overlay(agebs.loc[empty, ["geometry"]].reset_index(), polys[["geometry"]].reset_index(names="polygon"),
                             how="intersection", keep_geom_type=True)
        area = pieces.assign(a=pieces.area).groupby(["CVEGEO", "polygon"]).a.sum().unstack(fill_value=0.0)
        area[""] = agebs.loc[empty].area - area.sum(axis=1).reindex(agebs.index[empty]).fillna(0.0)
        by_polygon.loc[empty] = area.reindex(index=agebs.index[empty], columns=by_polygon.columns, fill_value=0.0).to_numpy()
    share = by_polygon.div(by_polygon.sum(axis=1), axis=0)
    majority = share.idxmax(axis=1)
    urban = pd.DataFrame({
        "unit": "ageb", "MUN": agebs.MUN.astype(int), "POBTOT": agebs.POBTOT.astype(float),
        "polygon_zone": majority.where(majority != ""),
        "polygon_share": share.max(axis=1),
        "straddles": (share > 0).sum(axis=1) > 1,
    })
    survey, dwellings = _dwelling_zones(codes)
    trip_zone, trip_ends = _trip_zones(codes)
    unknown = sorted({c for c in [*survey.index, *trip_zone.index] if len(c) == 13} - set(urban.index))
    if unknown:
        raise ValueError(f"AGEBs the survey records that the census does not have: {unknown}")
    urban["survey_zone"] = survey.reindex(urban.index)
    urban["sampled_dwellings"] = dwellings.reindex(urban.index).fillna(0).astype(int)
    urban["trip_zone"] = trip_zone.reindex(urban.index)
    urban["trip_ends"] = trip_ends.reindex(urban.index).fillna(0).astype(int)
    sampled, travelled = urban.survey_zone.notna(), urban.survey_zone.isna() & urban.trip_zone.notna()
    urban["zone"] = urban.survey_zone.where(sampled, urban.trip_zone.where(travelled, urban.polygon_zone))
    urban["rule"] = "outside"
    urban.loc[urban.polygon_zone.notna(), "rule"] = "majority"
    urban.loc[urban.polygon_zone.notna() & ~urban.straddles, "rule"] = "polygon"
    urban.loc[travelled, "rule"] = "trip ends"
    urban.loc[travelled & (urban.trip_zone != urban.polygon_zone), "rule"] = "trip ends over polygon"
    urban.loc[sampled, "rule"] = "survey"
    urban.loc[sampled & (urban.survey_zone != urban.polygon_zone), "rule"] = "survey over polygon"
    # Every urban AGEB is in a zone: one the survey never records and that lies mostly outside
    # every polygon goes to the polygon holding the most of what is inside, or, with nothing
    # inside, to the nearest polygon (all lie on or within a few km of one).
    for code in urban.index[urban.zone.isna()]:
        inside = share.loc[code].drop("", errors="ignore")
        if (inside > 0).any():
            urban.at[code, "zone"] = inside.idxmax()
        else:
            urban.at[code, "zone"] = polys.geometry.distance(agebs.geometry[code]).idxmin()
        urban.at[code, "rule"] = "nearest polygon"

    inside = gpd.sjoin(locs[["geometry"]], polys[["geometry"]].reset_index(names="polygon"), how="left", predicate="within").polygon
    rural_agebs = _mgn("ar", polys.crs).set_index("CVEGEO")[["geometry"]]
    in_ageb = gpd.sjoin(locs[["geometry"]], rural_agebs.reset_index(names="rural_ageb"), how="left", predicate="within").rural_ageb
    rural = pd.DataFrame({
        "unit": "locality", "MUN": locs.MUN.astype(int), "POBTOT": locs.POBTOT.astype(float),
        "polygon_zone": inside, "polygon_share": inside.notna().astype(float), "straddles": False,
        "rural_ageb": in_ageb,
    })
    rural["zone"] = rural.polygon_zone
    rural["rule"] = rural.polygon_zone.notna().map({True: "polygon", False: "outside"})
    rural_codes = codes[codes.code.str.len() == 9]
    coded = rural_codes.groupby("code").zone.first()
    has_dwellings = set(rural_codes.loc[rural_codes.source == "dwellings", "code"])
    recorded = rural.rural_ageb.isin(coded.index)
    in_frame = recorded & rural.polygon_zone.notna()
    none_inside = ~in_frame[recorded].groupby(rural.rural_ageb[recorded]).any()
    for ageb in none_inside.index[none_inside]:
        members = rural.index[rural.rural_ageb == ageb]
        distance = locs.geometry.reindex(members).distance(polys.geometry[coded[ageb]])
        size = rural.loc[members, "POBTOT"].fillna(0)
        in_frame[min(members, key=lambda c: (distance[c], -size[c]))] = True
    # A sampled rural AGEB's counted localities must be able to hold its sample: the survey
    # drew its dwellings there. While they hold fewer, the nearest left-out locality joins.
    dwellings_of = locs.TVIVPARHAB.astype(float).fillna(0.0)
    sampled_n = rural_codes[rural_codes.source == "dwellings"].groupby("code").n.sum()
    for ageb, n in sampled_n.items():
        members = rural.index[rural.rural_ageb == ageb]
        if len(members) == 0:
            continue
        distance = locs.geometry.reindex(members).distance(polys.geometry[coded[ageb]])
        size = rural.loc[members, "POBTOT"].fillna(0)
        for c in sorted((m for m in members if not in_frame[m]), key=lambda c: (distance[c], -size[c])):
            if dwellings_of.reindex(members[in_frame[members].to_numpy()]).sum() >= n:
                break
            in_frame[c] = True
    rural.loc[recorded, "zone"] = pd.NA
    rural.loc[recorded, "rule"] = "outside"
    rural.loc[in_frame, "zone"] = rural.loc[in_frame, "rural_ageb"].map(coded)
    rural.loc[in_frame, "rule"] = ["survey" if r in has_dwellings else "trip ends" for r in rural.loc[in_frame, "rural_ageb"]]

    # A rural AGEB the survey records but that holds no census locality (only unpopulated
    # places, e.g. a factory) is a unit of its own, with no population, so its code still
    # maps to the zone the survey coded for it.
    empty = sorted(set(coded.index) - set(rural.rural_ageb.dropna()))
    bare = pd.DataFrame({
        "unit": "rural ageb", "MUN": [int(c[2:5]) for c in empty], "POBTOT": 0.0,
        "polygon_zone": pd.NA, "polygon_share": 0.0, "straddles": False, "rural_ageb": empty,
        "zone": coded.reindex(empty).to_numpy(),
        "rule": ["survey" if c in has_dwellings else "trip ends" for c in empty],
    }, index=pd.Index(empty))

    out = pd.concat([urban, rural, bare])
    out.index.name = "CVEGEO"
    out["rule"] = pd.Categorical(out.rule, categories=RULES)
    out["sampled_dwellings"] = out.sampled_dwellings.astype("Int64")
    out["trip_ends"] = out.trip_ends.astype("Int64")
    return out[["unit", "MUN", "POBTOT", "polygon_zone", "polygon_share", "straddles", "survey_zone",
                "sampled_dwellings", "trip_zone", "trip_ends", "rural_ageb", "zone", "rule"]]


def check_assignment(assignment, viv, trips):
    """Where the assignment and the survey disagree; empty when every coded AGEB is placed.

    Every urban AGEB is in a zone; one with dwellings carries the zone the survey coded for
    them, and every other one with trip ends the zone coded for them. Every rural AGEB the
    survey records has a unit in the zone it coded (a locality, or the rural AGEB itself when
    it holds no census locality). Every assigned zone is a survey zone.
    """
    problems = []
    codes = survey_codes(viv, trips)
    survey, _ = _dwelling_zones(codes)
    trip_zone, _ = _trip_zones(codes)
    urban = assignment[assignment.unit == "ageb"].zone
    problems += [f"AGEB {code}: in no zone" for code in urban.index[urban.isna()]]
    for code, zone in survey.items():
        if len(code) == 13 and urban.get(code) != zone:
            problems.append(f"AGEB {code}: assigned {urban.get(code)}, its dwellings coded {zone}")
    for code, zone in trip_zone.items():
        if len(code) == 13 and code not in survey.index and urban.get(code) != zone:
            problems.append(f"AGEB {code}: assigned {urban.get(code)}, its trip ends coded {zone}")
    units = assignment[assignment.unit.isin(["locality", "rural ageb"])]
    rural_codes = codes[codes.code.str.len() == 9]
    for (code, source), g in rural_codes.groupby(["code", "source"]):
        for zone, n in zip(g.zone, g.n):
            members = units[(units.rural_ageb == code) & units.zone.notna()]
            if (members.zone == zone).sum() == 0:
                problems.append(f"rural AGEB {code}: {n} {source} coded {zone}, no unit of it there")
    unknown = set(assignment.zone.dropna()) - set(viv.centralidad.astype(str))
    if unknown:
        problems.append(f"zones no dwelling was coded with: {sorted(unknown)}")
    return problems


def trip_end_agreement(assignment, viv, trips):
    """Every trip end classified against the zone system: a count per class.

    ``in zone`` -- an urban AGEB whose assigned zone is the one coded for the end, or a
    rural AGEB with a locality in it; ``other zone`` -- an urban AGEB that straddles two
    zones, whose single zone is not this end's; ``access point``.
    """
    from eodgdl.chains import non_trips

    t = trips[~non_trips(trips)]
    ends = pd.concat([pd.DataFrame({"code": t.origen.astype(str), "zone": t.zona_origen.astype(str)}),
                      pd.DataFrame({"code": t.destino.astype(str), "zone": t.zona_destino.astype(str)})])
    urban = assignment[assignment.unit == "ageb"].zone
    locs = assignment[assignment.unit.isin(["locality", "rural ageb"]) & assignment.zone.notna()]
    held = set(map(tuple, locs[["rural_ageb", "zone"]].values))
    def kind(code, zone):
        if code.startswith("99999"):
            return "access point"
        if len(code) == 13:
            return "in zone" if urban.get(code) == zone else "other zone"
        return "in zone" if (code, zone) in held else "other zone"
    return pd.Series([kind(c, z) for c, z in zip(ends.code, ends.zone)]).value_counts()


def unit_shapes(state=STATE):
    """Every census unit's shape, indexed by CVEGEO: an urban AGEB's polygon, a rural
    locality's outline where the Marco Geoestadístico draws one, else its point."""
    agebs, locs, _ = _geography(state)
    outlines = _mgn("l", agebs.crs).set_index("CVEGEO").geometry
    loc_shape = locs.index.to_series().map(outlines)
    loc_shape = loc_shape.where(loc_shape.notna(), locs.geometry)
    return gpd.GeoDataFrame(
        {"unit": ["ageb"] * len(agebs) + ["locality"] * len(locs)},
        geometry=[*agebs.geometry.to_list(), *loc_shape.to_list()],
        index=pd.Index([*agebs.index, *locs.index], name="CVEGEO"), crs=agebs.crs,
    )


def zone_shapes(assignment):
    """The redrawn zones: each polygon with every urban AGEB and recorded rural AGEB moved whole.

    A zone keeps its polygon's remaining land, loses the urban AGEBs and the recorded rural
    AGEBs (their whole territory, less the ground urban AGEBs cover) assigned elsewhere or
    left outside, and gains those assigned to it; a rural AGEB the survey does not record
    stays split along the polygons, as its localities are. Indexed by zone code, one
    (multi)polygon each.
    """
    polys = zone_polygons()
    units = unit_shapes().geometry
    urban = assignment[assignment.unit == "ageb"]
    urban_shapes = units.reindex(urban.index)
    rural_units = assignment[assignment.unit.isin(["locality", "rural ageb"]) & assignment.rule.isin(["survey", "trip ends"])]
    rural_zone = rural_units.groupby("rural_ageb").zone.first()
    rural_shapes = _mgn("ar", polys.crs).set_index("CVEGEO").geometry.reindex(rural_zone.index)
    # INEGI's rural AGEBs leave holes for urban localities, but not always exactly: an urban
    # AGEB always wins the ground it covers
    rural_shapes = rural_shapes.difference(urban_shapes.union_all())
    moved = gpd.GeoSeries([*urban_shapes.to_list(), *rural_shapes.to_list()], crs=polys.crs)
    zone_of = pd.Series([*urban.zone.to_list(), *rural_zone.to_list()])
    out = {}
    for zone, poly in polys.geometry.items():
        others = moved[(zone_of != zone).to_numpy() & moved.intersects(poly).to_numpy()]
        keep = poly.difference(others.union_all()) if len(others) else poly
        own = moved[(zone_of == zone).to_numpy()].dropna()
        out[zone] = gpd.GeoSeries([keep, *own.to_list()], crs=polys.crs).union_all()
    return gpd.GeoDataFrame({"zone": list(out)}, geometry=list(out.values()), crs=polys.crs).set_index("zone")
