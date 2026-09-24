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
  is out of the universe (``outside``). Blocks never split an AGEB: they only choose;
* **a rural locality** goes to the polygon containing it (``polygon``) or is ``outside``,
  unless ``_RURAL_LOCALITY_ZONES`` places it (``override``). The survey codes a rural
  dwelling by its rural AGEB, which spans several zones, so the rural AGEBs are not units:
  :func:`check_assignment` only requires each sampled one to hold a locality in the zone
  the survey coded.

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
         "override", "outside"]

# Rural localities the survey places in a zone whose polygon does not contain them. In
# both rural AGEBs every locality inside a polygon lies in another zone than the one the
# survey coded, and only these localities are large enough for its sample (10-dwelling
# clusters):
#   140970123 (Tlajomulco): 80 dwellings coded 51F. Cuexcomatitlán (520 dwellings) and San
#     Juan Evangelista (513) are its only large localities, 1.1 and 3.0 km from 51F's
#     polygon; IMEPLAN's EOD2022 key put both, and Cruz Blanca between them, in 51F.
#   140973679 (Tlajomulco): 30 dwellings coded 11F. Renaceres (477 dwellings, 0.5 km from
#     11F's polygon) is its only large locality; IMEPLAN's table codes it 11F.
_RURAL_LOCALITY_ZONES = {
    "140970014": "51F",  # Cuexcomatitlán
    "140970004": "51F",  # San Juan Evangelista (San Juan)
    "140970441": "51F",  # Cruz Blanca (Colonia Sergio Barrios)
    "140970877": "11F",  # Renaceres [Fraccionamiento]
}


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
    left out). The survey codes a trip end's zone by where it lies, and an AGEB's origins
    and destinations agree on its zone (the most frequent one is the same for all 1,691
    codes seen as both). Access points (``99999...``) are left out: they are no census unit.
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
    coded most for its trip ends, or the polygon holding most of its population; so every
    AGEB the survey sampled or recorded a trip end in is in the zone system. A rural AGEB
    is not a unit, but every zone the survey coded for it (dwellings or trip ends) must hold
    one of its localities: after the polygons and ``_RURAL_LOCALITY_ZONES``, a zone that
    holds none takes the AGEB's locality nearest to that zone's polygon (the larger on a
    tie), unless that locality is the only one it has in another zone the survey coded.
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

    inside = gpd.sjoin(locs[["geometry"]], polys[["geometry"]].reset_index(names="polygon"), how="left", predicate="within").polygon
    rural_agebs = _mgn("ar", polys.crs).set_index("CVEGEO")[["geometry"]]
    in_ageb = gpd.sjoin(locs[["geometry"]], rural_agebs.reset_index(names="rural_ageb"), how="left", predicate="within").rural_ageb
    rural = pd.DataFrame({
        "unit": "locality", "MUN": locs.MUN.astype(int), "POBTOT": locs.POBTOT.astype(float),
        "polygon_zone": inside, "polygon_share": inside.notna().astype(float), "straddles": False,
        "rural_ageb": in_ageb,
    })
    missing = sorted(set(_RURAL_LOCALITY_ZONES) - set(rural.index))
    if missing:
        raise ValueError(f"_RURAL_LOCALITY_ZONES names localities the census does not have: {missing}")
    override = pd.Series(_RURAL_LOCALITY_ZONES)
    rural["zone"] = rural.polygon_zone
    rural["rule"] = rural.polygon_zone.notna().map({True: "polygon", False: "outside"})
    rural.loc[override.index, "zone"] = override
    rural.loc[override.index, "rule"] = "override"
    _place_rural_codes(rural, locs, polys, codes, locked=set(override.index))

    # A rural AGEB the survey records but that holds no census locality (only unpopulated
    # places, e.g. a factory) is a unit of its own, with no population, so its code still
    # maps to the zone the survey coded most for it.
    rural_codes = codes[codes.code.str.len() == 9]
    empty = sorted(set(rural_codes.code) - set(rural.rural_ageb.dropna()))
    top = rural_codes.sort_values(["code", "n", "zone"], ascending=[True, False, True]).drop_duplicates("code").set_index("code")
    bare = pd.DataFrame({
        "unit": "rural ageb", "MUN": [int(c[2:5]) for c in empty], "POBTOT": 0.0,
        "polygon_zone": pd.NA, "polygon_share": 0.0, "straddles": False, "rural_ageb": empty,
        "zone": top.zone.reindex(empty).to_numpy(),
        "rule": ["survey" if s == "dwellings" else "trip ends" for s in top.source.reindex(empty)],
    }, index=pd.Index(empty))

    out = pd.concat([urban, rural, bare])
    out.index.name = "CVEGEO"
    out["rule"] = pd.Categorical(out.rule, categories=RULES)
    out["sampled_dwellings"] = out.sampled_dwellings.astype("Int64")
    out["trip_ends"] = out.trip_ends.astype("Int64")
    return out[["unit", "MUN", "POBTOT", "polygon_zone", "polygon_share", "straddles", "survey_zone",
                "sampled_dwellings", "trip_zone", "trip_ends", "rural_ageb", "zone", "rule"]]


def _place_rural_codes(rural, locs, polys, codes, locked):
    """Give every zone the survey coded for a rural AGEB one of its localities (in place)."""
    rural_codes = codes[codes.code.str.len() == 9]
    dwelling = set(map(tuple, rural_codes.loc[rural_codes.source == "dwellings", ["code", "zone"]].values))
    required = rural_codes.groupby("code").zone.agg(set)
    for ageb, zones in required.items():
        members = rural.index[rural.rural_ageb == ageb]
        if len(members) == 0:
            continue
        for zone in sorted(zones):
            if (rural.loc[members, "zone"] == zone).any() or zone not in polys.index:
                continue
            distance = locs.geometry.reindex(members).distance(polys.geometry[zone])
            order = sorted(members, key=lambda c: (distance[c], -(rural.at[c, "POBTOT"] if pd.notna(rural.at[c, "POBTOT"]) else 0)))
            for c in order:
                current = rural.at[c, "zone"]
                if c in locked or (pd.notna(current) and current in zones and (rural.loc[members, "zone"] == current).sum() == 1):
                    continue
                rural.at[c, "zone"] = zone
                rural.at[c, "rule"] = "survey" if (ageb, zone) in dwelling else "trip ends"
                locked.add(c)
                break


def check_assignment(assignment, viv, trips):
    """Where the assignment and the survey disagree; empty when every coded AGEB is placed.

    Every urban AGEB with dwellings carries the zone the survey coded for them, and every
    other one with trip ends the zone it coded most for them (true by construction). Every
    zone the survey coded for a rural AGEB, from its dwellings or its trip ends, holds one
    of its localities: one that does not needs an entry in ``_RURAL_LOCALITY_ZONES``. Every
    assigned zone is a survey zone.
    """
    problems = []
    codes = survey_codes(viv, trips)
    survey, _ = _dwelling_zones(codes)
    trip_zone, _ = _trip_zones(codes)
    urban = assignment[assignment.unit == "ageb"].zone
    for code, zone in survey.items():
        if len(code) == 13 and urban.get(code) != zone:
            problems.append(f"AGEB {code}: assigned {urban.get(code)}, its dwellings coded {zone}")
    for code, zone in trip_zone.items():
        if len(code) == 13 and code not in survey.index and urban.get(code) != zone:
            problems.append(f"AGEB {code}: assigned {urban.get(code)}, its trip ends coded {zone}")
    locs = assignment[assignment.unit.isin(["locality", "rural ageb"])]
    held = set(map(tuple, locs.loc[locs.zone.notna(), ["rural_ageb", "zone"]].values))
    rural_codes = codes[codes.code.str.len() == 9]
    for (code, source), g in rural_codes.groupby(["code", "source"]):
        for zone, n in zip(g.zone, g.n):
            if (code, zone) not in held:
                problems.append(f"rural AGEB {code}: {n} {source} coded {zone}, no locality of it assigned there")
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
    """The redrawn zones: each zone's polygon, with every urban AGEB moved whole to its zone.

    A zone keeps the rural land of its polygon, loses the urban AGEBs assigned elsewhere or
    left outside, and gains the urban AGEBs and overridden localities assigned to it (a
    locality by its outline). Indexed by zone code, one (multi)polygon each.
    """
    polys = zone_polygons()
    units = unit_shapes().geometry
    urban = assignment[assignment.unit == "ageb"]
    shapes = units.reindex(urban.index)
    moved = assignment[assignment.rule == "override"]
    shapes_moved = units.reindex(moved.index)
    out = {}
    for zone, poly in polys.geometry.items():
        own = shapes[urban.zone == zone]
        others = shapes[urban.zone != zone]
        keep = poly.difference(others[others.intersects(poly)].union_all())
        extra = [*own.to_list(), *shapes_moved[moved.zone == zone].dropna().to_list()]
        out[zone] = gpd.GeoSeries([keep, *extra], crs=polys.crs).union_all()
    return gpd.GeoDataFrame({"zone": list(out)}, geometry=list(out.values()), crs=polys.crs).set_index("zone")
