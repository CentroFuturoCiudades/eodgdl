"""Zone-polygon and micro-zone loaders for the EOD 2023 study area.

The census units that make up each zone (urban AGEBs and rural localities) are assigned by
``eodgdl.reweight.zoning`` from these polygons and the survey's own coding.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from eodgdl.data._catalog import MICROZONAS_PARQUET, ZONIFICACION_PARQUET

# Zona-EOD access-point ids (airport / external gateways), dropped when drop_ap=True
_TAZ_ACCESS_POINTS = [
    "999990005",
    "999990004",
    "999990006",
    "999990001",
    "999990002",
    "999990003",
    "99999000A",
]
# Micro-zona access-point ids
_MZONA_ACCESS_POINTS = [603, 604, 605, 606, 607, 608]

# Manual fix (source: manual review; see README "Data provenance"): IMEPLAN's micro-zone
# totals count two localities twice, once whole and once as the AGEBs that subdivide them
# (IMEPLAN's AGEB table lists both); their populations are subtracted from these micro-zones.
_MTAZ_POBTOT_FIXES = {21: 3534.0, 442: 3103.0}


def _resolve(fpath: Path | None, filename: str) -> Path:
    if fpath is not None:
        return Path(fpath)
    from eodgdl.data import resolve

    return resolve(filename)


def load_zm_muns() -> dict[int, str]:
    """INEGI municipality code → name for the 9 Guadalajara metropolitan-area municipalities."""
    return {
        39: "Guadalajara",
        120: "Zapopan",
        98: "Tlaquepaque",
        97: "Tlajomulco",
        101: "Tonalá",
        70: "El Salto",
        51: "Juanacatlán",
        44: "Ixtlahuacán de los Membrillos",
        124: "Zapotlanejo",
    }


def load_taz(fpath: Path | None = None, drop_ap: bool = False) -> gpd.GeoDataFrame:
    """Load the EOD traffic-analysis zones (TAZ), indexed by ``ID_ZONAEOD``.

    With no ``fpath`` the zonification parquet is fetched from the mirror (or read from
    ``$EODGDL_DATA_DIR``); pass a path to read a local file.
    """
    taz = gpd.read_parquet(_resolve(fpath, ZONIFICACION_PARQUET)).set_index("ID_ZONAEOD")
    if drop_ap:
        access_points = _TAZ_ACCESS_POINTS
        taz = taz.query("~ID_ZONAEOD.isin(@access_points)")
    return taz


def load_mtaz(
    taz: gpd.GeoDataFrame, fpath: Path | None = None, drop_ap: bool = False
) -> gpd.GeoDataFrame:
    """Load the EOD micro-zones (MTAZ), aligned to ``taz``'s CRS and indexed by ``ID``."""
    mtaz = (
        gpd.read_parquet(_resolve(fpath, MICROZONAS_PARQUET))
        .to_crs(taz.crs)
        .set_index("ID")
    )
    if drop_ap:
        mtaz = mtaz.drop(_MZONA_ACCESS_POINTS).copy()

    # Remove duplicated-locality population double-counts
    for idx, delta in _MTAZ_POBTOT_FIXES.items():
        mtaz.loc[idx, "POBTOT"] -= delta

    return mtaz


def zone_system_report(taz, mtaz: gpd.GeoDataFrame) -> None:
    """Print diagnostics on disjoint micro-zone geometries and missing TAZ mappings."""
    # Disjoint (Multipolygon) micro-zones may cause problems when assigning centroids
    disjoint_geoms = list(mtaz[mtaz.count_geometries() > 1][["ZONAEOD202"]].index)
    print(
        f"The following {len(disjoint_geoms)} microzones have disjoint geometries:\n"
        f"    {disjoint_geoms}"
    )

    # Several micro-zones have no corresponding zone (some are access points)
    missing_taz = list(set(mtaz["ZONAEOD202"].unique()) - set(taz.index))
    print(
        f"The following zones {len(missing_taz)} zones do not exist, but 20 microzones "
        f"belong to them:\n    {missing_taz}"
    )
