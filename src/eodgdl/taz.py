"""Traffic Analysis Zone (TAZ) and AGEB zone-system loaders for the EOD 2023 study area."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from eodgdl.data._catalog import (
    AGEBS_ZONA_PARQUET,
    MICROZONAS_PARQUET,
    ZONIFICACION_PARQUET,
)

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
# Micro-zona / AGEB access-point ids
_MZONA_ACCESS_POINTS = [603, 604, 605, 606, 607, 608]

# Manual fixes (source: manual review; see README "Data provenance"):
# duplicated-locality population double-counts subtracted from these micro-zonas
_MTAZ_POBTOT_FIXES = {21: 3534.0, 442: 3103.0}
# wrong MZONA assignments corrected in the AGEB table
_AGEBS_MZONA_FIXES = {2163: 35, 2155: 624, 2161: 624}


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


def load_imeplan_agebs(
    taz: gpd.GeoDataFrame, fpath: Path | None = None, drop_ap: bool = False
) -> gpd.GeoDataFrame:
    """Load IMEPLAN's AGEB → zone table with census population, aligned to ``taz``'s CRS."""
    agebs_zones = (
        gpd.read_parquet(_resolve(fpath, AGEBS_ZONA_PARQUET))
        .set_index("ID")
        .to_crs(taz.crs)
    )

    # Correct wrong MZONA assignments
    for idx, mzona in _AGEBS_MZONA_FIXES.items():
        agebs_zones.loc[idx, "MZONA"] = mzona

    # NOTE: preserved from the original implementation — these drops lack an assignment and
    # are therefore currently no-ops (kept for behavioral parity, not effect).
    agebs_zones.drop(2255)
    agebs_zones.drop(2066)

    if drop_ap:
        # Drop access points and non-ageb geometries
        access_points = _MZONA_ACCESS_POINTS
        agebs_zones = agebs_zones.query("~MZONA.isin(@access_points)").copy()

    return agebs_zones


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
