"""Traffic Analysis Zone (TAZ) and AGEB zone-system loaders for the EOD 2023 study area."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

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
# duplicated-locality population double-counts subtracted from these micro-zonas. This is
# the micro-zone half of one correction; _drop_locality_ageb_overlap below is the AGEB half,
# and the two must agree — tests/test_smoke.py reconciles them.
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


def _locality_key(df: pd.DataFrame) -> pd.Series:
    """(CVE_ENT, CVE_MUN, CVE_LOC) zero-padded; NA where any part is missing.

    Not derived from CVEGEO: 12 AGEB rows carry a malformed one (ID 2066 stores
    '1407090134146' for locality 0134 / AGEB 1467), and CVE_LOC is inconsistently
    zero-padded across rows. Rows with an incomplete key are the access points and a
    few unidentified rows; none participates in an overlap.
    """

    def z(col: str, n: int) -> pd.Series:
        return df[col].astype("string").str.zfill(n)

    key = z("CVE_ENT", 2) + z("CVE_MUN", 3) + z("CVE_LOC", 4)
    return key.mask(df[["CVE_ENT", "CVE_MUN", "CVE_LOC"]].isna().any(axis=1))


def _drop_locality_ageb_overlap(agebs_zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop the coarser side where a locality row and its AGEB rows both appear.

    The table mixes locality rows (no CVE_AGEB) with AGEB rows. Where both cover the
    same locality its population is counted twice, so one side must go; keep whichever
    partitions the locality more finely:

        locality split into >1 AGEB  ->  keep the AGEBs, drop the locality row
        locality split into  1 AGEB  ->  keep the locality, drop the AGEB row, which
                                         adds no resolution and whose polygon covers
                                         only the urban part of the locality

    In the shipped data this removes IDs 2255 (a locality over 4 AGEBs), 2066 and 2265
    (single AGEBs). The first two are the AGEB-side half of the double-count that
    _MTAZ_POBTOT_FIXES corrects on the micro-zone side; 2265 carries no population.
    """
    key = _locality_key(agebs_zones)
    is_loc = agebs_zones.CVE_AGEB.isna()
    ageb_keys = key[~is_loc]

    drop: list = []
    for loc_id, loc_key in key[is_loc & key.notna()].items():
        members = list(ageb_keys.index[ageb_keys == loc_key])
        if len(members) > 1:
            drop.append(loc_id)
        elif members:
            drop.extend(members)
    return agebs_zones.drop(drop)


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

    # Locality rows and the AGEB rows that subdivide them both appear; keep the finer.
    agebs_zones = _drop_locality_ageb_overlap(agebs_zones)

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
