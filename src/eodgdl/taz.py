"""Zone-polygon loader for the EOD 2023 study area.

The census units that make up each zone (urban AGEBs and rural localities) are assigned by
``eodgdl.reweight.zoning`` from these polygons and the survey's own coding.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from eodgdl.data import ZONIFICACION_PARQUET

# Zona-EOD access-point ids (airport / external gateways), dropped when drop_ap=True; trip ends carry them as codes
ACCESS_POINTS = [
    "999990005",
    "999990004",
    "999990006",
    "999990001",
    "999990002",
    "999990003",
    "99999000A",
]


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

    With no ``fpath`` the zonification parquet is read from the clone's ``data/`` (or
    ``$EODGDL_DATA_DIR``); pass a path to read another file.
    """
    taz = gpd.read_parquet(_resolve(fpath, ZONIFICACION_PARQUET)).set_index("ID_ZONAEOD")
    if drop_ap:
        access_points = ACCESS_POINTS
        taz = taz.query("~ID_ZONAEOD.isin(@access_points)")
    return taz
