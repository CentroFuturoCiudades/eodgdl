"""eodgdl — IMEPLAN Guadalajara EOD 2023 origin-destination survey loader and TAZ tools."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eodgdl")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

from eodgdl.eod import EODTables, clean_eod, load_eod, rename_imeplan
from eodgdl.taz import (
    load_imeplan_agebs,
    load_mtaz,
    load_taz,
    load_zm_muns,
    zone_system_report,
)
from eodgdl.schemas import hab_schema, trips_schema, viv_schema
from eodgdl._resources import imeplan_rename_map
from eodgdl import data, tasha

__all__ = [
    "__version__",
    # Survey loader
    "EODTables",
    "load_eod",
    "rename_imeplan",
    "clean_eod",
    # Zone system
    "load_zm_muns",
    "load_taz",
    "load_mtaz",
    "load_imeplan_agebs",
    "zone_system_report",
    # Schemas
    "viv_schema",
    "hab_schema",
    "trips_schema",
    # Config / data
    "imeplan_rename_map",
    "data",
    # Model output schema
    "tasha",
]
