"""The workflow's outputs, read back: the cleaned survey, the zone system and the TASHA tables as the processing
(``workflow/Snakefile``) wrote them under the clone's ``output/``, faster than recomputing them, and the very tables the
later stages used. A reader that finds nothing names the target that writes it.

    from eodgdl import artifacts
    tables = artifacts.eod()             # EODTables, as load_eod() returns them    (uv run snakemake -c8 eod)
    zones = artifacts.zones()            # the zones redrawn along AGEB edges        (... zones)
    units = artifacts.zone_assignment()  # every urban AGEB and rural locality's zone (... zones)
    od = artifacts.tasha_tables()        # ODTables, the zone ids as text            (... tasha)

``$EODGDL_OUTPUT_DIR`` names another root (a run with ``--config output=ROOT``).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

from eodgdl.data import REPO


def output_dir() -> Path:
    """The workflow's root: ``$EODGDL_OUTPUT_DIR``, else the clone's ``output/``."""
    if value := os.environ.get("EODGDL_OUTPUT_DIR"):
        return Path(value).expanduser().resolve()
    return REPO / "output"


def _path(relative: str, target: str) -> Path:
    path = output_dir() / relative
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: `uv run snakemake -c8 {target}` writes it")
    return path


def eod():
    """The cleaned survey (:class:`eodgdl.EODTables`) the rule ``eod_tables`` wrote, exactly as ``load_eod()`` gave it
    then. Warns when the loader now gives other tables (the manifest's digest against
    :func:`eodgdl.impute.sources.eod.survey_versions`): ``snakemake eod`` brings them up to date."""
    import pandas as pd

    from eodgdl.eod import EODTables
    from eodgdl.impute.sources.eod import survey_versions
    from eodgdl.manifest import read_manifest

    tables = EODTables(*(pd.read_parquet(_path(f"eod/{name}.parquet", "eod")) for name in EODTables._fields))
    written = read_manifest(_path("eod/manifest.json", "eod"))["versions"].get("eod_tables")
    if written != survey_versions()["eod_tables"]:
        warnings.warn(f"{output_dir() / 'eod'} holds other tables than load_eod() gives now: `uv run snakemake -c8 eod`", stacklevel=2)
    return tables


def zones():
    """The zones redrawn along AGEB edges (:func:`eodgdl.reweight.zone_system`), indexed by zone, with each zone's urban
    AGEBs, rural localities and population."""
    import geopandas as gpd

    return gpd.read_file(_path("zones/zones.gpkg", "zones"), layer="zones").set_index("zone")


def zone_assignment():
    """Every urban AGEB and rural locality with its zone and the rule that placed it
    (:func:`eodgdl.reweight.assign_units`), indexed by CVEGEO."""
    import pandas as pd

    return pd.read_parquet(_path("zones/assignment.parquet", "zones"))


def tasha_tables():
    """The TASHA tables (:class:`eodgdl.tasha.ODTables`) the rule ``tasha_build`` wrote, every zone id read as text."""
    import pandas as pd

    from eodgdl import tasha

    return tasha.ODTables(*(pd.read_csv(_path(f"tasha/od_{table}.csv", "tasha"), dtype=dict.fromkeys(tasha.zone_columns(table), str))
                            for table in tasha.tables()))
