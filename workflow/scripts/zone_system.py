"""Rule zone_system: every urban AGEB and rural locality in its zone (reweight.assign_units) and the zones redrawn
along AGEB edges (reweight.zone_shapes), from the cleaned tables the eod_tables rule wrote. eodgdl.artifacts.zones()
and zone_assignment() read them back."""
from pathlib import Path

import pandas as pd

from eodgdl import reweight
from eodgdl.data import ZONIFICACION_PARQUET, resolve
from eodgdl.manifest import write_manifest

viv, trips = (pd.read_parquet(path) for path in (snakemake.input.viv, snakemake.input.trips))  # noqa: F821
assignment, shapes = reweight.zone_system(viv, trips)
problems = reweight.check_assignment(assignment, viv, trips)
if problems:
    raise SystemExit("\n".join(problems))
zones, units = Path(snakemake.output.zones), Path(snakemake.output.assignment)  # noqa: F821
shapes.reset_index().to_file(zones, layer="zones", driver="GPKG")
assignment.to_parquet(units)
write_manifest(zones.parent, "zone_system", inputs={"zones": resolve(ZONIFICACION_PARQUET)},
               upstream={"eod_tables": Path(snakemake.input.viv).parent}, outputs=[zones, units])  # noqa: F821
