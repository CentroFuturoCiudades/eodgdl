"""Rule zone_system: every urban AGEB and rural locality in its zone (reweight.assign_units) and the zones redrawn
along AGEB edges (reweight.zone_shapes), from the cleaned tables the eod_tables rule wrote, and the zone-system map
twice, plain and with the rural AGEBs drawn whole. eodgdl.artifacts.zones() and zone_assignment() read the zones back.

The map is the report's own fig-zone-map chunk (reports/reweight_inputs.qmd), read from the .qmd and run as written
with ``plt.show()`` swapped for a save, so the files and the rendered report cannot drift. The PDFs carry no creation
date, so the same map is the same bytes."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eodgdl import load_zm_muns, reweight  # noqa: E402
from eodgdl.data import ZONIFICACION_PARQUET, resolve  # noqa: E402
from eodgdl.manifest import write_manifest  # noqa: E402


def draw_map(out, viv, trips, *, rural_agebs=False):
    """Run the report's fig-zone-map chunk on ``viv`` and ``trips``, saving the figure to ``out`` (a PDF)."""
    chunk = re.search(r"#\| label: fig-zone-map\n(.*?)```", Path(snakemake.input.map_chunk).read_text(), re.S)  # noqa: F821
    if chunk is None:
        raise SystemExit(f"no fig-zone-map chunk in {snakemake.input.map_chunk}")  # noqa: F821
    code = chunk.group(1).replace(
        "plt.show()", f'plt.savefig({str(out)!r}, dpi=300, bbox_inches="tight", facecolor="white", metadata={{"CreationDate": None}})')
    # the names the chunk takes from the report's setup, build and zones chunks
    names = {"plt": plt, "np": np, "MUN_NAME": load_zm_muns(), "n_zones": 64, "viv": viv, "trips": trips, "reweight": reweight,
             "RURAL_AS_AGEB": rural_agebs}
    exec(compile(code, "reweight_inputs.qmd:fig-zone-map", "exec"), names)
    plt.close("all")
    return Path(out)


viv, trips = (pd.read_parquet(path) for path in (snakemake.input.viv, snakemake.input.trips))  # noqa: F821
assignment, shapes = reweight.zone_system(viv, trips)
problems = reweight.check_assignment(assignment, viv, trips)
if problems:
    raise SystemExit("\n".join(problems))
zones, units = Path(snakemake.output.zones), Path(snakemake.output.assignment)  # noqa: F821
shapes.reset_index().to_file(zones, layer="zones", driver="GPKG")
assignment.to_parquet(units)
maps = [draw_map(snakemake.output.map, viv, trips),  # noqa: F821
        draw_map(snakemake.output.map_rural, viv, trips, rural_agebs=True)]  # noqa: F821
write_manifest(zones.parent, "zone_system", inputs={"zones": resolve(ZONIFICACION_PARQUET)},
               upstream={"eod_tables": Path(snakemake.input.viv).parent}, outputs=[zones, units, *maps])  # noqa: F821
