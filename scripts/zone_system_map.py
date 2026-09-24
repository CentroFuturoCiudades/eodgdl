"""Save the zone-system map of reports/reweight_inputs.qmd (fig-zone-map) and the redrawn zones.

The map is drawn by the report's own chunk, read from the .qmd and run as written with
``plt.show()`` swapped for a save, so the PNG and the rendered report cannot drift. The
redrawn zones (``reweight.zone_shapes``: each centralidad polygon with every urban AGEB
moved whole to the zone ``reweight.assign_units`` gives it) go to a GeoPackage, one
feature per zone, with the assignment beside it as a CSV. Needs the ``reweight`` extra
(mxcensus fetches INEGI's census and Marco Geoestadístico layers once).

    uv run --extra reweight python scripts/zone_system_map.py [--out output/zone_system_map.png] [--dpi 300]
        [--shapes output/centralidades_redrawn.gpkg]
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "reweight_inputs.qmd"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="output/zone_system_map.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--shapes", default="output/centralidades_redrawn.gpkg")
    args = parser.parse_args()

    os.environ.setdefault("EODGDL_DATA_DIR", str(ROOT / "data"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from eodgdl import load_eod, load_zm_muns, reweight

    viv = load_eod(os.environ["EODGDL_DATA_DIR"]).viv
    chunk = re.search(r"#\| label: fig-zone-map\n(.*?)```", REPORT.read_text(), re.S)
    if chunk is None:
        raise SystemExit(f"no fig-zone-map chunk in {REPORT}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    code = chunk.group(1).replace(
        "plt.show()",
        f'plt.savefig({str(out)!r}, dpi={args.dpi}, bbox_inches="tight", facecolor="white")')
    # the names the chunk takes from the report's setup, build and zones chunks
    names = {"plt": plt, "np": np, "MUN_NAME": load_zm_muns(), "n_zones": 64, "viv": viv, "reweight": reweight}
    exec(compile(code, f"{REPORT.name}:fig-zone-map", "exec"), names)
    print(f"wrote {out}")

    assignment = reweight.assign_units(viv)
    shapes = reweight.zone_shapes(assignment)
    census = assignment[assignment.zone.notna()].groupby("zone").agg(
        agebs=("unit", lambda u: int((u == "ageb").sum())), localities=("unit", lambda u: int((u == "locality").sum())),
        POBTOT=("POBTOT", "sum"))
    shapes = shapes.join(census).reset_index()
    target = Path(args.shapes)
    target.parent.mkdir(parents=True, exist_ok=True)
    shapes.to_file(target, layer="centralidades_redrawn", driver="GPKG")
    assignment.to_csv(target.with_suffix(".csv"), float_format="%.4f")
    print(f"wrote {target} ({len(shapes)} zones) and {target.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
