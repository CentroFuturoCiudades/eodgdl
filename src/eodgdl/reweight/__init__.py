"""Inputs for TMG.SurveyReweight: a single household weight from the survey and the census.

The survey ships three nested expansion factors; the tool computes one per household by
matching record attributes to census targets. ``spec.yaml`` says what every attribute and
target is; ``records.py`` builds the record tables, ``targets.py`` the targets, ``check.py``
reads a written set back the way the tool will.

    from eodgdl import load_eod, reweight

    files = reweight.build(load_eod("data"), data_dir="data")
    reweight.write(files, "output/reweight")
    reweight.check("output/reweight")        # problems: must be empty

From the shell::

    eodgdl reweight build --data data --out output/reweight
    eodgdl reweight check output/reweight
"""
from eodgdl.reweight._spec import (
    attributes,
    check_spec,
    constraint_index,
    constraints,
    load_spec,
    matching_attributes,
)
from eodgdl.reweight.build import ReweightFiles, build, diagnostic, write, years
from eodgdl.reweight.check import check
from eodgdl.reweight.records import (
    build_households,
    build_people,
    build_trips,
    build_zones,
    zone_ids,
)
from eodgdl.reweight.targets import (
    Conapo,
    build_constraints,
    census_universe,
    coverage,
    crosswalk,
    load_conapo,
    reconcile,
)

__all__ = [
    "load_spec", "attributes", "constraints", "matching_attributes", "constraint_index", "check_spec",
    "ReweightFiles", "build", "write", "diagnostic", "years",
    "check",
    "build_zones", "zone_ids", "build_households", "build_people", "build_trips",
    "crosswalk", "census_universe", "reconcile", "coverage", "load_conapo", "Conapo", "build_constraints",
]
