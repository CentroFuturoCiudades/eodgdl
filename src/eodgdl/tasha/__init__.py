"""Target schema for the travel-demand model, and how the EOD GDL 2023 fills it.

The model consumes three tables — ``od_households.csv``, ``od_people.csv``,
``od_trips.csv``. ``model_schema.yaml`` says what they must contain;
``mappings.yaml`` says which raw survey answer produces each code. Both ship
inside this subpackage and are meant to be read and edited by hand.

    from eodgdl import tasha

    tasha.build_map("Mode")             # {'A PIE': 'W', 'CAMIÓN O AUTOBÚS': 'B', ...}
    tasha.mapping("Mode")               # ...plus the override, notes and caveats
    tasha.gaps()                        # what is assumed, constant or unresolved
    tasha.check_mappings()              # mappings.yaml vs. model_schema.yaml

    od = tasha.build(load_eod("data"))  # the three model input tables
    tasha.validate_all(*od)
"""
from eodgdl.tasha.build import (
    ODTables,
    build,
    build_households,
    build_people,
    build_trips,
)
from eodgdl.tasha._schema import (
    build_map,
    check_mappings,
    column_spec,
    columns,
    domain,
    find_table,
    gaps,
    load_mappings,
    load_schema,
    mapping,
    required_columns,
    survey_columns,
    tables,
    validate,
    validate_all,
    zone_columns,
)

__all__ = [
    # The contract
    "load_schema",
    "tables",
    "columns",
    "column_spec",
    "find_table",
    "domain",
    "required_columns",
    "zone_columns",
    # The mappings
    "load_mappings",
    "mapping",
    "build_map",
    "gaps",
    "check_mappings",
    "survey_columns",
    # Building
    "ODTables",
    "build",
    "build_households",
    "build_people",
    "build_trips",
    # Validation
    "validate",
    "validate_all",
]
