"""Build the three model input tables from the cleaned EOD GDL survey.

Every coded column is built from ``mappings.yaml`` rather than from a lookup
retyped here, so changing a mapping changes the output. What the mappings record
as ``derivation`` prose — the R/C demotion, the passenger override, the
work/school zone lookups — is implemented below, and the prose is its spec.

The input is what ``load_eod`` returns: trip chains already cleaned by
``eodgdl.eod.clean_trip_chains`` (persons with an untimed trip excluded,
mislabelled returns recoded, returns home made from home dropped, 12-hour-clock
start times moved). The builder refuses a trip table with untimed rows rather
than guess at them.

    from eodgdl import load_eod, tasha

    od = tasha.build(load_eod("data"))
    tasha.validate_all(*od)
    od.households.to_csv("output/od_households.csv", index=False)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from eodgdl.tasha._schema import build_map, mapping

PERSON = ["folio_vivienda", "folio_habitante"]
NO_ZONE = "0"  # sentinel for EmploymentZone / SchoolZone


class ODTables(NamedTuple):
    """The three tables the model consumes."""

    households: pd.DataFrame
    people: pd.DataFrame
    trips: pd.DataFrame


def _household_ids(viv: pd.DataFrame) -> pd.Series:
    """0-based dense counter over the sorted folio_vivienda index."""
    viv = viv.sort_index()
    return pd.Series(range(len(viv)), index=viv.index, name="HouseholdId")


def _purposes(trips: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(destination purpose, first-trip origin purpose), before the R/C demotion."""
    destination = (trips.motivo_viaje.map(build_map("PurposeDestination"))
                        .fillna(mapping("PurposeDestination")["default"]))
    origin = (trips.tipo_lugar_origen.map(build_map("PurposeOrigin"))
                   .fillna(mapping("PurposeOrigin")["default"]))
    return destination, origin


def build_households(viv: pd.DataFrame, hab: pd.DataFrame) -> pd.DataFrame:
    """od_households.csv, one row per dwelling."""
    veh, veh_default = build_map("Vehicles"), mapping("Vehicles")["default"]
    # Both lookups carry a default, so an answer the survey adds later widens the
    # column instead of raising an opaque cast error mid-build; `tasha check`
    # reports the unmapped level, and validate() the value it produced.
    reported = (viv.personas_en_vivienda.map(build_map("NumberOfPersons"))
                   .fillna(mapping("NumberOfPersons")["default"]).astype(int))
    observed = hab.groupby("folio_vivienda").size().reindex(viv.index).fillna(0).astype(int)

    return pd.DataFrame({
        "HouseholdId": _household_ids(viv),
        "HouseholdZone": viv.ageb.astype(str),
        # Reported size, raised to the observed member count when that is larger.
        "NumberOfPersons": np.maximum(reported, observed),
        "DwellingType": mapping("DwellingType")["constant"],
        "Vehicles": (viv.n_autos_camionetas.map(veh).fillna(veh_default).astype(int)
                     + viv.n_motos.map(veh).fillna(veh_default).astype(int)),
        "IncomeClass": (viv.ingreso_mensual_hogar.map(build_map("IncomeClass"))
                           .fillna(mapping("IncomeClass")["default"]).astype(int)),
        "ExpansionFactor": viv.ponderador.astype(float),
    }).sort_values("HouseholdId").reset_index(drop=True)


def build_people(hab: pd.DataFrame, trips: pd.DataFrame, viv: pd.DataFrame) -> pd.DataFrame:
    """od_people.csv, one row per person in ``hab``."""
    employment = (hab.trabajo_semana_pasada.map(build_map("EmploymentStatus"))
                     .fillna(mapping("EmploymentStatus")["default"]))
    # Occupation is O for exactly the non-workers, as the schema requires.
    occupation = (hab.giro_empresa.map(build_map("Occupation"))
                     .fillna(mapping("Occupation")["default"])
                     .mask(employment == "O", "O"))

    def first_destination(motives: list[str]) -> pd.Series:
        matching = trips[trips.motivo_viaje.isin(motives)]
        return (matching.groupby(level=PERSON).destino.first()
                        .reindex(hab.index).fillna(NO_ZONE).astype(str))

    made_school_trip = (trips.motivo_viaje.isin(["Estudiar", "Guardería"])
                             .groupby(level=PERSON).any()
                             .reindex(hab.index).fillna(False))
    student = ((hab.ocupacion == "Estudiante")
               | (hab.trabajo_semana_pasada == "Es estudiante")
               | made_school_trip)

    return pd.DataFrame({
        "HouseholdId": _household_ids(viv).reindex(
            hab.index.get_level_values("folio_vivienda")).to_numpy(),
        "PersonNumber": hab.index.get_level_values("folio_habitante"),
        "Age": hab.edad.to_numpy(),
        "Sex": hab.sexo_nacimiento.map(build_map("Sex")).to_numpy(),
        "License": np.where(hab.edad >= 18, "Y", "N"),
        "TransitPass": mapping("TransitPass")["constant"],
        "EmploymentStatus": employment.to_numpy(),
        "Formality": mapping("Formality")["constant"],
        "Occupation": occupation.to_numpy(),
        "FreeParking": mapping("FreeParking")["constant"],
        "StudentStatus": np.where(student, "S", "O"),
        "EmploymentZone": first_destination(["Trabajar"]).to_numpy(),
        "SchoolZone": first_destination(["Estudiar"]).to_numpy(),
        "ExpansionFactor": hab.ponderador.astype(float).to_numpy(),
    })


def build_trips(trips: pd.DataFrame, legs: pd.DataFrame, viv: pd.DataFrame) -> pd.DataFrame:
    """od_trips.csv, one row per trip, in chain (folio_viaje) order."""
    untimed = int((trips.hora_inicio_h.isna() | trips.hora_inicio_m.isna()).sum())
    if untimed:
        raise ValueError(
            f"{untimed:,} trips have no start time; load_eod() excludes their persons "
            "(clean_chains=True) — pass its tables rather than the survey as shipped"
        )
    trips = trips.sort_index()

    # Mode: the modo_principal lookup, then the passenger override.
    mode = trips.modo_principal.map(build_map("Mode"))
    override = mapping("Mode")["override"]
    is_auto = trips.modo_principal.isin(["AUTOMÓVIL PARTICULAR", "MOTOCICLETA"])
    passenger = trips[override["source"]] == "Acompañante"
    mode = mode.mask(is_auto & passenger, override["values"]["Acompañante"])

    # Purpose: the motivo_viaje lookup, then demote repeat work/school trips,
    # ranked in chain order.
    purpose, first_trip = _purposes(trips)
    repeat = trips.assign(_p=purpose).groupby(PERSON + ["_p"]).cumcount() > 0
    destination = (purpose.mask((purpose == "W") & repeat, "R")
                          .mask((purpose == "S") & repeat, "C"))

    # Origin purpose: trip 1 from the reported place, later trips from the
    # previous trip's destination purpose BEFORE the demotion, so R and C
    # never reach this column.
    origin = purpose.groupby(level=PERSON).shift(1).fillna(first_trip)

    duration = legs.groupby(level=[0, 1, 2]).traslado_min.sum().reindex(trips.index)

    return pd.DataFrame({
        "HouseholdId": _household_ids(viv).reindex(
            trips.index.get_level_values("folio_vivienda")).to_numpy(),
        "PersonNumber": trips.index.get_level_values("folio_habitante"),
        # Renumbered: load_eod's chain cleaning leaves gaps in folio_viaje.
        "TripNumber": trips.groupby(level=PERSON).cumcount().to_numpy() + 1,
        "StartTime": (trips.hora_inicio_h * 100 + trips.hora_inicio_m).astype(int).to_numpy(),
        "Mode": mode.to_numpy(),
        "PurposeOrigin": origin.to_numpy(),
        "ZoneOrigin": trips.origen.astype(str).to_numpy(),
        "PurposeDestination": destination.to_numpy(),
        "ZoneDestination": trips.destino.astype(str).to_numpy(),
        "ExpansionFactor": trips.ponderador.astype(float).to_numpy(),
        "Duration": duration.astype(int).to_numpy(),
    })


def build(tables) -> ODTables:
    """Build all three tables from an :class:`~eodgdl.EODTables` as ``load_eod`` returns it."""
    viv, hab, trips, legs = tables
    return ODTables(
        build_households(viv, hab),
        build_people(hab, trips, viv),
        build_trips(trips, legs, viv),
    )
