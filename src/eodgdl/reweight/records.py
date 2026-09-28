"""The survey as the tool's record tables: integer keys, numeric attributes.

Every attribute is read from spec.yaml (``attributes``); this module only realises the
entry kinds the spec documents (constant, values, range, when, count, sum, verbatim) and
the one prose derivation, ``Cyclist``.
"""
from __future__ import annotations

import pandas as pd

from eodgdl.chains import non_trips
from eodgdl.reweight._spec import GEOGRAPHY_COLUMN, attributes, load_spec
from eodgdl.taz import load_zm_muns

PERSON = ["folio_vivienda", "folio_habitante"]
TRIP = PERSON + ["folio_viaje"]


def zone_levels():
    """The centralidad codes in the survey schema's order; a zone's index is 1 + its position."""
    from eodgdl.schemas import viv_schema

    return list(viv_schema.columns["centralidad"].dtype.type.categories)


def zone_index():
    return {code: i + 1 for i, code in enumerate(zone_levels())}


def zone_ids(viv):
    """TAZ, Zone, Municipality, Region per dwelling (int), per the `zones` section of the spec."""
    zone = viv.centralidad.astype(str).map(zone_index())
    mun = viv.municipio.astype(str).map({name: code for code, name in load_zm_muns().items()})
    if zone.isna().any() or mun.isna().any():
        raise ValueError("a dwelling's centralidad or municipio is outside the known levels")
    return pd.DataFrame(
        {"TAZ": zone * 1000 + mun, "Zone": zone, "Municipality": mun, "Region": 1},
        index=viv.index,
    ).astype(int)


def build_zones(viv):
    """(ZoneSystem, ZoneLabels): the TAZ list with its maps, and what each TAZ is."""
    ids = zone_ids(viv)
    zone_system = ids.drop_duplicates("TAZ").sort_values("TAZ").reset_index(drop=True)
    labels = (
        ids.join(viv[["centralidad", "municipio"]])
        .groupby("TAZ")
        .agg(
            centralidad=("centralidad", "first"),
            municipio=("municipio", "first"),
            CVE_MUN=("Municipality", "first"),
            sampled_dwellings=("Zone", "size"),
        )
        .reset_index()
    )
    labels["centralidad"] = labels.centralidad.astype(str)
    labels["municipio"] = labels.municipio.astype(str)
    return zone_system, labels


def _attribute(df, name, entry):
    """One attribute of a table from its own columns (no legs / trips lookups)."""
    if "constant" in entry:
        return pd.Series(float(entry["constant"]), index=df.index)
    source = entry["source"]
    if "range" in entry:
        lo, hi = entry["range"]
        inside = df[source] >= lo
        if hi is not None:
            inside &= df[source] <= hi
        return inside.astype(float)
    if "values" in entry:
        out = df[source].astype(object).map(entry["values"]).fillna(entry.get("default", 0))
        out = out.astype(float)
        if "when" in entry:
            out = out.where(df.eval(entry["when"]), 0.0)
        return out
    if isinstance(source, str):
        return df[source].astype(float)
    raise ValueError(f"{name}: cannot build from {entry}")


def build_households(viv):
    ids = zone_ids(viv)
    out = pd.DataFrame({"HouseholdID": viv.index.to_numpy(), "HouseholdTAZ": ids.TAZ.to_numpy()})
    for name, entry in attributes("households").items():
        out[name] = _attribute(viv, name, entry).to_numpy()
    return out


def _cyclists(hab, trips, legs, entry):
    """Persons with a bicycle leg on a trip whose motive is in the entry's `motives`."""
    motive = trips.motivo_viaje.reindex(legs.index.droplevel("folio_traslado"))
    hit = legs.traslado_medio.astype(str).isin(entry["modes"]).to_numpy() & motive.astype(str).isin(entry["motives"]).to_numpy()
    persons = legs.index[hit].droplevel(["folio_viaje", "folio_traslado"]).unique()
    return pd.Series(hab.index.isin(persons), index=hab.index).astype(float)


def build_people(hab, trips, legs):
    trips = trips[~non_trips(trips)]
    legs = legs[legs.index.droplevel("folio_traslado").isin(trips.index)]
    folio_viv = hab.index.get_level_values("folio_vivienda")
    folio_hab = hab.index.get_level_values("folio_habitante")
    out = pd.DataFrame({"PersonID": folio_viv * 10 + folio_hab, "HouseholdID": folio_viv})
    for name, entry in attributes("persons").items():
        if name == "Cyclist":
            values = _cyclists(hab, trips, legs, entry)
        else:
            values = _attribute(hab, name, entry)
        out[name] = values.to_numpy()
    return out


def _count(trips, legs, entry):
    per_leg = legs.traslado_medio.astype(object).map(entry["count"]).fillna(0).astype(float)
    per_trip = per_leg.groupby(level=TRIP).sum().reindex(trips.index).fillna(0.0)
    return per_trip.clip(upper=1.0) if entry.get("any") else per_trip


def build_trips(trips, legs):
    trips = trips[~non_trips(trips)].sort_index()
    legs = legs[legs.index.droplevel("folio_traslado").isin(trips.index)]
    out = pd.DataFrame({
        "HouseholdID": trips.index.get_level_values("folio_vivienda"),
        "PersonID": trips.index.get_level_values("folio_vivienda") * 10
        + trips.index.get_level_values("folio_habitante"),
        "TripID": trips.index.get_level_values("folio_viaje"),
    })
    for name, entry in attributes("trips").items():
        if "count" in entry:
            values = _count(trips, legs, entry).to_numpy()
        elif "sum" in entry:
            values = out[entry["sum"]].sum(axis=1).to_numpy()
        else:
            values = _attribute(trips, name, entry).to_numpy()
        out[name] = values
    return out


def record_geography(records, table, zone_system, geography):
    """The geography id of every record of `table`, through its household's TAZ."""
    taz = records.households.set_index("HouseholdID").HouseholdTAZ
    column = GEOGRAPHY_COLUMN[geography]
    to_geo = zone_system.set_index("TAZ")[column]
    frame = getattr(records, table)
    hh = frame.HouseholdTAZ if table == "households" else frame.HouseholdID.map(taz)
    return hh.map(to_geo)
