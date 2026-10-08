"""Build the three model input tables from the cleaned EOD GDL survey.

Every coded column is built from ``mappings.yaml`` rather than from a lookup
retyped here, so changing a mapping changes the output. What the mappings record
as ``derivation`` prose — the R/C demotion, the passenger override, the
work/school zone lookups, the daycare trips by age, H only at the household's
zone — is implemented below, and the prose is its spec. Every zone column carries the TAZ, the integer of the
AGEB or access point in ``eodgdl/taz_ids.csv`` (``eodgdl._resources.taz_of``, which raises on a code without one);
the rules compare the survey's codes before that.

The columns a mapping marks ``imputed`` (IncomeClass; EmploymentStatus and StudentStatus where unanswered,
EmploymentStatus's P, a worker's unreported Occupation) read one completed dataset of the eodgdl.impute
pipeline ``tasha`` (persons and dwellings drawn jointly: a dwelling's NSE counts its drawn workers):
``load_completed(root, draw)`` (each level's ``<root>/tasha/<level>/completions.parquet``, draw ``draw``; it refuses a
run with the aggregates switched off and one drawn on other data than the survey's now), passed as
``build(tables, completed=...)``, where every row that reads a drawn value must have one. Without it those columns
take their mapping's default (IncomeClass 7, no P, the reported giro only).

``ExpansionFactor`` is the one column the reweighting reaches (mappings.yaml's derivation): each table's own
``ponderador`` by default, or, passed as ``build(tables, expansion=...)``, TMG.SurveyReweight's household weight on every
table (``load_expansion()``: the run's ``expansion`` in config/config.yaml). The imputations never read it (they train
on the design weight), so a new weight changes nothing but this column.

The input is what ``load_eod`` returns: trip chains already cleaned by
``eodgdl.chains.clean_trip_chains`` (untimed trips imputed, mislabelled returns
recoded, mistyped start hours repaired, the 38 home-to-home rows that
duplicate an untimed return dropped). The rows that cleaning marks as not
being trips — returns home made while already at home — stay in ``trips``
under ``problemas`` and the builder leaves them out, since the
contract forbids a trip from H to H. The builder refuses a trip table with
untimed rows rather than guess at them.

    from eodgdl import load_eod, tasha

    od = tasha.build(load_eod("data"), completed=tasha.load_completed("output/impute", draw=0), expansion=tasha.load_expansion())
    tasha.validate_all(*od)
    od.households.to_csv("output/od_households.csv", index=False)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from eodgdl._resources import taz_of
from eodgdl.chains import DAYCARE, ESCORT_FROM_AGE, days_past_midnight, non_trips
from eodgdl.tasha._schema import build_map, imputed_lookups, load_mappings, mapping

PERSON = ["folio_vivienda", "folio_habitante"]
NO_ZONE = 0  # sentinel for EmploymentZone / SchoolZone (TAZ ids start at 1)
# DAYCARE, ESCORT_FROM_AGE: a Guardería trip from this age on is an escort (PurposeDestination, StudentStatus);
# load_eod's chain rules already recode it so (motivo:guarderia), and the build reads the age the same way


class ODTables(NamedTuple):
    """The three tables the model consumes."""

    households: pd.DataFrame
    people: pd.DataFrame
    trips: pd.DataFrame


def load_completed(root="output/impute", draw=0) -> dict[str, dict[str, pd.DataFrame]]:
    """One completed dataset of every eodgdl.impute pipeline a mapping reads: ``{pipeline: {level: frame}}``, the rows
    of draw ``draw`` of each level's ``<root>/<pipeline>/<level>/completions.parquet`` (every value drawn, jointly
    across the levels), indexed by the level's keys. ``frame.attrs`` holds the draw and the pipeline's provenance.

    Refuses a run with its aggregates switched off (``--no-aggregate``: the parity check, whose levels do not hold
    together) and one drawn on other data than the pipeline's sources read now (:func:`_stale_levels`: a survey file,
    a chain rule or a hand decision changed since the run), or one whose levels' EOD-trained bundles were fitted on
    different weights, or on another than the levels' sources read (eodgdl.impute.bundle.weight_conflicts)."""
    pipelines = sorted({pipeline for entries in load_mappings().values() if isinstance(entries, dict)
                        for entry in entries.values() if isinstance(entry, dict)
                        for _, pipeline, _, _ in imputed_lookups(entry)})
    completed = {}
    for pipeline in pipelines:
        folder = Path(root) / pipeline
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        if not 0 <= draw < provenance["draws"]:
            raise ValueError(f"{folder}: draw {draw} outside its {provenance['draws']} completed datasets")
        rerun = f"rerun `eodgdl impute score {pipeline} --retrained <the root of its bundles> --out {folder}`"
        if not provenance.get("aggregate", True):
            raise ValueError(f"{folder}: run with its aggregates switched off (--no-aggregate: the parity check), so its levels' draws do not hold "
                             f"together: {rerun}")
        stale = _stale_levels(provenance)
        if stale:
            raise ValueError(f"{folder}: drawn on other data than the survey's now ({'; '.join(stale)}): {rerun}")
        from eodgdl.impute.bundle import weight_conflicts

        conflicts = weight_conflicts(provenance["levels"])
        if conflicts:
            raise ValueError(f"{folder}: drawn with bundles of mixed weights ({'; '.join(conflicts)}): {rerun}")
        completed[pipeline] = {}
        for level in provenance["levels"]:
            frame = pd.read_parquet(folder / level / "completions.parquet")
            frame = frame[frame["completion"] == draw].drop(columns=["completion", "weight"])
            frame = frame.set_index([key for key in ("folio_vivienda", "folio_habitante") if key in frame.columns])
            frame.attrs = {"root": str(root), "draw": draw, "level": level, "provenance": provenance}
            completed[pipeline][level] = frame
    return completed


def _stale_levels(provenance: dict) -> list[str]:
    """The levels of a pipeline run (its provenance) whose scoring source recorded other data versions than it reads
    now, or none (eodgdl.impute.sources.changed_versions; the eodgdl version aside)."""
    from eodgdl.impute.sources import changed_versions

    stale = []
    for level, entry in provenance["levels"].items():
        if "versions" not in entry or "source" not in entry:
            stale.append(f"{level}: no data versions recorded")
        elif changed := changed_versions(entry["source"], entry["versions"]):
            stale.append(f"{level}: {', '.join(changed)} changed")
    return stale


EXPANSIONS = ("design", "tmg")


def load_expansion(name=None, data_dir=None) -> pd.Series | None:
    """The household weight the tables expand with under ``name`` (default: config/config.yaml's ``expansion``):
    None under ``design`` (each table its own ``ponderador``), TMG.SurveyReweight's household weight under ``tmg``
    (``peso`` by ``folio_vivienda``, the file its sidecar records: eodgdl.reweight.weight.read_weight)."""
    if name is None:
        from eodgdl import config

        name = config.expansion()
    if name not in EXPANSIONS:
        raise ValueError(f"expansion {name!r}: not one of {', '.join(EXPANSIONS)}")
    if name == "design":
        return None
    from eodgdl.reweight.weight import read_weight

    return read_weight(data_dir)


def _expansion(own: pd.Series, index: pd.Index, expansion=None) -> pd.Series:
    """ExpansionFactor of the rows of ``index`` (mappings.yaml's derivation): the table's own ``ponderador``
    (``own``), or with ``expansion`` (:func:`load_expansion`) their household's weight, which every household must have."""
    if expansion is None:
        return own.astype(float)
    households = index.get_level_values("folio_vivienda")
    values = pd.Series(expansion.reindex(households).to_numpy(), index=index, dtype=float)
    if values.isna().any():
        missing = pd.unique(households[values.isna().to_numpy()])
        raise ValueError(f"no expansion weight for {len(missing):,} households (e.g. folio_vivienda {', '.join(map(str, missing[:5]))}): "
                         "a weight fitted on other household records than this survey's")
    return values


def _imputed(completed, pipeline: str, column: str, index: pd.Index, needed=None) -> pd.Series:
    """The completed ``column`` of ``pipeline`` (from the level whose completions hold it) for the rows of ``index``
    (NA where the pipeline has no value, or with no ``completed``). ``needed`` (a boolean mask over ``index``): the
    rows the build reads it for, each of which must have one (a completed dataset of other rows, from a stale or
    partial run, fails here instead of leaving them to the mapping's default)."""
    if completed is None:
        return pd.Series(pd.NA, index=index, dtype="object")
    frames = [frame for frame in completed[pipeline].values() if column in frame.columns]
    if len(frames) != 1:
        raise ValueError(f"{pipeline}: {len(frames)} levels hold the column {column!r}, not one")
    [frame] = frames
    keys = index.to_frame(index=False)[list(frame.index.names)]
    values = pd.Series(frame[column].reindex(pd.MultiIndex.from_frame(keys) if len(frame.index.names) > 1 else keys.iloc[:, 0]).to_numpy(),
                       index=index, dtype="object")
    if needed is not None:
        missing = np.asarray(needed, dtype=bool) & values.isna().to_numpy()
        if missing.any():
            raise ValueError(f"{pipeline}.{column}: no completed value for {int(missing.sum()):,} rows that read one (e.g. {list(index[missing][:3])}): "
                             "the completed datasets hold other rows than this survey's; rerun the pipeline")
    return values


def _imputed_codes(column: str, index: pd.Index, completed, which="imputed", needed=None) -> pd.Series:
    """The codes the mapping's imputed lookup (``which``: the entry's own, or its override's) gives the rows of
    ``index``; NA where the completed dataset has no value (``needed``: the rows that must have one, :func:`_imputed`)."""
    [(_, pipeline, name, values)] = [lookup for lookup in imputed_lookups(mapping(column)) if lookup[0] == which]
    return _imputed(completed, pipeline, name, index, needed).map(values)


def _household_ids(viv: pd.DataFrame) -> pd.Series:
    """0-based dense counter over the sorted folio_vivienda index."""
    viv = viv.sort_index()
    return pd.Series(range(len(viv)), index=viv.index, name="HouseholdId")


def _purposes(trips: pd.DataFrame, viv: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(destination purpose, first-trip origin purpose), before the R/C demotion.

    Either is H only where the survey's answer maps to H and the trip end is the
    household's zone: the first trip leaves from it (see PurposeOrigin's derivation),
    a return home reaches it (see PurposeDestination's).
    """
    home = viv.ageb.astype(str).reindex(trips.index.get_level_values("folio_vivienda")).to_numpy()
    destination = (trips.motivo_viaje.map(build_map("PurposeDestination"))
                        .fillna(mapping("PurposeDestination")["default"]))
    destination = destination.mask((destination == "H") & (trips.destino.astype(str).to_numpy() != home), "O")
    origin = (trips.tipo_lugar_origen.map(build_map("PurposeOrigin"))
                   .fillna(mapping("PurposeOrigin")["default"]))
    origin = origin.mask((origin == "H") & (trips.origen.astype(str).to_numpy() != home), "O")
    return destination, origin


def build_households(viv: pd.DataFrame, hab: pd.DataFrame, completed=None, expansion=None) -> pd.DataFrame:
    """od_households.csv, one row per dwelling (``completed``: :func:`load_completed`; ``expansion``:
    :func:`load_expansion`)."""
    veh, veh_default = build_map("Vehicles"), mapping("Vehicles")["default"]
    # Both lookups carry a default, so an answer the survey adds later widens the
    # column instead of raising an opaque cast error mid-build; `tasha check`
    # reports the unmapped level, and validate() the value it produced.
    reported = (viv.personas_en_vivienda.map(build_map("NumberOfPersons"))
                   .fillna(mapping("NumberOfPersons")["default"]).astype(int))
    observed = hab.groupby("folio_vivienda").size().reindex(viv.index).fillna(0).astype(int)

    return pd.DataFrame({
        "HouseholdId": _household_ids(viv),
        "HouseholdZone": taz_of(viv.ageb.astype(str)),
        # Reported size, raised to the observed member count when that is larger.
        "NumberOfPersons": np.maximum(reported, observed),
        "DwellingType": mapping("DwellingType")["constant"],
        "Vehicles": (viv.n_autos_camionetas.map(veh).fillna(veh_default).astype(int)
                     + viv.n_motos.map(veh).fillna(veh_default).astype(int)),
        # the AMAI level of the pipeline's completed dataset (its dwellings: the chain nse given the drawn workers)
        "IncomeClass": (_imputed_codes("IncomeClass", viv.index, completed, needed=np.ones(len(viv), dtype=bool))
                           .fillna(mapping("IncomeClass")["default"]).astype(int)),
        "ExpansionFactor": _expansion(viv.ponderador, viv.index, expansion),
    }).sort_values("HouseholdId").reset_index(drop=True)


def build_people(hab: pd.DataFrame, trips: pd.DataFrame, viv: pd.DataFrame, completed=None, expansion=None) -> pd.DataFrame:
    """od_people.csv, one row per person in ``hab`` (``completed``: :func:`load_completed`; ``expansion``:
    :func:`load_expansion`)."""
    # the answer, else the drawn situation (a person who did not answer), else the default; with a completed dataset
    # every row that reads a drawn value must have one (_imputed's needed)
    answered = hab.trabajo_semana_pasada.map(build_map("EmploymentStatus")).astype("object")
    employment = (answered.fillna(_imputed_codes("EmploymentStatus", hab.index, completed, needed=answered.isna()))
                     .fillna(mapping("EmploymentStatus")["default"]))
    # a worker's code from the drawn informality (P: informal), where the completed dataset has one
    informal = _imputed_codes("EmploymentStatus", hab.index, completed, "override imputed", needed=employment.eq("F"))
    employment = employment.mask((employment == "F") & informal.notna(), informal)
    # The reported giro, else the drawn one; O for exactly the non-workers, as the schema requires.
    reported = hab.giro_empresa.map(build_map("Occupation")).astype("object")
    occupation = (reported.fillna(_imputed_codes("Occupation", hab.index, completed, needed=reported.isna() & employment.ne("O")))
                     .fillna(mapping("Occupation")["default"])
                     .mask(employment == "O", "O"))

    def first_destination(motives: list[str]) -> pd.Series:
        matching = trips[trips.motivo_viaje.isin(motives)]
        first = matching.groupby(level=PERSON).destino.first().astype(str)
        return taz_of(first).reindex(hab.index).fillna(NO_ZONE).astype(int)

    age = hab.edad.reindex(trips.index.droplevel("folio_viaje")).to_numpy()
    made_school_trip = (((trips.motivo_viaje == "Estudiar")
                         | ((trips.motivo_viaje == DAYCARE) & (age < ESCORT_FROM_AGE)))
                             .groupby(level=PERSON).any()
                             .reindex(hab.index).fillna(False))
    unanswered = hab.trabajo_semana_pasada.isna()
    drawn_student = (unanswered & _imputed_codes("StudentStatus", hab.index, completed, needed=unanswered).eq("S")).fillna(False)
    student = ((hab.ocupacion == "Estudiante")
               | (hab.trabajo_semana_pasada == "Es estudiante")
               | made_school_trip
               | drawn_student)

    return pd.DataFrame({
        "HouseholdId": _household_ids(viv).reindex(
            hab.index.get_level_values("folio_vivienda")).to_numpy(),
        "PersonNumber": hab.index.get_level_values("folio_habitante"),
        "Age": hab.edad.to_numpy(),
        "Sex": hab.sexo_nacimiento.map(build_map("Sex")).to_numpy(),
        "License": np.where(hab.edad >= 18, "Y", "N"),
        "TransitPass": mapping("TransitPass")["constant"],
        "EmploymentStatus": employment.to_numpy(),
        "Occupation": occupation.to_numpy(),
        "FreeParking": mapping("FreeParking")["constant"],
        "StudentStatus": np.where(student, "S", "O"),
        "EmploymentZone": first_destination(["Trabajar"]).to_numpy(),
        "SchoolZone": first_destination(["Estudiar"]).to_numpy(),
        "ExpansionFactor": _expansion(hab.ponderador, hab.index, expansion).to_numpy(),
    })


def build_trips(
    trips: pd.DataFrame, legs: pd.DataFrame, viv: pd.DataFrame, hab: pd.DataFrame | None = None, expansion=None
) -> pd.DataFrame:
    """od_trips.csv, one row per trip, in chain (folio_viaje) order.

    ``hab`` gives the ages that tell a daycare trip (school) from an escort to one
    (F); without it every Guardería trip keeps the lookup's S. ``expansion``
    (:func:`load_expansion`): the household weight every trip carries instead of
    its own ``ponderador``.

    The rows ``load_eod`` marked as non-trips (``eodgdl.chains.non_trips``) are
    left out; ``TripNumber`` is renumbered over them and over the gaps in
    ``folio_viaje``. ``StartTime`` counts past 2400 once the person's day has
    passed midnight (``eodgdl.chains.days_past_midnight``).
    """
    untimed = int((trips.hora_inicio_h.isna() | trips.hora_inicio_m.isna()).sum())
    if untimed:
        raise ValueError(
            f"{untimed:,} trips have no start time; load_eod() imputes them "
            "(clean_chains=True) — pass its tables rather than the survey as shipped"
        )
    trips = trips[~non_trips(trips)].sort_index()

    # Mode: the modo_principal lookup, then the passenger override.
    mode = trips.modo_principal.map(build_map("Mode"))
    override = mapping("Mode")["override"]
    is_auto = trips.modo_principal.isin(["AUTOMÓVIL PARTICULAR", "MOTOCICLETA"])
    passenger = trips[override["source"]] == "Acompañante"
    mode = mode.mask(is_auto & passenger, override["values"]["Acompañante"])

    # Purpose: the motivo_viaje lookup, a daycare trip from ESCORT_FROM_AGE on as an
    # escort, then demote repeat work/school trips, ranked in chain order.
    purpose, first_trip = _purposes(trips, viv)
    if hab is not None:
        age = hab.edad.reindex(trips.index.droplevel("folio_viaje")).to_numpy()
        purpose = purpose.mask((trips.motivo_viaje == DAYCARE).to_numpy() & (age >= ESCORT_FROM_AGE), "F")
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
        # Renumbered over the non-trips left out and the gaps in folio_viaje.
        "TripNumber": trips.groupby(level=PERSON).cumcount().to_numpy() + 1,
        # hhmm on the diary's day, past 2400 from the trip where the day passes midnight on
        "StartTime": (trips.hora_inicio_h * 100 + trips.hora_inicio_m
                      + 2400 * days_past_midnight(trips)).astype(int).to_numpy(),
        "Mode": mode.to_numpy(),
        "PurposeOrigin": origin.to_numpy(),
        "ZoneOrigin": taz_of(trips.origen.astype(str)).to_numpy(),
        "PurposeDestination": destination.to_numpy(),
        "ZoneDestination": taz_of(trips.destino.astype(str)).to_numpy(),
        "ExpansionFactor": _expansion(trips.ponderador, trips.index, expansion).to_numpy(),
        "Duration": duration.astype(int).to_numpy(),
    })


def build(tables, completed=None, expansion=None) -> ODTables:
    """Build all three tables from an :class:`~eodgdl.EODTables` as ``load_eod`` returns it, with the imputed
    columns from ``completed`` (:func:`load_completed`; without it they take their mapping's default) and
    ``ExpansionFactor`` from ``expansion`` (:func:`load_expansion`: TMG's household weight on every table; without it
    each table's own ``ponderador``)."""
    viv, hab, trips, legs = tables
    return ODTables(
        build_households(viv, hab, completed, expansion),
        build_people(hab, trips, viv, completed, expansion),
        build_trips(trips, legs, viv, hab, expansion),
    )
