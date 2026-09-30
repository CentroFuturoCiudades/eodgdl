"""Worker-level feature frame for the giro model, built from the cleaned EOD tables (``eodgdl.load_eod``) by the
engine (:func:`eodgdl.impute.features.build_frame`): the ``eod.workers`` source (person and dwelling attributes),
the ``eod.work_trip`` builder (work-trip destination and mode) and the ``giro.destination`` builder here (the
destination's ámbito and DENUE establishment mix). The destination is a unit of the package's zone system
(``eodgdl.reweight.zoning``): the survey codes every trip end with its census unit, a 13-character urban AGEB or a
9-character rural AGEB (never a locality). DENUE and the census are fetched through ``mxcensus``."""

import numpy as np
import pandas as pd

from eodgdl.impute.features import build_frame, builder_config, register_builder
from eodgdl.impute.sources import Context
from eodgdl.impute.sources.eod import work_trip_destination

from ._config import DENUE_RELEASE, DENUE_STATE_CODE, KEYS, TASK, WORK_TRIP_PURPOSE


def compute_work_trip_destination(trips):
    """Most frequent destination type, destination code/zone and main mode of each person's work trips (purpose
    "Trabajar"); persons without a work trip on the survey day are absent."""
    return work_trip_destination(trips, WORK_TRIP_PURPOSE, KEYS)


def zone_units(tables, state=DENUE_STATE_CODE):
    """The zone system's urban AGEBs (13-character CVEGEO) and rural AGEBs (9-character key), from
    :func:`eodgdl.reweight.zoning.assign_units`: every code the survey records is one of them, in the zone it coded."""
    from eodgdl.reweight.zoning import assign_units

    units = assign_units(tables.viv, tables.trips, state=state)

    return set(units.index[units["unit"] == "ageb"]), set(units["rural_ageb"].dropna())


def _denue_aggregates(urban, config):
    """Establishment mix per destination unit -- an urban AGEB (13-character CVEGEO, in ``urban``) or else the rural
    AGEB DENUE places the establishment in (entity, municipality and AGEB: 9 characters) -- and per urban locality
    (9-character CVEGEO), the fallback for an urban AGEB with no establishment. A rural AGEB takes every establishment
    DENUE puts in it, also those in unpopulated places (industrial parks) that no census locality holds.

    The shares -- of large establishments and of each SCIAN sector (``denue_sectors``: two-digit code -> column
    suffix) -- weight each establishment by its staff (``employment_by_size``, a value per DENUE ``per_ocu`` band), or
    count it once with ``weighting: establishments``."""
    import mxcensus

    sectors = {str(code): str(suffix) for code, suffix in config["denue_sectors"].items()}
    denue = mxcensus.load_denue(state=config["state_code"], release=str(config["denue_release"]))
    key = (denue["cve_ent"].astype(str).str.zfill(2) + denue["cve_mun"].astype(str).str.zfill(3)
           + denue["cve_loc"].astype(str).str.zfill(4) + denue["ageb"].astype(str).str.zfill(4))
    is_urban = key.isin(urban)
    size = denue["per_ocu"].astype(str)
    staff = size.map({str(band): float(value) for band, value in config["employment_by_size"].items()})
    assert staff.notna().all(), f"DENUE size bands missing from the giro task's employment_by_size: {sorted(size[staff.isna()].unique())}"
    weighting = config["weighting"]
    assert weighting in ("employment", "establishments"), f"giro.destination weighting {weighting!r}: employment or establishments"
    frame = pd.DataFrame({
        "unit": key.where(is_urban, key.str[:5] + key.str[9:]),
        "sector": denue["codigo_act"].astype(str).str[:2].map(sectors),
        "large": ~size.str.startswith(tuple(config["small_establishment_levels"])),
        "weight": staff if weighting == "employment" else 1.0,
    })
    unmapped = frame["sector"].isna().sum()
    assert unmapped == 0, f"{unmapped} DENUE establishments with a SCIAN sector missing from the giro task's denue_sectors"
    frame["localidad"] = key.str[:9].where(is_urban)
    shares = {"dest_share_grandes": frame["large"], **{f"dest_share_scian_{suffix}": frame["sector"] == suffix for suffix in dict.fromkeys(sectors.values())}}

    def aggregate(key):
        total = frame.groupby(key)["weight"].sum()
        table = pd.DataFrame({"dest_establecimientos_log": np.log1p(frame.groupby(key).size())})
        for column, member in shares.items():
            table[column] = (member * frame["weight"]).groupby(frame[key]).sum() / total
        return table

    return aggregate("unit"), aggregate("localidad")


def _destination_columns(od, urban, rural, config):
    by_unit, by_localidad = _denue_aggregates(urban, config)
    numeric = list(by_unit.columns)

    code = od["destino_cvegeo"].astype("string")
    zona = od["destino_zona"].astype("string")
    ambito = pd.Series("desconocido", index=od.index, dtype=object)
    ambito[code.isin(urban).fillna(False).to_numpy()] = "ageb_urbana"
    ambito[code.isin(rural).fillna(False).to_numpy()] = "ageb_rural"
    ambito[zona.fillna("").str.startswith("Acceso")] = "fuera_zm"
    ambito[zona.eq("Aeropuerto").fillna(False)] = "aeropuerto"
    ambito[code.isna()] = "desconocido"
    unplaced = code[(ambito == "desconocido") & code.notna()]
    assert unplaced.empty, f"work destinations outside the zone system: {sorted(unplaced.unique())}"
    columns = pd.DataFrame({"destino_ambito": ambito}, index=od.index)

    features = by_unit.reindex(code.astype(object))
    fallback = by_localidad.reindex(code.astype(object).str[:9])
    fallback[(ambito != "ageb_urbana").to_numpy()] = np.nan
    features = features.where(features.notna(), fallback.to_numpy())
    features.index = od.index
    for column in numeric:
        columns[column] = features[column].astype(float)
    columns.loc[~columns["destino_ambito"].isin(["ageb_urbana", "ageb_rural"]), numeric] = np.nan

    return columns


def add_destination_features(od, urban, rural, state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """Attach ``destino_ambito`` and the DENUE establishment mix of the work-trip destination (columns
    ``destino_cvegeo`` and ``destino_zona`` must be present; ``urban`` and ``rural`` are :func:`zone_units`). Unknown,
    airport and out-of-metro destinations keep NaN in the DENUE columns (imputed inside the pipelines) and carry the
    information in ``destino_ambito``; so does a rural AGEB with no establishment."""
    config = {**builder_config(TASK, "giro.destination"), "state_code": state, "denue_release": release}
    od = od.copy()
    columns = _destination_columns(od, urban, rural, config)
    for column in columns.columns:
        od[column] = columns[column]

    return od


def _destination_versions(context, config):
    from importlib import resources
    from importlib.metadata import version
    from pathlib import Path

    from eodgdl.impute.sources import files_digest
    from eodgdl.impute.sources.eod import survey_versions

    package = Path(str(resources.files("eodgdl")))
    return {**survey_versions(), "mxcensus": version("mxcensus"), "zoning": files_digest([package / "reweight" / "zoning.py", package / "taz.py"])}


@register_builder("giro.destination", versions=_destination_versions)
def destination(frame, context, config, spec):
    """``destino_ambito`` and the DENUE establishment mix (``dest_establecimientos_log``, ``dest_share_grandes``,
    ``dest_share_scian_<sector>``) of each row's work-trip destination."""
    urban, rural = zone_units(context.eod(), state=config["state_code"])
    return _destination_columns(frame, urban, rural, config)


def build_worker_features(tables, state=DENUE_STATE_CODE, release=DENUE_RELEASE, context=None):
    """OD workers (``trabajo_semana_pasada`` in the employed categories) with every giro-model feature, the target
    ``giro`` (slug of ``giro_empresa``, NA when unobserved) and ``giro_desconocido``. Categoricals are plain strings.
    With ``tables=None`` the survey is loaded here and the frame read from (or written to) the feature cache."""
    context = context or Context(tables=tables)
    overrides = {"giro.destination": {"state_code": state, "denue_release": str(release)}}

    return build_frame(TASK, context, overrides=overrides).frame
