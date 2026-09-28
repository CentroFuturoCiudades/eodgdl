"""Worker-level feature frame for the giro model, built directly from the cleaned EOD tables (``eodgdl.load_eod``):
person and dwelling attributes, the work-trip destination and mode, and the destination's ámbito and DENUE
establishment mix. The destination is a unit of the package's zone system (``eodgdl.reweight.zoning``): the survey
codes every trip end with its census unit, a 13-character urban AGEB or a 9-character rural AGEB (never a locality).
DENUE and the census are fetched through ``mxcensus``."""

import numpy as np
import pandas as pd

from ._config import (
    DENUE_RELEASE, DENUE_SCIAN2, DENUE_STATE_CODE, DESTINATION_SHARE_FEATURES, DWELLING_COLUMNS, EMPLOYED_CATEGORIES,
    GIRO_CLASSES, GIRO_SLUGS, KEYS, SMALL_ESTABLISHMENT_LEVELS, WORK_TRIP_PURPOSE,
)

DESTINATION_NUMERIC_FEATURES = ["dest_establecimientos_log", "dest_share_grandes"] + DESTINATION_SHARE_FEATURES


def compute_work_trip_destination(trips):
    """Most frequent destination type, destination code/zone and main mode of each person's work trips (purpose
    "Trabajar"); persons without a work trip on the survey day are absent."""
    work_trips = trips.reset_index()
    work_trips = work_trips[work_trips["motivo_viaje"] == WORK_TRIP_PURPOSE]
    mode = lambda values: values.astype(str).value_counts().index[0]
    destination = work_trips.groupby(KEYS).agg(
        destino_trabajo=("tipo_lugar_destino", mode),
        destino_cvegeo=("destino", mode),
        destino_zona=("zona_destino", mode),
        modo_trabajo=("modo_principal", mode),
    )

    return destination.reset_index()


def zone_units(tables, state=DENUE_STATE_CODE):
    """The zone system's urban AGEBs (13-character CVEGEO) and rural AGEBs (9-character key), from
    :func:`eodgdl.reweight.zoning.assign_units`: every code the survey records is one of them, in the zone it coded."""
    from eodgdl.reweight.zoning import assign_units

    units = assign_units(tables.viv, tables.trips, state=state)

    return set(units.index[units["unit"] == "ageb"]), set(units["rural_ageb"].dropna())


def _denue_aggregates(urban, state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """Establishment mix per destination unit -- an urban AGEB (13-character CVEGEO, in ``urban``) or else the rural
    AGEB DENUE places the establishment in (entity, municipality and AGEB: 9 characters) -- and per urban locality
    (9-character CVEGEO), the fallback for an urban AGEB with no establishment. A rural AGEB takes every establishment
    DENUE puts in it, also those in unpopulated places (industrial parks) that no census locality holds."""
    import mxcensus

    denue = mxcensus.load_denue(state=state, release=release)
    key = (denue["cve_ent"].astype(str).str.zfill(2) + denue["cve_mun"].astype(str).str.zfill(3)
           + denue["cve_loc"].astype(str).str.zfill(4) + denue["ageb"].astype(str).str.zfill(4))
    is_urban = key.isin(urban)
    frame = pd.DataFrame({
        "unit": key.where(is_urban, key.str[:5] + key.str[9:]),
        "giro": denue["codigo_act"].astype(str).str[:2].map(DENUE_SCIAN2),
        "large": ~denue["per_ocu"].astype(str).str.startswith(SMALL_ESTABLISHMENT_LEVELS),
    })
    unmapped = frame["giro"].isna().sum()
    assert unmapped == 0, f"{unmapped} DENUE establishments with a SCIAN sector missing from config.yaml denue_scian2"
    frame["localidad"] = key.str[:9].where(is_urban)

    def aggregate(key):
        grouped = frame.dropna(subset=[key]).groupby(key)
        table = pd.DataFrame({"dest_establecimientos_log": np.log1p(grouped.size()), "dest_share_grandes": grouped["large"].mean()})
        for giro in GIRO_CLASSES:
            table[f"dest_share_{giro}"] = grouped["giro"].apply(lambda values: (values == giro).mean())
        return table

    return aggregate("unit"), aggregate("localidad")


def add_destination_features(od, urban, rural, state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """Attach ``destino_ambito`` and the DENUE establishment mix of the work-trip destination (columns
    ``destino_cvegeo`` and ``destino_zona`` must be present; ``urban`` and ``rural`` are :func:`zone_units`). Unknown,
    airport and out-of-metro destinations keep NaN in the DENUE columns (imputed inside the pipelines) and carry the
    information in ``destino_ambito``; so does a rural AGEB with no establishment."""
    od = od.copy()
    by_unit, by_localidad = _denue_aggregates(urban, state=state, release=release)

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
    od["destino_ambito"] = ambito

    features = by_unit.reindex(code.astype(object))
    fallback = by_localidad.reindex(code.astype(object).str[:9])
    fallback[(ambito != "ageb_urbana").to_numpy()] = np.nan
    features = features.where(features.notna(), fallback.to_numpy())
    features.index = od.index
    for column in DESTINATION_NUMERIC_FEATURES:
        od[column] = features[column].astype(float)
    od.loc[~od["destino_ambito"].isin(["ageb_urbana", "ageb_rural"]), DESTINATION_NUMERIC_FEATURES] = np.nan

    return od


def build_worker_features(tables, state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """OD workers (``trabajo_semana_pasada`` in the employed categories) with every giro-model feature, the target
    ``giro`` (slug of ``giro_empresa``, NA when unobserved) and ``giro_desconocido``. Categoricals are plain strings."""
    od = tables.hab.reset_index()
    dwelling_columns = [column for column in DWELLING_COLUMNS if column not in od.columns]
    od = od.merge(tables.viv[dwelling_columns], left_on="folio_vivienda", right_index=True, how="left", validate="many_to_one")
    od = od.merge(compute_work_trip_destination(tables.trips), on=KEYS, how="left", validate="one_to_one")
    od = od[od["trabajo_semana_pasada"].isin(EMPLOYED_CATEGORIES)].copy()
    od = add_destination_features(od, *zone_units(tables, state=state), state=state, release=release)
    categorical_columns = od.columns[od.dtypes.eq("category")]
    od[categorical_columns] = od[categorical_columns].astype("string")
    unknown_labels = set(od["giro_empresa"].dropna().unique()) - set(GIRO_SLUGS)
    assert not unknown_labels, f"giro_empresa labels missing from config.yaml giro_levels: {sorted(unknown_labels)}"
    od["giro"] = od["giro_empresa"].map(GIRO_SLUGS).astype("string")
    od["giro_desconocido"] = od["giro"].isna()

    return od.reset_index(drop=True)
