"""Worker-level feature frame for the giro model, built directly from the cleaned EOD tables (``eodgdl.load_eod``):
person and dwelling attributes, the work-trip destination and mode, and the destination's ámbito and DENUE
establishment mix (DENUE and the Marco Geoestadístico are fetched through ``mxcensus``)."""

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


def _denue_aggregates(state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """Per-AGEB (13-character CVEGEO) and per-locality (9-character) establishment mix from DENUE."""
    import mxcensus

    denue = mxcensus.load_denue(state=state, release=release)
    frame = pd.DataFrame({
        "ageb": denue["cve_ent"].astype(str).str.zfill(2) + denue["cve_mun"].astype(str).str.zfill(3)
        + denue["cve_loc"].astype(str).str.zfill(4) + denue["ageb"].astype(str).str.zfill(4),
        "giro": denue["codigo_act"].astype(str).str[:2].map(DENUE_SCIAN2),
        "large": ~denue["per_ocu"].astype(str).str.startswith(SMALL_ESTABLISHMENT_LEVELS),
    })
    unmapped = frame["giro"].isna().sum()
    assert unmapped == 0, f"{unmapped} DENUE establishments with a SCIAN sector missing from config.yaml denue_scian2"
    frame["localidad"] = frame["ageb"].str[:9]

    def aggregate(key):
        grouped = frame.groupby(key)
        table = pd.DataFrame({"dest_establecimientos_log": np.log1p(grouped.size()), "dest_share_grandes": grouped["large"].mean()})
        for giro in GIRO_CLASSES:
            table[f"dest_share_{giro}"] = grouped["giro"].apply(lambda values: (values == giro).mean())
        return table

    return aggregate("ageb"), aggregate("localidad")


def _destination_crosswalk(state=DENUE_STATE_CODE):
    """IMEPLAN destination code -> INEGI CVEGEO (via the eodgdl zone system) and the set of rural localities."""
    import eodgdl
    import mxcensus

    agebs = eodgdl.load_imeplan_agebs(eodgdl.load_taz())
    crosswalk = agebs[["CVEGEO_EOD", "CVEGEO"]].astype(str).drop_duplicates("CVEGEO_EOD").set_index("CVEGEO_EOD")["CVEGEO"]
    _, mg_loc_ageb = mxcensus.load_mg_census(state=state)
    marco = mg_loc_ageb.reset_index()
    rural = set(marco.loc[marco["AMBITO"].astype(str) == "Rural", "CVEGEO"].astype(str))

    return crosswalk, rural


def add_destination_features(od, state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """Attach ``destino_ambito`` and the DENUE establishment mix of the work-trip destination (columns
    ``destino_cvegeo`` and ``destino_zona`` must be present). Unknown, airport and out-of-metro destinations keep
    NaN in the DENUE columns (imputed inside the pipelines) and carry the information in ``destino_ambito``."""
    od = od.copy()
    crosswalk, rural = _destination_crosswalk(state=state)
    by_ageb, by_localidad = _denue_aggregates(state=state, release=release)

    code = od["destino_cvegeo"].astype("string")
    zona = od["destino_zona"].astype("string")
    resolved = code.map(crosswalk).fillna(code.where(code.str.len() == 13))
    ambito = pd.Series("desconocido", index=od.index, dtype=object)
    ambito[resolved.notna() & (resolved.str.len() == 13)] = "ageb_urbana"
    ambito[resolved.isin(rural)] = "localidad_rural"
    ambito[zona.fillna("").str.startswith("Acceso")] = "fuera_zm"
    ambito[zona.eq("Aeropuerto")] = "aeropuerto"
    ambito[code.isna()] = "desconocido"
    od["destino_ambito"] = ambito

    features = by_ageb.reindex(resolved.astype(object))
    fallback = by_localidad.reindex(resolved.astype(object).str[:9])
    features = features.where(features.notna(), fallback.to_numpy())
    features.index = od.index
    for column in DESTINATION_NUMERIC_FEATURES:
        od[column] = features[column].astype(float)
    od.loc[~od["destino_ambito"].isin(["ageb_urbana", "localidad_rural"]), DESTINATION_NUMERIC_FEATURES] = np.nan

    return od


def build_worker_features(tables, state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """OD workers (``trabajo_semana_pasada`` in the employed categories) with every giro-model feature, the target
    ``giro`` (slug of ``giro_empresa``, NA when unobserved) and ``giro_desconocido``. Categoricals are plain strings."""
    od = tables.hab.reset_index()
    dwelling_columns = [column for column in DWELLING_COLUMNS if column not in od.columns]
    od = od.merge(tables.viv[dwelling_columns], left_on="folio_vivienda", right_index=True, how="left", validate="many_to_one")
    od = od.merge(compute_work_trip_destination(tables.trips), on=KEYS, how="left", validate="one_to_one")
    od = od[od["trabajo_semana_pasada"].isin(EMPLOYED_CATEGORIES)].copy()
    od = add_destination_features(od, state=state, release=release)
    categorical_columns = od.columns[od.dtypes.eq("category")]
    od[categorical_columns] = od[categorical_columns].astype("string")
    unknown_labels = set(od["giro_empresa"].dropna().unique()) - set(GIRO_SLUGS)
    assert not unknown_labels, f"giro_empresa labels missing from config.yaml giro_levels: {sorted(unknown_labels)}"
    od["giro"] = od["giro_empresa"].map(GIRO_SLUGS).astype("string")
    od["giro_desconocido"] = od["giro"].isna()

    return od.reset_index(drop=True)
