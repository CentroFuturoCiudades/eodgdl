"""Catalog of fetchable IMEPLAN EOD 2023 data files.

The bare filenames are the Pooch registry keys; the data files live in the ``data/``
directory of the eodgdl repo (the Pooch ``base_url``). See ``_registry.py``.
"""
from __future__ import annotations

# Survey master tables (raw IMEPLAN CSV, ISO-8859-1)
VIVIENDAS_CSV = "IMEPLAN_Base_Viviendas_Master.csv"
HABITANTES_CSV = "IMEPLAN_Base_Habitantes_Master.csv"
VIAJES_CSV = "IMEPLAN_Base_Viajes_Master.csv"

# Zone-system geometries (parquet)
ZONIFICACION_PARQUET = "AMG_Zonificacion_para_encuesta.parquet"
MICROZONAS_PARQUET = "AMG_MicroZONAS2023.parquet"
AGEBS_ZONA_PARQUET = "RELACION_AGEBS-ZONA_con_datos_censales.parquet"

SURVEY_FILES = [VIVIENDAS_CSV, HABITANTES_CSV, VIAJES_CSV]
ZONE_FILES = [ZONIFICACION_PARQUET, MICROZONAS_PARQUET, AGEBS_ZONA_PARQUET]
FILES = SURVEY_FILES + ZONE_FILES
