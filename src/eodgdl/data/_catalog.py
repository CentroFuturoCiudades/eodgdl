"""Catalog of fetchable IMEPLAN EOD 2023 files: the input data and the fitted model bundles.

The bare filenames are the Pooch registry keys. The input data live in the ``data/``
directory of the eodgdl repo (the Pooch ``base_url``), the model bundles (``MODEL_FILES``)
in its ``models/`` directory (a URL of their own). See ``_registry.py``.
"""
from __future__ import annotations

# Survey master tables (raw IMEPLAN CSV, ISO-8859-1)
VIVIENDAS_CSV = "IMEPLAN_Base_Viviendas_Master.csv"
HABITANTES_CSV = "IMEPLAN_Base_Habitantes_Master.csv"
VIAJES_CSV = "IMEPLAN_Base_Viajes_Master.csv"

# Zone-system geometries (parquet)
ZONIFICACION_PARQUET = "AMG_Zonificacion_para_encuesta.parquet"

# CONAPO population projections for the 9 municipalities (municipality x sex x year x 5-year band)
CONAPO_CSV = "CONAPO_proyecciones_AMG.csv"

# INEGI registered motor vehicles (VMRC), annual, the 9 municipalities, 2015-2023
VMRC_CSV = "VMRC_AMG.csv"

# INEGI ENDUTIH share of dwellings with internet, 2020 and 2023 (scripts/endutih_internet.py)
ENDUTIH_CSV = "ENDUTIH_internet_AMG.csv"

# Fitted eodgdl.impute bundles, one per task (impute/tasks/<task>.yaml names its file; scikit-learn pickles, see the
# sklearn version in their metadata; written by `eodgdl impute retrain <task>`)
GIRO_MODEL_JOBLIB = "od_giro_hybrid_model.joblib"
INFORMALITY_MODEL_JOBLIB = "od_informality_model.joblib"
EDUCACION_JEFE_MODEL_JOBLIB = "od_educacion_jefe_model.joblib"
AMAI_BANOS_MODEL_JOBLIB = "od_amai_banos_model.joblib"
AMAI_DORMITORIOS_MODEL_JOBLIB = "od_amai_dormitorios_model.joblib"
AMAI_TRABAJADORES_MODEL_JOBLIB = "od_amai_trabajadores_model.joblib"   # diagnostic: read only by `evaluate nse`
EMPLEO_MODEL_JOBLIB = "od_empleo_model.joblib"   # the employment situation of the persons who did not answer it
# The same task trained on the survey's design weight, pinned for stage 1 of the processing: the reweight records'
# Employed / NotEmployed of the persons who did not answer (reweight/spec.yaml, `imputed`). v0.9.0's
# od_empleo_model.joblib, byte for byte; it stays when the tasks are retrained on the TMG weight.
EMPLEO_DESIGN_MODEL_JOBLIB = "od_empleo_design_model.joblib"

# AMAI's socioeconomic level (NSE) by AGEB, national, as delivered (read by eodgdl.impute.derive.amai for the AGEB
# rank calibration of the chain `nse`; brought over from informal-jobs-model's data/, 2026-09-03)
AMAI_NSE_AGEB_XLSX = "NSE_por_AGEB_AMAI.xlsx"

SURVEY_FILES = [VIVIENDAS_CSV, HABITANTES_CSV, VIAJES_CSV]
ZONE_FILES = [ZONIFICACION_PARQUET]
CENSUS_FILES = [CONAPO_CSV, VMRC_CSV, ENDUTIH_CSV]
MODEL_FILES = [
    GIRO_MODEL_JOBLIB, INFORMALITY_MODEL_JOBLIB, EDUCACION_JEFE_MODEL_JOBLIB, AMAI_BANOS_MODEL_JOBLIB,
    AMAI_DORMITORIOS_MODEL_JOBLIB, AMAI_TRABAJADORES_MODEL_JOBLIB, EMPLEO_MODEL_JOBLIB, EMPLEO_DESIGN_MODEL_JOBLIB,
]
REFERENCE_FILES = [AMAI_NSE_AGEB_XLSX]
FILES = SURVEY_FILES + ZONE_FILES + CENSUS_FILES + MODEL_FILES + REFERENCE_FILES
