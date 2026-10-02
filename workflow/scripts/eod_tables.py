"""Rule eod_tables: load_eod()'s four tables as parquet, exactly (dtypes, categorical levels and their order, the
index), with a manifest whose versions carry the tables' digest. eodgdl.artifacts.eod() reads them back."""
from pathlib import Path

from eodgdl.data import SURVEY_FILES, resolve
from eodgdl.impute.sources import Context
from eodgdl.impute.sources.eod import survey_versions
from eodgdl.manifest import write_manifest

out = Path(snakemake.output.manifest).parent  # noqa: F821  (snakemake is the rule's, injected)
context = Context()                            # loads the survey once, and lends it to survey_versions' digest
tables = context.eod()
for name, frame in zip(tables._fields, tables):
    frame.to_parquet(out / f"{name}.parquet")
write_manifest(out, "eod_tables", inputs={name: resolve(name) for name in SURVEY_FILES}, versions=survey_versions(context),
               outputs=[out / f"{name}.parquet" for name in tables._fields])
