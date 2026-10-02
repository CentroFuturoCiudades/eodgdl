"""eodgdl.artifacts reads back what the workflow wrote: the cleaned tables exactly, warning when the loader has moved on."""
import json

import pandas as pd
import pytest

from eodgdl import artifacts
from eodgdl.impute.sources.eod import survey_versions
from eodgdl.manifest import write_manifest


def _write_eod(root, tables, versions):
    out = root / "eod"
    out.mkdir(parents=True)
    for name, frame in zip(tables._fields, tables):
        frame.to_parquet(out / f"{name}.parquet")
    return write_manifest(out, "eod_tables", versions=versions)


def test_the_tables_come_back_as_load_eod_gave_them(stages, tmp_path, monkeypatch):
    monkeypatch.setenv("EODGDL_OUTPUT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="snakemake -c8 eod"):
        artifacts.eod()
    manifest = _write_eod(tmp_path, stages.revised, survey_versions())
    read = artifacts.eod()
    for written, back in zip(stages.revised, read):
        pd.testing.assert_frame_equal(written, back, check_exact=True)
    # a manifest from another loader: the tables still come back, with a warning to rebuild them
    recorded = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**recorded, "versions": {**recorded["versions"], "eod_tables": "0" * 64}}))
    with pytest.warns(UserWarning, match="snakemake -c8 eod"):
        artifacts.eod()


def test_a_missing_output_names_its_target(tmp_path, monkeypatch):
    monkeypatch.setenv("EODGDL_OUTPUT_DIR", str(tmp_path))
    for reader, target in ((artifacts.zones, "zones"), (artifacts.zone_assignment, "zones"), (artifacts.tasha_tables, "tasha")):
        with pytest.raises(FileNotFoundError, match=f"snakemake -c8 {target}"):
            reader()
