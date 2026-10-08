"""The workflow (workflow/Snakefile) parses and schedules the processing's stages: dry runs, nothing executed."""
import importlib.util
import os
import re
import subprocess
import sys

import pytest

from eodgdl.data import REPO

pytestmark = pytest.mark.skipif(importlib.util.find_spec("snakemake") is None, reason="the workflow extra is not installed")


def _dry_run(root, *config, env=None, targets=()):
    command = [sys.executable, "-m", "snakemake", *targets, "-n", "-c1", "--nolock", "--quiet", "rules", "--config", f"output={root}", *config]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return result, dict(re.findall(r"^(\w+)\s+(\d+)$", result.stdout, re.M))


def test_a_dry_run_schedules_every_stage(tmp_path):
    result, jobs = _dry_run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {rule: int(n) for rule, n in jobs.items() if rule != "total"} == {
        "eod_tables": 1, "zone_system": 1, "reweight_inputs": 2, "retrain": 7, "diagnostic_chain": 2, "pipeline_tasha": 1,
        "tasha_build": 1, "verify": 1, "all": 1}


def test_the_weight_is_the_run_config_s(tmp_path, run_config):
    # eodgdl reads the weight from the run config, so the workflow takes no other
    result, _ = _dry_run(tmp_path, "weight=tmg")
    assert result.returncode != 0 and "not with --config" in result.stdout + result.stderr
    # under weight tmg, nothing runs before TMG's weight is in the data
    run_config(weight="tmg")
    result, _ = _dry_run(tmp_path, env={"EODGDL_CONFIG": os.environ["EODGDL_CONFIG"], "EODGDL_DATA_DIR": str(tmp_path)})
    output = result.stdout + result.stderr
    assert result.returncode != 0 and "EOD_peso_hogar_TMG.csv" in output and "eodgdl reweight import-weight" in output


def test_only_the_tasha_build_waits_for_the_expansion_weight(tmp_path):
    # under expansion: tmg the TASHA build expands with TMG's weight, checked first (rule tmg_weight); the imputations
    # keep the design weight and wait for nothing
    result, _ = _dry_run(tmp_path / "out", "expansion=tmg", env={"EODGDL_DATA_DIR": str(tmp_path)})
    output = result.stdout + result.stderr
    assert result.returncode != 0 and "expansion: tmg, but there is no" in output and "eodgdl reweight import-weight" in output
    data = tmp_path / "data"
    data.mkdir()
    for path in (REPO / "data").iterdir():
        (data / path.name).symlink_to(path)
    (data / "EOD_peso_hogar_TMG.csv").write_text("folio_vivienda,peso\n1,1.0\n")
    (data / "EOD_peso_hogar_TMG.yaml").write_text("reweight_inputs: {base: mtazmun}\n")
    env = {"EODGDL_DATA_DIR": str(data)}
    (result, impute), (_, built) = (_dry_run(tmp_path / "out", "expansion=tmg", env=env, targets=[target]) for target in ("impute", "tasha"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert impute["retrain"] == "7" and impute["pipeline_tasha"] == "1" and "tmg_weight" not in impute
    assert built["tmg_weight"] == "1" and built["tasha_build"] == "1"
