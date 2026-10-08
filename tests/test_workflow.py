"""The workflow (workflow/Snakefile) parses and schedules the processing's stages: dry runs, nothing executed."""
import importlib.util
import os
import re
import subprocess
import sys

import pytest

from eodgdl.data import REPO

pytestmark = pytest.mark.skipif(importlib.util.find_spec("snakemake") is None, reason="the workflow extra is not installed")


def _dry_run(root, *config, env=None):
    command = [sys.executable, "-m", "snakemake", "-n", "-c1", "--nolock", "--quiet", "rules", "--config", f"output={root}", *config]
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
