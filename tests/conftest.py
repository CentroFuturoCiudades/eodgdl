"""Shared fixtures: the survey read and cleaned once for the whole session."""
import os
from pathlib import Path

import pytest

from eodgdl import load_stages

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "IMEPLAN_Base_Viajes_Master.csv").exists()
if HAS_DATA:   # the clone's data/ (eodgdl.data's default too), unless the caller points the tests elsewhere
    os.environ.setdefault("EODGDL_DATA_DIR", str(DATA_DIR))


@pytest.fixture(scope="session")
def stages():
    """``load_stages`` on the in-repo data — ``shipped``, ``rules``, ``revised`` — shared by every test: do not modify."""
    if not HAS_DATA:
        pytest.skip("in-repo data/ not present")
    return load_stages(DATA_DIR)


@pytest.fixture
def run_config(tmp_path, monkeypatch):
    """``run_config(**changes)`` points eodgdl.config at a copy of config/config.yaml with ``changes`` (a run of the
    processing with other parameters), for the rest of the test; returns the parameters written."""
    import yaml

    from eodgdl import config

    def write(**changes):
        values = {**config.load(), **changes}
        path = tmp_path / "run_config.yaml"
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        monkeypatch.setenv("EODGDL_CONFIG", str(path))
        return values

    return write
