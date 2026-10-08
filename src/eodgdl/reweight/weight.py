"""Stage 3's boundary: the household weight TMG.SurveyReweight fits, as it enters the data.

The tool runs outside eodgdl (a .NET XTMF2 model system, run by hand; it has no headless runner, and only the weight
comes back). Its ``UpdatedExpansionFactorsFile`` is one ``UpdatedExpansionFactor`` per household in the order of
``HouseholdRecords.csv``. :func:`import_weight` turns it into ``data/EOD_peso_hogar_TMG.csv`` (``folio_vivienda``,
``peso``) and writes the sidecar ``data/EOD_peso_hogar_TMG.yaml``: the sha256 of the file received and of the weight
file, the reweight inputs it was fitted on (stage 2's manifest: its ``outputs_digest``, :mod:`eodgdl.manifest`), the
constraint set (year), and whatever else is known about the run (the tool's commit, the exported ``.xmsys``
configuration's and the constraint report's sha256 when they are handed back; None otherwise).

The TASHA build expands with it under ``expansion: tmg`` in ``config/config.yaml`` (:func:`read_weight`, read by
eodgdl.tasha.load_expansion); the EOD sources read it only under ``weight: tmg``, the imputations' sensitivity run
(their ``run_weights``, :func:`eodgdl.impute.sources.with_weight`). :func:`check_weight` (run by ``eodgdl pipeline
verify`` and the workflow's rule ``tmg_weight`` before whatever reads the weight) fails where the weight file is not the
one the sidecar records, or the reweight inputs written now are not the ones the weight was fitted on.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import yaml

WEIGHT_CSV = "EOD_peso_hogar_TMG.csv"
SIDECAR = "EOD_peso_hogar_TMG.yaml"
KEY, COLUMN = "folio_vivienda", "peso"
TOOL = "https://github.com/TravelModellingGroup/TMG.SurveyReweight"


def _data_dir(data_dir):
    from eodgdl.data import data_dir as default

    return Path(data_dir) if data_dir is not None else default()


def import_weight(updated, reweight_dir, data_dir, year, *, tool_commit=None, configuration=None, report=None, notes=None):
    """Write ``<data_dir>/EOD_peso_hogar_TMG.csv`` and its sidecar from the tool's ``UpdatedExpansionFactorsFile``
    (``updated``), fitted on the reweight set in ``reweight_dir`` (its ``manifest.json``) with the constraint set of
    ``year``. ``configuration`` and ``report``: the exported model system and the constraint report, when handed back
    (their sha256 recorded, not copied). Returns the two paths. Fails where the tool's file does not hold one positive
    weight per household record."""
    from eodgdl.manifest import read_manifest, sha256

    reweight_dir, data_dir = Path(reweight_dir), Path(data_dir)
    households = pd.read_csv(reweight_dir / "HouseholdRecords.csv", usecols=["HouseholdID"])
    factors = pd.read_csv(updated)
    if list(factors.columns) != ["UpdatedExpansionFactor"]:
        raise ValueError(f"{updated}: expected the single column UpdatedExpansionFactor, got {list(factors.columns)}")
    if len(factors) != len(households):
        raise ValueError(f"{updated}: {len(factors)} weights for {len(households)} household records")
    weight = pd.to_numeric(factors["UpdatedExpansionFactor"], errors="coerce")
    if not (weight > 0).all():
        raise ValueError(f"{updated}: {int((~(weight > 0)).sum())} weights that are not positive numbers")
    manifest = read_manifest(reweight_dir)
    if str(year) not in {str(y) for y in manifest["parameters"].get("years", [])}:
        raise ValueError(f"{reweight_dir}: no constraint set for {year} (its years: {manifest['parameters'].get('years')})")
    out = data_dir / WEIGHT_CSV
    pd.DataFrame({KEY: households["HouseholdID"], COLUMN: weight}).to_csv(out, index=False)
    sidecar = {
        "weight_file": WEIGHT_CSV, "weight_sha256": sha256(out), "key": KEY, "column": COLUMN,
        "received": {"file": Path(updated).name, "sha256": sha256(updated), "imported": datetime.date.today().isoformat()},
        "reweight_inputs": {"outputs_digest": manifest["outputs_digest"], "manifest_sha256": sha256(reweight_dir / "manifest.json"),
                            "commit": manifest["code"]["commit"], "constraint_year": int(year),
                            "base": manifest["parameters"].get("base", "mtazmun")},
        "tool": {"repository": TOOL, "commit": tool_commit},
        "configuration_sha256": sha256(configuration) if configuration else None,
        "constraint_report_sha256": sha256(report) if report else None,
        "notes": notes,
    }
    path = data_dir / SIDECAR
    path.write_text(yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out, path


def read_sidecar(data_dir=None):
    """The sidecar of the TMG weight in ``data_dir`` (default :func:`eodgdl.data.data_dir`), or None where there is none."""
    folder = _data_dir(data_dir)
    if not (folder / SIDECAR).exists():
        return None
    return yaml.safe_load((folder / SIDECAR).read_text(encoding="utf-8"))


def read_weight(data_dir=None) -> pd.Series:
    """TMG's household weight in ``data_dir`` (default :func:`eodgdl.data.data_dir`): ``peso`` indexed by
    ``folio_vivienda``. Fails where it has not come back (no file, or no sidecar), is not the file its sidecar records,
    repeats a household or holds a weight that is not a positive number."""
    from eodgdl.manifest import sha256

    folder = _data_dir(data_dir)
    path, sidecar = folder / WEIGHT_CSV, read_sidecar(folder)
    if not path.exists() or sidecar is None:
        raise FileNotFoundError(f"{folder}: no TMG weight with its sidecar ({WEIGHT_CSV}, {SIDECAR}): bring it in with "
                                "`eodgdl reweight import-weight` (docs/pipeline.md, stage 3)")
    if sha256(path) != sidecar["weight_sha256"]:
        raise ValueError(f"{path}: not the weight its sidecar {SIDECAR} records")
    table = pd.read_csv(path)
    if table[KEY].duplicated().any():
        raise ValueError(f"{path}: {int(table[KEY].duplicated().sum())} repeated {KEY} values")
    weight = pd.to_numeric(table[COLUMN], errors="coerce")
    if not (weight > 0).all():
        raise ValueError(f"{path}: {int((~(weight > 0)).sum())} weights that are not positive numbers")
    return pd.Series(weight.astype(float).to_numpy(), index=pd.Index(table[KEY], name=KEY), name=COLUMN)


def set_directory(sidecar):
    """The folder, under a root, of the reweight set a sidecar's weight was fitted on: ``reweight`` (the MTAZMun
    set; a sidecar from before the TAZ set names no base) or ``reweight_taz`` (spec.yaml's ``zones.bases``)."""
    from eodgdl.reweight._spec import bases

    base = (sidecar or {}).get("reweight_inputs", {}).get("base", "mtazmun")
    return bases()[base]["directory"]


def check_weight(data_dir=None, root="output") -> list[str]:
    """What does not hold of the TMG weight in ``data_dir`` (none where it has not come back yet): the weight file is
    the one its sidecar records, and the reweight inputs under ``<root>/<set>`` (their manifest; the set the sidecar
    names, :func:`set_directory`) are the ones it was fitted on. The workflow's rule ``tmg_weight`` runs it on the
    stage 2 it just wrote, before any imputation."""
    from eodgdl.manifest import read_manifest, sha256

    folder = _data_dir(data_dir)
    sidecar = read_sidecar(folder)
    if sidecar is None:
        return [] if not (folder / WEIGHT_CSV).exists() else [f"{folder / WEIGHT_CSV}: no sidecar {SIDECAR}"]
    problems = []
    weight = folder / sidecar["weight_file"]
    if not weight.exists():
        problems.append(f"{weight}: missing (its sidecar is there)")
    elif sha256(weight) != sidecar["weight_sha256"]:
        problems.append(f"{weight}: not the weight its sidecar records")
    manifest = Path(root) / set_directory(sidecar) / "manifest.json"
    if not manifest.exists():
        problems.append(f"{manifest}: no reweight inputs to check the weight against (eodgdl reweight build --out {manifest.parent})")
    elif read_manifest(manifest)["outputs_digest"] != sidecar["reweight_inputs"]["outputs_digest"]:
        problems.append(f"{manifest.parent}: other reweight inputs than the TMG weight was fitted on: the weight is stale")
    return problems
