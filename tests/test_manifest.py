"""Manifests (eodgdl.manifest) and stage 3's boundary (eodgdl.reweight.weight): each stage records what it read and
wrote, `pipeline verify` lists every broken link, and TMG's weight enters the data only with the reweight inputs it was
fitted on."""
import json

import pandas as pd
import pytest

from eodgdl.manifest import check_manifest, read_manifest, verify, write_manifest
from eodgdl.reweight import weight


def _stage(folder, files, **kw):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)
    return write_manifest(folder, "test", **kw)


def test_a_manifest_records_outputs_inputs_and_upstream(tmp_path, monkeypatch):
    from eodgdl import manifest as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_repo", lambda: None)        # outside a clone: paths relative to the working directory
    (tmp_path / "input.csv").write_text("a\n1\n")
    upstream = _stage(tmp_path / "up", {"x.csv": "1"})
    path = _stage(tmp_path / "down", {"y.csv": "2", "sub/z.csv": "3"}, inputs={"input": tmp_path / "input.csv"}, upstream={"up": upstream.parent},
                  parameters={"draws": 2})
    manifest = read_manifest(path)
    assert set(manifest["outputs"]) == {"y.csv", "sub/z.csv"} and manifest["inputs"]["input"]["path"] == "input.csv"
    assert manifest["upstream"]["up"] == {"path": "../up/manifest.json", "outputs_digest": read_manifest(upstream)["outputs_digest"]}
    assert {"eodgdl", "commit", "dirty", "uv_lock", "platform"} <= set(manifest["code"]) and manifest["parameters"] == {"draws": 2}
    assert verify(tmp_path, data_dir=tmp_path) == []

    # outputs named relative to the working directory, as the CLI's writers name them, or to the stage's directory
    again = write_manifest("down", "test", outputs=["down/y.csv", "sub/z.csv"], name="relative_manifest.json")
    assert set(read_manifest(again)["outputs"]) == {"y.csv", "sub/z.csv"}
    # a rebuild of the upstream with the same outputs keeps the link (content, not the manifest's bytes)
    _stage(tmp_path / "up", {"x.csv": "1"}, parameters={"rebuilt": True})
    assert check_manifest(path) == []
    # what breaks: an upstream with other outputs, an output edited, an input changed, an output gone
    _stage(tmp_path / "up", {"x.csv": "other"})
    (tmp_path / "down" / "y.csv").write_text("edited")
    (tmp_path / "input.csv").write_text("a\n2\n")
    (tmp_path / "down" / "sub" / "z.csv").unlink()
    problems = check_manifest(path)
    assert any("upstream up" in p and "other outputs" in p for p in problems)
    assert any("output y.csv changed" in p for p in problems) and any("output sub/z.csv is missing" in p for p in problems)
    assert any("input input (input.csv) changed" in p for p in problems)


def _reweight_set(folder, households=(1, 2, 3)):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"HouseholdID": list(households), "HouseholdMTAZMun": 1001}).to_csv(folder / "HouseholdRecords.csv", index=False)
    return write_manifest(folder, "reweight_inputs", parameters={"years": [2020, 2023]})


def test_the_tmg_weight_enters_with_its_sidecar(tmp_path, monkeypatch):
    from eodgdl import manifest as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_repo", lambda: None)
    data, root = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    _reweight_set(root / "reweight")
    pd.DataFrame({"UpdatedExpansionFactor": [10.5, 20.0, 30.25]}).to_csv(tmp_path / "updated.csv", index=False)
    csv, sidecar = weight.import_weight(tmp_path / "updated.csv", root / "reweight", data, 2023, tool_commit="960c5ed")
    assert pd.read_csv(csv).to_dict("list") == {"folio_vivienda": [1, 2, 3], "peso": [10.5, 20.0, 30.25]}
    recorded = weight.read_sidecar(data)
    assert recorded["reweight_inputs"]["outputs_digest"] == read_manifest(root / "reweight")["outputs_digest"]
    assert recorded["reweight_inputs"]["constraint_year"] == 2023 and recorded["tool"]["commit"] == "960c5ed"
    assert recorded["reweight_inputs"]["base"] == "mtazmun"   # a manifest that names no base is the MTAZMun set's
    assert recorded["configuration_sha256"] is None and recorded["constraint_report_sha256"] is None   # only the weight came back
    assert weight.check_weight(data, root) == [] and verify(root, data) == []

    # reweight inputs rebuilt with other content: the weight was fitted on something else
    (root / "reweight" / "HouseholdRecords.csv").write_text("HouseholdID,HouseholdMTAZMun\n1,1001\n2,1001\n3,2001\n")
    write_manifest(root / "reweight", "reweight_inputs", parameters={"years": [2020, 2023]})
    assert any("the weight is stale" in p for p in weight.check_weight(data, root))
    # the weight file edited after the import
    csv.write_text("folio_vivienda,peso\n1,1\n2,1\n3,1\n")
    assert any("not the weight its sidecar records" in p for p in weight.check_weight(data, root))


def test_a_weight_fitted_on_the_taz_set_is_checked_against_it(tmp_path, monkeypatch):
    from eodgdl import manifest as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_repo", lambda: None)
    data, root = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    _reweight_set(root / "reweight")
    (root / "reweight_taz").mkdir(parents=True)
    pd.DataFrame({"HouseholdID": [1, 2, 3], "HouseholdTAZ": [7, 7, 8]}).to_csv(root / "reweight_taz" / "HouseholdRecords.csv", index=False)
    write_manifest(root / "reweight_taz", "reweight_inputs", parameters={"years": [2020, 2023], "base": "taz"})
    pd.DataFrame({"UpdatedExpansionFactor": [10.5, 20.0, 30.25]}).to_csv(tmp_path / "updated.csv", index=False)
    weight.import_weight(tmp_path / "updated.csv", root / "reweight_taz", data, 2023)
    recorded = weight.read_sidecar(data)
    assert recorded["reweight_inputs"]["base"] == "taz" and weight.set_directory(recorded) == "reweight_taz"
    assert weight.check_weight(data, root) == []
    # the MTAZMun set rebuilt does not touch it; the TAZ set rebuilt makes it stale
    (root / "reweight" / "HouseholdRecords.csv").write_text("HouseholdID,HouseholdMTAZMun\n1,1001\n2,1001\n3,2001\n")
    write_manifest(root / "reweight", "reweight_inputs", parameters={"years": [2020, 2023]})
    assert weight.check_weight(data, root) == []
    (root / "reweight_taz" / "HouseholdRecords.csv").write_text("HouseholdID,HouseholdTAZ\n1,7\n2,8\n3,8\n")
    write_manifest(root / "reweight_taz", "reweight_inputs", parameters={"years": [2020, 2023], "base": "taz"})
    assert any("reweight_taz: other reweight inputs" in p for p in weight.check_weight(data, root))


def test_import_weight_refuses_what_does_not_fit_the_records(tmp_path):
    _reweight_set(tmp_path / "reweight")
    for frame, message in ((pd.DataFrame({"UpdatedExpansionFactor": [1.0, 2.0]}), "2 weights for 3 household records"),
                           (pd.DataFrame({"UpdatedExpansionFactor": [1.0, 0.0, 2.0]}), "1 weights that are not positive"),
                           (pd.DataFrame({"weight": [1.0, 1.0, 1.0]}), "the single column UpdatedExpansionFactor")):
        frame.to_csv(tmp_path / "updated.csv", index=False)
        with pytest.raises(ValueError, match=message):
            weight.import_weight(tmp_path / "updated.csv", tmp_path / "reweight", tmp_path, 2023)
    pd.DataFrame({"UpdatedExpansionFactor": [1.0, 1.0, 1.0]}).to_csv(tmp_path / "updated.csv", index=False)
    with pytest.raises(ValueError, match="no constraint set for 2021"):
        weight.import_weight(tmp_path / "updated.csv", tmp_path / "reweight", tmp_path, 2021)


def test_a_source_refuses_a_weight_its_sidecar_does_not_record(tmp_path, monkeypatch):
    from eodgdl.impute.sources import with_weight

    monkeypatch.setenv("EODGDL_DATA_DIR", str(tmp_path))
    pd.DataFrame({"folio_vivienda": [1, 2], "peso": [1.0, 2.0]}).to_csv(tmp_path / "w.csv", index=False)
    (tmp_path / "w.yaml").write_text(json.dumps({"weight_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="not the weight its sidecar w.yaml records"):
        with_weight(pd.DataFrame({"folio_vivienda": [1, 2]}), {"weight": {"file": "w.csv", "column": "peso"}})
