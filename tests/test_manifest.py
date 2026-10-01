"""Manifests (eodgdl.manifest): each stage records what it read and wrote, and `pipeline verify` lists every broken
link."""
from eodgdl.manifest import check_manifest, read_manifest, verify, write_manifest


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

