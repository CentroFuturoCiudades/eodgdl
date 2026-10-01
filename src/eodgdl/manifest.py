"""Manifests: what each stage of the processing read and wrote, so the chain of inputs and outputs can be verified.

The processing (``docs/pipeline.md``): load → ``empleo`` on the design weight → reweight inputs → TMG.SurveyReweight
(external) → every task retrained on the TMG weight → pipeline ``tasha`` → TASHA build. Every stage that writes files
writes a manifest beside them (:func:`write_manifest`):

- ``stage`` and ``code``: the eodgdl version, the git commit and whether the tree had uncommitted changes to tracked
  files, Python, the platform and ``uv.lock``'s sha256 (:func:`code_identity`). Same platform and lock: the same bytes
  are expected; elsewhere, equivalence within Monte Carlo error.
- ``parameters``: the stage's settings (seeds, draws, bootstrap refits).
- ``versions``: the data versions the stage's sources declare (the survey files and loader code, the weight file's
  sha256) and ``environment``: the packages whose data or models it reads (mxcensus and its registry's sha256, which
  pins every census, ENOE, ENIGH and DENUE file it fetches; scikit-learn).
- ``inputs``: ``{label: {path, sha256}}`` of every file read (survey files, model bundles, weight file), the path
  relative to the clone's root (``data/...``, ``models/...``) where it lies there.
- ``upstream``: ``{label: {path, outputs_digest}}``, the manifests of the stages it read, linked by their outputs'
  content (:func:`outputs_digest`), not by the manifest's own bytes, so a rebuild that reproduces the same outputs
  keeps the link.
- ``outputs``: ``{relative path: sha256}`` of every file it wrote, and ``outputs_digest`` over them.

:func:`verify` walks every manifest under a directory (and TMG's weight sidecar, :mod:`eodgdl.reweight.weight`) and
lists every broken link: an output that changed or went missing, an upstream whose outputs are no longer what was read,
an input whose bytes changed. Like ``reweight check`` it returns a list, empty when the chain is sound.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

MANIFEST = "manifest.json"
FORMAT = "eodgdl.manifest/1"


def sha256(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _repo():
    """The repository a clone runs from (src/eodgdl -> the root), or None for an installed package."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / ".git").exists() else None


def code_identity() -> dict:
    """The eodgdl version, the commit and a ``dirty`` flag (uncommitted changes to tracked files; None outside a
    clone), Python, the platform and ``uv.lock``'s sha256."""
    import eodgdl

    root, commit, dirty, lock = _repo(), None, None, None
    if root is not None:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)

        head = git("rev-parse", "HEAD")
        commit = head.stdout.strip() or None if head.returncode == 0 else None
        status = git("status", "--porcelain", "--untracked-files=no")
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        lock = sha256(root / "uv.lock") if (root / "uv.lock").exists() else None
    return {"eodgdl": eodgdl.__version__, "commit": commit, "dirty": dirty, "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(), "uv_lock": lock}


def environment() -> dict:
    """The installed packages whose data or models a stage reads: mxcensus (its version and its registry's sha256: the
    sha256 of every file it fetches, checked by Pooch on download) and scikit-learn; absent ones are left out."""
    from importlib import resources
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    try:
        registry = resources.files("mxcensus.data") / "registry.txt"
        with resources.as_file(registry) as path:
            out["mxcensus"] = {"version": version("mxcensus"), "registry": sha256(path)}
    except (ModuleNotFoundError, PackageNotFoundError, FileNotFoundError):
        pass
    try:
        out["scikit-learn"] = version("scikit-learn")
    except PackageNotFoundError:
        pass
    return out


def outputs_digest(outputs: dict) -> str:
    """One sha256 over a manifest's outputs (their relative paths and sha256s, in order)."""
    digest = hashlib.sha256()
    for name in sorted(outputs):
        digest.update(f"{name}\0{outputs[name]}\n".encode())
    return digest.hexdigest()


def _base() -> Path:
    """What a manifest's input paths are relative to: the clone's root, else the working directory."""
    return (_repo() or Path.cwd()).resolve()


def _shown(path) -> str:
    """``path`` relative to :func:`_base` where it lies under it (``data/...``, ``models/...``), else absolute."""
    path = Path(path).resolve()
    return str(path.relative_to(_base())) if path.is_relative_to(_base()) else str(path)


def _files(out_dir, outputs, exclude):
    """``{path relative to out_dir: sha256}`` of ``outputs``: paths as given (absolute, or relative to the working
    directory) where they lie under ``out_dir``, else relative to it."""
    out_dir = Path(out_dir).resolve()
    if outputs is None:
        outputs = [p for p in sorted(out_dir.rglob("*")) if p.is_file()]
    names = {}
    for path in outputs:
        path = Path(path).resolve() if Path(path).resolve().is_relative_to(out_dir) else out_dir / path
        name = path.relative_to(out_dir).as_posix()
        if name != MANIFEST and name not in exclude and not name.endswith(f"/{MANIFEST}"):
            names[name] = sha256(path)
    return names


def read_manifest(path) -> dict:
    path = Path(path)
    return json.loads((path / MANIFEST if path.is_dir() else path).read_text(encoding="utf-8"))


def write_manifest(out_dir, stage, *, inputs=None, upstream=None, parameters=None, versions=None, outputs=None,
                   exclude=(), name=MANIFEST) -> Path:
    """Write ``<out_dir>/<name>`` for ``stage``: ``inputs`` ({label: path}), ``upstream`` ({label: an upstream
    manifest or its directory}; a missing one is recorded as such), ``parameters``, ``versions`` (data versions), and
    the sha256 of ``outputs`` (paths under ``out_dir``; default: every file under it but manifests and ``exclude``).
    Returns the manifest's path."""
    out_dir = Path(out_dir)
    files = _files(out_dir, outputs, set(exclude))
    links = {}
    for label, path in (upstream or {}).items():
        path = Path(path)
        path = path / MANIFEST if path.is_dir() or not path.suffix else path
        links[label] = ({"path": os.path.relpath(path.resolve(), out_dir.resolve()), "outputs_digest": read_manifest(path)["outputs_digest"]}
                        if path.exists() else {"path": _shown(path), "outputs_digest": None})
    manifest = {"format": FORMAT, "stage": stage, "code": code_identity(), "environment": environment(), "parameters": parameters or {},
                "versions": versions or {}, "inputs": {label: {"path": _shown(path), "sha256": sha256(path)} for label, path in (inputs or {}).items()},
                "upstream": links, "outputs": files, "outputs_digest": outputs_digest(files)}
    path = out_dir / name
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def check_manifest(path) -> list[str]:
    """What no longer holds in one manifest: its outputs (each file's sha256, none missing, ``outputs_digest``), its
    upstream links (the upstream manifest's ``outputs_digest`` now) and its inputs (each file's sha256 now, read from
    the recorded path: relative to the clone's root, else to the working directory; one not found is reported)."""
    path = Path(path)
    where = path.parent
    try:
        manifest = read_manifest(path)
    except (OSError, ValueError) as error:
        return [f"{path}: unreadable ({error})"]
    problems = []
    outputs = manifest.get("outputs", {})
    for name, digest in outputs.items():
        file = where / name
        if not file.exists():
            problems.append(f"{path}: output {name} is missing")
        elif sha256(file) != digest:
            problems.append(f"{path}: output {name} changed since it was written")
    if outputs_digest(outputs) != manifest.get("outputs_digest"):
        problems.append(f"{path}: outputs_digest does not match its outputs")
    for label, link in manifest.get("upstream", {}).items():
        if link.get("outputs_digest") is None:
            problems.append(f"{path}: upstream {label} had no manifest when this stage ran ({link['path']})")
            continue
        upstream = (where / link["path"]).resolve()
        if not upstream.exists():
            problems.append(f"{path}: upstream {label} not found at {link['path']}")
        elif read_manifest(upstream).get("outputs_digest") != link["outputs_digest"]:
            problems.append(f"{path}: upstream {label} ({link['path']}) holds other outputs than this stage read")
    for label, entry in manifest.get("inputs", {}).items():
        file = _base() / entry["path"]
        if not file.exists():
            problems.append(f"{path}: input {label} not found at {entry['path']}")
        elif sha256(file) != entry["sha256"]:
            problems.append(f"{path}: input {label} ({entry['path']}) changed since this stage read it")
    return problems


def verify(root="output", data_dir=None) -> list[str]:
    """Every broken link under ``root``: each manifest's (:func:`check_manifest`), and TMG's weight sidecar in
    ``data_dir`` (default :func:`eodgdl.data.data_dir`) against the reweight inputs it was fitted on
    (:func:`eodgdl.reweight.weight.check_weight`). Empty when the chain is sound."""
    root = Path(root)
    problems = []
    manifests = sorted(root.rglob(MANIFEST)) + sorted(p for p in root.rglob(f"*_{MANIFEST}"))
    if not manifests:
        problems.append(f"{root}: no manifest found")
    for path in manifests:
        problems += check_manifest(path)
    from eodgdl.reweight.weight import check_weight

    problems += check_weight(data_dir, root)
    return problems
