"""Compare two roots of ``eodgdl.impute`` outputs, file by file and model by model: the parity check of the engine's
reviews (``docs/handoff_impute_engine.md`` §0).

    uv run python scripts/impute/parity.py ROOT_A ROOT_B [--models] [--tolerance 1e-12] [--data data] [--strict] [--ignore-columns PATTERN ...]

A root has the layout of ``output/impute/`` (``scripts/impute/rerun.sh ROOT`` writes one): ``<task>/`` retrains (the
bundle, ``scores.parquet``, ``scenarios.parquet``, ``evaluation/*.parquet``, ``summary.json``), ``<chain>/`` runs
(``scores.parquet``, ``completions.parquet``, ``provenance.json``, ``evaluation/``) and ``giro_scores.parquet``.

- Every parquet in either root: identical (``assert_frame_equal(check_exact=True)``), else equal within the tolerance
  (``|a - b| <= tolerance * max(1, |b|)``: absolute below one, relative above), else different, with each column that
  differs and by how much. The held-out tables (a retrain's evaluation of its held-out rows, which moves whenever that
  evaluation does) are reported apart from the rest.
- Every JSON as a dict, key by key, numbers within the tolerance, without what records the run rather than its result:
  the timings, the versions, the bundles' sha256 and the eodgdl version a chain records for its bundles. A retrain
  summary's held-out entries apart, as the tables.
- With ``--models``: each task's two bundles, unless their bytes are equal, model by model (the arms, the auxiliary
  models, the label's components) by ``predict_proba`` on the task's frames (its training source's and, when another,
  its scoring source's, built from ``--data`` through the feature cache), with their classes and training level shares.

Exit status 1 when anything outside the held-out tables and entries differs beyond the tolerance (``--strict``:
anything at all), else 0.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# a retrain's evaluation of its held-out rows (run._heldout_tables, _hybrid_tables, the components, populations and
# target profile): ``evaluation/<name>.parquet`` or ``<name>__<arm>.parquet``
HELDOUT_TABLES = {"test_metrics", "test_uncertainty", "class_metrics", "confusion", "test_distribution", "calibration", "reliability", "isotonic",
                  "isotonic_calibration", "isotonic_reliability", "components_heldout", "populations", "target_profile"}
HELDOUT_KEYS = {"test_metrics", "test_uncertainty", "isotonic"}     # a retrain summary's entries under arms.<arm>
RUN_KEYS = {"timings_s", "versions"}                                 # what records the run, anywhere in a JSON


def retrain_dirs(root):
    """The task directories of a root: those holding a retrain's ``summary.json``."""
    return {path.parent.relative_to(root) for path in root.glob("*/summary.json")}


def is_heldout_table(relative, retrains):
    return relative.parent.name == "evaluation" and relative.parent.parent in retrains and relative.stem.split("__")[0] in HELDOUT_TABLES


def within(a, b, tolerance):
    return np.abs(a - b) <= tolerance * np.maximum(1.0, np.abs(b))


def numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def compare_frames(a, b, tolerance):
    """``("identical" | "within" | "different", [detail, ...])``."""
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        return "identical", []
    except AssertionError:
        pass
    details = []
    only_a, only_b = [c for c in a.columns if c not in b.columns], [c for c in b.columns if c not in a.columns]
    if only_a or only_b:
        details.append(f"columns only in A {only_a}, only in B {only_b}")
    elif list(a.columns) != list(b.columns):
        details.append("the same columns in another order")
    if len(a) != len(b):
        return "different", details + [f"{len(a):,} rows against {len(b):,}"]
    if not a.index.equals(b.index):
        details.append("another index")
    status = "different" if details else "within"
    for column in [c for c in a.columns if c in b.columns]:
        x, y = a[column].reset_index(drop=True), b[column].reset_index(drop=True)
        dtype = f" (dtype {x.dtype} against {y.dtype})" if x.dtype != y.dtype else ""
        if numeric(x) and numeric(y):
            xv, yv = x.to_numpy(float, na_value=np.nan), y.to_numpy(float, na_value=np.nan)
            nan = np.isnan(xv) | np.isnan(yv)
            unequal = (np.isnan(xv) != np.isnan(yv)) | (~nan & (xv != yv))
            if not unequal.any():
                if dtype:
                    details.append(f"{column}{dtype}")
                    status = "different"
                continue
            beyond = (np.isnan(xv) != np.isnan(yv)) | (~nan & ~within(xv, yv, tolerance))
            gap = np.abs(xv - yv)[~nan]
            shown = f"; {xv[0]!r} -> {yv[0]!r}" if len(xv) == 1 else ""
            details.append(f"{column}: {int(unequal.sum()):,} rows differ, max |d| {gap.max() if gap.size else float('nan'):.3g}{shown}{dtype}")
            if beyond.any() or dtype:
                status = "different"
        else:
            xs, ys = x.astype("string").fillna("<NA>"), y.astype("string").fillna("<NA>")
            unequal = (xs != ys).to_numpy()
            if unequal.any() or dtype:
                first = int(np.flatnonzero(unequal)[0]) if unequal.any() else None
                shown = f" (first at row {first}: {xs[first]!r} -> {ys[first]!r})" if first is not None else ""
                details.append(f"{column}: {int(unequal.sum()):,} rows differ{shown}{dtype}")
                status = "different"
    return status, details


def flatten(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, (*path, str(index)))
    else:
        yield path, value


def ignored(path):
    return bool(RUN_KEYS & set(path)) or path[-2:] == ("bundle", "sha256") or (len(path) == 3 and path[0] == "bundles" and path[2] == "eodgdl_version")


def compare_json(a, b, tolerance):
    """``[(path, value A, value B, beyond tolerance)]`` of the keys that differ."""
    flat_a, flat_b = dict(flatten(a)), dict(flatten(b))
    out = []
    for path in sorted(set(flat_a) | set(flat_b)):
        if ignored(path):
            continue
        x, y = flat_a.get(path, "<absent>"), flat_b.get(path, "<absent>")
        if x == y or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y)):
            continue
        both_numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        out.append((path, x, y, not (both_numbers and within(float(x), float(y), tolerance))))
    return out


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bundle_models(bundle):
    """``{name: model}``: every fitted model of a v2 bundle."""
    models = {f"arm {name}": entry["model"] for name, entry in bundle["arms"].items()}
    models.update({f"auxiliary {arm}/{feature}": model for arm, entries in bundle["auxiliary"].items() for feature, model in entries.items()})
    models.update({f"component {name}/{arm}": model for name, entry in (bundle.get("components") or {}).items() for arm, model in entry["arms"].items()})
    return models


def compare_shares(a, b):
    a, b = getattr(a, "training_level_shares_", {}) or {}, getattr(b, "training_level_shares_", {}) or {}
    if set(a) != set(b):
        return f"training level shares of other features: {sorted(set(a) ^ set(b))}"
    changed = [feature for feature in a if not a[feature].equals(b[feature])]
    return f"training level shares differ for {changed}" if changed else None


def compare_models(root_a, root_b, tolerance):
    """``[(task, model, status, detail)]`` for the bundle of every task directory in both roots."""
    import joblib

    from eodgdl.impute.bundle import as_v2
    from eodgdl.impute.features import build_frame
    from eodgdl.impute.sources import Context
    from eodgdl.impute.spec import load_task

    context, rows = Context(), []
    for task in sorted(str(path) for path in retrain_dirs(root_a) | retrain_dirs(root_b)):
        spec = load_task(task)
        path_a, path_b = root_a / task / spec.bundle_name, root_b / task / spec.bundle_name
        if not (path_a.exists() and path_b.exists()):
            rows.append((task, "bundle", "different", f"only in {'A' if path_a.exists() else 'B'}"))
            continue
        if sha256(path_a) == sha256(path_b):
            rows.append((task, "bundle", "identical", "the same bytes"))
            continue
        a, b = as_v2(joblib.load(path_a), spec), as_v2(joblib.load(path_b), spec)
        frames = {spec.source: build_frame(spec, context).frame}
        if spec.score_source != spec.source:
            frames[spec.score_source] = build_frame(spec, context, role="score").frame
        models_a, models_b = bundle_models(a), bundle_models(b)
        for name in sorted(set(models_a) | set(models_b)):
            if name not in models_a or name not in models_b:
                rows.append((task, name, "different", f"only in {'A' if name in models_a else 'B'}"))
                continue
            model_a, model_b = models_a[name], models_b[name]
            classes_a, classes_b = list(model_a.named_steps["classifier"].classes_), list(model_b.named_steps["classifier"].classes_)
            if classes_a != classes_b:
                rows.append((task, name, "different", f"classes {classes_a} against {classes_b}"))
                continue
            status, details = "identical", []
            for source, frame in frames.items():
                try:
                    pa, pb = model_a.predict_proba(frame), model_b.predict_proba(frame)
                except (KeyError, ValueError) as error:          # a column or a level the other contract lacks
                    status, details = "different", details + [f"{source}: {type(error).__name__}: {error}"]
                    continue
                gap = float(np.abs(pa - pb).max())
                if gap > 0:
                    details.append(f"{source}: max |d| {gap:.3g} over {len(frame):,} rows")
                    if not within(pa, pb, tolerance).all():
                        status = "different"
                    elif status == "identical":
                        status = "within"
            shares = compare_shares(model_a, model_b)
            if shares:
                status, details = "different", details + [shares]
            rows.append((task, name, status, "; ".join(details)))
    return rows


def drop_columns(frame, patterns):
    """``frame`` without the columns matching any of the shell ``patterns``."""
    import fnmatch

    return frame.drop(columns=[column for column in frame.columns if any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_a", type=Path)
    parser.add_argument("root_b", type=Path)
    parser.add_argument("--models", action="store_true", help="Also compare every fitted model of each task's two bundles by predict_proba")
    parser.add_argument("--tolerance", type=float, default=1e-12, help="Absolute below one, relative above (default 1e-12)")
    parser.add_argument("--data", default=None, help="Local data directory for --models (else $EODGDL_DATA_DIR or fetch)")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any difference beyond the tolerance, held-out tables included")
    parser.add_argument("--ignore-columns", nargs="+", default=[], metavar="PATTERN",
                        help="Drop the columns matching these shell patterns from every parquet of both roots before comparing (e.g. a "
                             "pipeline level's given columns against its chain run alone: 'trabajadores_14*' 'prob_trabajadores_14*')")
    args = parser.parse_args()
    if args.data:
        os.environ["EODGDL_DATA_DIR"] = str(Path(args.data).resolve())
    root_a, root_b = args.root_a, args.root_b
    retrains = retrain_dirs(root_a) | retrain_dirs(root_b)
    failed = False

    files_a = {path.relative_to(root_a) for path in root_a.rglob("*.parquet")}
    files_b = {path.relative_to(root_b) for path in root_b.rglob("*.parquet")}
    counts, reports = {"identical": 0, "within": 0, "different": 0}, {"heldout": [], "other": []}
    for relative in sorted(files_a & files_b):
        status, details = compare_frames(*(drop_columns(pd.read_parquet(root / relative), args.ignore_columns) for root in (root_a, root_b)), args.tolerance)
        counts[status] += 1
        if status != "identical":
            heldout = is_heldout_table(relative, retrains)
            reports["heldout" if heldout else "other"].append((relative, status, details))
            failed |= status == "different" and (args.strict or not heldout)
    print(f"parquet: {len(files_a & files_b)} in both: {counts['identical']} identical, {counts['within']} within {args.tolerance:g}, "
          f"{counts['different']} different; only in A: {len(files_a - files_b)}, only in B: {len(files_b - files_a)}")
    for relative in sorted(files_a ^ files_b):
        print(f"  only in {'A' if relative in files_a else 'B'}: {relative}")
        failed = True
    for kind, title in (("other", "tables that moved (not held-out)"), ("heldout", "held-out tables that moved")):
        if reports[kind]:
            print(f"{title}: {len(reports[kind])}")
            for relative, status, details in reports[kind]:
                print(f"  {relative} [{status}]")
                for detail in details[:12]:
                    print(f"      {detail}")
                if len(details) > 12:
                    print(f"      ... {len(details) - 12} more columns")

    json_a = {path.relative_to(root_a) for path in root_a.rglob("*.json")}
    json_b = {path.relative_to(root_b) for path in root_b.rglob("*.json")}
    print(f"json: {len(json_a & json_b)} in both; only in A: {sorted(map(str, json_a - json_b))}, only in B: {sorted(map(str, json_b - json_a))}")
    failed |= bool(json_a ^ json_b)
    for relative in sorted(json_a & json_b):
        differences = compare_json(json.loads((root_a / relative).read_text()), json.loads((root_b / relative).read_text()), args.tolerance)
        if not differences:
            continue
        print(f"  {relative}: {len(differences)} keys differ")
        for path, x, y, beyond in differences:
            heldout = relative.name == "summary.json" and relative.parent in retrains and bool(HELDOUT_KEYS & set(path))
            failed |= beyond and (args.strict or not heldout)
            print(f"      {'.'.join(path)}: {x!r} -> {y!r}{' (held-out)' if heldout else ''}{'' if beyond else ' (within tolerance)'}")

    if args.models:
        rows = compare_models(root_a, root_b, args.tolerance)
        print(f"models: {sum(status == 'identical' for *_, status, _ in rows)} identical, {sum(status == 'within' for *_, status, _ in rows)} within "
              f"{args.tolerance:g}, {sum(status == 'different' for *_, status, _ in rows)} different")
        for task, name, status, detail in rows:
            if status != "identical" or name == "bundle":
                print(f"  {task} {name}: {status}{': ' + detail if detail else ''}")
            failed |= status == "different"

    print("PARITY: differences beyond the tolerance" + (" (held-out tables included)" if args.strict else " outside the held-out tables") if failed else "PARITY: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
