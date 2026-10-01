"""Command-line interface for eodgdl."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eodgdl",
        description="eodgdl — IMEPLAN Guadalajara EOD 2023 survey tools",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch_p = sub.add_parser("fetch", help="Pre-download data files and model bundles from the mirror")
    fetch_p.add_argument(
        "--dataset",
        choices=["survey", "zones", "models", "all"],
        default="all",
        help="Which file group to fetch (default: all)",
    )

    sub.add_parser("info", help="Show cache directory and mirror info")

    tasha_p = sub.add_parser("tasha", help="Model output schema: mappings and validation")
    tasha_sub = tasha_p.add_subparsers(dest="tasha_cmd", required=True)
    tasha_sub.add_parser("check", help="Check mappings.yaml against model_schema.yaml")
    tasha_sub.add_parser("gaps", help="List assumed, constant and unresolved columns")
    val_p = tasha_sub.add_parser("validate", help="Validate produced od_*.csv files")
    val_p.add_argument("directory", help="Directory holding the od_*.csv files")
    val_p.add_argument(
        "--suffix", default="", help="Filename suffix, e.g. _v2 for od_trips_v2.csv"
    )

    build_p = tasha_sub.add_parser("build", help="Build od_*.csv from the survey")
    build_p.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    build_p.add_argument("--out", default="output", help="Where to write (default: output/)")
    build_p.add_argument(
        "--suffix", default="", help="Filename suffix, e.g. _v2 for od_trips_v2.csv"
    )
    build_p.add_argument("--impute", default="output/impute",
                         help="Root of the eodgdl.impute outputs whose completed datasets fill the imputed columns "
                              "(default: output/impute; the pipeline's <root>/tasha/<level>/)")
    build_p.add_argument("--draw", type=int, default=0, help="Which completed dataset (default 0)")
    build_p.add_argument("--no-impute", action="store_true",
                         help="Leave the imputed columns at their mapping's default (IncomeClass 7, no P, Formality O)")

    review_p = sub.add_parser("review", help="Review sheets: the trip chains pending a fix, to edit by hand")
    review_sub = review_p.add_subparsers(dest="review_cmd", required=True)
    exp_p = review_sub.add_parser(
        "export", help="Write the chains pending a fix as a review sheet (CSV); the selections add up"
    )
    exp_p.add_argument("--out", default="chain_review.csv", help="Where to write (default: chain_review.csv)")
    exp_p.add_argument(
        "--codes", default=None,
        help="Comma-separated problemas codes, or a group: breaking (the chain is inconsistent, "
        "fix by hand) or tolerated; a person is exported if a row carries one (default, with no other "
        "selection: any code)",
    )
    exp_p.add_argument("--persons", default=None, help="Comma-separated household/person, e.g. 8992/1,11303/2")
    exp_p.add_argument(
        "--screen", default=None,
        help="Comma-separated screens of a consistent chain gone implausible (eodgdl.review.SCREENS): zero_stay (a "
        "trip leaves the minute the previous one arrives), zero_work, short_work (under 30 min at work or school), "
        "long_workday (over 14 h at work), long_day (over 20 h), early_start (a first non-work trip before 05:00), "
        "long_errand (8 h or more at an errand), companion_apart (a start a household member reported alike, read "
        "otherwise), companion_12h (a trip made with a household member, 12 h from theirs)",
    )
    exp_p.add_argument(
        "--stale", action="store_true",
        help="The persons with hand decisions a rule change left stale, each pre-filled for a new look",
    )
    exp_p.add_argument(
        "--since", default=None,
        help="A snapshot (`eodgdl review snapshot`): the persons whose values moved since it was taken",
    )
    exp_p.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    snap_p = review_sub.add_parser(
        "snapshot", help="Write every trip's values as load_eod() gives them, to compare after a rule change"
    )
    snap_p.add_argument("--out", required=True, help="Where to write (a .csv.gz)")
    snap_p.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    for name, help_text in (
        ("verify", "Check an edited sheet as freeze does and say what it would change, writing nothing"),
        ("freeze", "Merge an edited sheet into the hand decisions load_eod applies (eodgdl/revisions/chains.csv.gz)"),
    ):
        frz_p = review_sub.add_parser(name, help=help_text)
        frz_p.add_argument("sheet", help="The edited review sheet, exported from the tables load_eod() returns now")
        frz_p.add_argument("--data", default=None, help="Local survey directory (else fetch)")
        frz_p.add_argument("--source", default=None, help="The review round's name on its decisions (default: the sheet's name)")
        frz_p.add_argument("--dry-run", action="store_true", default=name == "verify",
                           help="Check and report, write no decisions (what verify does)")
        frz_p.add_argument("--edits", default=None, help="Write the edits recovered from the sheet to this CSV")
        frz_p.add_argument("--sheet-out", default=None,
                           help="Write the edited persons' sheet, after the edits, to this CSV")
        frz_p.add_argument("--out", default=None, help="Where to write the decisions (default: eodgdl/revisions/chains.csv.gz)")

    rew_p = sub.add_parser("reweight", help="Inputs for TMG.SurveyReweight: records, zone system, census targets")
    rew_sub = rew_p.add_subparsers(dest="reweight_cmd", required=True)
    rb_p = rew_sub.add_parser("build", help="Build the record, zone and constraint files from the survey and the census")
    rb_p.add_argument("--data", default=None, help="Local data directory (else fetch)")
    rb_p.add_argument("--out", default="output/reweight", help="Where to write (default: output/reweight/)")
    rc_p = rew_sub.add_parser("check", help="Read a written set back the way the tool will; list what would fail")
    rc_p.add_argument("directory", help="Directory holding the set")

    imp_p = sub.add_parser("impute", help="Imputation tasks, chains and pipelines (eodgdl.impute): score, retrain, evaluate, compare")
    imp_sub = imp_p.add_subparsers(dest="impute_cmd", required=True)
    for name, help_text in (("score", "Score a task's or a chain's rows with the fitted bundles"),
                            ("retrain", "Select, evaluate, refit and score a task (or every task of a chain); write the bundles and evaluation tables"),
                            ("evaluate", "Diagnostics of a chain (its evaluation: section)"),
                            ("compare", "Fold log losses of candidate specifications against the task's, at its bundle's winning hyperparameters")):
        cmd_p = imp_sub.add_parser(name, help=help_text)
        cmd_p.add_argument("task", metavar="chain" if name == "evaluate" else "task",
                           help="Chain (src/eodgdl/impute/chains/<chain>.yaml)" if name == "evaluate" else
                           "Task (src/eodgdl/impute/tasks/<task>.yaml)" + (", chain (impute/chains/<chain>.yaml) or pipeline (impute/pipelines/<pipeline>.yaml)" if name != "compare" else ""))
        cmd_p.add_argument("--data", default=None, help="Local data directory (else $EODGDL_DATA_DIR or fetch)")
        cmd_p.add_argument("--refresh", action="store_true", help="Rebuild the cached feature frames")
        if name in ("retrain", "compare"):
            cmd_p.add_argument("--jobs", type=int, default=-1, help="Parallel workers for the CV fits (default: all cores; 1 = in process)")
        if name != "retrain":
            default = "output/impute" if name == "evaluate" else None
            cmd_p.add_argument("--retrained", default=default, help="Take each task's bundle from <dir>/<task>/ when a retrain wrote one there, else the installed one"
                                                                     + (" (default output/impute; also where the upstream scenarios are read)" if default else ""))
    imp_sub.choices["score"].add_argument("--out", default=None, help="For a task, the parquet of keys and outputs (default: output/impute/<task>_scores.parquet); for a chain or a pipeline, a directory (default: output/impute/<name>/)")
    imp_sub.choices["retrain"].add_argument("--out", default="output/impute", help="Each task is written to <out>/<task>/, where --retrained <out> finds it (default: output/impute)")
    imp_sub.choices["retrain"].add_argument("--bootstrap", type=int, default=0, metavar="B",
                                            help="Also refit the selected configurations on B cluster bootstraps of the training rows (<out>/<task>/bootstrap/), for a chain's uncertainty: bootstrap")
    imp_sub.choices["score"].add_argument("--draws", type=int, default=None, metavar="M",
                                          help="For a chain: M multiple imputations (propagation draws), every value drawn; written to <out> (default: output/impute/<chain>/multiple_imputation/). For a pipeline: M instead of its draws")
    imp_sub.choices["score"].add_argument("--bootstrap", action="store_true",
                                          help="For a chain: draw each imputation's models from the tasks' bootstrap bundles under --retrained (uncertainty: bootstrap; needs draws)")
    imp_sub.choices["score"].add_argument("--no-aggregate", action="store_true",
                                          help="For a pipeline: leave the drawn values out of its aggregates (each level then reproduces its chain alone: the parity "
                                               "check); written to <out> (default: output/impute/<pipeline>/no_aggregate/, apart from what tasha build reads)")
    imp_sub.choices["evaluate"].add_argument("--out", default=None, help="Directory (default: <retrained>/<chain>/); tables go to <out>/evaluation/")
    imp_cmp = imp_sub.choices["compare"]
    imp_cmp.add_argument("--spec", required=True, help="YAML mapping each candidate name to a spec fragment merged into the task's")
    imp_cmp.add_argument("--seeds", type=int, nargs="+", default=[42], help="CV seeds (repeated CV; default 42)")
    imp_cmp.add_argument("--out", default=None, help="CSV prefix for <out>_folds.csv and <out>_paired.csv")

    args = parser.parse_args()

    if args.cmd == "fetch":
        from eodgdl.data import POOCH
        from eodgdl.data._catalog import FILES, MODEL_FILES, SURVEY_FILES, ZONE_FILES

        fnames = {"survey": SURVEY_FILES, "zones": ZONE_FILES, "models": MODEL_FILES, "all": FILES}[args.dataset]
        for fname in fnames:
            path = POOCH.fetch(fname, progressbar=True)
            print(f"  {fname} → {path}")
        print(f"\nFetched {len(fnames)} file(s).")

    elif args.cmd == "info":
        from eodgdl.data._paths import get_pooch_cache_dir
        from eodgdl.data._registry import _BASE_URL, _MODELS_URL

        print(f"Cache directory : {get_pooch_cache_dir()}")
        print(f"Mirror base URL : {_BASE_URL}")
        print(f"Models URL      : {_MODELS_URL}")
        print("Overrides: $EODGDL_CACHE_DIR (cache), $EODGDL_DATA_DIR (local data dir),")
        print("           $EODGDL_MODELS_DIR (local models dir; default: models/ beside the data dir),")
        print("           $EODGDL_BASE_URL (mirror base), $EODGDL_MODELS_URL (models base).")

    elif args.cmd == "tasha":
        raise SystemExit(_tasha(args))

    elif args.cmd == "review":
        raise SystemExit(_review(args))

    elif args.cmd == "reweight":
        raise SystemExit(_reweight(args))

    elif args.cmd == "impute":
        raise SystemExit(_impute(args))


def _report(problems: list[str], ok_message: str) -> int:
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(ok_message)
    return 0


def _chain_notes(trips) -> None:
    """Print tasha.chain_report: informational, never changes the exit code."""
    from eodgdl import tasha

    notes = tasha.chain_report(trips)
    if notes:
        print("chain diagnostics (informational; the cleaning rules are in eodgdl.chains):")
        for note in notes:
            print(f"  - {note}")
        print()


def _tasha(args) -> int:
    from pathlib import Path

    import pandas as pd

    from eodgdl import tasha

    if args.tasha_cmd == "check":
        return _report(
            tasha.check_mappings(), "mappings.yaml agrees with model_schema.yaml"
        )

    if args.tasha_cmd == "gaps":
        print(tasha.gaps()[["table", "column", "status"]].to_string(index=False))
        return 0

    if args.tasha_cmd == "build":
        import os

        from eodgdl import load_eod

        if args.data:   # the survey load_completed checks the completed datasets against: the one built from
            os.environ["EODGDL_DATA_DIR"] = str(Path(args.data).resolve())
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        completed = None if args.no_impute else tasha.load_completed(args.impute, args.draw)
        od = tasha.build(load_eod(args.data), completed=completed)
        for table, df in zip(tasha.tables(), od):
            path = out / f"od_{table}{args.suffix}.csv"
            df.to_csv(path, index=False)
            print(f"wrote {path}  ({len(df):,} rows)")
        path = out / f"od_provenance{args.suffix}.json"
        if completed is not None:
            import json

            # which completed dataset filled the imputed columns, and the bundles behind it
            provenance = {"impute_root": str(args.impute), "draw": args.draw,
                          "pipelines": {pipeline: next(iter(levels.values())).attrs["provenance"] for pipeline, levels in completed.items()}}
            path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"wrote {path}  (draw {args.draw} of {', '.join(completed)})")
        elif path.exists():   # an imputed build's record, which these tables no longer are
            path.unlink()
            print(f"removed {path}  (the tables take the mappings' defaults)")
        print()
        _chain_notes(od.trips)
        return _report(tasha.validate_all(*od), "conforms to model_schema.yaml")

    directory = Path(args.directory)
    frames = {}
    for table in tasha.tables():
        path = directory / f"od_{table}{args.suffix}.csv"
        if path.exists():
            # Zone ids are all-digit strings; without this they read as int64.
            frames[table] = pd.read_csv(
                path, dtype=dict.fromkeys(tasha.zone_columns(table), str)
            )
            print(f"read {path}  ({len(frames[table]):,} rows)")
        else:
            print(f"skip {path}  (not found)")
    print()
    if not frames:
        print(f"no od_*{args.suffix}.csv files found in {directory}")
        return 1
    if "trips" in frames:
        _chain_notes(frames["trips"])
    return _report(tasha.validate_all(**frames), "conforms to model_schema.yaml")


def _persons_arg(text: str, hab) -> list[tuple[int, int]]:
    """``household/person,…`` as keys, each checked against ``hab``."""
    persons = []
    for part in text.split(","):
        household, _, person = part.strip().partition("/")
        key = (int(household), int(person))
        if key not in hab.index:
            raise ValueError(f"no person {household}/{person} in the survey")
        persons.append(key)
    return persons


def _review(args) -> int:
    from pathlib import Path

    import pandas as pd

    from eodgdl import load_stages, review
    from eodgdl.chains import BREAKING_ISSUES, ISSUE_CODES, PERSON, has_code

    # the survey as shipped, after the rules and after the hand decisions, from one read; a decision a rule
    # change left stale is set aside rather than stopping the tools that let a round decide it again
    stages = load_stages(args.data, skip_stale=True)
    shipped, rules, cleaned = stages
    stale = review.stale_decisions(rules, shipped)
    if len(stale):
        print(f"{len(stale):,} hand decisions no longer apply to the rules' output (a rule changed under them) "
              "and are set aside; `eodgdl review export --stale` puts them on a sheet\n")

    if args.review_cmd == "snapshot":
        path = review.write_snapshot(review.snapshot(cleaned, shipped), args.out)
        print(f"wrote {path}  ({len(shipped.trips):,} trips); after the change: eodgdl review export --since {path}")
        return 0

    if args.review_cmd == "export":
        rows = review.chain_rows(cleaned, shipped)
        try:
            chosen = []
            if args.codes or not (args.persons or args.screen or args.stale or args.since):
                chosen += list(review.pending_persons(rows, [c.strip() for c in args.codes.split(",")] if args.codes else None))
            if args.persons:
                chosen += _persons_arg(args.persons, cleaned.hab)
            if args.screen:
                chosen += list(review.screened_persons(cleaned, [s.strip() for s in args.screen.split(",")], shipped))
            if args.stale:
                chosen += list(zip(stale.household.astype(int), stale.person.astype(int)))
            if args.since:
                moved = review.changed_persons(review.read_snapshot(args.since), review.snapshot(cleaned, shipped))
                print(f"{len(moved):,} persons' values moved since {args.since}")
                chosen += list(moved)
        except ValueError as err:
            print(err)
            return 1
        chosen = sorted(set(map(tuple, chosen)))
        persons = pd.MultiIndex.from_arrays([[k[0] for k in chosen], [k[1] for k in chosen]], names=PERSON)
        sheet = review.prefill_stale(review.chain_sheet(rows, cleaned.hab, persons), stale)
        path = review.write_sheet(sheet, args.out)
        print(f"wrote {path}  ({len(persons):,} persons, {len(sheet):,} rows)")
        shown = stale[pd.MultiIndex.from_arrays([stale.household, stale.person]).isin(persons)]
        if len(shown):
            print(f"{len(shown):,} stale decisions written into their new … cells, with why in the note")
        print()
        selected = rows[rows.index.droplevel("folio_viaje").isin(persons)]
        counts = pd.DataFrame({
            "rows": {c: int(has_code(selected.problemas, c).sum()) for c in ISSUE_CODES},
            "persons": {c: int(has_code(selected.problemas, c).groupby(level=PERSON).any().sum()) for c in ISSUE_CODES},
        }).rename_axis("problemas")
        counts["group"] = ["breaking" if c in BREAKING_ISSUES else "tolerated" for c in counts.index]
        left = counts.loc[counts["rows"] > 0]
        print(left.to_string() if len(left) else "no chain carries those codes")
        picked = review.screens(cleaned, shipped).reindex(persons)
        print("\nscreens over the persons exported: "
              + ", ".join(f"{name} {int(picked[name].fillna(False).sum()):,}" for name in review.SCREENS))
        return 0

    # verify and freeze: the same checks; verify (freeze --dry-run) writes no decisions
    try:
        edited = review.read_sheet(args.sheet)
        edits = review.sheet_edits(edited)
    except ValueError as err:
        print(err)
        return 1
    print(f"read {args.sheet}  ({len(edited):,} rows, {len(edits):,} edits)")
    if args.edits:
        edits.to_csv(args.edits, index=False, encoding="utf-8-sig")
        print(f"wrote {args.edits}")
    try:
        done = review.freeze_round(edits, stages, args.source or Path(args.sheet).stem, shown=review.sheet_keys(edited))
    except ValueError as err:
        print(err)
        return 0 if args.dry_run and str(err).startswith("nothing to freeze") else 1
    change = done.changes
    what = "would write" if args.dry_run else "wrote"
    path = args.out or review.decisions_path()
    if not args.dry_run:
        review.write_decisions(done.decisions, path)
    print(f"{what} {path}: {len(done.decisions):,} decisions ({change['added']:,} added, {change['changed']:,} "
          f"changed, {change['removed']:,} removed, {change['notes']:,} with a new note) from {len(done.frozen):,} of "
          f"the sheet's {len(edits):,} edits; the rest already held, were moot or were notes")
    if len(done.stale):
        print(f"{len(done.stale):,} stale decisions decided again or let go")
    if len(done.dead):
        print(f"{len(done.dead):,} decisions that no longer changed anything removed")
    if done.verified is not None:
        print()
        print(done.verified.persons.to_string(index=False))
        print()
        for key, value in done.verified.summary.items():
            print(f"  {key:<34} {value:>7,}")
        if args.sheet_out:
            review.write_sheet(done.verified.sheet, args.sheet_out)
            print(f"\nwrote {args.sheet_out}  ({len(done.verified.sheet):,} rows)")
    return 0


def _reweight(args) -> int:
    from eodgdl import reweight

    if args.reweight_cmd == "build":
        from eodgdl import load_eod

        files = reweight.build(load_eod(args.data), data_dir=args.data)
        for path in reweight.write(files, args.out):
            print(f"wrote {path}")
        print()
        return _report(reweight.check(args.out), "the set is loadable and every constraint is feasible")

    return _report(reweight.check(args.directory), "the set is loadable and every constraint is feasible")


def _impute(args) -> int:
    import os
    from pathlib import Path

    if args.data:
        os.environ["EODGDL_DATA_DIR"] = str(Path(args.data).resolve())
    from eodgdl.impute import run
    from eodgdl.impute.chain import is_chain, load_chain, retrain_chain, run_chain, write_chain
    from eodgdl.impute.pipeline import is_pipeline, load_pipeline, retrain_pipeline, run_pipeline, write_pipeline
    from eodgdl.impute.sources import Context
    from eodgdl.impute.spec import load_task

    context = Context(refresh=args.refresh)
    if args.impute_cmd == "evaluate":
        from eodgdl.impute.diagnostics import evaluate_chain, write_evaluation

        tables, summary = evaluate_chain(args.task, retrained=args.retrained, context=context)
        out = write_evaluation(tables, summary, args.out or f"{args.retrained}/{args.task}")
        shown = tables["headline"] if "headline" in tables else tables["distribution"].pivot_table(index=["variant", "column"], columns="level", values="share")
        print(shown.round(4).to_string())
        print(f"wrote {len(tables)} tables to {out}")
        return 0

    if is_pipeline(args.task) and args.impute_cmd in ("score", "retrain"):
        pipeline = load_pipeline(args.task)
        if args.impute_cmd == "score":
            # with the aggregates off, not a completed dataset to build from: apart from what tasha build reads
            out = Path(args.out or f"output/impute/{pipeline.name}" + ("/no_aggregate" if args.no_aggregate else ""))
            result = run_pipeline(pipeline, context=context, retrained=args.retrained, draws=args.draws, aggregate=not args.no_aggregate)
            write_pipeline(result, out)
            sizes = ", ".join(f"{name} {len(level.frame):,} rows" for name, level in result.levels.items())
            print(f"{pipeline.name} ({result.provenance['draws']} draws, aggregate {'on' if result.provenance['aggregate'] else 'off'}): {sizes}; wrote {out}")
        else:
            for task, (path, digest) in retrain_pipeline(pipeline, args.out, context=context, n_jobs=args.jobs, bootstrap=args.bootstrap).items():
                print(f"{task}: wrote {path} (sha256 {digest})")
        return 0
    if is_chain(args.task) and args.impute_cmd in ("score", "retrain"):
        chain = load_chain(args.task)
        if args.impute_cmd == "score":
            default = f"output/impute/{chain.name}"
            if args.draws is not None or args.bootstrap:
                from eodgdl.impute.chain import parse_chain

                changed = {**({"propagation": "draws", "draws": args.draws} if args.draws is not None else {}), **({"uncertainty": "bootstrap"} if args.bootstrap else {})}
                chain, default = parse_chain({**chain.raw, **changed}), f"{default}/multiple_imputation"
            out = Path(args.out or default)
            result = run_chain(chain, context=context, retrained=args.retrained)
            write_chain(result, out)
            print(f"{chain.name} ({chain.mode}, {chain.propagation}): {len(result.frame):,} rows, {len(result.completions):,} completions; wrote {out}")
        else:
            for task, (path, digest) in retrain_chain(chain, args.out, context=context, n_jobs=args.jobs, bootstrap=args.bootstrap).items():
                print(f"{task}: wrote {path} (sha256 {digest})")
        return 0
    spec = load_task(args.task)
    if args.impute_cmd == "score":
        scored = run.score_task(spec, retrained=args.retrained, context=context)
        out = Path(args.out or f"output/impute/{spec.name}_scores.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        run.output_frame(spec, scored).to_parquet(out, index=False)
        imputed = scored[f"{spec.prefix}_fue_imputado"]
        print(f"scored {len(scored):,} rows, {int(imputed.sum()):,} imputed; wrote {out}")
        return 0
    if args.impute_cmd == "retrain":
        result = run.retrain(spec, context=context, n_jobs=args.jobs)
        path, digest = run.write_retrain(result, Path(args.out) / spec.name)
        if args.bootstrap:
            paths = run.write_bootstrap(spec, run.bootstrap_bundles(spec, result.bundle, args.bootstrap, context=context, n_jobs=args.jobs), Path(args.out) / spec.name)
            print(f"wrote {len(paths)} bootstrap bundles to {paths[0].parent}")
        for arm, entry in result.summary["arms"].items():
            chosen = entry["selected"]
            print(f"{arm}: {chosen['model']} {chosen['best_params']} CV log loss {chosen['weighted_log_loss']:.4f}, "
                  f"held-out {entry['test_metrics']['weighted_log_loss']:.4f}")
        print(f"wrote {path} (sha256 {digest}); to install it, copy it under models/ and update src/eodgdl/data/registry.txt")
        return 0
    import yaml

    with open(args.spec, encoding="utf-8") as handle:
        candidates = yaml.safe_load(handle)
    folds, paired = run.compare(spec, candidates, seeds=tuple(args.seeds), retrained=args.retrained, context=context, n_jobs=args.jobs)
    print(paired.round(5).to_string(index=False))
    if args.out:
        folds.to_csv(f"{args.out}_folds.csv", index=False)
        paired.to_csv(f"{args.out}_paired.csv", index=False)
    return 0


if __name__ == "__main__":
    main()
