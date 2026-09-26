"""Command-line interface for eodgdl."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eodgdl",
        description="eodgdl — IMEPLAN Guadalajara EOD 2023 survey tools",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch_p = sub.add_parser("fetch", help="Pre-download data files from the mirror")
    fetch_p.add_argument(
        "--dataset",
        choices=["survey", "zones", "all"],
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
        help="Comma-separated screens of a consistent chain gone implausible: zero_stay (a trip leaves the "
        "minute the previous one arrives), long_workday (over 14 h at work), long_day (over 20 h)",
    )
    exp_p.add_argument(
        "--stale", action="store_true",
        help="The persons with hand decisions a rule change left stale, each pre-filled for a new look",
    )
    exp_p.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    ver_p = review_sub.add_parser("verify", help="Read an edited sheet: recover the edits, apply them, recompute problemas")
    ver_p.add_argument("sheet", help="The edited review sheet")
    ver_p.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    ver_p.add_argument("--edits", default=None, help="Write the recovered edits to this CSV")
    ver_p.add_argument("--out", default=None, help="Write the edited persons' sheet, after the edits, to this CSV")
    frz_p = review_sub.add_parser(
        "freeze", help="Merge an edited sheet into the hand decisions load_eod applies (eodgdl/revisions/chains.csv.gz)"
    )
    frz_p.add_argument("sheet", help="The edited review sheet, exported from the tables load_eod() returns now")
    frz_p.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    frz_p.add_argument("--source", default=None, help="The review round's name on its decisions (default: the sheet's name)")
    frz_p.add_argument("--out", default=None, help="Where to write (default: eodgdl/revisions/chains.csv.gz)")

    rew_p = sub.add_parser("reweight", help="Inputs for TMG.SurveyReweight: records, zone system, census targets")
    rew_sub = rew_p.add_subparsers(dest="reweight_cmd", required=True)
    rb_p = rew_sub.add_parser("build", help="Build the record, zone and constraint files from the survey and the census")
    rb_p.add_argument("--data", default=None, help="Local data directory (else fetch)")
    rb_p.add_argument("--out", default="output/reweight", help="Where to write (default: output/reweight/)")
    rc_p = rew_sub.add_parser("check", help="Read a written set back the way the tool will; list what would fail")
    rc_p.add_argument("directory", help="Directory holding the set")

    args = parser.parse_args()

    if args.cmd == "fetch":
        from eodgdl.data import POOCH
        from eodgdl.data._catalog import FILES, SURVEY_FILES, ZONE_FILES

        fnames = {"survey": SURVEY_FILES, "zones": ZONE_FILES, "all": FILES}[args.dataset]
        for fname in fnames:
            path = POOCH.fetch(fname, progressbar=True)
            print(f"  {fname} → {path}")
        print(f"\nFetched {len(fnames)} file(s).")

    elif args.cmd == "info":
        from eodgdl.data._paths import get_pooch_cache_dir
        from eodgdl.data._registry import _BASE_URL

        print(f"Cache directory : {get_pooch_cache_dir()}")
        print(f"Mirror base URL : {_BASE_URL}")
        print("Overrides: $EODGDL_CACHE_DIR (cache), $EODGDL_DATA_DIR (local data dir),")
        print("           $EODGDL_BASE_URL (mirror base).")

    elif args.cmd == "tasha":
        raise SystemExit(_tasha(args))

    elif args.cmd == "review":
        raise SystemExit(_review(args))

    elif args.cmd == "reweight":
        raise SystemExit(_reweight(args))


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
        from eodgdl import load_eod

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        od = tasha.build(load_eod(args.data))
        for table, df in zip(tasha.tables(), od):
            path = out / f"od_{table}{args.suffix}.csv"
            df.to_csv(path, index=False)
            print(f"wrote {path}  ({len(df):,} rows)")
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

    if args.review_cmd == "export":
        rows = review.chain_rows(cleaned, shipped)
        try:
            chosen = []
            if args.codes or not (args.persons or args.screen or args.stale):
                chosen += list(review.pending_persons(rows, [c.strip() for c in args.codes.split(",")] if args.codes else None))
            if args.persons:
                chosen += _persons_arg(args.persons, cleaned.hab)
            if args.screen:
                chosen += list(review.screened_persons(cleaned, [s.strip() for s in args.screen.split(",")]))
            if args.stale:
                chosen += list(zip(stale.household.astype(int), stale.person.astype(int)))
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
        picked = review.screens(cleaned).reindex(persons)
        print("\nscreens over the persons exported: "
              + ", ".join(f"{name} {int(picked[name].fillna(False).sum()):,}" for name in review.SCREENS))
        return 0

    try:
        edited = review.read_sheet(args.sheet)
        edits = review.sheet_edits(edited)
    except ValueError as err:
        print(err)
        return 1
    print(f"read {args.sheet}  ({len(edited):,} rows, {len(edits):,} edits)")
    if args.review_cmd == "freeze":
        try:
            done = review.freeze_round(
                edits, stages, args.source or Path(args.sheet).stem, shown=review.sheet_keys(edited)
            )
        except ValueError as err:
            print(err)
            return 1
        path = review.write_decisions(done.decisions, args.out or review.decisions_path())
        change = done.changes
        print(f"wrote {path}: {len(done.decisions):,} decisions ({change['added']:,} added, {change['changed']:,} "
              f"changed, {change['removed']:,} removed) from {len(done.frozen):,} of the sheet's {len(edits):,} "
              "edits; the rest already held, were moot or were notes")
        if len(done.stale):
            print(f"{len(done.stale):,} stale decisions decided again or let go")
        if len(done.dead):
            print(f"{len(done.dead):,} decisions that no longer changed anything removed")
        if done.verified is not None:
            print()
            for key, value in done.verified.summary.items():
                print(f"  {key:<26} {value:>7,}")
        return 0
    if args.edits:
        edits.to_csv(args.edits, index=False, encoding="utf-8-sig")
        print(f"wrote {args.edits}")
    if edits.empty:
        return 0
    try:
        verified = review.verify_edits(cleaned, shipped, edits, rules)
    except ValueError as err:
        print(err)
        return 1
    print()
    print(verified.persons.to_string(index=False))
    print()
    for key, value in verified.summary.items():
        print(f"  {key:<26} {value:>7,}")
    if args.out:
        review.write_sheet(verified.sheet, args.out)
        print(f"\nwrote {args.out}  ({len(verified.sheet):,} rows)")
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


if __name__ == "__main__":
    main()
