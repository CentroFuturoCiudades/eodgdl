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


def _report(problems: list[str], ok_message: str) -> int:
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(ok_message)
    return 0


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
    return _report(tasha.validate_all(**frames), "conforms to model_schema.yaml")


if __name__ == "__main__":
    main()
