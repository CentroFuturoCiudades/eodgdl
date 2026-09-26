"""Fold the seven hand passes into one table of decisions (one-off, 2026-09-25; kept as the table's history).

Passes 1-7 were frozen one after another, each against the tables the passes before it
left, so later passes undid earlier ones: 12,344 frozen edits for 5,262 standing
decisions. They are archived under ``notebooks/revisions/`` and nothing loads them. This
script merged them, in order, exactly as a review round merges (``review.freeze_round``,
which also checks that the decisions, applied once to the chain rules' output, give what
the passes gave one after another). Each decision keeps the note of the pass that set it
and names that pass's review sheet as its source.

The table has moved on since (review rounds 8-10 in ``src/eodgdl/revisions/README.md``,
and the rules with them), so the script writes only where it is told: run against
today's rules it shows how the passes fold, not the table ``load_eod`` applies.

    uv run python scripts/revisions/squash_passes.py --data data --out /tmp/passes_1_7.csv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eodgdl import EODStages, load_eod, review

ARCHIVE = Path(__file__).resolve().parents[2] / "notebooks" / "revisions"


def archived_passes() -> list[tuple[int, pd.DataFrame]]:
    """The frozen passes, in the order they were applied: [(n, edits)]."""
    out = []
    for path in sorted(ARCHIVE.glob("chains_*.csv.gz"), key=lambda p: int(p.name[len("chains_"):-len(".csv.gz")])):
        edits = pd.read_csv(path, dtype=str, keep_default_na=False)
        edits[review.KEYS] = edits[review.KEYS].astype(int)
        out.append((int(path.name[len("chains_"):-len(".csv.gz")]), edits))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    parser.add_argument("--out", required=True, help="Where to write the folded table (never the live one by default)")
    args = parser.parse_args()

    shipped = load_eod(args.data, clean_chains=False)
    rules = load_eod(args.data, revise_chains=False)
    decisions = review.decision_frame([])
    tables = rules
    for n, edits in archived_passes():
        sheet = "chain_review" if n == 1 else f"chain_review_{n}"   # notebooks/<sheet>.csv, the pass's review sheet
        done = review.freeze_round(edits, EODStages(shipped, rules, tables), sheet, decisions=decisions)
        tables = review.apply_edits(tables, edits, shipped, rules)
        print(f"pass {n} ({sheet}): {len(edits):,} frozen edits, {len(done.frozen):,} that change the tables -> "
              f"{done.changes['added']:,} decisions added, {done.changes['changed']:,} changed, "
              f"{done.changes['removed']:,} removed; {len(done.decisions):,} standing")
        decisions = done.decisions
    path = review.write_decisions(decisions, args.out)
    print(f"wrote {path}: {len(decisions):,} decisions, {decisions.set_index(review.KEYS[:2]).index.nunique():,} persons")


if __name__ == "__main__":
    main()
