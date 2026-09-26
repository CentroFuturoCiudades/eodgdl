"""Fold the seven hand passes into the one table of decisions ``load_eod`` applies (one-off, 2026-09-25).

Passes 1-7 were frozen one after another, each against the tables the passes before it
left, so later passes undid earlier ones: 12,344 frozen edits for 5,262 standing
decisions. They are archived under ``notebooks/revisions/`` and nothing loads them. This
script merges them, in order, into ``src/eodgdl/revisions/chains.csv.gz`` exactly as a
review round is merged (``review.freeze_edits`` then ``review.merge_decisions``), and
after every pass checks that the decisions, applied once to the chain rules' output,
give what the passes gave one after another. Each decision keeps the note of the pass
that set it and names that pass's review sheet as its source.

    uv run python scripts/revisions/squash_passes.py --data data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eodgdl import load_eod, review

ARCHIVE = Path(__file__).resolve().parents[2] / "notebooks" / "revisions"


def archived_passes() -> list[tuple[int, pd.DataFrame]]:
    """The frozen passes, in the order they were applied: [(n, edits)]."""
    out = []
    for path in sorted(ARCHIVE.glob("chains_*.csv.gz"), key=lambda p: int(p.name[len("chains_"):-len(".csv.gz")])):
        edits = pd.read_csv(path, dtype=str, keep_default_na=False)
        edits[review.KEYS] = edits[review.KEYS].astype(int)
        out.append((int(path.name[len("chains_"):-len(".csv.gz")]), edits))
    return out


def same(a, b) -> bool:
    return a.trips.equals(b.trips) and a.legs.equals(b.legs) and a.hab.equals(b.hab)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Local survey directory (else fetch)")
    args = parser.parse_args()

    shipped = load_eod(args.data, clean_chains=False)
    rules = load_eod(args.data, revise_chains=False)
    decisions = review.decision_frame([])
    tables = rules
    for n, edits in archived_passes():
        sheet = "chain_review" if n == 1 else f"chain_review_{n}"   # notebooks/<sheet>.csv, the pass's review sheet
        frozen = review.freeze_edits(edits, tables, shipped, rules)
        merged = review.merge_decisions(decisions, frozen, rules, shipped, sheet)
        tables = review.apply_edits(tables, edits, shipped, rules)
        if not same(review.apply_revisions(rules, shipped, merged), tables):
            raise SystemExit(f"pass {n}: the merged decisions do not give what the passes give")
        change = review.decision_changes(decisions, merged)
        print(f"pass {n} ({sheet}): {len(edits):,} frozen edits, {len(frozen):,} that change the tables -> "
              f"{change['added']:,} decisions added, {change['changed']:,} changed, {change['removed']:,} removed; "
              f"{len(merged):,} standing")
        decisions = merged
    path = review.write_decisions(decisions, review.decisions_path())
    print(f"wrote {path}: {len(decisions):,} decisions, {decisions.set_index(review.KEYS[:2]).index.nunique():,} persons")


if __name__ == "__main__":
    main()
