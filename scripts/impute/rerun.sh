#!/usr/bin/env bash
# Rerun every eodgdl.impute output into ROOT, the layout of output/impute/, for scripts/impute/parity.py: retrain the
# tasks (each to ROOT/<task>/), score giro (ROOT/giro_scores.parquet, as `eodgdl impute score giro` writes
# output/impute/giro_scores.parquet) and both chains, and evaluate the chains, all with --retrained ROOT (a task without
# a retrain in ROOT scores from its installed bundle, which a change to its spec leaves unusable until reinstalled).
# Each retrain also writes BOOTSTRAP bootstrap bundles (ROOT/<task>/bootstrap/), and each chain its completed datasets
# with a bootstrap model per draw (ROOT/<chain>/multiple_imputation/), which the evaluations' Rubin's rules read too.
#
#   scripts/impute/rerun.sh ROOT [TASK ...]      (from the repository root; ~25 minutes for the six tasks)
#
# TASK ...: the tasks to retrain (default: all six). A task not retrained into ROOT scores from its installed bundle
# in the chains (--retrained falls back to it); `evaluate sector_informality` needs giro's retrain in ROOT.
# Environment: DATA (default data), JOBS (the CV workers of a retrain, default -1: every core), BOOTSTRAP (the bootstrap
# bundles per task, default 50; 0 writes none, and the chains' multiple imputations are then skipped).
#
# Every step must succeed (set -o pipefail: a retrain piped through grep would otherwise exit 0), and every retrain
# must leave its bundle.
set -euo pipefail

root=${1:?usage: scripts/impute/rerun.sh ROOT [TASK ...]}
shift
tasks=("$@")
if [ ${#tasks[@]} -eq 0 ]; then
    tasks=(giro informality educacion_jefe amai_banos amai_dormitorios amai_trabajadores)
fi
data=${DATA:-data}
jobs=${JOBS:--1}
bootstrap=${BOOTSTRAP:-50}
mkdir -p "$root"

for task in "${tasks[@]}"; do
    echo "== retrain $task"
    uv run eodgdl impute retrain "$task" --data "$data" --out "$root" --jobs "$jobs" --bootstrap "$bootstrap"
    bundle=$(uv run python -c "from eodgdl.impute.spec import load_task; print(load_task('$task').bundle_name)")
    if [ ! -f "$root/$task/$bundle" ]; then
        echo "retrain $task left no bundle at $root/$task/$bundle" >&2
        exit 1
    fi
done

echo "== score giro"
uv run eodgdl impute score giro --data "$data" --retrained "$root" --out "$root/giro_scores.parquet"
for chain in sector_informality nse; do
    echo "== score $chain"
    uv run eodgdl impute score "$chain" --data "$data" --retrained "$root" --out "$root/$chain"
    if [ "$bootstrap" -gt 0 ]; then
        echo "== multiple imputations of $chain"
        uv run eodgdl impute score "$chain" --data "$data" --retrained "$root" --draws 50 --bootstrap --out "$root/$chain/multiple_imputation"
    fi
    echo "== evaluate $chain"
    uv run eodgdl impute evaluate "$chain" --data "$data" --retrained "$root"
done
echo "== done: $root"
