#!/usr/bin/env bash
# Rerun every eodgdl.impute output into ROOT, the layout of output/impute/, for scripts/impute/parity.py: retrain the
# tasks (each to ROOT/<task>/), score giro from its installed bundle (ROOT/giro_scores.parquet, as `eodgdl impute
# score giro` writes output/impute/giro_scores.parquet), then score and evaluate both chains with --retrained ROOT.
#
#   scripts/impute/rerun.sh ROOT [TASK ...]      (from the repository root; ~25 minutes for the six tasks)
#
# TASK ...: the tasks to retrain (default: all six). A task not retrained into ROOT scores from its installed bundle
# in the chains (--retrained falls back to it); `evaluate sector_informality` needs giro's retrain in ROOT.
# Environment: DATA (default data), JOBS (the CV workers of a retrain, default -1: every core).
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
mkdir -p "$root"

for task in "${tasks[@]}"; do
    echo "== retrain $task"
    uv run eodgdl impute retrain "$task" --data "$data" --out "$root" --jobs "$jobs"
    bundle=$(uv run python -c "from eodgdl.impute.spec import load_task; print(load_task('$task').bundle_name)")
    if [ ! -f "$root/$task/$bundle" ]; then
        echo "retrain $task left no bundle at $root/$task/$bundle" >&2
        exit 1
    fi
done

echo "== score giro (installed bundle)"
uv run eodgdl impute score giro --data "$data" --out "$root/giro_scores.parquet"
for chain in sector_informality nse; do
    echo "== score $chain"
    uv run eodgdl impute score "$chain" --data "$data" --retrained "$root" --out "$root/$chain"
    echo "== evaluate $chain"
    uv run eodgdl impute evaluate "$chain" --data "$data" --retrained "$root"
done
echo "== done: $root"
