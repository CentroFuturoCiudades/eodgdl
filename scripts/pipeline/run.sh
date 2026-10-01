#!/usr/bin/env bash
# The whole processing into ROOT, stage by stage (docs/pipeline.md), each stage writing its manifest:
#
#   0. load            load_eod() on DATA (every later stage reads it; its versions are in each manifest)
#   1. empleo, design  P(trabaja) of the 4,370 unanswered from models/od_empleo_design_model.joblib (inside stage 2)
#   2. reweight inputs eodgdl reweight build -> ROOT/reweight/ (+ manifest.json)
#   3. TMG weight      external: TMG.SurveyReweight, run by hand on ROOT/reweight/. Without the weight in DATA the
#                      script stops here, naming what to hand over and what must come back.
#   4. retrain         every task on the TMG weight, config's bootstrap refits each -> ROOT/impute/<task>/ (scripts/impute/rerun.sh)
#   5. completed data  pipeline tasha, 50 draws -> ROOT/impute/tasha/ (rerun.sh's last step)
#   6. TASHA build     eodgdl tasha build --draw DRAW -> ROOT/tasha/od_*.csv (DRAW: config's draw)
#   then               eodgdl pipeline verify ROOT
#
#   scripts/pipeline/run.sh ROOT          (from the repository root)
#
# The run parameters are config/config.yaml's (eodgdl.config): its weight, its bootstrap (rerun.sh), its draw. Environment:
# DATA (default data), JOBS (rerun.sh's CV workers, default -1), and DRAW / BOOTSTRAP to override config's for one run.
# Stage 4 runs only when data/EOD_peso_hogar_TMG.csv is there with its sidecar, was fitted on the reweight inputs stage 2
# just wrote (the same outputs_digest), and config's weight is tmg; with weight tmg and no weight file nothing runs.
set -euo pipefail

root=${1:?usage: scripts/pipeline/run.sh ROOT}
data=${DATA:-data}
export EODGDL_DATA_DIR
EODGDL_DATA_DIR=$(cd "$data" && pwd)
draw=${DRAW:-$(uv run python -m eodgdl.config draw)}
weight_name=$(uv run python -m eodgdl.config weight)
weight="$data/EOD_peso_hogar_TMG.csv"
if [ "$weight_name" = tmg ] && [ ! -f "$weight" ]; then
    echo "the run's weight is tmg (config/config.yaml, or \$EODGDL_CONFIG), but there is no $weight: import it first (eodgdl reweight import-weight)" >&2
    exit 1
fi
mkdir -p "$root"

echo "== stages 0-2: the reweight inputs (empleo on the design weight inside) -> $root/reweight"
uv run eodgdl reweight build --data "$data" --out "$root/reweight"

if [ ! -f "$weight" ]; then
    cat <<EOF

== stage 3 is external: stop here.
Hand over: $root/reweight/ (README.md says how to configure the XTMF2 model system; manifest.json records what it holds).
Must come back: the tool's UpdatedExpansionFactorsFile (one UpdatedExpansionFactor per row of HouseholdRecords.csv), and
which constraint year was used; if possible the tool's commit, the exported .xmsys and the ConstraintReportFile. Then:

  uv run eodgdl reweight import-weight UPDATED.csv --year YEAR --reweight $root/reweight --data $data [--tool-commit C] [--configuration X.xmsys] [--report R.csv]

set weight: tmg in config/config.yaml, and run this script again.
EOF
    exit 0
fi

echo "== stage 3: the TMG weight against the reweight inputs just written"
uv run python - "$root" <<'EOF'
import sys

from eodgdl import config
from eodgdl.impute.sources import get_source, weight_file
from eodgdl.reweight.weight import WEIGHT_CSV, check_weight

problems = check_weight(None, sys.argv[1])
if config.weight() != "tmg":
    problems.append(f"{config.path()}: weight is {config.weight()!r}; set it to tmg for stages 4-6")
eod = [name for name in ("eod.workers", "eod.persons", "eod.labour", "eod.dwellings")
       if (weight_file(get_source(name).config) or {}).get("file") != WEIGHT_CSV]
if eod and config.weight() == "tmg":
    problems.append(f"under weight tmg the sources {eod} do not read {WEIGHT_CSV}: see run_weights in src/eodgdl/impute/sources/eod.yaml")
if problems:
    sys.exit("\n".join(problems))
print("the weight was fitted on these reweight inputs, and config's weight tmg makes the EOD sources read it")
EOF

echo "== stages 4-5: every task retrained on the TMG weight, the pipeline tasha -> $root/impute"
DATA="$data" scripts/impute/rerun.sh "$root/impute"

echo "== stage 6: the TASHA tables of draw $draw -> $root/tasha"
uv run eodgdl tasha build --data "$data" --impute "$root/impute" --draw "$draw" --out "$root/tasha"

echo "== verify"
uv run eodgdl pipeline verify "$root" --data "$data"
