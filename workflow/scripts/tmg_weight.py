"""Rule tmg_weight: TMG's household weight is the file its sidecar records, fitted on the reweight inputs this root
holds (reweight.weight.check_weight). Under `weight: tmg` every imputation waits for it."""
from pathlib import Path

from eodgdl.reweight.weight import check_weight

problems = check_weight(None, Path(snakemake.input.reweight).parent)  # noqa: F821
if problems:
    raise SystemExit("\n".join(problems))
Path(snakemake.output[0]).write_text("the TMG weight was fitted on these reweight inputs\n", encoding="utf-8")  # noqa: F821
