# The processing, from the shipped survey to the TASHA tables

What runs, in which order, on which weight, and what each stage records so the chain can be checked from end to end.
The processing is a Snakemake workflow, `workflow/Snakefile`, one rule per stage; `eodgdl.artifacts` reads its outputs
back. What a run chooses is `config/config.yaml` (`eodgdl.config`, which eodgdl itself reads): the weight the EOD sources
read (`design` or `tmg`), the bootstrap refits per task, the diagnostic chains' draws and the completed dataset the
TASHA build reads; what defines a model (seeds, grids, the pipeline's draws) stays in its YAML. The plan and its
decisions: `docs/handoff_reproducible_pipeline.md`.

## Stages

Paths are under the workflow's root, `output/` (`--config output=ROOT` for another).

| stage | rule | what | weight | reads | writes (and its manifest) |
|---|---|---|---|---|---|
| 0. load | `eod_tables` | `load_eod()`: rename, clean, schemas, repeated diaries, chain rules, hand decisions, legs | — | `data/` survey files, the loader's code and `src/eodgdl/revisions/` (`eod.loader_files()`) | `eod/{viv,hab,trips,legs}.parquet` + `manifest.json`; its `versions` carry the tables' digest |
| — | `zone_system` | every urban AGEB and rural locality in its zone, the zones redrawn (`reweight.zone_system`) | — | stage 0, the zone polygons, the census via `mxcensus` | `zones/zones.gpkg`, `zones/assignment.parquet`, the map `zones/zone_system_map.pdf` and `zone_system_map_rural_agebs.pdf` (`reports/reweight_inputs.qmd`'s fig-zone-map chunk, run by the rule) + `manifest.json` |
| 1. employment before reweighting | (inside 2) | P(`trabaja`) of the 4,370 persons 16+ who did not answer the activity question | design (`ponderador`) | stage 0, `models/od_empleo_design_model.joblib` | the records' `Employed` = P, `NotEmployed` = 1 − P (`reweight/spec.yaml`, `imputed:`) |
| 2. reweight inputs | `reweight_inputs` | `eodgdl reweight build` | — | stages 0–1, census via `mxcensus`, CONAPO / VMRC / ENDUTIH, the zone polygons | `reweight/` + `manifest.json` (stage `reweight_inputs`) |
| 3. household weight | (external), `tmg_weight` | TMG.SurveyReweight (XTMF2, run by hand); the rule checks what comes back | — | stage 2's set | `data/EOD_peso_hogar_TMG.csv` + sidecar `.yaml` (`eodgdl reweight import-weight`); `checks/tmg_weight.txt` |
| 4. final models | `retrain` (one job per task) | every task retrained, `bootstrap` refits each | the run's (`tmg` for the delivery) | stage 0, stage 3 under `tmg`, ENOE, ENIGH, DENUE (through `mxcensus`) | `impute/<task>/` + `manifest.json` (stage `retrain`) |
| — | `diagnostic_chain` | `sector_informality` and `nse` scored alone, `draws` multiple imputations, evaluated | the run's | stage 4's bundles | `impute/<chain>/` (scores, `multiple_imputation/`, `evaluation/`), read by the imputation reports |
| 5. completed datasets | `pipeline_tasha` | pipeline `tasha`, its 50 draws each scored with a bootstrap model | the run's | stage 0, stage 4's bundles | `impute/tasha/` + `manifest.json` (stage `imputation`), linked upstream to each retrain |
| 6. TASHA tables | `tasha_build` | `eodgdl tasha build --draw d` | TMG for the expansion (pending) | stage 0, stage 5 | `tasha/od_*.csv`, `od_provenance.json` + `od_manifest.json` (stage `tasha_build`) |
| — | `verify` | every manifest walked (`eodgdl.manifest.verify`) | — | every stage | `checks/verify.txt` |

Under `weight: design` stages 4–6 run on the survey's own weight (what `output/` holds until TMG's weight comes back);
under `weight: tmg` every imputation waits for the weight and its check (`tmg_weight`), and the workflow refuses to
start without the file.

**One pass.** The weights depend on an imputation (stage 1) and the final imputations on the weights (stage 4); the
loop is not iterated. Stage 1's model is trained on the design weight because the TMG weight does not exist yet; both
`empleo` models are part of the processing, each with its role (`od_empleo_design_model.joblib` for stage 1, the
stage-4 retrain for the TASHA build). Training on the design weight is defensible: it is constant within the AGEB in
98.9% of AGEBs, the EOD tasks carry the design variables as features, and the NSE calibration ranks within the AGEB.

**Stage 1 is a probability, not a draw.** The tool reads every attribute as a float (`LoadSurveyRecords`,
`float.Parse`) and sums weight × value, so the 4,370 count as their probability of working. At the design weight the
survey then gives 2,366,943 employed against the census's 2,695,199 (0.88; 0.84 with them counted as non-workers).

## When a rule reruns

Snakemake reruns a job when an input is newer than its outputs, or its parameters (`weight`, `bootstrap`, `draws`,
`draw`) or its command changed, and then everything downstream of it. The inputs list the data, the models and the
library code each stage runs (the Snakefile's `code(...)`: the loader for stage 0, `impute/` and `giro/` with the zoning
for the imputations, `reweight/`, `tasha/`); the CLI and `manifest.py`, which print and record, are left out. So an
edit of the chain rules reruns everything, an edit of a task YAML the retrains and what follows, an edit of
`tasha/mappings.yaml` the TASHA build alone. `uv run snakemake -n` says what would run, and why, before anything does.

Timestamps are what carry a change downstream (Snakemake's `input` trigger only sees the list of inputs change), so they
stay among the triggers, and for a rule that runs a script its code trigger is the script's timestamp too: an edit that
changes no result (a comment, a docstring), or a branch switch or merge that rewrites files with the same content, still
reruns what reads them, unless `uv run snakemake -c1 --touch` marks the outputs current (check first with `git diff`
that no code changed since the outputs were built). The recorded versions do not have that problem: the survey's is
a digest of the tables `load_eod()` returns (`eod.tables_digest`, computed once per loader code and kept under `.cache/`),
so an edit that leaves the tables as they were moves no version and stales no completed dataset.

## How a run records itself

Each stage that writes files writes a manifest beside them (`eodgdl.manifest.write_manifest`):

- `code`: eodgdl version, git commit, whether tracked files had uncommitted changes, Python, platform, `uv.lock`'s
  sha256; `environment`: `mxcensus` (version and its registry's sha256, which pins every census, ENOE, ENIGH and DENUE
  file it fetches: Pooch checks each download against it) and scikit-learn.
- `parameters` (years, seeds, draws, bootstrap refits), `versions` (the data versions the sources declare: the survey
  files, the cleaned tables' digest, the weight file's sha256 once the TMG weight is in), `inputs` (path and sha256 of
  every file read).
- `upstream`: each stage it read, by path and **outputs digest** (one sha256 over the upstream's outputs), so a rebuild
  that reproduces the same files keeps the link and one that does not breaks it.
- `outputs`: every file written and its sha256.

**Reproducibility, layered** (decided 2026-10-01): the hashes chain every input and output. On the same platform with the
same `uv.lock`, a rerun is expected to give the same bytes (compare the outputs digests); elsewhere, the stages agree
within Monte Carlo error (`scripts/impute/parity.py` compares two roots with a tolerance). The bundles refuse another
scikit-learn version.

**Guards in the code**, beyond the manifests:

- A source's weight file is one of its data versions (`Source.data_versions`): the feature cache, a bundle's training
  data version and `tasha.load_completed`'s staleness check all move when the weight's values do.
- A chain records each bundle's file sha256, training source and training data (`bundle.bundle_identity`); a chain, a
  pipeline and `tasha.load_completed` refuse EOD-trained bundles fitted on another weight than their source reads, or
  on mixed weights (`bundle.weight_conflicts`; ENOE's and ENIGH's tasks train on their surveys' own weights).
- The weight file is refused where its sidecar records other bytes; `reweight.weight.check_weight` (the rules
  `tmg_weight` and `verify`) fails where the reweight inputs written now are not those the weight was fitted on.

## Stage 3: the boundary

TMG.SurveyReweight (`github.com/TravelModellingGroup/TMG.SurveyReweight`; read at `960c5ed`, 2026-10-01) is an XTMF2
model system with no headless runner (`XTMF2.Run` serves runs sent by the GUI or a RunServer; `-config` is "not
supported yet"); a colleague runs it and **only the weight comes back** (the user, 2026-10-01; adopting the
reweighting inside the pipeline is planned for later). What is handed over: `output/reweight/` with its README (how to
configure the model system) and manifest. What must come back: the `UpdatedExpansionFactorsFile` and the constraint
year used; the tool's commit, the exported `.xmsys` and the `ConstraintReportFile` when available. Then:

```bash
uv run eodgdl reweight import-weight UPDATED.csv --year 2023 --reweight output/reweight --data data
```

writes `data/EOD_peso_hogar_TMG.csv` (`folio_vivienda`, `peso`, in `HouseholdRecords.csv`'s order) and
`data/EOD_peso_hogar_TMG.yaml` (the sha256 of the file received and written, stage 2's outputs digest and commit, the
year, the tool's commit and the configuration's and report's sha256 or null). The EOD sources then read it once `config/config.yaml`
says `weight: tmg` (each source's `run_weights` in `src/eodgdl/impute/sources/eod.yaml` names the file under `tmg`); the
NSE calibration follows (`weight: source`). The tool parses numbers with the machine's culture: it must run under a
`.`-decimal locale.

## Commands

```bash
uv sync --extra workflow                     # Snakemake, and every extra a rule needs
uv run snakemake -n                          # what would run, and why
uv run snakemake -c8 reweight                # stages 0-2 under weight: design: the set to hand over
uv run eodgdl reweight import-weight UPDATED.csv --year 2023 --data data   # stage 3's result into the data
# then set `weight: tmg` in config/config.yaml (and commit it with the run's outputs)
uv run snakemake -c8                         # everything: checks the weight, then stages 4-6 (~75 min) and verify
uv run snakemake -c8 eod zones               # one target and what it needs: eod, zones, reweight, impute, tasha
uv run snakemake -c8 --config output=ROOT    # another root, e.g. to compare with scripts/impute/parity.py
uv run snakemake -c1 --touch                 # after an edit that changes no result: mark the outputs current
```

## Pending

- Stage 6's expansion: `tasha/mappings.yaml`'s `ExpansionFactor` reads `ponderador` (dwelling, person and trip weights
  of the survey). With the TMG weight it should read the household weight for every table; a mapping change, to make
  when the weight arrives.
- `reports/pipeline.qmd` (item G): the manifests read and asserted, stage 1 against stage 4's `empleo` on the 4,370,
  and what the TMG weight moved in the models (`parity.py` between the design-weight and TMG-weight roots).
