# The processing, from the shipped survey to the TASHA tables

What runs, in which order, on which weight, and what each stage records so the chain can be checked from end to end.
The driver is `scripts/pipeline/run.sh ROOT`; `eodgdl pipeline verify ROOT` checks a finished (or partial) run. The plan
and its decisions: `docs/handoff_reproducible_pipeline.md`.

## Stages

| stage | what | weight | reads | writes (and its manifest) |
|---|---|---|---|---|
| 0. load | `load_eod()`: rename, clean, schemas, repeated diaries, chain rules, hand decisions, legs | — | `data/` survey files, `src/eodgdl/revisions/`, the loader's code | nothing on disk; every later manifest records its `versions` (`sources.eod.survey_versions`: survey sha256, loader digest) |
| 1. employment before reweighting | P(`trabaja`) of the 4,370 persons 16+ who did not answer the activity question | design (`ponderador`) | stage 0, `models/od_empleo_design_model.joblib` | inside stage 2: the records' `Employed` = P, `NotEmployed` = 1 − P (`reweight/spec.yaml`, `imputed:`) |
| 2. reweight inputs | `eodgdl reweight build` | — | stages 0–1, census via `mxcensus`, CONAPO / VMRC / ENDUTIH, the zone polygons | `ROOT/reweight/` + `manifest.json` (stage `reweight_inputs`) |
| 3. household weight | TMG.SurveyReweight (XTMF2, external, run by hand) | — | stage 2's set | `data/EOD_peso_hogar_TMG.csv` + sidecar `.yaml` (`eodgdl reweight import-weight`) |
| 4. final models | every task retrained (`scripts/impute/rerun.sh`), 50 bootstrap refits each | TMG | stage 0, stage 3, ENOE, ENIGH, DENUE (through `mxcensus`) | `ROOT/impute/<task>/` + `manifest.json` (stage `retrain`) |
| 5. completed datasets | pipeline `tasha`, 50 draws | TMG | stage 0, stage 4's bundles | `ROOT/impute/tasha/` + `manifest.json` (stage `imputation`), linked upstream to each retrain |
| 6. TASHA tables | `eodgdl tasha build --impute ROOT/impute --draw d` | TMG for the expansion (pending) | stage 0, stage 5 | `od_*.csv`, `od_provenance.json` + `od_manifest.json` (stage `tasha_build`) |

**One pass.** The weights depend on an imputation (stage 1) and the final imputations on the weights (stage 4); the
loop is not iterated. Stage 1's model is trained on the design weight because the TMG weight does not exist yet; both
`empleo` models are part of the processing, each with its role (`od_empleo_design_model.joblib` for stage 1, the
stage-4 retrain for the TASHA build). Training on the design weight is defensible: it is constant within the AGEB in
98.9% of AGEBs, the EOD tasks carry the design variables as features, and the NSE calibration ranks within the AGEB.

**Stage 1 is a probability, not a draw.** The tool reads every attribute as a float (`LoadSurveyRecords`,
`float.Parse`) and sums weight × value, so the 4,370 count as their probability of working. At the design weight the
survey then gives 2,366,943 employed against the census's 2,695,199 (0.88; 0.84 with them counted as non-workers).

## How a run records itself

Each stage that writes files writes a manifest beside them (`eodgdl.manifest.write_manifest`):

- `code`: eodgdl version, git commit, whether tracked files had uncommitted changes, Python, platform, `uv.lock`'s
  sha256; `environment`: `mxcensus` (version and its registry's sha256, which pins every census, ENOE, ENIGH and DENUE
  file it fetches: Pooch checks each download against it) and scikit-learn.
- `parameters` (years, seeds, draws, bootstrap refits), `versions` (the data versions the sources declare: survey,
  loader, the weight file's sha256 once the TMG weight is in), `inputs` (path and sha256 of every file read).
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
- The weight file is refused where its sidecar records other bytes; `reweight.weight.check_weight` (in `pipeline
  verify` and the driver) fails where the reweight inputs written now are not those the weight was fitted on.

## Stage 3: the boundary

TMG.SurveyReweight (`github.com/TravelModellingGroup/TMG.SurveyReweight`; read at `960c5ed`, 2026-10-01) is an XTMF2
model system with no headless runner (`XTMF2.Run` serves runs sent by the GUI or a RunServer; `-config` is "not
supported yet"); a colleague runs it and **only the weight comes back** (the user, 2026-10-01; adopting the
reweighting inside the pipeline is planned for later). What is handed over: `ROOT/reweight/` with its README (how to
configure the model system) and manifest. What must come back: the `UpdatedExpansionFactorsFile` and the constraint
year used; the tool's commit, the exported `.xmsys` and the `ConstraintReportFile` when available. Then:

```bash
uv run eodgdl reweight import-weight UPDATED.csv --year 2023 --reweight output/reweight --data data
```

writes `data/EOD_peso_hogar_TMG.csv` (`folio_vivienda`, `peso`, in `HouseholdRecords.csv`'s order) and
`data/EOD_peso_hogar_TMG.yaml` (the sha256 of the file received and written, stage 2's outputs digest and commit, the
year, the tool's commit and the configuration's and report's sha256 or null). The EOD sources then read it by switching
their `weight:` in `src/eodgdl/impute/sources/eod.yaml` to `{file: EOD_peso_hogar_TMG.csv, column: peso}`; the NSE
calibration follows (`weight: source`). The tool parses numbers with the machine's culture: it must run under a
`.`-decimal locale.

## Commands

```bash
scripts/pipeline/run.sh output                        # stages 0-2; stops at 3 until the weight is in data/
uv run eodgdl reweight import-weight UPDATED.csv --year 2023 --data data   # stage 3's result into the data
scripts/pipeline/run.sh output                        # again: checks the weight, then stages 4-6 (~75 min)
uv run eodgdl pipeline verify output --data data      # every broken link, or none
```

## Pending

- Stage 6's expansion: `tasha/mappings.yaml`'s `ExpansionFactor` reads `ponderador` (dwelling, person and trip weights
  of the survey). With the TMG weight it should read the household weight for every table; a mapping change, to make
  when the weight arrives.
- `reports/pipeline.qmd` (item G): the manifests read and asserted, stage 1 against stage 4's `empleo` on the 4,370,
  and what the TMG weight moved in the models (`parity.py` between the design-weight and TMG-weight roots).
