# The processing, from the shipped survey to the TASHA tables

What runs, in which order, on which weight, and what each stage records so the chain can be checked from end to end.
The processing is a Snakemake workflow, `workflow/Snakefile`, one rule per stage; `eodgdl.artifacts` reads its outputs
back. What a run chooses is `config/config.yaml` (`eodgdl.config`, which eodgdl itself reads): the weight the EOD sources
read (`weight`: `design`, or `tmg` for a sensitivity run), the weight the TASHA tables expand with (`expansion`:
`design` or `tmg`), the bootstrap refits per task, the diagnostic chains' draws and the completed dataset the TASHA
build reads; what defines a model (seeds, grids, the pipeline's draws) stays in its YAML. The plan and its decisions:
`docs/handoff_reproducible_pipeline.md`.

## Stages

Paths are under the workflow's root, `output/` (`--config output=ROOT` for another).

| stage | rule | what | weight | reads | writes (and its manifest) |
|---|---|---|---|---|---|
| 0. load | `eod_tables` | `load_eod()`: rename, clean, schemas, repeated diaries, chain rules, hand decisions, legs | — | `data/` survey files, the loader's code and `src/eodgdl/revisions/` (`eod.loader_files()`) | `eod/{viv,hab,trips,legs}.parquet` + `manifest.json`; its `versions` carry the tables' digest |
| — | `zone_system` | every urban AGEB and rural locality in its zone, the zones redrawn (`reweight.zone_system`) | — | stage 0, the zone polygons, the census via `mxcensus` | `zones/zones.gpkg`, `zones/assignment.parquet`, the map `zones/zone_system_map.pdf` and `zone_system_map_rural_agebs.pdf` (`reports/reweight_inputs.qmd`'s fig-zone-map chunk, run by the rule) + `manifest.json` |
| 1. employment before reweighting | (inside 2) | P(`trabaja`) of the 4,370 persons 16+ who did not answer the activity question | design (`ponderador`) | stage 0, `models/od_empleo_design_model.joblib` | the records' `Employed` = P, `NotEmployed` = 1 − P (`reweight/spec.yaml`, `imputed:`) |
| 2. reweight inputs | `reweight_inputs` (one job per set) | `eodgdl reweight build [--base taz]` | — | stages 0–1, census via `mxcensus`, CONAPO / VMRC / ENDUTIH, the zone polygons | `reweight/` (by MTAZ × municipality) and `reweight_taz/` (by TAZ, the working group's TAZ targets), each + `manifest.json` (stage `reweight_inputs`, `parameters.base`) |
| 3. household weight | (external), `tmg_weight` | TMG.SurveyReweight (XTMF2, run by hand); the rule checks what comes back | — | stage 2's set | `data/EOD_peso_hogar_TMG.csv` + sidecar `.yaml` (`eodgdl reweight import-weight`); `checks/tmg_weight.txt` |
| 4. final models | `retrain` (one job per task) | every task retrained, `bootstrap` refits each | `weight` (design; `tmg` only in a sensitivity run) | stage 0, ENOE, ENIGH, DENUE (through `mxcensus`); stage 3 under `weight: tmg` | `impute/<task>/` + `manifest.json` (stage `retrain`) |
| — | `diagnostic_chain` | `sector_informality` and `nse` scored alone, `draws` multiple imputations, evaluated | `weight` | stage 4's bundles | `impute/<chain>/` (scores, `multiple_imputation/`, `evaluation/`), read by the imputation reports |
| 5. completed datasets | `pipeline_tasha` | pipeline `tasha`, its 50 draws each scored with a bootstrap model | `weight` | stage 0, stage 4's bundles | `impute/tasha/` + `manifest.json` (stage `imputation`), linked upstream to each retrain |
| 6. TASHA tables | `tasha_build` | `eodgdl tasha build --draw d --expansion E` | `expansion` (`tmg` for the delivery): `ExpansionFactor` only | stage 0, stage 5; stage 3 under `expansion: tmg` | `tasha/od_*.csv`, the data dictionary `od_dictionary.md` and code tables `od_<table>_codes.csv`, `od_provenance.json` + `od_manifest.json` (stage `tasha_build`; the weight file among its `inputs` under `tmg`) |
| — | `verify` | every manifest walked (`eodgdl.manifest.verify`) | — | every stage | `checks/verify.txt` |

**Two weights, two roles** (decided 2026-10-08). The imputations (stages 1, 4 and 5) train on the survey's design
weight in every delivery run, and TMG's household weight reaches the TASHA tables only as their `ExpansionFactor`
(stage 6, `expansion: tmg`). So stages 4–5 do not wait for the reweighting: when the weight comes back, only the TASHA
build reruns (seconds, not ~75 min), after `tmg_weight` checks the weight against the reweight inputs. The workflow
refuses to start under `expansion: tmg` (or `weight: tmg`) without the file.

There is no loop left to iterate. Stage 1 scores the 4,370 unanswered with `od_empleo_design_model.joblib`, and stage
4's `empleo` retrain on the design weight predicts the same probabilities (largest difference 0, measured 2026-10-08):
the employment TMG's weight is fitted against is the employment the TASHA tables draw from. Before 2026-10-08 the plan
was to retrain stage 4 on the TMG weight, one pass, with two `empleo` models apart.

What the training weight moves was measured on 2026-10-08, each EOD-trained task of the pipeline `tasha` refitted at
its selected configuration under other weights (the design refit reproduced the shipped bundles exactly). With no
weights at all (a large change: the design weight's Kish efficiency is 0.55), each imputed row moves less than one
bootstrap refit moves it (mean total variation 0.068 / 0.074 / 0.060 for `empleo` / `giro` / `educacion_jefe`, against
0.081 / 0.084 / 0.067), and the imputed rows' class shares by at most 0.5 pp, except the heads' licenciatura in
`educacion_jefe` (20.6% to 22.3% of the 3,819 imputed, about 0.36 pp of all heads). The dwelling weight in place of the
person weight moves them by at most 0.1 pp. The NSE rank calibration with uniform weights changes 0.08% of the
dwelling-draws, one level each: the design weight is constant within the AGEB in 98.9% of AGEBs. TMG's weight calibrates
the same MTAZ × municipality cells and varies within them by household attributes the tasks mostly carry as features,
so it is expected to move them no more; `weight: tmg` stays as the check, a sensitivity run into another root
(`--config output=ROOT` with `weight: tmg` in the config, then `scripts/impute/parity.py`).

**Stage 1 is a probability, not a draw.** The tool reads every attribute as a float (`LoadSurveyRecords`,
`float.Parse`) and sums weight × value, so the 4,370 count as their probability of working. At the design weight the
survey then gives 2,366,943 employed against the census's 2,695,199 (0.88; 0.84 with them counted as non-workers).

## When a rule reruns

Snakemake reruns a job when an input is newer than its outputs, or its parameters (`weight`, `expansion`, `bootstrap`, `draws`,
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
  files, the cleaned tables' digest, the weight file's sha256 under `weight: tmg`), `inputs` (path and sha256 of
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
configure the model system) and manifest, or `output/reweight_taz/` for the TAZ set. What must come back: the `UpdatedExpansionFactorsFile` and the constraint
year used; the tool's commit, the exported `.xmsys` and the `ConstraintReportFile` when available. Then:

```bash
uv run eodgdl reweight import-weight UPDATED.csv --year 2023 --reweight output/reweight --data data   # or --reweight output/reweight_taz
```

writes `data/EOD_peso_hogar_TMG.csv` (`folio_vivienda`, `peso`, in `HouseholdRecords.csv`'s order) and
`data/EOD_peso_hogar_TMG.yaml` (the sha256 of the file received and written, stage 2's outputs digest and commit, the
year, the tool's commit and the configuration's and report's sha256 or null). The TASHA build then expands with it once
`config/config.yaml` says `expansion: tmg`: every table's `ExpansionFactor` is the household's weight
(`tasha.load_expansion`, `reweight.weight.read_weight`; the person and trip calibrations of the design weights,
cyclists and bus boardings, are among the tool's targets). The EOD sources read it only under `weight: tmg`, the
sensitivity run (each source's `run_weights` in `src/eodgdl/impute/sources/eod.yaml` names the file under `tmg`; the
NSE calibration follows, `weight: source`). The tool parses numbers with the machine's culture: it must run under a
`.`-decimal locale.

## Commands

```bash
uv sync --extra workflow                     # Snakemake, and every extra a rule needs
uv run snakemake -n                          # what would run, and why
uv run snakemake -c8 reweight                # stages 0-2: the set to hand over
uv run eodgdl reweight import-weight UPDATED.csv --year 2023 --data data   # stage 3's result into the data
# then set `expansion: tmg` in config/config.yaml (and commit it with the run's outputs)
uv run snakemake -c8                         # everything: checks the weight, then reruns the TASHA build alone and verify
uv run snakemake -c8 eod zones               # one target and what it needs: eod, zones, reweight, impute, tasha
uv run snakemake -c8 --config output=ROOT    # another root, e.g. to compare with scripts/impute/parity.py
uv run snakemake -c1 --touch                 # after an edit that changes no result: mark the outputs current
```

## Pending

- When TMG's weight comes back: `expansion: tmg`, and the sensitivity run (`weight: tmg` into another root, `parity.py`
  against `output/`) for what the weight would have moved in the models; `reports/pipeline.qmd` reads it.
