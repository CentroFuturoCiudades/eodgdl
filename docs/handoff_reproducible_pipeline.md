# Handoff: the whole processing pipeline, reweighting included, documented and reproducible

Written 2026-10-01 at `9a48d54` (branch `impute-chains`), for the next session. The user's decision (2026-10-01): document
the whole data processing, from the shipped survey to the TASHA tables, **including the reweighting by
TMG.SurveyReweight**, and make it reproducible. Read `CLAUDE.md` (local), `src/eodgdl/reweight/README.md`,
`docs/impute.md` (Pipelines) and the top of `docs/handoff_impute_engine.md` (session 11) first. `docs/handoff_pipeline.md`
is the earlier brief for the staged imputation pipeline (done in session 10); this file does not replace it.

## Why

Two things the reweighting adds to what is already reproducible:

1. **The weights depend on an imputation, and the final imputations on the weights.** The reweight records count the
   4,370 persons aged 16+ who did not answer the employment question as `NotEmployed` (`reweight/spec.yaml`,
   `Employed` / `NotEmployed`), while the TASHA build gives them the employment the pipeline `tasha` draws (task
   `empleo`: 33.5% working, weighted, ~1,480 persons per draw; ~115,000 at the design weight). The weights would be fitted
   on one picture of employment and the model would use another. Decision: the reweight records read the
   imputed employment from an `empleo` model trained on the **design** weight (stage 1); TMG fits the weight (stage 3);
   every task is then retrained on the **TMG** weight (stage 4). One pass, no iteration. Both `empleo` models are part of
   the pipeline, each with its role, not two versions of which one replaces the other.
2. **TMG.SurveyReweight is an external .NET (XTMF2) tool** run by a colleague. Whatever happens there must be recorded
   so the chain of inputs and outputs can be verified across it.

Effect on the targets (2020 set, design weight, from `output/reweight/README.md` and the 50 draws): `Employed` survey
2,252,572 vs census 2,695,199 (0.84) → ~2,367,000 (0.88) with the imputation; `NotEmployed` 2,181,263 vs 1,472,937 (1.48)
→ ~2,066,000 (1.40). Most of the gap remains (the survey's home-presence bias); the imputation removes a known error from
the margin that does the most work.

**`output/reweight/` must not be handed over until item C is done.** It is current (rebuilt 2026-10-01 before and after
`9a48d54`: byte-identical, `reweight check` clean, `tests/test_reweight.py` passes) but still counts the 4,370 as
`NotEmployed`.

## Is training on the design weight acceptable? (asked and answered 2026-10-01)

Yes; measured on `load_eod()`:

- `viv.ponderador` is constant within the AGEB in 98.9% of AGEBs (median within-AGEB CV 0); `hab.ponderador` differs
  within a dwelling in 3.5% of dwellings. The design weight is a property of the AGEB's sample, nothing about the
  person.
- The EOD tasks already carry the design variables as features (`municipio`, `centralidad`), the standard advice for
  multiple imputation under a complex design (Reiter, Raghunathan & Kinney 2006). Weighting then matters only where
  AGEBs of one zone differ given the features; it costs efficiency (Kish effective sample share 0.55 for both weights).
- The NSE calibration ranks dwellings within the AGEB (`derive/amai.py`): a weight constant within the AGEB drops out.
- ENOE / ENIGH tasks use those surveys' own weights (`sources/enoe.yaml`, `enigh.yaml`), as they should.
- The TMG weight will vary with household composition (age, sex, who works), which the EOD tasks mostly see as
  features (`hogar_otros_*`, age, sex), so stage 4 is expected to move the conditional models little. Measure it (item
  G), do not assume it. Totals (TASHA's expansion, diagnostic shares, the delta scenario's target, the workers check)
  move first-order and must use the TMG weight.

## The stages

| stage | what | weight | inputs | outputs | today |
|---|---|---|---|---|---|
| 0. load | `load_eod()` | — | `data/` survey files, `revisions/`, loader code | the cleaned tables | hashed: `sources.eod.survey_versions()` |
| 1. imputation before reweighting | `empleo` only: P(`trabaja`) for the 4,370 | design | stage 0, the pinned bundle | probabilities per person | **new** (C) |
| 2. reweight inputs | `eodgdl reweight build` | — | stage 0, stage 1, census via `mxcensus`, CONAPO / VMRC / ENDUTIH | `output/reweight/` | no manifest |
| 3. **TMG.SurveyReweight** | external | — | stage 2's set + the XTMF2 configuration | one household weight | **new** (E) |
| 4. final imputation | all seven tasks retrained, 50 bootstrap bundles each (`scripts/impute/rerun.sh`, ~70 min) | TMG | stage 0, stage 3, ENOE, ENIGH, DENUE | bundles | `retrain` writes `summary.json` with the bundle's sha256 |
| 5. completed datasets | pipeline `tasha`, 50 draws | TMG | stage 0, stage 4's bundles | `output/impute/tasha/` | `provenance.json` per level (`9a48d54`) |
| 6. TASHA build | `eodgdl tasha build --impute --draw d` | TMG for the expansion | stage 0, stage 5 | `od_*.csv` | `od_provenance.json` |

## What `9a48d54` (session 11) already gives, and what it leaves

Checked 2026-10-01: the commit touches no reweight file, no loader, no revision, no `data/` or `models/` file; it does not
change the plan's logic, and it narrows item B.

- A chain's (and so each pipeline level's) provenance records its scoring `source` and data `versions` (survey files,
  loader code and revisions, each task's feature-cache key); `tasha.load_completed` refuses a run whose data changed
  since (`sources.changed_versions`). **Once the weight is part of a source's versions (A), this check refuses
  completed datasets drawn under another weight with no further code.**
- `bundle.check_bootstrap` refuses bootstrap refits of another bundle; `write_retrain` clears an earlier `bootstrap/`.
- **Still missing**: the provenance records each bundle's `spec_hash`, `scoring_hash`, sklearn and eodgdl versions,
  none of which depends on the weight. Two bundles trained on different weights leave identical entries; the bundle's
  sha256 and its `metadata["data_versions"]["training"]` are not copied. `check_bundle` (scikit-learn, level contract,
  scoring hash) accepts either, correctly for scoring, so nothing stops a pipeline mixing weights.
- The bundle's training data version already covers the weight's **column name** (the feature-cache key hashes the
  source's YAML entry, `weight: ponderador` included) but **not its values**.

## Work items, in order

**0. Settle the tool's two unknowns first.** Clone `github.com/TravelModellingGroup/TMG.SurveyReweight` (read only, not
vendored) and read its record loaders: (a) does it accept a fractional attribute value (`Employed` = 0.34)? It forms
weighted sums, so likely yes, but nobody has checked; (b) can it run headless from a saved configuration? Then ask the
user who runs stage 3 and whether the configuration can be saved and handed back. (a) decides C's form; (b) decides
whether stage 3 is a rerunnable stage or a recorded boundary (E).

**A. The weight as a versioned input.** A source's `weight` may name a column of a weight file under `data/` (e.g.
`weight: {file: EOD_peso_hogar_TMG.csv, column: peso}`, joined on `folio_vivienda`), and the source's `versions` include
that file's sha256. Then the feature-cache key, the bundle's training data version and `changed_versions` all move when
the weight's values do. `weight: ponderador` stays the default and keeps today's hashes (no retrain forced). Tests: a
changed weight file changes the versions; `load_completed` refuses a run drawn under the other.

**B. Bundle identity in the provenance, and a guard.** `run_chain` copies, per bundle, its file sha256 and
`metadata["data_versions"]["training"]` into the provenance. `parse_pipeline` / `run_pipeline` and
`tasha.load_completed` refuse a run whose EOD-trained bundles (`source` an `eod.*`) were trained on different weights;
ENOE / ENIGH tasks are exempt (their own survey weights). `parity.py` must keep ignoring what it ignores today.

**C. Stage 1 and the reweight records.** (Unblocks handing over the reweight set.)
- Pin the design-weight `empleo` bundle under a name of its own: `models/od_empleo_design_model.joblib`, today the same
  bytes as v0.9.0's `od_empleo_model.joblib` (catalog `MODEL_FILES`, registry sha256).
- `reweight/spec.yaml`: `Employed` / `NotEmployed` of a person with no answer read P(`trabaja`) from that bundle, scored
  by `run.score_task("empleo", path=...)` (its scoring source `eod.persons` holds every person 16+; keep the unanswered) (the probability, not a draw: no seed, no dependence on a draw's index). A
  declared key in the spec (e.g. `imputed: {task: empleo, bundle: od_empleo_design_model.joblib, class: trabaja}`),
  realised in `records.py` with no retyped lookup, as `tasha/mappings.yaml` does for `imputed:`. If item 0 says the tool
  takes integers only, use one draw instead, the draw the TASHA build uses, and say so in the spec.
- `Unemployed` / `Inactive` stay 0 for the 4,370 (no target reads them); `Employed + NotEmployed` still partitions the
  12+ population (`test_employed_is_tasha_s_worker`'s second assert).
- `test_employed_is_tasha_s_worker`: the answered persons as now; the unanswered `Employed` equals the pinned bundle's
  P(`trabaja`). `reweight check` must accept the fractional column (it reads the files as the tool does: mirror item 0).
- Rewrite the README's "4,370 persons" caveat and the diagnostic table; rebuild `output/reweight/`; pin the new
  `Employed` / `NotEmployed` survey totals in `tests/test_reweight.py`.
- The reweight set records the bundle's sha256 (stage 2's manifest, D).

**D. A manifest per stage.** `manifest.json` beside each stage's outputs: stage name, code (eodgdl version, git commit,
dirty flag), the sha256 of every input file and of the upstream manifests, parameters (seeds, draws, bootstrap B), the
sha256 of every output. Build on what exists rather than beside it: `summary.json` (retrain), `provenance.json`
(pipeline), `od_provenance.json` (TASHA). A command (`eodgdl pipeline verify ROOT`, or under each subcommand) walks the
chain and lists every broken link; like `reweight check` it returns a list, empty when sound.

**E. Stage 3's boundary.** TMG's result enters as `data/EOD_peso_hogar_TMG.csv` (`folio_vivienda`, the weight; a catalog
group of its own beside `SURVEY_FILES`, registry) with a sidecar (`.yaml`) recording: the sha256 of stage 2's manifest it was fitted on, the tool's
repository commit and XTMF2 version, the configuration file used (which constraints were on, whether `BusBoardings`
stayed: the README calls it optional; the sex × age crosses), and the tool's convergence report (copied under `data/`
or hashed). Reading the weight checks the sidecar's stage-2 hash against a fresh stage 2: if today's reweight inputs
differ from what TMG was given, it fails. If item 0(b) says the tool runs headless, stage 3 is also a script that
reruns it from the saved configuration and compares.

**F. One driver.** `scripts/pipeline/run.sh ROOT`: stages 0–2, then stops naming what to hand over (the reweight set
and its manifest) and what must come back (E's two files); with the weight file in `data/`, it resumes at stage 4
(`rerun.sh` with the TMG weight), 5 and 6, each stage writing its manifest, and ends with `verify`.

**G. A report, `reports/pipeline.qmd`** (in the repo's style: recomputed at render time, structural claims `assert`ed):
the stage graph; the manifests read and every hash link asserted; the one-pass compromise measured (stage 1 vs stage 4
`empleo` on the 4,370: shares, and the `Employed` total each implies under the TMG weight); what the TMG weight moved in
the stage-4 models (`parity.py` between the design-weight and TMG-weight roots: drawn shares by task, held-out metrics).
If the drawn shares move by less than a point, record it and say so: the design-weight bundles could then serve stage
4 too, and the user decides.

**H. Releases.** Two, each one commit that bumps the version and sets `REF` to the new tag (CLAUDE.md, data access):
- after A–D (and E's format): both `empleo` bundles in `models/`, the reweight set reproducible from the tag, handed over;
- after the weight returns: the weight file and sidecar in `data/`, the stage-4 bundles in `models/`, the report.
v0.9.0 stays the design-weight model set. Merge `impute-chains` into `main` after each, as before.

## Open questions for the user (ask at the start)

1. Who runs TMG.SurveyReweight, and can the configuration and its report come back with the weight? (item 0(b), E)
2. Fractional `Employed` or one draw, once item 0(a) is known. Recommended: fractional.
3. Bit-for-bit reproducibility (same platform, `uv.lock`, the census cache of `mxcensus` pinned) or equivalence within
   Monte Carlo error across platforms? The bundles already refuse another scikit-learn version.
4. Stage 4 on the TMG weight even if G shows it moves little? Recommended: yes, for one consistent weight end to end,
   unless the user prefers fewer model sets.

## Item 0 and the answers (2026-10-01, session 12)

Read at TMG.SurveyReweight `960c5ed`, TMG-Framework `2a328dc`, XTMF2 `588a866` (2026-10-01):

- **(a) Fractional attributes: accepted.** `LoadSurveyRecords.LoadTable` reads every non-key column with
  `CsvReader.Get(out float)` (`float.Parse`); a constraint sums `weight × Π(matched attributes)`. The records are already
  written as `1.0` / `0.0` and the targets are fractional, so nothing in the format changes. A record with a value > 0
  takes the constraint's full correction (`UpdateConstraint`, step 3), so a household with an unanswered person is
  corrected by both `Employed` and `NotEmployed`, as one with a worker and a non-worker is. `float.Parse` uses the current
  culture: on a decimal-comma locale the files (today's included) would be misread; say so to whoever runs it.
- **(b) Headless: no.** `XTMF2.Run` serves a run sent over a bus by the GUI or a RunServer; its `-config` prints "Custom
  configurations are not supported yet". The GUI exports a model system as a `.xmsys` file (`ExportModelSystem`); running
  one without the GUI would take a C# driver against XTMF2's library.

The user's answers: (1) a colleague runs it and **only the weight comes back**; the plan is to adopt the reweighting in
the pipeline later, so E's sidecar records what is known (stage-2 manifest hash, tool commit if given) and leaves the
configuration and report optional. (2) **Fractional** `Employed` = P(`trabaja`). (3) **Both, layered**: hashes chain every
input and output; on the same platform with `uv.lock`, identical bytes are expected; elsewhere a tolerance check on the
stage summaries. (4) Stage 4 **always** retrains on the TMG weight.

## Progress (session 12, 2026-10-01; uncommitted until the user asks)

- **A done.** `sources.with_weight` / `weight_file` / `weight_versions`, `Source.data_versions` (read by
  `features.build_frame` and `changed_versions`); `weight: ponderador` keeps every hash. nse.yaml's calibration weight
  is `weight: source` (`chain.derive_config`, also read by `reports/imputation_nse.qmd`): only the chain hash moved.
  Tests: `tests/test_weights.py`.
- **B done.** `bundle.bundle_identity` (sha256, training source and versions, bootstrap set sha256) in every chain
  provenance; `bundle.weight_conflicts` in `run_chain`, `run_pipeline`, `tasha.load_completed`; `check_bootstrap` checks
  the training weight and bootstrap bundles now record their training versions. `parity.py` ignores the new run keys
  (`BUNDLE_RUN_KEYS`). `output/impute/tasha/` rerun: parquet byte-identical, only provenance keys added (the old one has
  no training sources, which `load_completed` now refuses).
- **C done.** `models/od_empleo_design_model.joblib` (catalog, registry; same sha256 as v0.9.0's empleo bundle);
  spec.yaml `imputed:`; `records.imputed_scores`; `reweight` extra brings scikit-learn; the CLI's `--data` reaches the
  models; `check` partitions with a float32 tolerance. Survey at the design weight: `Employed` 2,366,943 (0.88 of the
  2020 target), `NotEmployed` 2,066,892 (1.40). `output/reweight/` rebuilt: only `PersonRecords.csv` (4,370 rows),
  the diagnostics and README changed. `tests/conftest.py` defaults `EODGDL_DATA_DIR` to the clone's `data/`.
- **D done.** `src/eodgdl/manifest.py` (`write_manifest`, `check_manifest`, `verify`): stage 2 (`reweight.write`), each
  retrain (`run.write_retrain_manifest`, CLI and `retrain_chain`), a pipeline run (`pipeline.write_pipeline_manifest`,
  upstream = each task's retrain manifest), the TASHA build (`od_manifest.json`); links by outputs digest, so a rebuild
  with the same outputs keeps them. `eodgdl pipeline verify ROOT`. Checked end to end in a scratch root (retrain
  manifests on clones of `output/impute/<task>`, the pipeline, the TASHA build, the reweight set): clean, and an edited
  retrain output is reported. `output/reweight/` rebuilt with its manifest (verify clean); `output/impute/` has none yet
  (its retrains predate D; stage 4 writes them).
- **E done (format and entry).** `src/eodgdl/reweight/weight.py`: `import_weight` (the tool's
  `UpdatedExpansionFactorsFile` in `HouseholdRecords.csv`'s order -> `data/EOD_peso_hogar_TMG.csv` + `.yaml` sidecar:
  received/written sha256, stage 2's outputs digest and commit, year, tool commit, configuration/report sha256 or null),
  `eodgdl reweight import-weight`; `check_weight` in `verify`; `with_weight` refuses a weight its sidecar does not
  record. Not headless (item 0), so no rerun script. The catalog entry and registry line come with the file (release 2).
- **F done.** `scripts/pipeline/run.sh ROOT`: stages 0-2, stops at 3 naming the handover; with the weight, checks it
  against the fresh stage 2 and that the EOD sources read it, then `rerun.sh` (4-5), `tasha build` (6), `verify`.
- **G partly.** `reports/pipeline.qmd` renders now: stage graph, manifests asserted (`verify == []`), stage 1 measured
  (median P 0.215, 33.6% working at the person weight; the activity table). The stage-1 vs stage-4 comparison and the
  parity of the two roots are written to run once stage 4 has (the parity section still to add then).
- **Pending, not in the brief:** stage 6's expansion (`tasha/mappings.yaml` `ExpansionFactor` reads each table's
  `ponderador`; with the TMG weight every table should read the household weight) — a mapping change when it arrives.
- **Committed** per item (7fe9adf A, 5a4ebc3 B, a08de68 C, c5bafc0 D, 7f59d26 E, d23bc67 F, 645771a G).
- **H, first release done** (2026-10-01): the `v0.9.0` tag, never pushed until then (installs fetched 404 for every
  bundle under `.../v0.9.0/models/`), pushed; `v0.10.0` (9c5e6d9: version, `REF`, lock) tagged and pushed, every model
  bundle fetched from the mirror and sha256-checked; `impute-chains` merged into `main` (a31de77). The reweight set can
  be handed over (`output/reweight/`, reproducible from v0.10.0). The second release waits for the weight.

## The simplification (decided 2026-10-01, session 12)

The user's decision: eodgdl is an academic data pipeline, not a user-facing package. Keep the library (the chain rules,
review, impute engine, reweight, tasha: ~90% of the code, untouched) and replace the convenience and orchestration layer
with a Snakemake workflow in the same repo; the package stays installed editable, so `load_eod()`, `load_taz()` and the
rest still read the clone. The release ritual (a registry sha256, `REF`, version and lock per data or model change: nine
releases in five days) was the most frequent non-scientific chore. Four steps, the suite and output parity green at
each (`scripts/impute/parity.py`, the manifests):

1. **Drop distribution — done (uncommitted).** `src/eodgdl/data/` (Pooch, `registry.txt`, `_catalog.py`, `_registry.py`,
   `_paths.py`) replaced by `src/eodgdl/data.py`: `data_dir()`, `models_dir()`, `cache_dir()` (`<clone>/.cache/`, the
   feature cache moved there from the platform cache dir), `resolve()` (`*.joblib` → `models/`), the input files' names.
   `eodgdl info|fetch` removed, `pooch` / `platformdirs` dropped from the dependencies, the reweight spec's `imputed`
   bundle checked against `models/` instead of a catalog list. The "second release" of H is now a commit; tags mark
   milestones (Zenodo at paper time, with the published outputs).
2. **One `config.yaml`**: the weight choice (today a hand edit of `impute/sources/eod.yaml` that `run.sh` checks), seeds,
   draws, bootstrap B, read by the code; it makes the hand-switch guards hold by construction.
3. **The Snakefile** (`workflow/`, rules at stage level: load → EOD parquet, zones, reweight inputs, retrain per task,
   the pipeline `tasha`, the TASHA build per draw, reports) replacing `scripts/pipeline/run.sh` and
   `scripts/impute/rerun.sh`; the TMG weight is a rule input under `data/`, so the graph stops there by itself. Declare
   the library modules each rule imports as its inputs (Snakemake does not see imported code), and leave mtime out of
   the rerun triggers (a checkout must not retrain). Chain internals and draws stay in the engine, not in rules. A thin
   `eodgdl.artifacts` reads the materialized outputs (cleaned tables, zones) for quick access.
4. **Delete what became redundant**: most of the CLI (the review round's commands stay: interactive hand work, its
   decisions table an input), `pipeline verify`, the staleness guards; `manifest.py` reduced to one final manifest
   (commit plus the results' sha256), since Snakemake's provenance is weaker than the manifests'.

## Do not

- Change a target or attribute in Python: `spec.yaml` holds every definition (`reweight/README.md`).
- Retrain the shipped bundles before the weight returns, or overwrite `output/reweight/` before C's tests pass.
- Iterate weights and imputation to a fixed point: one pass, documented (decided 2026-10-01).
- Read zone ids as numbers; drop `diario_repetido` persons (they stay, user 2026-09-25).
