# Handoff: `eodgdl.impute`, one repository for every imputation model on the survey

## 0. Status at the end of session 5 (2026-09-28): v0.4.0 released; informal-jobs-model's figures next

Read this section first, starting with **Pending for session 6**; §1–§11 below are the original plan (session 1), kept
as the record of the design. Then read `CLAUDE.md` (local, gitignored: its `eodgdl.impute` section describes every
module as built) and `src/eodgdl/impute/__init__.py`.

### Pending for session 6 (start here)

1. **informal-jobs-model's figures into a report here, then archive it (user, 2026-09-28).** The user chose the
   figures of its notebooks 03 and 05 (ENOE vs EOD predictor distributions, ENOE informality profiles, the
   known-sector comparison, the informality models' calibration, ENOE vs EOD informality overall and by sector), 06
   and 08 (the head's education observed vs modelled, with the AMAI education score; the predominant NSE by AGEB,
   AMAI's and the chain's before and after calibration, with the aggregate bars) and `informal_job_plots` (the
   presentation versions: "OD duro" with the combined calibration curve; original vs modelled sector shares). Not
   the old notebook-04 sector charts: `reports/imputation_giro.qmd` has them. In the style of eodgdl's reports (plain
   matplotlib), shown in the report and written as PDF + 600-dpi PNG to `output/figures/`. The calibration figure's
   isotonic curve is not saved today (`isotonic__<arm>.parquet` holds its metrics only): the retrain is to write it,
   and the combined curve (each held-out row scored by the arm that scores it, informal-jobs-model's "hybrid")
   beside it; then retrain informality. What session 5's inventory found:
   - ENOE: `build_frame(load_task("informality"), Context(), role="train").frame` (`survey_weight` already / 8,
     `informalidad`, `sector`, `edad_cat`, `municipio`); informal-jobs-model's metro benchmark is `municipio !=
     "otro"` (38,933 rows, 39.35% informal, identical). EOD workers: the `role="score"` frame, weight `ponderador`
     from `output/impute/{giro,informality}/scores.parquet`; the four sectors are `prob_giro_*` collapsed with
     `chains/sector_informality.yaml`'s transform. "OD muestreo" is one seeded Bernoulli draw per worker, "OD duro"
     `informalidad_final`, "esperado" the probabilities.
   - NSE maps: per AGEB, the arg-max of the weighted `prob_nse_*` / `prob_nse_calibrado_*`; AMAI's
     `nivel_predominante` through `derive.amai.read_amai_ageb`; polygons from mxcensus's Marco Geoestadístico 2020
     (as `reweight.zoning` reads it), which holds every urban AGEB the EOD records. "Before calibration" is the
     draws' distribution, not informal-jobs-model's plug-in cut.
   - informal-jobs-model's `outputs/` (its figures, bundles, tables) is gitignored and stays on disk; its tracked
     `data/GUADALAJARA_SHP/` is replaced by the Marco Geoestadístico. Then its README points to eodgdl and its GitHub
     repository is archived: ask the user before each GitHub step.
2. **Review notes left as they are** (design points from session 4, not bugs; raise them if the user wants to go
   further):
   - Held-out metrics and scoring treat a missing feature differently: `run.heldout` and the evaluators use the
     model's `predict_proba`, where the missing label is a category it was trained with, while scoring
     (`marginalize`) never counts the missing label as supported and averages over the observed levels. The published
     metrics describe a slightly different predictor for rows with a missing value (as the retired notebook did).
   - With `evaluation.isotonic.ship: true`, the shift scenario would refit uncalibrated arms beside a calibrated
     bundle (latent: shipping is off in every task).
   - An enumerated chain step's other outputs (`<p>_marginalized_features`, `<p>_model_used`) come from each row's
     first scenario (`chain._conditionals`); the same in every scenario today.
   - `expected` propagation leaves NA in the completions for a task without `target.scores` that only a derive step
     reads (latent).
   - `chain._aggregate` stops the run on a NA derived level ("derived values outside the declared levels"): an NSE
     dwelling missing its internet, cars or workers answer would do it (none today).
   - Feature-cache keys include the task's classes, so `amai_banos`, `amai_dormitorios` and `amai_trabajadores` cache
     the same ENIGH frame three times (disk only).
   - `adjust_imputed_share` clips at one (§6.4).
   - ruff warns that `pyproject.toml`'s top-level `[tool.ruff] ignore` should move to `[tool.ruff.lint]` (not impute).

### Session 5 (2026-09-28): session 4's pending list, resolved

1. **Pushed, released, merged** (user). 3539cee pushed; then b321096 (the NSE ties, below) and 2ecfcdd (version
   0.4.0, `REF = "v0.4.0"`, `uv lock`), tagged `v0.4.0` (lightweight) and pushed; `impute-chains` merged into `main`
   (`--no-ff`) with the commit that wrote this handoff, and pushed. `data/` is v0.3.0's, unchanged. The mirror check
   passed (an empty `EODGDL_CACHE_DIR`, no `EODGDL_DATA_DIR`): `giro.load_model()` and `score` of both chains fetched
   the three survey files, the zone file, five bundles and the AMAI file from the tag, each checksum-verified, and the
   chains' scores and completions equal the local run's exactly (the provenance differs only in the eodgdl version
   the bundles recorded: 0.2.0 for the installed ones, 0.3.0 for the retrained ones).
2. **The NSE calibration ranks tied points at random per draw** (user's choice; b321096). A derive step's config
   carries `seed`, a `SeedSequence` that `chain._derive` spawns from the chain's seed by the step's position, apart
   from the draws' stream. The alternative, tie-averaged mid-ranks, moved the AGEB mixes away from AMAI's in a
   simulation on the v0.3.0 completions (mean total variation per draw and AGEB 0.089 → 0.104). On the survey against
   the v0.3.0 run: draws, points and uncalibrated levels identical; `nse_calibrado` changes in 2.7% of the
   completions; the first dwelling of a tie now averages level 3.08 against 3.06 for the others (was 2.58 against
   3.44); 8,612 dwellings' calibrated distributions move (mean total variation 0.041) and 619 of the 14,887
   calibrated dwellings change their most probable level; the weighted shares, the Monte Carlo table and the AMAI
   reference move by at most 5e-6. No retrain.
3. **`output/impute/` refreshed** (local, gitignored). All six tasks retrained on the release's code (~24 min), both
   chains scored and evaluated with `--retrained output/impute`, `giro_scores.parquet` rescored from the installed
   bundle (it was the legacy bundle's), the three reports re-rendered. Against a copy of the v0.3.0 outputs, every
   selection, CV loss, held-out metric and table is identical except the intended changes: giro's `missingness` row
   for education (1.84% / 46.48% → 2.50% / 46.54%), giro's `scenarios.parquet` (the delta rows' confidence), the
   stale `amai_trabajadores/evaluation/missingness.parquet` gone, and the NSE calibration (item 2). The retrained
   bundles predict as the installed ones (identical scores); their bytes differ, so they were not copied to `data/`.
4. **informal-jobs-model**: the user decided (pending item 1).

### Where things are

| phase | commit | branch |
|---|---|---|
| 0 mxcensus v0.4.0 | 80e4936 | merged into `main` at 00bf5f5 (pushed) |
| 1 engine core, batched marginalization | 98cf6aa | merged |
| 2 specs, sources, feature cache, bundles v2 | e766c54 | merged |
| 3 score / retrain / compare, CLI, `reports/imputation_giro.qmd`, notebook deleted, §6.1 fixed | 310d953 | merged |
| 4 chains (parallel / sequential; enumerate / draws / expected; derive steps) | 845a97d | `impute-chains` (pushed) |
| 5 ENOE source, YAML harmonization, `informality`, chain `sector_informality` | 82d86f5 | `impute-chains` |
| 5b diagnostics (`eodgdl impute evaluate`), `reports/imputation_informality.qmd` | bf112ce | `impute-chains` |
| 6 NSE: `eod.dwellings`, `enigh.households`, four tasks, chain `nse`, AMAI data file, `reports/imputation_nse.qmd` | 7cfef45 | `impute-chains` |
| 7 release: six bundles installed in `data/`, giro parity fixture from the v2 bundle, docs, `v0.3.0` | 609938f, 48ae9b3 (tag `v0.3.0`) | `impute-chains`, merged into `main` |
| review and simplification (session 4) | 3539cee | merged into `main` |
| NSE ties at random, release `v0.4.0` (session 5) | b321096, 2ecfcdd (tag `v0.4.0`) | merged into `main` |

`impute-chains` branches from `main` at 00bf5f5 and holds phases 4–7 and sessions 4–5, with the `v0.3.0` and `v0.4.0`
tags on it. It is merged into `main` (`--no-ff`) up to the commit that wrote this handoff. `uv run pytest`: 158 passed,
~110 s.

Parity reached (details in the commit messages):
- giro: the engine's scores equal the notebook's reference bit for bit except one row (a numpy row-sum layout
  artifact of the old recursion, 3e-16); `retrain giro` reproduces its selections, CV losses and held-out metrics
  exactly.
- informality: the ENOE frame equals informal-jobs-model's `enoe_harmonized.parquet` (51,707 rows); the retrain
  its bundle's selections and all held-out metrics bit for bit; P(informal | x, sector) its EOD scores exactly
  where the harmonized features agree; its notebook-05 diagnostics exactly or within 0.001 (stale EOD inputs).
- NSE: not a parity target (deliberate changes, `reports/imputation_nse.qmd` #sec-differences); the calibrated
  distribution matches AMAI's by AGEB within 0.4 pp except E, as informal-jobs-model found.

### Decisions (user, 2026-09-28): every §10 question is answered

- §10.1 sector → informality: `enumerate`; NSE: `draws`, 50 (50, 200 and enumerate agree within 0.05 pp; the plug-in
  compresses the tails). AMAI order: workers → bathrooms → bedrooms; workers are now *counted from the EOD* (not
  imputed), bathrooms condition on the head's education, bedrooms on both (bathrooms' AMAI points).
- §10.2 ENIGH population: households in cities of 100k+ (`selection.population: {tam_loc: ["1"]}`; evidence in each
  retrain's `evaluation/populations.parquet`).
- §10.3 isotonic: diagnostic always, shipped only with `evaluation.isotonic.ship: true` (off everywhere).
- §10.4 one-SE only; ROPE later if wanted (the rule is a spec field, `selection.rule`).
- §10.5 informal-jobs-model's fate: decide **after phase 7**.
- §10.6 mxcensus: public loaders suffice (keys are ordinary columns); no mxcensus change.
- §10.7 phases 0–3 merged to `main`; phases 4–7 on `impute-chains`.
- Also: ENIGH vehicles = cars + vans + pickups; `data/NSE_por_AGEB_AMAI.xlsx` (21 MB, AMAI's NSE by AGEB) may be
  redistributed through the data mirror (it is in `_catalog.py` and `registry.txt`, not yet on a tag).

### Phase 7: the release (done, session 3, 2026-09-28)

What it found:
- **Every retrain reproduced the previous run.** All six tasks were retrained on the final code, and each
  selection, CV loss, held-out metric and evaluation table equals the phase 3–6 run's. The four NSE bundles are
  byte-identical to it. giro's and informality's bundles differ only in metadata: the provenance fields phases 5–6
  added (`source`, `training_population`, `versions.training`), and the feature-code hashes.
- **The installed bundles** are `data/od_{giro_hybrid,informality,educacion_jefe,amai_banos,amai_dormitorios,
  amai_trabajadores}_model.joblib`, in `MODEL_FILES` and `registry.txt`. The user chose to ship the diagnostic
  `amai_trabajadores` bundle, so `evaluate nse` works from the mirror.
- **The chains scored from the installed bundles** (no `--retrained`) reproduce every table of the earlier runs:
  `sector_informality` and `nse` scores, completions and evaluations. The three reports re-rendered.
- **`tests/data/giro_parity.parquet`'s `expected__*` columns are now the v2 bundle's `score_frame`**, and the test
  is exact on every row (no layout special case). Against the legacy bundle, 10 of its 384 rows moved beyond 1e-12
  (max 0.07), all marginalized over `destino_trabajo`; no `giro_final` changed. `giro.load_model()` + `impute_giro`
  on the legacy view equals the engine exactly.
- `test_bundle_checks_and_v2_round_trip` now converts the legacy *view* of the v2 file. No test reads a real legacy
  pickle any more; the legacy bundle is in git (6d1d1d0, sha 227afc71…) if one is wanted.
- `evaluate sector_informality` reads `<retrained>/giro/scenarios.parquet`, which only a giro retrain writes; the
  README says so.
- The mirror check passed, with no `EODGDL_DATA_DIR` and an empty cache. `giro.load_model()` and `score` of both
  chains fetched the survey, the six bundles and the AMAI file from the `v0.3.0` tag, each checksum-verified, and
  the chains' scores and completions equal the local run's exactly.

The checklist as it was run (the original text, kept for the record):

The committed `data/od_giro_hybrid_model.joblib` is still the notebook's legacy bundle (sha 227afc71…, converted in
memory by `bundle.as_v2`); no other task's bundle is in `data/`, so today `score` of `informality` / the chains needs
`--retrained output/impute`. The user decided (2026-09-28) that the retrained bundles are installed at this release.

1. **Retrain every task on the final code** (the bundles under `output/impute/` were written during phases 3–6, the
   giro one before later changes to `run.retrain`; regenerate all for consistency):
   `for t in giro informality educacion_jefe amai_banos amai_dormitorios amai_trabajadores; do uv run eodgdl impute
   retrain $t --data data; done` (~25 min in all; giro ~7, informality ~3, each ENIGH task ~4–8). Check each
   selection against the previous run (the commit messages and `output/impute/<task>/summary.json`).
2. **Install the bundles**: copy each `output/impute/<task>/<spec bundle file>` to `data/` (`od_giro_hybrid_model.joblib`,
   `od_informality_model.joblib`, `od_educacion_jefe_model.joblib`, `od_amai_banos_model.joblib`,
   `od_amai_dormitorios_model.joblib`; ask whether to ship the diagnostic `od_amai_trabajadores_model.joblib`, which
   only `evaluate nse` reads); add the new names to `src/eodgdl/data/_catalog.py` (`MODEL_FILES`) and their sha256 to
   `registry.txt`. Sizes: 36 KB – 6.3 MB each.
3. **The giro parity fixture**: installing the v2 giro bundle moves 31 workers (its auxiliary destination models now
   carry training shares, phase 1). Regenerate `tests/data/giro_parity.parquet`'s `expected__*` columns from the new
   bundle and drop the one-row layout special case in `tests/test_giro.py::test_scoring_reproduces_the_reference`
   (the test then pins the installed bundle exactly). `giro.load_model()` returns the old keys for a v2 file
   (`legacy_view`); `test_bundle_checks_and_v2_round_trip` covers it.
4. **Chains and reports from the installed bundles**: `eodgdl impute score sector_informality` and `score nse` without
   `--retrained`; `evaluate` both; re-render the three reports (`QUARTO_PYTHON=.venv/bin/python quarto render
   reports/imputation_{giro,informality,nse}.qmd`) and look at the figures.
5. **Docs**: README (the impute sections exist; add a short overview of `eodgdl.impute`, the three chains and the
   outputs' columns), `src/eodgdl/giro/__init__.py` docstring, this handoff (mark phase 7 done), local `CLAUDE.md`
   (release notes). The giro config's old rationale ("nothing refers to ENOE, which is why it lives in the survey
   package") is gone with `giro/config.yaml`; check nothing else repeats it.
6. **Release** (CLAUDE.md "Data access"): one commit that bumps `pyproject.toml` to `0.3.0`, sets
   `src/eodgdl/data/_registry.py` `REF = "v0.3.0"`, `uv lock`; tag `v0.3.0` (lightweight, as `v0.1.0` / `v0.2.0`) on
   it; push branch and tag; merge `impute-chains` into `main` (`--no-ff`, as 00bf5f5) and push. Then check the mirror
   with no `EODGDL_DATA_DIR`: `uv run python -c "from eodgdl import giro; print(sorted(giro.load_model()))"` and a
   `score` of each chain fetch their bundles from the tag (`resolve` / Pooch, checksums from `registry.txt`).
7. **After the release**: ask the user about informal-jobs-model (§10.5: archive, or reduce to paper figures reading
   eodgdl's outputs).

### After the release: review and simplification (session 4, 2026-09-28)

A full review of `eodgdl.impute` and a refactor for one workflow; every shipped output is reproduced (below).

- **Fixed**: `eodgdl impute score <task>` selected the *training* source's keys and crashed for every task scored on
  another source (informality, the AMAI tasks); `retrain <chain>` wrote `output/impute/<chain>/<task>/`, where
  `--retrained output/impute` never looked; the feature-cache keys missed the engine's own code (`harmonize.py`,
  `features.py`), so an edit there served stale frames; a component of the label (`evaluation.components`) was not
  masked where the task's label is unknown (latent: ENOE has no such rows); the `missingness` table ignored the task's
  `missing_values` (giro's "No sabe" education counted as observed); a `scoring.level_subsets` feature absent from one
  arm raised; the delta-adjusted scenario kept the unadjusted `prediction_confidence`; a boosting grid with early
  stopping on would have been scored as stages of one chain; a spec whose last arm requires something is refused;
  `write_retrain` / `write_evaluation` left an earlier run's tables and `scenarios.parquet` behind, to be read later as
  current (`output/impute/amai_trabajadores/evaluation/missingness.parquet` is such a leftover, from 17:21).
- **One workflow**: `bundle.load_bundle(spec, path=None, retrained=None)` is the only bundle lookup (explicit file,
  else `<retrained>/<task>/`, else the data file); `score`, `evaluate`, `compare` take `--retrained`, `retrain --out
  ROOT` writes `ROOT/<task>/` (for a chain too). `--bundle` is gone.
- **Reused machinery**: one per-arm prediction (`arms.predict_arms`, behind scoring and `run.predict_rows`), one
  writer of probabilities (`arms.set_class_probabilities`: scoring, chains, delta); evaluators take a probability
  matrix, predicted once per arm (`run.heldout`); sources return plain frames; the four harmonization builders are one
  factory (`harmonize.register_harmonization`); one cached YAML reader (`spec.read_yaml`); AMAI's 21 MB file is read
  once (`derive/amai.read_amai_ageb`); `select.tune` fits only the selected configuration; `run.retrain` and
  `chain.run_chain` are split into named steps.
- **`eodgdl.giro`** keeps `impute`, `load_model`, `impute_giro`, the column names, the constants and the feature
  builders; the notebook-era training / evaluation wrappers (`giro/model.py`) are gone, and `giro/_ml.py` keeps only
  `prepare_model_features`, which the pre-v0.3.0 pickles reference (227afc71 still loads and scores exactly as before).
- **Parity** (against a snapshot of `output/impute/` from the v0.3.0 runs; `uv run pytest`: 157 passed): scoring
  giro, informality and both chains from the installed bundles gives identical scores, completions, provenance and
  evaluation tables; the six retrains reproduce every selection, CV loss, held-out metric, evaluation table and every
  fitted model's predictions (arms, auxiliary, components), except the two intended changes (giro's `missingness` row
  for education, 1.84% / 46.48% → 2.50% / 46.54%, now equal to the arm without education's share; the delta
  scenario's confidence); both chains scored and evaluated from those retrains (`--retrained`) are identical again,
  and the three reports render from them. `output/impute/` itself was left as it was (refreshed in session 5).
- **How the parity was checked** (reuse it after any engine change): copy `output/impute/` aside; `score` each task
  and chain and `retrain` each task into another root (`--out`, then `score` / `evaluate --retrained` that root);
  compare every parquet with `pandas.testing.assert_frame_equal(check_exact=True)`, the summaries as dicts (timings,
  versions and the bundle's sha256 aside), and each fitted model of the two bundles by `predict_proba` on the task's
  frames. Mind that `output/impute/giro_scores.parquet` of 14:41 was scored with the legacy bundle, not the installed
  one; the retrain's own `scores.parquet` is the reference.

### Not done (deliberately deferred; mention to the user if relevant)

- Auxiliary level models are fitted with fixed hyperparameters (§5.7 wanted them tunable).
- ROPE / baycomp selection, SHAP: not implemented (§10.4).
- `compare` holds hyperparameters at the published winner and does not vary a training population.
- The head-of-household education model uses `centralidad` for place: the AGEB's ~1,700 levels exceed
  HistGradientBoosting's 255-category limit; census AGEB covariates (e.g. mean schooling, via mxcensus) are an
  untried alternative (`compare` can test them once a builder exists).
- Where a dwelling holds several reported heads, `eod.dwellings` takes the first in the survey's order
  (informal-jobs-model took the first non-null value per column).

### Practical notes (additions to §11)

- Retrain outputs: `output/impute/<task>/` (bundle, `scores.parquet`, `scenarios.parquet`, `evaluation/*.parquet`,
  `summary.json`); chain runs: `output/impute/<chain>/` (`scores.parquet`, `completions.parquet`, `provenance.json`,
  `evaluation/`). All gitignored.
- The feature cache lives under the eodgdl cache dir (`<EODGDL_CACHE_DIR>/impute/`); keys cover code, config and data
  versions; `--refresh` rebuilds.
- ENOE and ENIGH come from mxcensus's cache (`~/Library/Caches/mxcensus`); building the ENOE frame takes ~10 s, the
  ENIGH one ~6 s.
- Run long retrains with `run_in_background`; several retrains at once oversubscribe the cores (each tuner already
  uses every core). Every `eodgdl impute` process imports the working tree when it starts: do not edit or `git stash`
  sources while a background run of several tasks is going.
- Render a report in place (`QUARTO_PYTHON=.venv/bin/python quarto render reports/<report>.qmd`); `--output-dir` renders
  in place and then *moves* the HTML away from `reports/`. Point a report at other outputs with its environment
  variables (`EODGDL_IMPUTE_GIRO`, `EODGDL_IMPUTE_INFORMALITY`, `EODGDL_IMPUTE_SECTOR_INFORMALITY`, `EODGDL_IMPUTE_DIR`).


Written 2026-09-28 on branch `giro-model`, for a new session. Read this file whole, then `CLAUDE.md`, then
`src/eodgdl/giro/` (≈1,000 lines) before changing anything. Every claim about `../informal-jobs-model` comes from
a read-only analysis made that day; re-check the ones you rely on (file:line references below are into that
repository unless stated).

## 1. Goal (original plan, session 1)

Build a **categorical imputation engine inside eodgdl** and bring into this repository **every** imputation model
that today lives in `notebooks/giro_model.ipynb` and in `../informal-jobs-model` (ijm): giro, informality, place of
work, head-of-household education, the AMAI components and the NSE level. Decisions (user, 2026-09-28):

- **Everything lives in eodgdl.** ijm's models, harmonization maps and post-steps move here; ijm is left with
  nothing it must maintain (its fate is an open question, §10).
- **eodgdl may train on mxcensus data**: ENOE and ENIGH through mxcensus's public loaders, alongside DENUE and the
  census it already reads.
- **Imputation runs sequentially, not only in parallel.** *Parallel*: each target is imputed from observed
  covariates alone, independently of the others (ijm's AMAI components). *Sequential*: targets are imputed in a
  declared order, each model conditioning on the targets imputed before it, so the joint uncertainty is carried
  (P(y₁, y₂, … | x) = P(y₁ | x) · P(y₂ | x, y₁) · …). Both modes are first-class. (Separately, every fit can run
  single-process with `n_jobs=1`.)
- All targets are **categorical** (ordinal ones carry an expected score as well); no regression.
- The random forest stays in the default grid.
- `eodgdl.giro` stays as a thin, backward-compatible wrapper (`giro.impute()`, `giro.load_model()`,
  `OUTPUT_COLUMNS`, column names).

The engine offers three modes:

- **score**: load the published winners and impute, in chain order (seconds, once §3's bottleneck is fixed);
- **compare**: evaluate feature specifications cheaply while iterating (fixed hyperparameters, same grouped folds,
  paired fold differences);
- **retrain**: grouped-CV selection → held-out evaluation → final fit → bundle + evaluation artifacts for a report.

## 2. State of the branch when this was written

`giro-model`, pushed: the merge of `main` (v0.2.0) at `b05e8e5`, then four commits (their messages lost their
subject lines through a shell quoting slip; the user may have rebuilt them with corrected messages — contents are
what matters):

| change | files | why |
|---|---|---|
| Destination crosswalk on the zone system | `giro/features.py`, `giro/__init__.py`, `giro/config.yaml`, `tests/test_giro.py`, README, pyproject comment | `main` retired `eodgdl.load_imeplan_agebs`. A destination code *is* a zone-system unit: 13-char urban AGEB or 9-char rural AGEB (`giro.zone_units(tables)` over `reweight.zoning.assign_units`). DENUE establishments count to their urban AGEB, else to the rural AGEB DENUE places them in (agrees 99.89% with the zoning's; keeps establishments in unpopulated places, e.g. `14097059A`: 217 establishments, 373 workers). `destino_ambito` level `localidad_rural` → `ageb_rural`. 1,737 rural-destination workers: before, 1,121 were `desconocido` and 616 had a rural AGEB key read as a locality; now 1,590 carry their AGEB's mix. A code outside the zone system raises. `add_destination_features(od, urban, rural, …)` changed signature. |
| Parallel tuning + staged boosting | `giro/model.py` (`tune_models`, `_fit_fold`, `_cv_tasks`, `_fold_metrics`) | Each (configuration, fold) fit in its own single-threaded joblib worker; a HistGradientBoosting grid over `max_iter` fitted once per fold and scored at each stage. Identical to the serial loop (fold log losses to 1e-16, same selection). |
| Notebook fixes | `notebooks/giro_model.ipynb` | Cell 20 shadowed the `giro` module with a loop variable; cell 10's fit count. |
| Refit | `data/od_giro_hybrid_model.joblib`, `registry.txt` | §3. |
| This handoff | `docs/handoff_impute_engine.md` | |

mxcensus is pinned at **v0.2.0** (`[tool.uv.sources]`). The moved models need **v0.4.0** (ENIGH loaders arrived
in v0.3.0; ijm already runs 0.4.0); bump it in phase 0.

## 3. The giro refit (2026-09-28)

Refit with the notebook as it stands (full grid, random forest kept), on `load_eod()` of v0.2.0 and the zone-system
crosswalk. Bundle `data/od_giro_hybrid_model.joblib`, sha256 `227afc71…` in `registry.txt` (metadata: eodgdl 0.2.0,
sklearn 1.9.0, DENUE 202211). The scored workers and the sensitivity scenarios are kept as local references in
`outputs/reference/od_giro_imputed.parquet` and `…_sensitivity.parquet` (gitignored).

- 26,913 OD workers; 17,429 with a reported giro (13,944 train / 3,485 held-out, by household); 9,484 imputed.
- Both arms select **GradientBoosting** (lr 0.05, 50 iterations, 31 leaves, l2 1.0), as the previous bundle.
  CV log loss with education: GB 0.910, RF 0.925, LR 0.940; without: GB 0.926, RF 0.933 (+0.0070 ± 0.0040, just
  outside one SE), LR 0.955.
- Held-out, weighted: with education log loss 0.8824 → 0.8863, accuracy 64.3% → 63.5% (95% CI 61.0–65.8%);
  without education 0.9078 → 0.9089, 62.0% → 62.0%. 31.2% / 29.4% below the weighted-marginal baseline (1.288).
  Within noise of the old bundle, which was trained on a pre-release eodgdl.
- Imputed shares (weighted, %): comercio 24.7, servicio 35.7, educacion 1.3, industria 32.5, gobierno 5.9 (known:
  33.4 / 34.8 / 1.6 / 26.2 / 4.1). Shift-weighted within ~1 pp; delta factor for gobierno 0.692.
- 222 imputed rows have an `ocupacion` level absent from training (Desempleado, Estudiante, Hogar, Jubilado…) and
  are marginalized over it.

**Where the 20 minutes go** (per-cell times): features 29 s; tuning 85 s + 79 s; evaluation 6 s + 6 s; **final
refit + scoring 495 s; covariate-shift refit + scoring 522 s**. Scoring is the bottleneck:
`predict_proba_marginalizing` recurses feature by feature and, for rows with an unsupported value, calls the whole
pipeline's `predict_proba` once per supported level on a small slice. Scoring 9,484 workers made **16,619
`predict_proba` calls** (314,806 rows; only 535 workers need any marginalization, mostly
`destino_trabajo+modo_trabajo` for workers without a work trip), while one call on all 9,484 rows takes 0.04 s.
Expanding every row's combinations into one frame, with weights, and calling `predict_proba` once per model gives
identical probabilities in seconds. This matters more now: sequential chains multiply the rows to score (§5.6).

## 4. What exists today in `../informal-jobs-model` (ijm), all of which moves here

A uv package `informal_jobs_model` (no tests) depending on `eodgdl[giro]` by **git branch `giro-model`, locked at
`9aa68d0`** (pre-merge) and mxcensus 0.4.0 through an `override-dependencies`. Outputs are regenerated by notebooks
01–08 and are stale (written before eodgdl's chain rounds 16–17; stages 4–5 older than stages 1–2).

### 4.1 Informality chain (package code + notebooks 01–05)

- **S1 base frames** (`generate_enoe_od_dataframes.py`): pooled ENOE 2022t1–2023t4, Jalisco (`ent=14`); employed
  filter `clase2==1 & r_def==0 & c_res∈{1,3} & 12≤eda≤98` (l.75–91; note mxcensus's canonical filter floors age at
  15); `survey_weight = fac_tri / 8` (l.107); cross-quarter household key from the *private*
  `mxcensus.enoe._DWELLING_KEY_SPEC` / `_level_key` (l.12); household size from the SDEM roster. OD workers from
  `eodgdl.load_eod()` (`generate_od_dataframe` l.156) with a copy of giro's work-trip destination
  (`compute_od_work_trip_destination` l.140) and worker set (`config/od.yaml`).
- **S2 harmonization** (`harmonize_enoe_od_dataframes.py`, 568 lines): pairs of `harmonize_{enoe,od}_<attr>`
  functions (gender, occupation, age, education, municipality, marital status, relationship, household size,
  sector, workplace) with inline dicts; `assert_mapping_covers(values, mapping, allowed_unmapped)` (l.33) fails on
  unmapped codes (allowed: ENOE `e_con` 9, OD marital "Otros (especifique)"; ENOE municipality and the workplace
  rule unchecked). Target levels in `config/harmonization.yaml`; `mappings/sector.yaml` has `enoe_scian` (21 codes →
  4 sectors), `od_giro_empresa`, and `od_giro` (5 → 4). ENOE `lugar_trabajo` is a rule over `p4, p4b, p4e, p4f,
  p4h, scian` (l.463); OD `lugar_trabajo` from `OD_WORKPLACE_MAPPING` over `destino_trabajo` (l.492). Target
  `informal` from `emp_ppal`; components `informal_sector` / `informal_unprotected` from `tue2 ∈ {5,6,7}` (l.519).
- **M1 giro → sector**: eodgdl's giro, collapsed 5→4 by summing `prob_giro_*` (comercio→comercio;
  servicio, educacion→servicios_transporte; industria→manufactura_construccion; gobierno→gobierno_otro_agricultura);
  observed rows one-hot (`attach_sector_probabilities`, l.425–459).
- **M2 informality** (`informality_model.py`): binary, trained on all labelled ENOE workers, held-out = grouped fold
  0 restricted to the metro municipalities; features `genero, ocupacion, edad_num, escolaridad, municipio,
  estado_civil, parentesco, tamano_viv_cat, sector, lugar_trabajo`, robust arm without `escolaridad`; weights
  normalized to mean 1; same families and one-SE rule as giro (serial tuner). Evaluation: ECE, calibration
  slope/intercept, household bootstrap, PSU-grouped CV, isotonic calibration on out-of-fold predictions (not
  shipped, `SHIP_CALIBRATED=False`), reweighting held-out ENOE to the OD missing-education profile, an ENOE–OD gap
  decomposition, a raking diagnostic, component models. Unsampled metro municipalities are averaged over
  `training_municipalities` (`level_subsets`).
- **M2a place of work**: fixed-hyperparameter HGB P(lugar_trabajo | x, sector) per arm on ENOE; marginalized within
  each sector scenario.
- **Scoring**: P(I | x) = Σ_s P(s | x) · P(I | x, s) over the 4 sectors, writing `prob_informal_given_sector_<s>`,
  `prob_informal_sector_<s>`, `prob_informal`, `informal_predicted` (0.5), `informal_sampled` (one Bernoulli draw),
  expected informal/formal population. Bundle `informality_hybrid_model.joblib`.

### 4.2 NSE chain (notebooks 06–08, all inline)

- **M3 head-of-household education**: heads = reported `Jefe del hogar` else the oldest member; 7 AMAI classes from
  `escolaridad`; trained on the OD, weight **`ponderador_x` (the person weight, for a dwelling-level target)**,
  groups `folio_vivienda`; feature sets Base/Extended/Spatial (one-hot `ageb`) × LR/RF/sklearn GB chosen by strict
  argmin at default hyperparameters, then a small grid; `SimpleImputer(most_frequent)` +
  `OneHotEncoder(handle_unknown="ignore")`. Output `prob_education_*` and an expected score.
- **M4 AMAI components**: workers aged 14+ `{0,1,2,3,4+}`, complete bathrooms `{0,1,2+}`, bedrooms `{1,2,3,4+}`,
  each with and without income (dispatch on missing `ingreso_hogar`); trained on **national ENIGH 2022** (no Jalisco
  or urban filter), weight `factor`, groups `folioviv`; strict argmin selection; income bins from `ing_cor/3`;
  vehicles from `num_auto` only (vans/pickups possibly missed, uncertain); ENIGH↔OD maps without coverage checks
  (unmapped → NaN → mode). **The three components are imputed independently** ("parallel"); the head's education
  enters as a plug-in expected score. Output `prob_*` and expected scores. Bundle 368 MB.
- **M5 NSE** (not fitted): the sum of the six component expected scores cut at AMAI's thresholds into a **hard**
  level, then `calibrate_ageb_nse`: a weighted mid-rank within each AGEB (viv `ponderador`) read against the AMAI
  AGEB cumulative distribution (`data/NSE_por_AGEB_AMAI.xlsx`), giving a hard calibrated level. The distribution
  over NSE levels is never carried.

### 4.3 Shared machinery, drifted

ijm's `common.py` is a near-copy of eodgdl's `giro/_ml.py`, drifted both ways. **ijm's marginalization is newer**
(`predict_proba_marginalizing`: `level_subsets` only narrows the averaging, the missing label is never supported;
auxiliary `fit_level_model` attaches `training_level_shares_` and marginalizes its own inputs). ijm also has an
isotonic wrapper (`IsotonicCalibratedPipeline`) and `cross_validate_grouped`. eodgdl has the parallel/staged tuner.
`select_one_se`, `complexity_key`, `fold_table`, `bootstrap_by_group`, `marginal_log_loss`,
`reweight_to_target_profile`, `make_tree_preprocessor`, `prepare_model_features` are the same in both.

### 4.4 `../informal_jobs` is not a source

It is the 2023–24 Monterrey prototype (ENOE 2019 Nuevo León + the Monterrey OD survey 2019). Lessons, all avoided
here: hard labels between models (its informality rate: 29% by hard labels vs 37% by mean probability); no
weights; CV not grouped; balanced accuracy as selector; value maps inline and drifting between R and Python.
Ideas worth keeping as options: a Bayesian practical-equivalence (ROPE, `baycomp`) rule; SHAP.

## 5. Design

### 5.1 Layout

```
src/eodgdl/impute/                  the engine: pandas + scikit-learn; knows no survey vocabulary
  spec.py         TaskSpec / ChainSpec: parse + validate YAML; spec hashes
  sources.py      registry of source builders → a frame with declared keys, weight, CV group, population
  features.py     registry of feature builders; on-disk cache keyed by (builder, version, input versions)
  levels.py       category-level contract: declared levels + missing label; assert_known_levels
  harmonize.py    YAML value maps (per source → common levels), fail-on-unmapped with allowed_unmapped
  models.py       families/grids from the spec → pipelines (prepare → preprocess → classifier)
  select.py       grouped CV (parallel, staged boosting), one-SE within/across families, fold tables, compare()
  marginalize.py  batched marginalization (ijm semantics) + auxiliary level models with training shares
  arms.py         hybrid dispatch: N feature arms chosen per row by covariate missingness
  chain.py        parallel and sequential execution (§5.6): scenario enumeration, draws, transforms, derive steps
  evaluate.py     held-out metrics, group bootstrap, multiclass reliability/ECE, isotonic option, shift checks
  bundle.py       fit/save/load; metadata; load-time checks
  run.py          score(chain), retrain(task), compare(task, candidates); outputs
src/eodgdl/impute/tasks/            one YAML per task: giro, destino_trabajo (aux), informality, lugar_trabajo
                                    (aux), educacion_jefe, amai_trabajadores, amai_banos, amai_dormitorios
src/eodgdl/impute/chains/           one YAML per chain: sector_informality.yaml, nse.yaml
src/eodgdl/impute/harmonization/    common levels + value maps per source: enoe.yaml, enigh.yaml, eod.yaml,
                                    sector.yaml (SCIAN → 4 sectors, giro 5 → 4)
src/eodgdl/impute/sources/          source builders: eod (workers, heads, dwellings), enoe (pooled workers),
                                    enigh (households/dwellings)
src/eodgdl/impute/derive/           deterministic post-steps: AMAI points, NSE cut, AGEB rank calibration
src/eodgdl/giro/                    thin wrapper (backward compatible) + giro's feature builders (worker frame,
                                    work-trip destination, DENUE mix)
src/eodgdl/giro/_ml.py              re-export shim so bundles pickled against it still load
reports/imputation_<task>.qmd       replace the notebooks: read a retrain's artifacts, train nothing
```

Extras: `impute` = scikit-learn, joblib, pyyaml, mxcensus>=0.4.0 (training on ENOE/ENIGH and giro's DENUE
features need it; scoring from bundles on the OD needs it only for giro's DENUE features). `giro` becomes an alias
of `impute` for compatibility. The data files ijm reads (`NSE_por_AGEB_AMAI.xlsx`; check what else) move under
`data/` and into the catalog/registry (CLAUDE.md "Data access").

This reverses the old rationale in giro's docstrings ("knows nothing about ENOE, which is why it lives in the
survey package"): update README/CLAUDE.md and the giro docstrings when the ENOE tasks land.

### 5.2 Sources

A source builder returns a frame plus its contract: row keys, weight column (normalized to mean 1 inside the
engine), **CV group key** (composite allowed), population filter, and the source's data versions (for the cache and
bundle metadata). Planned sources:

| source | builder | keys / group | weight |
|---|---|---|---|
| `eod.workers` | `load_eod()` hab ⋈ viv, employed categories (giro's worker set; ijm's copy goes away) | person / `folio_vivienda` | `ponderador` (hab) |
| `eod.heads` | household heads (reported, else oldest) | dwelling / `folio_vivienda` | `ponderador` (viv): the right weight for a dwelling-level target (ijm used the person weight) |
| `enoe.workers` | pooled quarters via `mxcensus.load_enoe_persons` / `load_enoe_hogares`, employed filter as ijm's (age floor 12, not the canonical 15) | person / cross-quarter household, from the **index levels the public loaders return** (no private imports) | `fac_tri / n_quarters` |
| `enigh.households` | `load_enigh_hogares` + `load_enigh_viviendas` + personas (workers 14+) | household / `folioviv` | `factor` |

Harmonization maps every source's raw codes to the common levels before a task sees them; a code outside a map
raises unless listed in `allowed_unmapped` (fix ijm's unchecked ENOE municipality, workplace rule and all ENIGH maps
this way).

### 5.3 Task spec (giro, sketch)

```yaml
task: giro
source: eod.workers                 # train and score on the same source
target:
  column: giro_empresa
  classes: {Comercio: comercio, Servicio: servicio, Educación: educacion, Industria: industria,
            Gobierno/sector público: gobierno}     # label → slug; order = output order
features:
  builders: [giro.person, giro.dwelling, giro.work_trip, giro.destination_denue]
  numeric: [edad, dest_establecimientos_log, dest_share_grandes, dest_share_*]
levels: {destino_ambito: [ageb_urbana, ageb_rural, aeropuerto, fuera_zm, desconocido]}   # + schema-derived
missing_label: no_especificado
arms:                               # first arm whose `requires` are observed scores the row
  with_education: {features: [...], requires: [escolaridad]}
  without_education: {features: [...]}
auxiliary: {destino_trabajo: {features: arm features minus the destination block}}
selection:
  cv: {splits: 5, seed: 42, test_fold: 0}
  metric: weighted_log_loss
  rule: one_se
  families:
    LogisticRegression: {params: {C: [0.1, 1.0, 10.0]}}
    RandomForest: {n_estimators: 500, params: {max_leaf_nodes: [25, 50, 100], max_features: [sqrt, 0.7], min_samples_leaf: [1, 5, 10]}}
    GradientBoosting: {native_categoricals: true, params: {max_iter: [50, 100, 200, 400], learning_rate: [0.05, 0.1], max_leaf_nodes: [15, 31], l2_regularization: [0.0, 1.0]}}
evaluation:
  bootstrap: 500
  calibration_bins: 10
  shift: {profile: [sexo_nacimiento, edad, municipio, estado_civil, parentesco, personas_en_vivienda, ocupacion, trabajo_semana_pasada]}
  delta: {class: gobierno}
outputs: {prefix: giro}
```

An ENOE-trained task declares `source` (training) and `score_source` (scoring) separately: informality trains on
`enoe.workers` and scores `eod.workers`, both harmonized to the same levels; the held-out evaluation can be
restricted (`test_filter: metro municipalities`), and `level_subsets` averages unsampled municipalities over the
training ones.

### 5.4 Modes and CLI

| mode | API | CLI |
|---|---|---|
| score | `impute.score("sector_informality", tables)`; `giro.impute()` | `eodgdl impute score <chain> --out …` |
| compare | `impute.compare(task, candidates={name: spec_or_features}, seeds=(42,))` | `eodgdl impute compare <task> --spec candidate.yaml` |
| retrain | `impute.retrain(task)` (one task; a chain retrains in order) | `eodgdl impute retrain <task|chain> --out output/impute/<task>` |

**compare**: hyperparameters fixed at the published bundle's winner (from its metadata), the same grouped folds on
the training rows only (the held-out fold is never touched), each candidate's fold log losses paired against the
baseline's: mean difference ± SE, and repeated-CV means with several seeds. Features come from the cache, so a
feature-list change costs one CV of one configuration. The winning spec is committed, then `retrain` reselects
hyperparameters and writes the bundle.

### 5.5 Outputs

Per task (prefix `<p>`; giro keeps its current names): `prob_<p>_<slug>` (rows sum to 1, observed rows one-hot),
`<p>_observado`, `<p>_imputado`, `<p>_final` (convenience arg-max), `<p>_fue_imputado`, `<p>_model_used` (arm),
`<p>_prediction_confidence`, `<p>_marginalized_features`; optional expected score (ordinal, from a score map);
per-scenario conditionals `prob_<p>_given_<upstream>_<s>` in enumerated chains; draws (§5.6). Scenario outputs
(shift, delta) share the schema plus `scenario`. `validate_probability_rows` on every output.

### 5.6 Parallel and sequential execution

A **chain spec** lists steps in order; each step is a task (or a derive step) and says how it receives earlier
targets:

```yaml
chain: nse
mode: sequential                  # or: parallel (every task sees observed covariates only)
propagation: draws                # sequential only: enumerate | draws | expected
draws: 50                         # M completed datasets, seeded
steps:
  - educacion_jefe
  - amai_trabajadores:  {uses: [educacion_jefe]}
  - amai_banos:         {uses: [educacion_jefe, amai_trabajadores]}
  - amai_dormitorios:   {uses: [educacion_jefe, amai_trabajadores, amai_banos]}
  - derive: amai_puntos           # sum of the component scores (per draw)
  - derive: nse_nivel             # AMAI cut points
  - derive: nse_ageb_calibrado    # AGEB rank calibration against data/NSE_por_AGEB_AMAI.xlsx
```

```yaml
chain: sector_informality
mode: sequential
propagation: enumerate            # 4 sectors: exact
steps:
  - giro
  - informality:
      uses:
        giro: {as: sector, transform: {comercio: comercio, servicio: servicios_transporte,
               educacion: servicios_transporte, industria: manufactura_construccion,
               gobierno: gobierno_otro_agricultura}}      # many-to-one collapse by summation
      auxiliary: {lugar_trabajo: {conditions_on: [sector]}}   # marginalized within each scenario
```

- **Parallel**: each task is scored from observed covariates; tasks are independent (today's AMAI components).
- **Sequential**: task k is *trained* with the earlier targets as features (observed values in its training source;
  in ENIGH all three components are observed together) and *scored* conditioning on the earlier imputations:
  - `enumerate`: exact scenario sum over the product of earlier classes (P(y | x) = Σ_s P(s | x) P(y | x, s)); rows
    with an observed earlier target are one-hot. Feasible while the product of class counts stays small (sector: 4).
  - `draws`: M seeded multiple imputations: draw y₁ ~ P(y₁ | x), then y₂ ~ P(y₂ | x, y₁), …; derive steps run per
    draw, so the NSE level's *distribution* is carried (ijm carries only a hard level). Outputs: marginal
    probabilities estimated over draws, plus the M completed datasets (long format, `draw` column) for
    variance-aware downstream use.
  - `expected`: plug-in expected score of an ordinal upstream (what ijm does for education → AMAI); allowed but
    flagged as plug-in in the outputs.
- Cross-source sequences work because every source is harmonized to common levels (giro trained on the OD feeds
  informality trained on ENOE through the sector transform; in ENOE the sector is observed from SCIAN).
- Retraining a task invalidates every later step of the chains it is in (upstream bundle hashes, §5.8).
- A cyclic, MICE-style iteration is out of scope; the order is declared.

### 5.7 Machinery to port (and from where)

- Marginalization: **ijm's semantics** (`averaging_shares`, `level_subsets`, auxiliary models with training shares
  that marginalize their own inputs), implemented **batched** (§3): expand all combinations into one frame, one
  `predict_proba` per model, weighted sum per row. Enumerated chains reuse the same expansion.
- Tuner: eodgdl's parallel/staged `tune_models`; `n_jobs=1` supported. `select_one_se`, `complexity_key`,
  `fold_table` as they are. ROPE optional later.
- Evaluation: eodgdl's held-out metrics, `test_metrics_with_uncertainty`, `bootstrap_by_group`,
  `marginal_log_loss`, `reweight_to_target_profile`, delta adjustment; ijm's isotonic wrapper (ship toggle),
  grouped robustness CV, gap decomposition and raking diagnostic; reliability/ECE made **multiclass**.
- Auxiliary models become tunable and live in the task bundle.
- ijm's notebooks 06–07 selection (strict argmin, feature set chosen at default hyperparameters) is replaced by the
  engine's grouped CV + one-SE; `handle_unknown="ignore"` + mode imputation by the level contract (raise on unknown,
  marginalize unsupported, `no_especificado` for missing).

### 5.8 Bundles

One joblib per task: fitted arms, auxiliary models, category levels, class map, metadata = spec hash, feature
builder versions + level-contract hash + harmonization-map hash, eodgdl version, data versions (survey sha256s,
DENUE release, ENOE periods, ENIGH year), winner params and fold losses (for `compare`), test metrics, sklearn
version, random state, and the hashes of the upstream bundles it was trained with. On load: refuse on sklearn
version mismatch, level-contract mismatch (the `localidad_rural`→`ageb_rural` class of error surfaces here), and
upstream-hash mismatch within a chain. Pickled callables live at `eodgdl.impute.*`; `eodgdl.giro._ml` stays as a
shim. Bundles ship under `data/` with sha256s in `registry.txt` and a release tag (`REF`): **the `v0.2.0` tag has no
giro bundle**, so today `giro.load_model()` only works with `$EODGDL_DATA_DIR` pointing at a clone. Size: ijm's AMAI
bundle is 368 MB — check what makes it large (probably random forests) before shipping it through the mirror.

### 5.9 Feature cache

Parquet under the eodgdl cache dir, keyed by builder name + version + input versions (eodgdl version and survey
sha256s, DENUE release, ENOE periods, ENIGH year). The chain rules move trips into and out of
`motivo_viaje == "Trabajar"`, which moves `destino_trabajo` and `lugar_trabajo`: an unversioned cache goes stale
silently. `--refresh` rebuilds.

## 6. Known defects to fix during the migration

1. `giro.impute_under_covariate_shift` calls `impute_giro` **without `destination_models`** (verified): in the
   shift scenario, workers without a work trip are marginalized by global shares, not P(destination | x). ijm
   reports this scenario as a headline.
2. eodgdl's auxiliary destination models have no `training_level_shares_` (verified in `giro/_ml.py`
   `fit_level_model`); fixed by porting ijm's version.
3. Scoring cost (§3): batch the marginalization.
4. `adjust_imputed_share` clips at 1, so the target share is not reached when the clip binds (minor).
5. From ijm, to fix as the models move: person weight for the dwelling-level education model;
   `handle_unknown="ignore"` + mode imputation; ENIGH trained nationally (rural included) and applied to urban
   Guadalajara with no shift check (add one; consider a filter); ENIGH maps without coverage checks; `num_auto` only
   for vehicles (verify); hard NSE level only (fixed by draws).

## 7. Migration plan

Each phase ends with `uv run pytest` green and is committed separately (ask the user before committing or
pushing). Parity tests guard behaviour: exact where no change is intended, documented differences otherwise.

0. **Groundwork.** Bump mxcensus to `v0.4.0` in `[tool.uv.sources]`, `uv lock`, run the suite (reweight and giro use
   `load_census`, `load_mg_census`, `load_denue`; signatures reportedly unchanged, not yet run). Add a small
   committed parity fixture from `outputs/reference/` (a few hundred workers' features and probabilities, including
   marginalized ones).
1. **Engine core from giro, no behaviour change.** `eodgdl.impute` levels/harmonize/models/select/marginalize/arms/
   evaluate; generic code out of `giro/_ml.py` and `giro/model.py`; `giro/_ml.py` becomes a shim. First batch the
   marginalization (exact parity, minutes → seconds), then port ijm's semantics and auxiliary shares (parity except
   documented rows). Unit tests on synthetic data: marginalization (batched = recursive), one-SE, staged = unstaged
   tuning, harmonize fail-on-unmapped.
2. **Specs, sources, features, cache, bundles v2** for giro. `tasks/giro.yaml` absorbs `giro/config.yaml`;
   converter from the current bundle so `score` works before any retrain; `giro.load_model()` keeps the old dict
   keys readable.
3. **Modes and CLI**: `score`, `compare`, `retrain`; `retrain giro` reproduces the notebook's selection and metrics.
   Evaluation artifacts + `reports/imputation_giro.qmd`; delete `notebooks/giro_model.ipynb`; fix §6.1.
4. **Chains**: `chain.py` with parallel and sequential modes, `enumerate` / `draws` / `expected`, derive steps.
   Synthetic two- and three-task chains in tests (e.g. check that enumerate = the mean over many draws).
5. **ENOE sources and the informality chain.** `sources/enoe.py` (public mxcensus API only), `harmonization/*.yaml`
   ported from ijm's inline dicts with coverage checks, `lugar_trabajo` rule, tasks `informality` + `lugar_trabajo`,
   chain `sector_informality`. Parity: retrain on the same ENOE periods and compare metrics and imputed rates with
   ijm's (regenerate ijm's outputs on its locked eodgdl first, or compare against a fresh ijm run; exact equality is
   not expected because of eodgdl's newer data).
6. **ENIGH sources and the NSE chain.** `sources/enigh.py`, tasks `educacion_jefe` + the three AMAI components, derive
   steps (points, cut, AGEB calibration; move `NSE_por_AGEB_AMAI.xlsx` into `data/` + catalog), chain `nse` in
   sequential/draws mode, with a parallel run of the same chain as the comparison to ijm's approach.
7. **Release.** Docs (README, CLAUDE.md, reports), version bump, `REF` to the new tag, bundles in `registry.txt`, tag
   and push. Then decide ijm's fate (§10).

## 8. Division of work across repositories

- **eodgdl**: everything above — engine, sources, harmonization, tasks, chains, derive steps, bundles, reports.
- **mxcensus**: nothing required if the public loaders suffice (their returned index levels give the ENOE
  cross-quarter household key). If a public pooled-quarters loader or a public key accessor is wanted, that is a
  separate change in mxcensus (open question).
- **ijm**: nothing new; after phase 7 it can be archived or reduced to paper-specific figures reading eodgdl's
  outputs.

## 9. Invariants (keep these true throughout)

- No retyped lookup tables in Python: every value map and level list lives in YAML (like `tasha/mappings.yaml`).
- Hard labels never feed a downstream model; probabilities, enumerations, draws or flagged expected scores do.
- Every model is trained with survey weights and household-grouped CV.
- Unknown category → raise; unsupported → marginalize (recorded per row); missing → `no_especificado`.
- Scoring never refits; bundles carry everything needed plus the hashes that prove what they were trained on.

## 10. Open questions for the user

1. Sequential propagation defaults: `enumerate` for sector → informality and `draws` (how many?) for the NSE chain?
   And the order of the AMAI components in the sequence (workers → bathrooms → bedrooms is a guess)?
2. ENIGH population: keep national training (ijm) or restrict (Jalisco / urban localities), given the shift to urban
   Guadalajara?
3. Ship isotonic-calibrated models when they win on held-out calibration, or keep them diagnostic (ijm: not shipped)?
4. One-SE only, or add the ROPE (baycomp) rule?
5. ijm after the migration: archive, or keep for paper figures only?
6. mxcensus: are the public loaders enough, or add a public pooled-ENOE loader / key accessor there?
7. When does `giro-model` merge into `main`, and should the NSE/informality work be a separate branch?

## 11. Practical notes

- In this shell `cp`, `rm` and `mv` are aliased to interactive versions and hang a non-interactive command: use
  `command cp -f`, `command rm -f`. Chaining heredocs inside `&& \` lines broke four commit messages here: write
  commit messages to a file and use `git commit -F`.
- Run anything that reads the survey with `EODGDL_DATA_DIR=$PWD/data` (the mirror tag lacks the giro bundle).
- To execute a notebook headless, copy it to a scratch directory **with a `data/` subdirectory** (the giro
  notebook's last cell writes `ROOT/data/…`), then `uv run jupyter nbconvert --to notebook --execute --allow-errors
  --ExecutePreprocessor.timeout=-1 --output executed.ipynb <copy>`.
- ijm's notebooks are JSON; extract code cells with a short Python script rather than reading them raw.
