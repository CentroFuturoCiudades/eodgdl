# Handoff: `eodgdl.impute`

State at the end of session 7 (2026-09-29, the second complete review). Read `docs/impute.md` first (the design:
what the engine does and why), then this file (state, decisions, pending), then `CLAUDE.md`'s `eodgdl.impute` section
(local). The history of sessions 1–6 and the original plan (§1–§11 of the earlier handoff: the migration from
informal-jobs-model, the phases, the parity it reached) is in git: `git show 19e9d95:docs/handoff_impute_engine.md`.

## Where things are

- **`v0.7.0`** (session 9, 2026-09-30): the head's education (`educacion_jefe`) reads the other household members'
  highest education (`eod.household_education`) in an arm of its own (`con_hogar`, requires it; `sin_hogar` is the model
  as it was): CV -0.0465 against the single arm; one model reading it with the median where missing lost 0.030 on the
  heads whose household skipped it (53% of the imputed, weighted). Retrained: held-out 1.420 -> 1.372 (each dwelling by
  its arm), the NSE shares within 0.1 pp; parity: only `educacion_jefe` and `nse` moved. The census builder
  `census.home` (census ratios, DENUE establishments per adult, AMAI's NSE by AGEB for the dwelling's AGEB) and the
  household builder were measured on giro and educacion_jefe (numbers in the task YAMLs): nothing for giro, census and
  AMAI -0.003 for education and -0.0008 beyond the household's, not taken. `eod.dwellings` gained
  `jefe_folio_habitante`. The run is `output/impute_v070/` (`rerun.sh ... educacion_jefe`); the user swaps it into
  `output/impute/`.
- **TASHA gaps** (session 9): after imputation 10 required columns (+ `Formality`) still hold defaults for some rows:
  `IncomeClass` 58% unknown, `DwellingType`/`License`/`TransitPass`/`FreeParking` not surveyed, `EmploymentStatus`
  4,370 adults unanswered (the socio-economic block skipped; `StudentStatus` 3,954 of them), `Occupation` 9,484 workers
  (giro's imputation not read by `tasha.build`), `EmploymentZone` 3,286 workers without a work trip, `SchoolZone`
  2,965 students (mostly the Easter weeks, 3-16 April). The plan proposed to the user (pending decisions: one seeded
  draw vs arg-max, the meaning of a worker's "0" zone, License/TransitPass sources): wire giro and informality by a
  draw; an employment/student task first in the chain; income from ENIGH in the NSE chain; work and school location
  models; dwelling type fused from the census sample questionnaire or ENIGH; parking from the trips' answers.
- **`v0.6.0`** (session 9, 2026-09-30): giro's destination features are the staff-weighted shares of the 20 SCIAN
  sectors plus the staff share in large establishments (DENUE 2022-11, each establishment weighted by the midpoint of its
  `per_ocu` band, 500 for the top one), replacing the five shares of establishments per giro; `denue_scian2` became
  `denue_sectors` (code -> share column). Measured with `eodgdl impute compare` (three seeds, the bundle's winners; the
  numbers in `giro.yaml`): sector shares by staff -0.0080 / -0.0064 CV log loss (with / without education), by
  three-digit subsector -0.0098 / -0.0093 (not taken, the user's choice: 69 more features for about 0.002, small classes'
  F1 lower); the three missing-value alternatives tried (the booster's NaN branch, the dwelling's AGEB for workers
  without a work trip, zero establishments) did not help and were removed. Retrained: the same selections, the combined
  held-out loss 0.8896 -> 0.8783 (with education 0.8894 -> 0.8776, without 0.9061 -> 0.8958), imputed shares within
  0.5 pp, the informality rate 35.908% -> 35.932%; parity: only giro and `sector_informality` moved. The giro bundle
  reinstalled, `tests/data/giro_parity.parquet` regenerated. `output/impute/` is its run (`rerun.sh ... giro`, 50 bootstrap
  bundles, the other tasks copied from v0.5.1's root), v0.5.1's kept as `output/impute_v051/`; the giro, informality
  and figures reports re-rendered from it. With the new features isotonic recalibration no longer lowers the
  with-education arm's mean ECE (0.0187 -> 0.0191; the share-weighted ECE still falls), so the giro report's sentence
  now follows the table.
- **Paused** (session 8, 2026-09-29): none of the open questions below is needed for the delivery (none moves an
  imputed value, a bundle or a model input), and the user paused the work; reopen the variance question if a
  deliverable quotes the reports' intervals. Session 8 added the bootstrap caveat to `docs/impute.md`'s limits and
  deleted the old comparison roots.
- **Released `v0.5.1`** (2026-09-29): 8d8e7c8 (giro's workers without a work trip take `sin_viaje`, below) and 6d0e9b7
  (version 0.5.1, `REF = "v0.5.1"`), tagged, pushed and merged into `main`; the mirror check passed again (giro and both
  chains from the tag equal `output/impute/` exactly).
- **Released `v0.5.0`** (2026-09-29, the user's request): c89d20b (version 0.5.0, `REF = "v0.5.0"`, `uv lock`), tagged
  `v0.5.0` (lightweight, as before), pushed with `impute-chains`, which is merged into `main` (`--no-ff`) and pushed.
  Session 7's commits: 2bf14e0 (parity tool), 41cd5a7 (selection on the scoring path), 0c0b120 (copied records
  documented), 855c09f (focus 3), ce40f35 (bootstrap multiple imputation), 83df4f2 (tests, a YAML bug, the legacy giro
  path removed), ae8c836 (the final run, the six bundles reinstalled, the docs).
- The mirror check passed: with an empty `EODGDL_CACHE_DIR` and no `EODGDL_DATA_DIR`, `giro.load_model()` and `eodgdl
  impute score` of giro and both chains fetched the survey, the zone file, AMAI's table and the bundles from the tag,
  each checksum-verified, and giro's scores and both chains' scores and completions equal `output/impute/`'s exactly.
- `output/impute/` is session 7's final run (`scripts/impute/rerun.sh`: six retrains with 50 bootstrap bundles each,
  giro and both chains scored, each chain's 50 completed datasets with bootstrap models under
  `<chain>/multiple_imputation/`, both evaluations with Rubin's rules). The older roots kept for comparison
  (before session 7, after its selection change, before session 6's held-out change, with their figures and reports)
  were deleted in session 8; to compare with an earlier state, rerun it from its commit into a new root.
- `uv run pytest`: 176 passed (~4 min).

`v0.5.0` is **breaking**: `giro.load_model` returns the v2 bundle; `giro.impute_giro`, `bundle.from_legacy_giro`,
`legacy_view` and `giro/_ml.py` are gone (pickles from before 0.3.0 no longer load); informality's sector level
`gobierno_otro_agricultura` is `gobierno`; giro reads the interview month.

**Giro's workers without a work trip** (option B, decided 2026-09-29 after `v0.5.0`, released as `v0.5.1`): not
having made a work trip on the survey day is a state, not a missing value. `eod.work_trip` takes a `no_trip` level
(default null: missing, as informality keeps it, since ENOE knows no such state); giro names `sin_viaje` and declares it
(`levels.extra`, a new spec field: levels appended to a schema column's), and its auxiliary destination model is gone.
Measured first on the published model: over all 2,895 known workers without a work trip, out of fold (three seeds),
averaging over destinations predicted commerce 36.0% and services 40.7% against 42.3% and 34.4% observed; the level
42.9% and 34.3%, log loss lower by 0.014–0.043 on those workers (the held-out fold's gain the largest). Retrained: the
same selections, the combined held-out loss 0.8966 → 0.8896 (accuracy 63.3% → 63.7%), imputed shares within 0.3 pp
(commerce +0.2, services −0.3), the informality rate 35.917% → 35.908%; parity: only giro and `sector_informality`
moved. The giro bundle reinstalled, `tests/data/giro_parity.parquet` regenerated (its 131 workers without a trip take
`sin_viaje`).

## Pending for the user

1. Nothing decided is pending. The chains publish the single fit's outputs (the user's choice, 2026-09-29: the
   bootstrap datasets stay a separate product for analyses that need intervals).

## Session 7, by focus

Every change that moved an output was decided by the user from measured options; everything else was proved with the
parity tool. Scratch measurements are not kept; their method and numbers are here and in the commit messages.

**Step 0.** `scripts/impute/parity.py ROOT_A ROOT_B [--models]` (every parquet exactly, then within 1e-12, the
held-out tables apart; the summaries key by key without timings, versions and hashes; every fitted model by
`predict_proba`) and `scripts/impute/rerun.sh ROOT [TASK ...]` (every output into a root, `set -o pipefail`).

**1. Copied worker records** (decided: keep household CV groups, document). The 1,793 persons of
`hab.diario_repetido` are 1,238 pairs (`eod.repeated_diary_pairs`, the loader's own rule): person records copied with
the age changed (giro agrees 98.8% in 164 pairs of known workers, occupation 98.8%, age 12%). The model memorizes them
(out-of-fold 0.726 with the twin in training, 0.903 without): household-grouped CV optimistic by ~0.006, the held-out
loss by ~0.003–0.005 (50 of 3,485 held-out workers). The selection is not driven by it (10 seeds: boosting ahead of the
forest by 0.013–0.024), but a fresh twin-grouped split flipped giro's without-education arm on a knife-edge, a
fragility of the one-SE rule on five folds (below). `reports/imputation_giro.qmd` #sec-selection counts it at render
time. How the copies train (both, half, one) changed held-out losses and shares by noise only.

**2. Missing values in selection** (decided). Selection now scores each CV fold as the bundle scores it
(`run.fold_expander`, 41cd5a7): educacion_jefe moved from boosting to a logistic regression (C 0.1; CV as scored
1.456, boosting 1.501; held-out 1.4696 → 1.4204); no other selection moved. Asked whether ROPE would be steadier than
the one-SE rule: measured, no (the Bayesian correlated t-test calls more ties, which the family order LR < RF < GB then
decides; stable only where its width decides); three pooled repeats would be (0 of 120 seed sets flip): the user kept
one five-fold run. Two income arms for the education task (the AMAI tasks' layout) would carry the refusal pattern:
not adopted.

**3. Labels and features** (decided). The income refusers' reported education stays and trains the model: ENIGH 2022's
metro heads (licenciatura 22.1%, 19.9–24.4%) sit between the EOD's reporters (15.4%) and refusers (56.7%), and the EOD
departs from ENIGH at both ends (primaria 13% vs 24%), so neither correction is supported (`imputation_nse.qmd`
#sec-education-benchmark). Every ENOE and ENIGH code the maps read matches INEGI's labels; every AMAI point and cut
matches AMAI's 2022 rule. The EOD records the level of schooling, not its completion: documented with its bound (the
calibrated NSE within 0.28 pp, #sec-completion). ENOE's agriculture moved from the government sector to manufacturing
(the EOD's government workers 13.1% → 9.8% expected informality, the overall rate unchanged). April's capture change
reaches giro (an office as the work destination half as often, within zones too; giro unreported by 6% of January's
workers and 41–43% of March's and April's): the interview month is a giro feature (CV −0.0013 / −0.0020; the held-out
difference within noise).

**4. Report claims.** The CV and held-out losses now describe the same predictor, and the reports say so; asserts
back the comparative claims (hard vs probabilistic shares, isotonic's trade, the tails of the plug-in, the benchmark
gap, the combined curve); the informality report states what grouping by sampling unit changes instead of presuming
it; the NSE report's draws are described as predictive uncertainty, with Rubin's rules beside them.

**5. Bootstrap bundles** (decided: opt-in, built, ce40f35). `retrain --bootstrap B`, a chain's `uncertainty:
bootstrap` (draw d scored with bootstrap bundle d mod B), `score <chain> --draws M --bootstrap` (completed datasets),
Rubin's rules in the diagnostics (`evaluate.rubin`, `evaluate.cluster_shares`). Measured first on 20 refits: point
estimates within 0.4 pp; the informality rate's standard error 0.51 → 0.77 pp (fraction of missing information 0.73);
giro's shares ×1.04–1.15; the NSE before calibration ×0.99–1.10, after calibration ×1.00. The final run (50 bootstrap
bundles per task, 50 completed datasets per chain): the informality rate 35.83%, 95% interval 34.44–37.23%, standard
error 0.52 → 0.70 pp with bootstrap models (×1.35; fraction of missing information 0.40 → 0.67), by drawn giro ×1.03–1.33;
giro's shares ×1.01–1.09; the NSE ×1.01–1.11 before calibration, ×1.00 after (fraction 0.01–0.06). In the reports:
`imputation_informality.qmd` #sec-uncertainty, `imputation_nse.qmd` #sec-level. The bootstrap bundles make
`output/impute/` weigh 840 MB (giro's 391 MB): local only.

**6. Tests.** Coverage of `eodgdl.impute` and `eodgdl.giro` 84% → 89% and beyond: the diagnostics of a task chain and
of a distribution chain (synthetic), every refusal of `check_bundle`, AMAI's points, levels and calibration, the EOD
sources on the survey, the bootstrap refits and chains, Rubin's rules. The source test found a bug (below).

**7. Code and docs** (decided). eod.yaml listed the dwellings' employed categories in flow style, which splits "Tenía
trabajo, pero no trabajó" at its comma: 104 workers aged 14+ uncounted (94 dwellings' NSE, ≤0.06 pp before calibration);
fixed, and both EOD sources now refuse a category the schema lacks. The legacy giro path removed (breaking, above).
B023 closures bound; `eodgdl.giro` declares `__all__`. `docs/impute.md` written; this file trimmed.

## Open questions and limits (not asked; for a later session)

- Why the EOD's heads' education departs from ENIGH's metro distribution at both ends (the question's wording, the
  sample, the capture).
- The within-imputation variance ignores the survey's strata and design stages (`reports/expansion_factors.qmd`).
  The technical report's Metodología calls the sample "Aleatorio Estratificado Trietápico": centralidad strata, and its
  sample replacements are listed by AGEB and block; the survey files carry the AGEB, not the block. The bootstrap
  bundles resample the training sources' dwellings and households, not their primary sampling units, so the between
  variance ignores the designs too (ENOE's frame keeps `survey_stratum` and `survey_psu`). Neither is measured.
- The one-SE rule on one five-fold run can flip near ties (recorded; the user kept it).
- Copied worker records leave the CV and held-out losses slightly optimistic (documented).
- `run.retrain` is long; auxiliary models have fixed hyperparameters; SHAP, ROPE and a varying training population in
  `compare` are not implemented; the education task uses the centrality, not the AGEB.

## Practical notes

- Prove an engine change: `scripts/impute/rerun.sh ROOT` (≈70 min with 50 bootstrap bundles per task; `BOOTSTRAP=0`
  for none), then `uv run python scripts/impute/parity.py output/impute ROOT --models`. A root rerun for a few tasks:
  copy `output/impute` and `rerun.sh ROOT task ...`.
- Every CLI step imports the working tree when it starts: don't edit `src/` or `git stash` while a run goes.
- A change to a task's features or levels makes its installed bundle unusable (`check_bundle`) until the retrained one
  is installed (`data/`, `registry.txt`; for giro also `tests/data/giro_parity.parquet`, regenerated by scoring its
  384 workers with the new bundle).
- Render a report in place: `QUARTO_PYTHON=.venv/bin/python quarto render reports/imputation_<name>.qmd`
  (`--output-dir` moves the HTML away). Point a report at another root with `EODGDL_IMPUTE_GIRO`,
  `EODGDL_IMPUTE_INFORMALITY`, `EODGDL_IMPUTE_SECTOR_INFORMALITY` or `EODGDL_IMPUTE_DIR`.
- The feature cache is under `<EODGDL_CACHE_DIR>/impute/` (keys over code, configuration and data); `--refresh`
  rebuilds. ENOE and ENIGH come from mxcensus's cache; mxcensus's labels of ENOE's `par_c` 406–411 look shifted (the
  map reads the hundreds, so it is unaffected).
- In this shell `cp`, `rm` and `mv` are interactive aliases: use `command cp -f` and the like.
