# Handoff: `eodgdl.impute`, one repository for every imputation model on the survey

## 0. Status at the end of session 6 (2026-09-28): session 5's review notes resolved; next, a second complete review

Read this section first, starting with **Session 7: a second complete review**; §1–§11 below are the original plan (session 1), kept
as the record of the design. Then read `CLAUDE.md` (local, gitignored: its `eodgdl.impute` section describes every
module as built) and `src/eodgdl/impute/__init__.py`.

### Session 7: a second complete review of `eodgdl.impute` (start here)

The user asked (2026-09-29) for a second complete review, in a new session. Session 4 made the first (a review and a
refactor for one workflow, "After the release: review and simplification" below); sessions 5 and 6 resolved its notes.
This one reads the engine as it stands at 7f783b3 with fresh eyes: whether the models, their evaluation and their
outputs say what the reports say, and what the first review missed. Scope: `src/eodgdl/impute/` (sources,
harmonizations, tasks, chains, derive steps), `src/eodgdl/giro/`, the `eodgdl impute` CLI, the four
`reports/imputation_*.qmd` and their tests. The survey loading, the chain rules, `tasha` and `reweight` are out of
scope except where an impute source reads them.

**The user's rules** (standing): ask before pushing, merging or releasing, and before any change that moves published
outputs (scores, held-out metrics, the reports' numbers): bring the numbers and the options. Fix what moves nothing,
with tests, and prove it with the parity recipe ("After the release" below). The branch is `impute-chains`, five
commits ahead of `origin` at the end of session 6 (70eab10, c5f905f, eee00ad, be127dc, 7f783b3; check `git status
-sb`: the user may have pushed since). `output/impute/` is session 6's run and matches 7f783b3.

**How to run it.**
- Read this §0, `CLAUDE.md`'s `eodgdl.impute` section and `src/eodgdl/impute/__init__.py`, then the modules in the
  order data flows: `spec`, `sources/`, `features`, `harmonize` (with `harmonization/*.yaml`), `levels`, `models`,
  `select`, `marginalize`, `arms`, `bundle`, `run`, `chain`, `derive/`, `diagnostics`; then the task and chain YAMLs;
  then the reports.
- Every finding gets the claim, the evidence (file:line, a number from a script) and a class: a defect in today's
  outputs, a latent defect, a decision for the user, or a cleanup. Record them here, as session 4 did.
- **Step 0, before changing anything: commit the parity check as a tool.** Session 6 wrote it twice in a scratchpad,
  now gone: `scripts/impute/parity.py ROOT_A ROOT_B [--models]` comparing every parquet exactly, then within 1e-12,
  the held-out tables apart (they move whenever the evaluation does), the summaries as dicts (timings, versions, the
  bundles' sha256 and the chains' recorded eodgdl version aside) and every fitted model (arms, auxiliary, components)
  by `predict_proba` on its task's frames; and a runner that retrains the six tasks, scores giro and both chains and
  evaluates the chains into a root, under `set -o pipefail` (piped through `grep`, a failed retrain exits 0).

**Focus, in the order of what could change a conclusion.**

1. **Copied worker records leak into giro's evaluation** (found 2026-09-29, not fixed). The 1,793 persons of
   `hab.diario_repetido` form 794 groups of diaries copied across households (`eod.flag_repeated_diaries`; the groups
   come from `eod._diary_signatures` by AGEB, date and signature). Among giro's 17,429 known workers, 333 are flagged;
   the 162 pairs of them in different dwellings share the giro 98.8% of the time (chance 30.6%), the sex 94% and the
   destination zone 100%, the age only 12%: copied worker records, not only copied trips. The CV groups are households,
   so 53 of the 3,485 held-out workers have their twin in training, and the CV folds split pairs the same way. Measure
   the effect of grouping twins with each other (a CV group column on `eod.workers`: the twin group where there is
   one, else the household) on the held-out metrics and the selections, and decide how the copies train (both, one,
   or half weight each). The dwelling tasks are barely touched: dwellings that share a copied diary agree on the
   head's education 38% of the time and on income 53%, random dwellings of the same AGEB and date 34% and 50%.
2. **Missing values across training, selection, evaluation and scoring.** Since be127dc the held-out evaluation scores
   as the bundles do, but `select.tune`'s grouped CV still compares `predict_proba`, where the missing label is the
   category the model learned: the selection optimizes another predictor than the one that scores (educacion_jefe's
   CV log loss 1.390 against its held-out 1.470). Score the CV folds through the scoring path (fold training shares,
   fold auxiliary models) and see whether a selection changes; if one does it moves bundles and scores (numbers to the
   user). Income non-response is handled three ways: educacion_jefe averages it over the training shares (one arm),
   the AMAI tasks score it with an arm trained without income (`sin_ingreso`), giro averages a missing destination
   over an auxiliary model. Decide whether educacion_jefe should take the AMAI tasks' two arms (`run.compare` measures
   it). Option B for giro (the no-trip destination and mode as the learned category) stays the user's decision.
3. **The labels and features the EOD tasks learn from.** A third of educacion_jefe's training heads live in dwellings
   that refused the income question, whose reported licenciatura looks inflated (session 6, item 1;
   `reports/imputation_nse.qmd` #sec-income), and the known heads' education enters the AMAI points directly. Measure
   what training without those labels (or with them down-weighted) does to the imputed education and to the NSE before
   and after calibration. Re-check the harmonization maps (`harmonization/*.yaml`) against the ENOE, ENIGH and EOD
   questionnaires: the parity with informal-jobs-model proved the port, not the mappings. Check whether the April
   capture change (trips 3+, `reports/loading.qmd` #sec-april) reaches a feature (giro's destination is the work
   trip's).
4. **What the reports claim.** CV and held-out log losses now describe different predictors (focus 2): read every
   sentence that compares or interprets them, and every structural claim that no `assert` holds. The chains carry the
   draws' Monte Carlo error but no model uncertainty (one fitted bundle per task): say so where intervals are shown
   until focus 5 gives them.
5. **Proper multiple imputation: bootstrap bundles** (the user asked for it, 2026-09-29, after asking whether MICE
   would fit: it would not replace the engine, since most of it is data fusion under conditional independence, which
   chained equations cannot improve and, cycling EOD-only against ENOE- or ENIGH-only variables, would fill with
   artifacts; its multiple-imputation discipline is what is missing). Today each task has one fitted bundle, so the
   NSE chain's 50 draws carry the predictive randomness only: the imputations are improper and every interval
   computed from them is too narrow. Design, to measure and then bring to the user:
   - Per task, B refits of each arm's selected configuration (no new selection) and of its auxiliary models on a
     household-cluster bootstrap of the training rows (the source's CV group, survey weights kept; ENOE's `hogar`,
     ENIGH's dwelling), seeded from the task's seed. They are a retrain product (`<root>/<task>/bootstrap/`, or
     refitted on demand from the seeds): at B = 50 they would weigh ~50 times today's 16 MB of installed bundles, too
     much for the data mirror, so the shipped bundle stays the single fit.
   - A chain option (say `uncertainty: bootstrap`, default off so no output moves until the user decides): with
     `draws`, draw m scores every step with bootstrap model m mod B (bootstrap multiple imputation: a posterior draw
     of the model, then a draw from its predictive); with `enumerate` or `expected`, the marginals per bootstrap model,
     whose spread is the model's share of the uncertainty. The rank calibration already runs per draw.
   - Estimates combine by Rubin's rules: the mean over imputations, and a variance of the within-imputation part
     (the survey's own design variance of the estimate, which needs a design-based estimator for the EOD's three
     nested weights: `reports/expansion_factors.qmd`) plus (1 + 1/M) times the between-imputation part. The
     diagnostics (`evaluate_distribution`'s Monte Carlo table) and the reports gain the intervals: giro's imputed
     shares, the informality rate by sector and zone, the NSE mix by AGEB before and after calibration.
   - Check that the point estimates barely move (the mean over bootstrap models against the single fit) while the
     intervals widen, and by how much per estimate; then numbers and options to the user.
   - For the travel-demand model: M completed datasets (discrete sector, informality and NSE per person or dwelling)
     that the model runs M times, so the imputation's uncertainty reaches its outputs; the NSE chain already writes
     its draws (`completions.parquet`), `sector_informality` would need `draws` next to `enumerate`.
   - Congeniality: where an analysis relates an imputed variable to travel behaviour (trip rates by sector or NSE,
     say), the imputation models must see that behaviour, or the association comes out attenuated; `run.compare`
     measures candidate features (e.g. trip counts or modes for the education and NSE tasks).
6. **Tests.** The suite runs 84% of `eodgdl.impute`'s and `eodgdl.giro`'s statements (2026-09-29: `uv run --with
   coverage python -m coverage run --source=src/eodgdl/impute,src/eodgdl/giro -m pytest`; `uv run --with pytest-cov`
   fails on a second numpy import). Not run by any test: `diagnostics.py`
   (212 statements, all of `eodgdl impute evaluate`: only the reports run it), most of `check_bundle`'s refusals
   (other classes, arms, features, auxiliary predictors; `bundle.py` 88–101), the real-data paths of the sources
   (`sources/eod.py` 62%, `sources/enigh.py` 36%) and AMAI's file readers (`derive/amai.py` 56–81), and the CLI's
   `impute` commands. The diagnostics and the bundle guard matter most: a synthetic chain like `tests/test_chain.py`'s
   can carry an `evaluation:` section.
7. **Code and docs.** `run.retrain` (144 lines) holds most of the logic; ruff's B023 in `features.build_frame` (a
   closure over loop variables, called at once: correct but fragile) and in `giro/features.py`; the legacy giro bundle
   path (`bundle.from_legacy_giro`, `legacy_view`, `giro/_ml.py`) could go in a breaking release; the installed bundles
   in `data/` carry pre-be127dc held-out metrics in their metadata (reinstall at the next release). This handoff holds
   850 lines of history, and `CLAUDE.md`'s impute section is 92 dense lines: propose a design document
   (`docs/impute.md`: what the engine does and why) and keep only the state and pointers here.

Still waiting on the user from session 6: push `impute-chains`, merge into `main`, whether to release, and option B
for giro. The deliberately deferred work under "Not done" below is separate and was not asked for.

### Session 7 (2026-09-29): findings, in the order of the focus list

**Step 0 (2bf14e0).** `scripts/impute/parity.py ROOT_A ROOT_B [--models]` and `scripts/impute/rerun.sh ROOT [TASK ...]`.
Checked on session 6's roots (`output/impute_70eab10` → `output/impute`): 144 of 246 parquets identical, the other 102
all held-out tables, the JSON differences all held-out entries, the 19 fitted models identical.

**Focus 1: copied worker records in giro's CV folds.** Measured with scratch scripts (the twin pairs rebuilt from the
shipped trips by the loader's own rule reproduce `hab.diario_repetido` exactly: 1,238 pairs, 1,793 persons, 797 twin
groups; linking households through them merges 1,501 dwellings into 664 components, the largest of 10).

- *The copies are person records* (a defect in today's evaluation, small). 548 of giro's 26,913 workers are flagged,
  333 of them with a reported giro. In the 281 twin pairs of two workers, the giro agrees in 98.8% of the 164 pairs
  where both report it (chance 30.6%), the occupation 98.8%, education 93%, marital status and relationship 94.5%,
  sex 94%, but the age 12% and the household size 24%. Only 8 pairs have one giro reported and the other not, so a
  twin is almost never a donor for an imputed worker.
- *The model memorizes them.* Out-of-fold log loss of the 333 flagged known workers (GB as selected, 10 seeds of
  5-fold CV over every known worker): 0.726 under household groups, where the twin sits in the training folds 76% of
  the time, against 0.903 when twins share a fold (households linked by a copied diary kept together). Household
  grouping makes the CV optimistic by 0.0067 (with education; 0.0057 without; RandomForest 0.0030): 0.0034 from the
  flagged rows, 0.0014 from the unflagged members of linked households (copies the diary rule cannot see, e.g.
  workers without trips), the rest within noise. The published held-out split has 50 of its 3,485 workers (1.5% of
  the weight) with a twin in training: the published held-out log loss (0.8934 with education) is optimistic by some
  0.003–0.005.
- *The leak does not drive the selection* (10 seeds over every known worker: GB beats RF by 0.013–0.024 in both arms
  under both groupings; the one-SE rule picks RF in none). *But the selection is fragile*: on the retrain's own
  training split (13.9k rows) RF trails GB by 0.0086 on average and falls within one SE in 2 of 10 seeds with
  household groups, 1 of 10 with twin groups — and that one is seed 42. The paired SE from five folds (four degrees of
  freedom) ranges from 0.001 to 0.008 across seeds. So both ways of grouping twins tried flip the without-education arm
  to RandomForest: a fresh split over the components (RF − GB 0.008083 against an SE of 0.008119) and today's split
  with each linked household moved to its component's first household (0.0035 against 0.0068; 631 rows change fold).
  The without-education arm scores 4,522 of the 9,484 imputed workers: RF moves the imputed shares by up to 0.54 pp
  (servicio 35.69% → 35.15%, gobierno 5.86% → 6.17%, comercio 24.68% → 24.96%), 284 arg-max labels, 0.50 at most on
  one worker's probabilities. The with-education arm keeps GB and its refit is unchanged.
- *The held-out numbers move by split noise.* One fold's log loss has an SD of 0.018 across random splits; the fresh
  twin-grouped split gives 0.9171 / 0.9375 (with / without education) against the published 0.8934 / 0.9161, the
  repaired one 0.8976 / 0.9305.
- *How the copies train barely matters* (held-out on the twin-grouped split, the published configurations): hybrid log
  loss 0.9174 (both copies, as today), 0.9204 (half weight each), 0.9166 (one kept); imputed shares within 0.08 pp.
- *The dwelling tasks are barely touched*: 47 twin pairs are both the heads `eod.dwellings` picks (37 with the
  education reported, agreeing 97%); 18 of educacion_jefe's 2,818 held-out dwellings are linked to a training dwelling
  through a copied head.

Options (a decision for the user; the group column would come from the loader, which records each flagged person's
twin group in `hab`, with `eod.workers` and `eod.dwellings` grouping the households it links): (A) twin-linked groups
with a fresh split; (B) today's split repaired; (C) keep household groups and document the leak. A and B both move the
without-education arm to RandomForest under today's selection, so the choice goes with focus 2's question of how
selection compares configurations (the scoring path, and repeated CV against the five-fold SE's noise).

**Focus 2: missing values across training, selection, evaluation and scoring.**

- *Selecting on the predictor that scores changes one selection* (a defect in today's outputs: educacion_jefe's
  selection optimizes a predictor the bundle never uses). Every task's grid re-run with each CV fold scored as the
  bundle scores (fold training shares; fold auxiliary models fitted on the fold's training rows and the rows whose
  target is unknown; staged boosting scored through the expansion), on today's splits; the `predict_proba` fold losses
  reproduce the published grid to the bit. educacion_jefe (1,045–1,100 validation rows marginalized per fold, nearly
  all over income): GradientBoosting 1.3902 → 1.5013, LogisticRegression (C 0.1) 1.3942 → 1.4562, RandomForest → 1.4944,
  so LR is selected by 0.038 (SE 0.003). Retrained with LR: held-out as the bundle scores 1.4204 (accuracy 44.3%)
  against GB's published 1.4696 (41.0%); imputed shares licenciatura 16.7% → 19.9%, preparatoria 29.8% → 28.4%,
  secundaria 27.5% → 26.4%, posgrado 3.4% → 2.6%; the NSE chain moves by at most 0.16 pp before calibration and 0.03 pp
  after. Unchanged: giro (GB both arms; CV 0.9099 → 0.9113 and 0.9260 → 0.9276, 450–550 marginalized rows per fold),
  informality (GB both arms; +0.001), amai_banos, amai_dormitorios and amai_trabajadores (no validation row
  marginalized, identical).
- *Two income arms for educacion_jefe* (a decision; measured by a full retrain, today's selection): `con_ingreso` GB as
  today, `sin_ingreso` LR (C 0.1). Hybrid held-out 1.3885, the best of the three, but the arm without income is trained
  on the refusers too (a third of the training heads) and carries the refusal → licenciatura pattern session 6 chose not
  to carry: imputed licenciatura 29.8% (16.7% today, 19.9% with LR), posgrado 1.6%. The held-out set rewards it because
  its refusers carry the same pattern, so it cannot arbitrate (focus 3).
- *The five-fold SE is noisy* (focus 1): on giro's training split, the one-SE rule with the SE pooled over three
  repeats of the five-fold CV (the same scale as today's SE, estimated with 12 degrees of freedom instead of 4) never
  puts RandomForest within one SE of GradientBoosting for the without-education arm (0 of 120 sets of three seeds,
  either grouping), where a single run does in 1–2 of 10 seeds.
- Option B for giro (the missing destination and mode as learned categories) is still the user's.

**Decided (user, 2026-09-29) and done.** (1) The CV groups stay the households; the copied records are documented:
`eod.repeated_diary_pairs` returns the pairs `flag_repeated_diaries` reads (exported; the flag unchanged, pinned at
1,238 pairs in `tests/test_eod.py`), and `reports/imputation_giro.qmd` #sec-selection counts at render time what the
copies do to the split (333 flagged known workers, giro agreeing 98.8% in 164 pairs, 50 held-out workers with their copy
in training) and cites the measured optimism. (2) Selection on the scoring path, for every task (41cd5a7):
`select.tune(expand=)`, `run.fold_expander`, the robustness CV and `run.compare`. (3) One five-fold run: asked whether
ROPE would be steadier, measured that it would not (the Bayesian correlated t-test on the same folds, "keep the simpler
family unless the best is credibly better by more than r", keeps RandomForest in 90% of single runs at r = 0 and in
100% of three-run sets at r ≥ 0.005: stable only where the width decides, and every tie goes to the family order LR <
RF < GB, which says little about complexity); the user kept the single run. (4) One income arm.
The rerun (`scripts/impute/rerun.sh`, then `parity.py output/impute_7f783b3 output/impute --models`) moved exactly
what was intended: educacion_jefe's selection (LogisticRegression C 0.1; CV as scored 1.4562; held-out 1.4204, accuracy
44.3%) and with it its scores, tables and the `nse` chain (the calibrated shares by at most 0.03 pp); giro's and
informality's CV tables and the summaries' CV losses (giro 0.9113 / 0.9276, informality 0.4386 / 0.4588), their fitted
models predicting identically and their scores and `sector_informality` unchanged; the three ENIGH bundles byte-identical.
`output/impute/` holds the new run; the one before is `output/impute_7f783b3/` (with `output/figures_7f783b3/` and the
four reports' HTML in `output/reports_7f783b3/`). The four impute reports were re-rendered.

**Focus 3: the labels and features the EOD tasks learn from.**

- *The income refusers' education* (decided: keep, option A). Training the education task without the refusers'
  answers (34% of the known heads' weight) lowers the held-out loss on the reporters (1.4744 → 1.4645, half weight
  1.4659) and the imputed licenciatura (19.9% → 14.1%); the NSE moves by at most 0.26 pp before calibration and 0.03 pp
  after. Imputing the refusers' own education too takes the survey's licenciatura from 27.8% to 18.0% and its posgrado
  from 4.1% to 6.6%; the calibrated NSE by at most 0.5 pp. ENIGH 2022's heads in the nine municipalities (1,342
  households) hold a licenciatura in 22.1% (19.9–24.4%), between the EOD's reporters (15.4%) and refusers (56.7%), a
  posgrado in 2.5% (the reporters 5.6%, the refusers 2.1%) and a primaria in 24.1% (the EOD's known heads 13.4%): the EOD
  departs from ENIGH at both ends, so refusing likely goes with more schooling as well as with inflated answers, and the
  benchmark supports neither correction. `reports/imputation_nse.qmd` #sec-education-benchmark computes it at render
  time.
- *The harmonization maps against the questionnaires* (checked; a cleanup). Every ENOE code the maps read (sex,
  position, schooling, marital status, relationship `par_c` by hundreds, SCIAN sector, the place-of-work questions
  p4–p4h, `emp_ppal`, `tue2`) and every ENIGH code (`educa_jefe`, `sexo_jefe`, `tam_loc`, `conex_inte`, `tenencia`,
  `trabajo_mp`) matches mxcensus's labels, and every AMAI point and cut matches AMAI's 2022 rule (its questionnaire:
  education 0/6/11/12/18/23/27/36/59/85, bathrooms 0/24/47, cars 0/22/43, internet 32, workers 0/15/31/46/61, bedrooms
  0/8/16/24/32; levels at 48/95/116/141/168/202). mxcensus's labels of `par_c` 406–411 look shifted (1,052
  "great-great-grandparents"): its catalog's problem, not the map's (the map reads the hundreds).
- *Level, not completion* (decided: document). The EOD records the level of schooling; AMAI's points assume it
  completed. With ENIGH's incomplete shares among Jalisco's city heads (primaria 34.5%, secundaria 11.9%, preparatoria
  20.3%, licenciatura 18.3%) the uncalibrated NSE moves by up to 1.8 pp (C +1.76, C+ −1.69, A/B −0.92), the calibrated by
  0.28 pp. `reports/imputation_nse.qmd` #sec-completion computes the bound on the metro heads at render time.
- *The sector map mixed agriculture into government* (a defect in the by-sector outputs; decided: fixed). In the metro
  area ENOE's `gobierno_otro_agricultura` was 77% government (5.7% informal) and 23% agriculture (63.5%); the EOD's
  *Gobierno/sector público* workers took 13.1% expected informality. With agriculture (SCIAN 1) grouped with
  manufacturing and the level renamed `gobierno`: 9.8%, the overall rate 35.92% → 35.93%, ENOE's held-out +0.002.
- *The April capture change reaches giro's features* (decided: the interview month as a feature). From April a work
  trip's destination is an office half as often at every trip position (15.8% in March, 6.3% in April; within the 14
  zones surveyed on both sides 16.2% → 8.4%), and the giro goes unreported by 6.4% of January's workers, 28.8%
  of February's, 42.6% of March's and 40.8% of April's (fieldwork moved from Guadalajara in January to Zapopan in March
  and the outer municipalities in April). `mes_entrevista` (the `eod.workers` source) in both arms: `run.compare`, paired
  over seeds 42, 7 and 11, −0.0013 (with education) and −0.0020 (without), every seed. `reports/imputation_giro.qmd`
  #sec-month.

Both spec changes (giro's features, informality's sector levels) left the installed bundles unusable (`check_bundle`
refused them). Retrained (`scripts/impute/rerun.sh`, giro and informality into a copy of the outputs): giro keeps
GradientBoosting in both arms, CV 0.9113 → 0.9101 and 0.9276 → 0.9251, held-out 0.8934 → 0.8963 and 0.9161 → 0.9146
(paired on the same held-out rows, the combined arms +0.0026, 95% household bootstrap −0.0020 to +0.0078: noise, the CV
the larger sample), imputed shares within 0.14 pp; informality keeps its selections, held-out 0.4161 → 0.4179 and 0.4368
→ 0.4385, the government sector's expected informality 13.1% → 9.8%, the overall rate unchanged; parity: nothing else
moved (the nse chain, educacion_jefe and the ENIGH tasks identical). **Installed locally** (user, 2026-09-29): the
retrained giro, informality and educacion_jefe bundles in `data/` with their `registry.txt` sha256 (giro 0bd57e15…,
informality 5ba4ea56…, educacion_jefe b39d55ab…), and `tests/data/giro_parity.parquet` regenerated (its 384 workers
gained `mes_entrevista`; expected probabilities up to 0.11 from v0.4.0's bundle); scoring giro and both chains from
`data/` equals the retrains exactly. The mirror still serves v0.4.0 until a release (version, `REF`, tag and push are
the user's); the three ENIGH bundles in `data/` are v0.3.0's (the same predictions, their metadata's held-out metrics
older than be127dc). `output/impute/` holds this run; `output/impute_41cd5a7/` the one after focus 2 alone.

**Focus 5: bootstrap bundles, measured** (scratch scripts; not in the engine). 20 refits per task of each arm's
selected configuration and its auxiliary models on a cluster bootstrap of the training rows (the source's group,
weights kept): giro 100 s, informality 36 s, the ENIGH tasks and the education seconds. Completed datasets draw each
model before drawing its predictions (giro and informality: 20; NSE: 20 × 3 draws); within variance by dwelling-cluster
linearization (no strata), Rubin's rules. Point estimates move by at most 0.4 pp. Total standard errors, single fit
(today's draws) → bootstrap models: giro's shares 0.45 → 0.47–0.53 pp (fraction of missing information 0.20–0.37);
the informality rate 0.51 → 0.77 pp (×1.5, FMI 0.73: the ENOE model's uncertainty dominates; the chain enumerates and
publishes no interval at all); informality by sector ×1.24–1.55 (FMI 0.52–0.75); the NSE before calibration ×0.99–1.10
(FMI ~0.4); after calibration ×1.00 (FMI 0.02–0.05: the calibration pins each AGEB's mix). Design for the engine as
the brief has it (bootstrap bundles a retrain product under `<root>/<task>/bootstrap/`, a chain option `uncertainty:
bootstrap` off by default, Rubin's rules in the diagnostics); pending the user's decision.

### Session 6 (2026-09-28): session 5's pending list

The user asked for the numbers and the options of items 1 and 2 and the fixes of items 3–8; later that night they
delegated items 1 and 2 ("finish all pending items without my input"), keeping pushing, merging and releasing for
themselves. Session 6 chose option A for item 1 and option (a) for item 2.

1. **Missing values: held-out metrics against scoring. Measured, then option A: the held-out evaluation scores as the
   bundles do** (be127dc). First measured with a scratch script: each arm's selected configuration refitted on
   the train split exactly as `select.tune` does (the published `test_metrics__<arm>` and `__hybrid` reproduced
   exactly), the held-out rows scored through `arms.predict_arms` with the train split's training shares and auxiliary
   models fitted on every row outside the held-out fold (as the bundle's are fitted on every row; on the known-target
   train rows instead, the numbers move by < 0.002). Weighted log loss (accuracy):

   | | rows the scoring path marginalizes | `predict_proba` (published) | scoring path | difference, household bootstrap 95% |
   |---|---|---|---|---|
   | giro, with education | 633 (17.3%): 561 without a work trip, 91 without education (21 both) | 0.8863 (63.5%) | 0.8934 (62.6%) | +0.0071 (+0.0015, +0.0134) |
   | giro, without education | 563 (16.3%): 561 without a work trip | 0.9089 (62.0%) | 0.9161 (61.3%) | +0.0072 (+0.0014, +0.0138) |
   | giro, hybrid (as it scores) | 563 (16.3%) | 0.8873 (63.5%) | 0.8940 (62.6%) | +0.0067 (+0.0009, +0.0132) |
   | educacion_jefe | 1,384 (42.7%), 1,376 over income | 1.3531 (47.7%) | 1.4696 (41.1%) | +0.1165 (+0.0968, +0.1360) |

   On the rows the scoring path marginalizes the gap is +0.04 for giro (0.986 → 1.027, the marginal baseline 1.246)
   and +0.27 for educacion_jefe (1.182 → 1.455, the marginal baseline 1.505). giro: on the held-out workers without a
   work trip (the arm without education) the observed shares are comercio 44.0%, servicio 31.4%; `predict_proba` gives
   43.3% / 34.3%, the scoring path 35.4% / 42.1% (it averages the destination over P(destination | x), which knows
   nothing of the missing trip). Among the imputed workers 4.3% lack a work trip, so scoring with the missing label kept
   for destination and mode moves the imputed shares by at most 0.3 pp (comercio 24.68% → 24.88%). educacion_jefe: on
   the held-out dwellings the scoring path marginalizes (all but 8 without income) the observed licenciatura share is
   49.7%; `predict_proba` 47.5%, the scoring path 24.7%. Among the 3,819 imputed dwellings 97% lack income (96% "No
   quiso responder"), so keeping the missing label would move the imputed education a lot: licenciatura 16.7% → 31.9%,
   preparatoria 29.8% → 19.8%, posgrado 3.4% → 1.3%, the arg-max of 39% of them, the expected AMAI points 28.8 → 32.5.
   An auxiliary income model P(income | x) instead of the training shares barely helps (held-out 1.4696 → 1.4667;
   imputed shares within 0.2 pp).

   **The income refusal looks like a fieldwork pattern**, so the learned "refused → licenciatura" association should
   not be carried to the dwellings whose head's education is unknown. Among the heads whose education is known,
   refusal rises from 15% (January) to 28%, 49% and 51% (April). The heads who refused hold a licenciatura in 57% of
   cases against 16% of those who reported an income, yet a posgrado less often (2.1% against 5.6%), and both gaps hold
   at every AMAI level of the AGEB (licenciatura 50% against 10% and posgrado 0.6% against 3.9% where AMAI's mean level
   of the AGEB is below 2.5, D+ and below; 70% against 27% and 5.4% against 10.9% from 4, C and up) and in every
   municipality: one answer inflated rather than a better-educated group. AMAI's NSE of their AGEBs differs little
   (mean level 3.36 for the refusers, 3.16 for the reporters), and the unknown-education dwellings live in poorer AGEBs
   (3.03). The survey records no interviewer, so the pattern cannot be traced further. It also touches the known
   heads: a third of the education task's training rows are refusers whose reported education may carry it.
   `reports/imputation_nse.qmd` #sec-income computes all of it at render time.

   The options as put to the user: (A) evaluate through the scoring path, so the published held-out metrics describe
   the predictor that scores (scores unchanged; the held-out tables and the reports move); (B) let scoring keep the
   missing label as a category for chosen features (a per-feature spec option; scores change, so a retrain, the
   bundles reinstalled and a release): defensible for giro's destination and mode, not for educacion_jefe's income;
   (C) document only. **Chosen: A** (B for giro stays a decision for the user: "Pending" above). `run.heldout` scores
   a held-out row through `arms.score_arm`: the train split's model carries its training shares (attached after
   `select.tune`), and each arm's auxiliary models are refitted on every row outside the held-out fold (the training
   fold and the rows whose target is unknown, as the bundle's are fitted on every row). Every held-out table goes that
   way: the arms' metrics, intervals, class tables, calibration and the isotonic diagnostic (its maps still learned
   from out-of-fold `predict_proba`, scored as a shipped isotonic arm would be), the `__hybrid` tables, the label's
   components, the target profile and the population comparison (training shares only: no task with populations has
   auxiliary models). Model selection is unchanged: the grouped CV compares `predict_proba` on its validation folds,
   where the missing label is the category the model learned (so the selections, and with them the bundles and the
   scores, do not move). The reports say how held-out rows are scored; `reports/imputation_nse.qmd` gains #sec-income.
   `tests/test_impute.py::test_heldout_rows_are_predicted_as_scoring_predicts_them`.

   Parity (the recipe; a second root against the first, which equals `output/impute/` at 70eab10): of 246 parquets 144
   are identical, 26 differ by less than 1e-12 and 76 are held-out tables that moved; no other table, score, scenario
   or chain output moved, every fitted model predicts identically, and the summaries differ only in the arms' held-out
   entries. The moves: giro's arms 0.8863 → 0.8934 and 0.9089 → 0.9161, the hybrid 0.8873 → 0.8940 (weighted
   accuracy 63.5% → 62.6%); educacion_jefe 1.3531 → 1.4696 (47.7% → 41.1%), both equal to the scratch measurement's
   to 1e-16; informality 0.4161 → 0.4161 and 0.4366 → 0.4368, its components' log losses +0.005 (the few ENOE rows
   without a place of work, now averaged over P(place | x)); the ENIGH tasks by rounding only (relative 1e-15).
   `output/impute/` now holds this run; the one it replaced is `output/impute_70eab10/` (with `output/figures_70eab10/`
   and the four impute reports' HTML in `output/reports_70eab10/`). The four impute reports were re-rendered from it.
2. **`chain._aggregate` stops on a missing derived level: option (a)** (eee00ad). The options as put: (a) keep
   failing, with an error that names the derive step, the column and the rows; (b) carry a declared missing level
   through `nse` and `nse_calibrado` (a new all-zero `prob_nse_<missing>` column today; the diagnostics and reports
   would have to leave it out of the AMAI comparison); (c) impute the answer (tasks for internet and cars trained on
   ENIGH, like the bathrooms). No EOD dwelling lacks an internet, car or worker answer (17,901 checked), so (a) keeps
   today's outputs and makes a future missing answer a visible decision: `chain._check_levels` stops the chain where a
   categorical derived value is missing or outside the step's declared levels, naming the chain, the step, the column,
   the values, the completions and up to five rows by their keys (the assertion in `_aggregate` is now an internal
   invariant). `test_a_derived_value_outside_its_levels_names_the_rows`.
3. **`expected` propagation and a task without `target.scores`** (c5f905f, as items 4–8): `parse_chain` refuses it
   (every task `_needed` returns must declare scores; the per-`uses` check it replaces was a subset), and `_advance` no
   longer writes NA. `tests/test_chain.py::test_chain_validation`, and a derive step reading the expected score in
   `test_expected_propagation_plugs_in_the_upstream_score`.
4. **The shift scenario ships as the bundle does**: `run.shipped_arm` (a refit, isotonic-wrapped under
   `evaluation.isotonic.ship`) builds both the bundle's arms and the shift scenario's. The `isotonic: {ship: True}` part
   of `test_retrain_writes_a_bundle_that_scores_the_same` checks the shift bundle's arms.
5. **An enumerated or drawn step's `<p>_model_used` and `<p>_marginalized_features`** join the row's scenarios (the
   value they share, else the union of the names; `chain._joined_over_scenarios`), no longer the first scenario's.
   `test_arms_and_marginalized_features_are_joined_over_a_rows_scenarios` (a two-arm task whose upstream fills a level
   without training support in one scenario and the missing label in another).
6. **The delta scenario reaches its target**: where a scaled probability would pass one it is held at one and the
   factor solved (`evaluate.clipped_scale_factor`: the largest probabilities at one, as many as the target needs; a
   target the rows with a positive probability cannot hold raises). Where nothing passes one the factor is the plain
   ratio, computed as before, so giro's scenario (factor 0.69) is unchanged. The synthetic task's delta scenario in the
   retrain test does take the new path. `test_delta_adjustment_reaches_its_target_where_probabilities_reach_one`.
7. **The feature cache keys a builder's columns on the task's classes only when the builder reads them**
   (`register_builder(..., reads_classes=True)`: `giro.destination` alone), so the three AMAI tasks share ENIGH's
   harmonized columns. Every cache key moved once (rebuilt on first use); the frames are the same.
   `test_feature_cache_hits_misses_and_keys`.
8. **ruff**: `pyproject.toml`'s `ignore` moved to `[tool.ruff.lint]` (the deprecation warning is gone; ruff's findings
   are the same 12 as before, none in session 6's lines).

`uv run pytest`: 163 passed (~125 s). Items 3–8 changed no output: c5f905f's parity run (the recipe of "After the
release") found 256 parquet and JSON files equal to `output/impute/`'s (the chains' provenance aside, which records the
bundles' eodgdl version: 0.4.0 against the 0.3.0 of the bundles the earlier run read), and every fitted model predicting
identically; items 1 and 2 moved only the held-out tables (item 1).

### Session 5's review notes, as handed to session 6

The list as session 5 wrote it (the user asked, 2026-09-28, that session 4's review notes go to a new session); session
6 above resolved it. None was a defect in the outputs; ordered by how much they matter, with what session 5 measured.

1. **Missing values: held-out metrics against scoring (a modelling decision for the user).** Every arm is trained
   with the missing label (`no_especificado`) as a category. The held-out evaluation (`run.heldout`, the model's
   `predict_proba`) uses that category; scoring (`arms.score_arm` → `marginalize.predict_proba_marginalizing`) never
   treats the missing label as supported and averages such a row over the observed levels, by the auxiliary model's
   P(level | x) where the task has one, else by the training shares. So the published held-out metrics describe
   another predictor than the one that scores, and scoring never uses the "missing" category the model learned.
   Weighted share of the held-out rows with a missing categorical feature (session 5):
   - giro: 17.6% (arm with education) / 16.3% (without). `destino_trabajo` and `modo_trabajo` are missing for the
     16.2% without a work trip (the destination averaged by its auxiliary model, the mode by training shares);
     `escolaridad` 1.6%.
   - educacion_jefe: 42.7%, nearly all `ingreso_hogar` (42.1%; no auxiliary model, so training shares), where
     non-response may well be informative.
   - informality 0.7%; the ENIGH tasks 0%.

   Steps: (a) measure each arm's held-out metrics through the scoring path (`arms.predict_arms` with the train-split
   model, its training shares and auxiliary models fitted on the train split) against today's `predict_proba`, for
   giro and educacion_jefe; (b) take the numbers to the user, with the choices: evaluate through the scoring path
   (the metrics follow the shipped predictor; scores unchanged), let scoring keep the missing label where it is
   informative (e.g. a per-feature spec option; scores change, so a retrain and a release), or document only.
2. **`chain._aggregate` stops on a missing derived level** (latent; the behaviour is the user's call).
   `derive.amai.amai_points` gives NaN where the dwelling's internet, cars or workers answer is missing; `nse_nivel`
   then gives NA and `_aggregate` fails ("derived values outside the declared levels"). No EOD dwelling lacks one
   today. Options: keep failing but name the rows and the column; carry a missing level through `nse` and
   `nse_calibrado` (the calibration already leaves NA points uncalibrated); or impute the answer.
3. **`expected` propagation and a task without `target.scores`** (latent, small): `chain._advance` writes NA for such
   a task, which only a derive step would read. Make `parse_chain` refuse it (with `propagation: expected`, every task
   `_needed` returns declares `target.scores`), with a test in `tests/test_chain.py`.
4. **The shift scenario ignores `evaluation.isotonic.ship`** (latent, small): `run.retrain`'s `shift_arms` refits the
   arms uncalibrated even when the bundle ships isotonic arms (off in every task). Wrap those refits in
   `evaluate.fit_isotonic` as the shipped arms are; extend the `isotonic: {ship: True}` part of
   `tests/test_impute.py::test_retrain_writes_a_bundle_that_scores_the_same`.
5. **An enumerated step's other outputs come from each row's first scenario** (`chain._conditionals`, `firsts`):
   `<p>_model_used` and `<p>_marginalized_features` of a task scored under several upstream scenarios are the first
   scenario's; the same in every scenario today. A cheap guard: assert they agree across a row's scenarios, or keep
   their union.
6. **`adjust_imputed_share` clips at one** (§6.4; latent): the delta scenario misses its target share where a scaled
   probability passes one. giro, the only task with `evaluation.delta` (`gobierno`), scales down (factor 0.69), and no
   row reaches the clip. A version that never clips (e.g. scaling the class's odds, the factor solved for the target)
   changes the scenario only where the clip binds.
7. **The feature cache stores the ENIGH frame three times** (disk only): `features.build_frame` keys every builder's
   columns on `spec.classes`, so `amai_banos`, `amai_dormitorios` and `amai_trabajadores` cache the same
   `harmonize.enigh` columns under three keys. Keying on the classes only for builders that read them needs builders
   to declare it; low value.
8. **ruff** (not impute): move `pyproject.toml`'s top-level `[tool.ruff] ignore` to `[tool.ruff.lint]`.

The deliberately deferred work under "Not done" below is separate and was not asked for.

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
4. **informal-jobs-model's figures in `reports/imputation_figures.qmd`** (the user's choice: those of its notebooks 03,
   05, 06 and 08 and the presentation versions, not the old notebook-04 sector charts, which the giro report has).
   Nine figures under informal-jobs-model's names, in the style of eodgdl's reports with its Spanish labels, shown in
   the report and written as PDF + 600-dpi PNG to `output/figures/` (`$EODGDL_FIGURES_DIR`). They read the retrains and
   the chains: ENOE is the informality task's training frame in the metro municipalities (38,933 rows, 39.35%
   informal, as informal-jobs-model's benchmark); "OD muestreo" is one seeded draw per worker, "OD duro" the arg-max,
   a worker's hard sector its most probable one; an AGEB's EOD level on the NSE maps is the level with the largest
   weighted probability mass, "before calibration" the draws' distribution (not informal-jobs-model's plug-in); the
   polygons are the Marco Geoestadístico 2020's urban AGEBs (mxcensus). The ramp for the seven NSE levels is the dataviz
   skill's blue, steps 100–700: no seven steps from 250 up clear its ordinal gate (adjacent ΔL ≥ 0.06), so the lightest
   step recedes as a choropleth's may, with the legend and a table as relief. For the calibration figure the retrain now
   writes each arm's `isotonic_calibration` / `isotonic_reliability` and, for a task with several arms,
   `test_metrics__hybrid`, `calibration__hybrid` and `reliability__hybrid` (each held-out row predicted by the arm that
   would score it, informal-jobs-model's "hybrid"). The evaluators take a probability matrix in sorted class order (the
   model's, as sklearn's `log_loss` reads it); a first version combined the arms in the task's order and got the log
   loss wrong (1.77 where the arms give 0.416 and 0.437), caught before the commit and pinned by `test_hybrid_heldout_predicts_each_row_with_its_arm`
   with an unsorted task. All six tasks were retrained again: every earlier table is unchanged.
5. **informal-jobs-model archived** (user): its README points to eodgdl (ca2c67c, pushed) and its GitHub repository is
   archived (read-only, history kept; `gh repo unarchive` reverses it). Its `outputs/` (gitignored: figures, bundles,
   tables) stays on disk; its tracked `data/GUADALAJARA_SHP/` is replaced by the Marco Geoestadístico.

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
| informal-jobs-model's figures (`reports/imputation_figures.qmd`), the isotonic and combined held-out tables (session 5) | ce1fedb | merged into `main` at b27a0e6 |
| session 5's review notes (session 6): 3–8, 2, 1; this handoff | c5f905f, eee00ad, be127dc, then the handoff | `impute-chains`, local (with 70eab10: not pushed) |

`impute-chains` branches from `main` at 00bf5f5 and holds phases 4–7 and sessions 4–6, with the `v0.3.0` and `v0.4.0`
tags on it. It is merged into `main` (`--no-ff`) up to ce1fedb, at b27a0e6; 70eab10 and session 6's commits are local.
`uv run pytest`: 163 passed, ~125 s.

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

- `output/impute_70eab10/`, `output/figures_70eab10/` and `output/reports_70eab10/` (session 6) keep the outputs,
  figures and impute reports' HTML from before item 1 moved the held-out tables, for a comparison; delete them when
  done. The four impute reports and `output/figures/` were re-rendered from the new `output/impute/`.
- A background run that pipes `eodgdl impute retrain` through `grep` exits 0 even when the retrain fails (zsh without
  `setopt pipefail`): check that every task wrote its bundle, as session 6's runs did.
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
