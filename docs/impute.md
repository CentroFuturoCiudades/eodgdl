# `eodgdl.impute`: design

What the imputation engine does and why. The state of the work, what is pending and the history of the decisions are
in `docs/handoff_impute_engine.md`; the evidence for each modelling choice is in the four reports under `reports/`.

## Purpose

The EOD 2023 leaves some variables a travel-demand model needs unreported (a worker's giro), or never asks them (whether
a job is formal, the dwelling's bathrooms and bedrooms), or asks them of only some rows (the head of household's
education). `eodgdl.impute` fills them with **categorical probabilities**, never a hard label: every imputed row keeps a
probability for each class, and downstream users collapse, relabel or draw from them.

Two kinds of model live side by side:

- **within the survey**: the target is observed for some EOD rows and imputed for the others (`giro`,
  `educacion_jefe`);
- **data fusion**: the target is observed in another survey and imputed for every EOD row (`informality` from INEGI's
  ENOE; the AMAI components `amai_banos`, `amai_dormitorios` and the diagnostic `amai_trabajadores` from ENIGH). Both
  surveys are first **harmonized** to common levels, and the imputation assumes the target is independent of the survey
  given the shared covariates (conditional independence): nothing in the EOD can test it.

Imputations are chained where one depends on another: `sector_informality` (the giro's sector, then informality given
it) and `nse` (education, bathrooms given it, bedrooms given both, then AMAI's points, level and calibration).

## Building blocks

A **task** is one YAML under `src/eodgdl/impute/tasks/`, parsed by `spec.parse_task` into a `TaskSpec`: its training
source and scoring source, target classes (and ordinal scores), features and their groups, declared levels, **arms**,
auxiliary models, candidate grid and evaluation settings. The engine knows no survey vocabulary: all of it comes from
the task.

A **source** (`sources/`) returns a task's rows as a plain frame, with its keys, weight and CV group declared in the
source's YAML (`eod.workers`, `eod.dwellings`, `enoe.workers`, `enigh.households`). **Feature builders** add columns
(`eod.work_trip`, `giro.destination`), and **harmonizations** are builders that map a source's raw codes to the common
levels of `harmonization/common.yaml`, failing on any code the map does not cover. Frames are cached as parquet under
the eodgdl cache directory with keys over every input that could change them (the code, the configuration, the data
files' sha256).

The **level contract** (`levels.py`) fixes every categorical feature's levels (declared, or the eodgdl schema's) plus
the missing label. A value outside the contract raises; a declared level with no training rows is marginalized at
scoring time.

## Scoring: what a bundle does with a row

`run.score_frame` sends each row whose target is unknown to the **first arm whose required features it has observed**
(giro: with education, else without; the AMAI tasks: with income, else without). The arm's model predicts
`P(class | x)`. Where a feature's value has no training support (a level the training rows never hold) or is missing,
the prediction is **averaged over the supported levels** (`marginalize.py`): with the training shares, or with a
row-specific `P(level | x)` from an **auxiliary model** where the task declares one (informality's place of work, for
the EOD workers without a work trip: ENOE knows no such state). All combinations are scored in one batched call.

A missing value is never read as a category, even where the model saw it as one in training: non-response carries
fieldwork patterns (the income refusals, `reports/imputation_nse.qmd` #sec-income) that should not steer an
imputation. What is a state rather than a missing answer gets a level of its own instead: a giro worker who made no
work trip on the survey day takes `sin_viaje` for the destination and mode (the builder's `no_trip`, the task's
`levels.extra`), learned from the reported workers who made none. The outputs are `prob_<prefix>_<class>` (rows sum to one; observed rows are one-hot), the arg-max as a
convenience, its probability, the arm used, the marginalized features, and for ordinal targets the expected score.

## Training: `run.retrain`

1. **Split**: the rows with an observed target are split by the source's CV group (households, or dwellings; ENOE's
   cross-quarter household), stratified by the target; one fold is held out.
2. **Selection** (`select.tune`): for each arm, a grid over logistic regression, random forest and gradient boosting,
   5-fold grouped CV on the training rows with the survey weights, **each validation fold scored as the bundle scores
   it** (`run.fold_expander`: the fold's training shares and auxiliary models fitted outside the fold). Within each
   family and then across families the simplest configuration whose paired fold difference to the best is within one
   standard error is kept. Scoring the folds as the bundle does matters where many rows are marginalized: it moved the
   education task from boosting to a logistic regression.
3. **Held-out evaluation**: the selected configuration refitted on the training rows and scored on the held-out fold
   exactly as the bundle would score it (`run.heldout`): metrics with household-bootstrap intervals, per-class tables,
   calibration, an isotonic recalibration (evaluated always, shipped only on request), each arm's rows and the arms
   combined as scoring combines them.
4. **Refit** on every row with an observed target, the auxiliary models on every row, and the **bundle** (below).
5. **Imputation tables and scenarios**: the known against the imputed rows' profiles, the arms used, the confidence;
   a covariate-shift scenario (the training rows reweighted to the imputed rows' profile, the arms refitted) and a
   delta scenario (one class's imputed share scaled to its observed one).

`run.compare` evaluates candidate feature sets cheaply: the published winner's configuration, the same folds scored the
same way, paired fold differences over several seeds.

## Bundles

A bundle (`bundle.py`, format `eodgdl.impute.bundle/2`) holds the fitted arms, auxiliary models, classes, the level
contract and metadata: hashes of the spec, of what decides scoring and of the level contract, the data versions, the
selection and held-out metrics, the scikit-learn and eodgdl versions. `check_bundle` **refuses** a bundle that could not
score the task as it stands: another scikit-learn, other levels, classes, arms, features or auxiliary predictors, or a
scoring or level-contract hash that differs. `load_bundle` reads an explicit file, else the one a retrain wrote under
`<root>/<task>/`, else the installed data file (`data/od_<task>_model.joblib`, fetched from the data mirror). A change
to a task's features or levels therefore requires retraining and reinstalling its bundle.

## Chains

A chain (`chains/<chain>.yaml`, `chain.py`) lists task steps, each optionally **using** earlier tasks as features, and
**derive** steps (deterministic functions of the values, `derive/`). A downstream task is trained on the upstream as
observed in its own training source (ENOE's sector, ENIGH's education), so retraining an upstream task does not
invalidate it. Every mode builds **completions**, one row per (row, scenario) or (row, draw) with a weight:

- `enumerate`: every combination of the upstream classes, weighted by their probabilities: the exact
  `P(y | x) = Σ_s P(s | x) P(y | x, s)` (`sector_informality`);
- `draws`: seeded multiple imputations, each value drawn given the draws before it; a step's marginal is the mean of
  its conditional probabilities over the draws, and derive steps run per draw (`nse`, 50 draws);
- `expected`: the upstream's expected score plugged in (a comparison only: it compresses the tails).

The NSE's derive steps sum AMAI's points (`derive/amai.py`, the AMAI 2022 rule: education, bathrooms, cars, internet,
workers, bedrooms), cut AMAI's levels, and **calibrate** within each AGEB against AMAI's NSE by AGEB: in each draw the
dwellings, ranked by points (weighted mid-rank, ties in a random order per draw), take the levels of AMAI's cumulative
distribution. The ordering is the model's; the mix is AMAI's.

## Uncertainty

Each task ships one fitted bundle, so draws from it carry the **predictive** uncertainty only. For **proper multiple
imputation**, `retrain --bootstrap B` refits each arm's selected configuration and its auxiliary models on B cluster
bootstraps of the training rows (`run.bootstrap_bundles`, under `<root>/<task>/bootstrap/`). A chain with
`uncertainty: bootstrap` (off by default: the published outputs are the single fit's) scores draw d with the d-th
bootstrap bundle of every task, a draw of the models and then of their predictions; `eodgdl impute score <chain>
--draws M --bootstrap` writes M such completed datasets for a model that needs values. The diagnostics combine each
estimate over completed datasets by **Rubin's rules** (`evaluate.rubin`): the within variance by cluster linearization
with the dwelling as the primary unit (`evaluate.cluster_shares`, no strata), plus `(1 + 1/M)` times the between
variance. The informality rate's uncertainty is mostly the ENOE model's; after the AGEB calibration the NSE's is
almost all sampling.

## Diagnostics and reports

`diagnostics.evaluate_chain` runs a chain's `evaluation:` section. For a task chain (`sector_informality`): the
training source's benchmark, the rate within each upstream level, the upstream task's retrain scenarios, bounds for the
marginalized feature, the components of the label, a direct standardization of the gap to the benchmark, raking, and
Rubin's rules. For a distribution chain (`nse`): the distribution under other propagations, the draws' Monte Carlo
error, agreement, AMAI's reference by AGEB, a check against a component the EOD does observe, and Rubin's rules.

The reports read what the retrains and evaluations wrote and train nothing; their structural claims are asserted at
render time. `reports/imputation_giro.qmd`, `imputation_informality.qmd`, `imputation_nse.qmd`, and
`imputation_figures.qmd` (informal-jobs-model's figures redrawn).

## Decisions that shape the outputs

| decision | evidence |
|---|---|
| Missing values are averaged over their observed levels, never read as a learned category | income refusal pattern: `imputation_nse.qmd` #sec-income |
| No work trip is a state: giro's destination and mode take `sin_viaje` | `tasks/giro.yaml` levels.extra; `imputation_giro.qmd` #sec-arms |
| Selection on CV folds scored as the bundle scores | handoff §0 session 7, focus 2 |
| CV groups: the household; copied worker records straddle folds, documented | `imputation_giro.qmd` #sec-selection |
| One-SE rule on one five-fold run (ROPE and repeated runs measured, not adopted) | handoff §0 session 7 |
| The income refusers' education kept as reported | `imputation_nse.qmd` #sec-education-benchmark |
| AMAI points assume a completed level (the EOD records the level) | `imputation_nse.qmd` #sec-completion |
| ENOE's agriculture grouped with manufacturing; the EOD's government meets government alone | `harmonization/enoe.yaml`; `imputation_informality.qmd` |
| The interview month is a giro feature (April's capture change) | `imputation_giro.qmd` #sec-month |
| ENIGH training population: cities of 100,000+ | `imputation_nse.qmd`, the populations table |
| NSE by 50 draws, calibration ties at random per draw | `imputation_nse.qmd` #sec-level, #sec-calibration |

## Working with it

```bash
uv run eodgdl impute retrain <task|chain> --data data [--bootstrap 50]   # -> output/impute/<task>/
uv run eodgdl impute score <task|chain> --data data [--retrained output/impute] [--draws M --bootstrap]
uv run eodgdl impute evaluate <chain> --data data                        # -> output/impute/<chain>/evaluation/
uv run eodgdl impute compare <task> --spec candidates.yaml --seeds 42 7 11
scripts/impute/rerun.sh ROOT [TASK ...]                                  # every output into ROOT
uv run python scripts/impute/parity.py ROOT_A ROOT_B [--models]          # what moved between two roots
```

Any change to the engine is proved with the last two: rerun into a new root and compare it with `output/impute/`
(every parquet exactly, the held-out tables apart, the summaries key by key, every fitted model by its predictions).

## Limits

- Data fusion rests on conditional independence given the harmonized covariates.
- The within-imputation variance ignores the survey's strata and the stages of its design
  (`reports/expansion_factors.qmd`).
- Auxiliary level models have fixed hyperparameters; SHAP, ROPE and a varying training population in `compare` are not
  implemented.
- The education task uses the dwelling's centrality, not its AGEB (1,700 levels).
