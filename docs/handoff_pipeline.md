# Handoff: the staged imputation pipeline

Session 9 (2026-09-30) ended with the user's decision to replace the two chains that feed the TASHA build (`labour`,
`nse`) with one **staged pipeline**: within-survey imputation, then data fusion, then derived values, drawn jointly
per completed dataset. This file is the brief for the session that builds it. Read `docs/impute.md` (the engine's
design) and `docs/handoff_impute_engine.md` (state, session 9's entries at the top) first; `CLAUDE.md` is local.

## Why

The chains are drawn independently, and they share a variable: the dwelling's workers aged 14+.

- `nse` takes it from the survey's answers (`sources/eod.py` `dwellings` → `trabajadores_14_n`; harmonized to
  `trabajadores_14` in `harmonization/eod_viviendas.yaml`). It enters the AMAI points (`chains/nse.yaml`, the
  `amai_puntos` step's `observed.trabajadores_14`: 15 points a worker) and is a feature of both ENIGH models
  (`tasks/amai_banos.yaml`, `amai_dormitorios.yaml`, group `household`).
- `labour` draws the employment of the 4,370 persons aged 16+ who did not answer (task `empleo`): about 1,480 per draw
  are workers, uncounted by `nse`.
- Measured (session 9, the 50 MI draws of both chains): 3,549 dwellings (20%) hold an unanswered member aged 14+;
  per draw about 1,315 dwellings would gain points and about 690 (636-751; 3.6% of the dwelling weight) would move up
  one uncalibrated NSE level. `IncomeClass` (TASHA) is that level, calibrated.

MICE was considered and rejected (the user agreed): the missing answers co-occur (the skipped socio-economic block), so
a fixed order is as good as iterating; arms beat imputing a predictor (giro: 0.8958 vs 0.9010); each task keeps its
selection, held-out evaluation and bootstrap. A multilabel / `sklearn.multioutput.ClassifierChain` fusion model was
also discussed: the bathrooms → bedrooms chain already is a classifier chain with joint draws, ENOE and ENIGH never
observe the same rows, and ClassifierChain plugs in predictions (our `expected` variant) instead of enumerating or
drawing. Not pursued; a 12-class bathrooms × bedrooms model on ENIGH is an optional, low-priority experiment.

## The design

Each completed dataset *d* is stage 1's draw *d*, stage 2 drawn given it, stage 3 computed from both:

| stage | assumption, and how it is checked | persons | dwellings |
|---|---|---|---|
| 1. within the survey | MAR given the observed covariates; EOD held-out households | `empleo` → `giro` on `eod.labour` (giro read where the person works, reported or drawn) | `educacion_jefe` on `eod.dwellings` |
| 1→2. aggregate | deterministic | | `trabajadores_14` = reported workers aged 14+ plus the drawn ones (`situacion_laboral == trabaja`, age ≥ 14), capped at "4 o más" as the harmonization does |
| 2. data fusion | conditional independence given the shared covariates; donor held-out rows, benchmarks | `informality` (ENOE) given giro's sector (`transform: giro_sector`) | `amai_banos` → `amai_dormitorios` (ENIGH) given the education points and the completed `trabajadores_14` |
| 3. derive | deterministic | | `amai_puntos` (with the completed worker count), `nse_nivel`, `nse_ageb_calibrado` |

No task is retrained: only what each is scored on changes. The draws keep `uncertainty: bootstrap` (draw *d* scored
with bootstrap bundle *d* mod B of every task) and the seeds stay reproducible.

## Engine work (the parts that are new)

1. **A pipeline spec and runner** (e.g. `impute/pipelines/<name>.yaml`, `pipeline.py`): ordered stages; each stage one
   or more chains at one row level (persons, dwellings), propagation `draws` only; stage *k* draw *d* conditions on
   stage *k−1* draw *d*. Reuse `chain.run_chain` per stage where possible: it already takes `upstream_outputs` and
   `specs`, and a step's `score_source` (session 9) lets a task score other rows.
2. **An aggregate step** from person completions to dwelling features, per draw (the only new operation): the
   completed `trabajadores_14_n` replaces the source's before the dwelling stage's harmonization reads it (or a
   derive-like step writes the harmonized `trabajadores_14` directly; keep the level contract: `"0".."4 o más"`).
   Decide where it lives so that `harmonize.eod_viviendas`'s cached columns are not silently stale: the cache keys
   cover the source frame, not per-draw inputs, so a per-draw feature must enter after the cache.
3. **Output**: completed datasets per level, e.g. `<root>/<pipeline>/persons.parquet` and `dwellings.parquet` (one row
   per row × draw: keys, `completion`, weight, every imputed and derived column), plus `provenance.json` (pipeline
   hash, draws, seed, every bundle's hashes, bootstrap counts). Marginals (`scores.parquet`) per level as the chains
   write them.
4. **`tasha.load_completed`** reads the pipeline's output instead of `<root>/<chain>/multiple_imputation/`: today it
   collects the chains named by `imputed: {chain, column}` in `tasha/mappings.yaml` (nse, labour). Either name the
   pipeline in those entries (`imputed: {pipeline: tasha, column: ...}`) or map chain → pipeline level; keep
   `check_mappings` checking the keys against the columns' levels (`_schema.chain_levels`).
5. **CLI**: `eodgdl impute score <pipeline> [--draws M] [--bootstrap]`, and `retrain <pipeline>` retraining its tasks
   as `retrain <chain>` does. `scripts/impute/rerun.sh` should write the pipeline too.

What stays: the tasks and bundles; `sector_informality` (its enumerate marginals and `evaluate sector_informality`)
and `nse` (its variants and `evaluate nse`, the AMAI benchmark) remain diagnostic chains, run as today; their reports
keep reading them. Decide whether `labour` survives as a chain (it becomes stage 1 + the person half of stage 2) or is
removed with its YAML; `reports/imputation_empleo.qmd` reads `output/impute/labour/multiple_imputation/` and must move
to the pipeline's output.

## Validation plan

- Unit tests (`tests/test_chain.py` style, synthetic): stage alignment (draw *d* of stage 2 sees draw *d* of stage
  1), the aggregation (a drawn worker aged ≥ 14 raises the count; under 14 or not drawn does not; the cap), the
  bootstrap index per draw, the output's row counts per level.
- Parity, with `scripts/impute/parity.py` against today's `output/impute/`: with the aggregation switched off the
  pipeline must reproduce the `labour` and `nse` MI completions exactly (same seeds, same models). With it on, only
  dwellings with a drawn worker may move (bathrooms, bedrooms, points, levels), and persons not at all.
- Report the change: NSE levels moved per draw (expect ~690 uncalibrated, upward), the calibrated distribution
  (`nse_calibrado`; the AGEB mix is AMAI's, so moves are re-rankings within AGEBs), `IncomeClass` counts in the TASHA
  build (draw 0), bathrooms / bedrooms shares.
- `tasha.validate_all(*od) == []` on the new build; `uv run pytest` green.

## State at handoff

- Branch `impute-chains`, HEAD `ba271a3`, ten commits after `main` (`8dc067d`). Local tags **v0.6.0, v0.7.0, v0.8.0**
  are not pushed (`REF = "v0.8.0"`, so installed users cannot fetch the giro, educacion_jefe and empleo bundles until
  they are); pushing and merging into `main` waits on the user's go-ahead, then the mirror check.
- `output/impute/`: v0.7.0's run (`rerun.sh`, the educacion_jefe retrain) plus `empleo/` (retrain, 50 bootstrap
  bundles) and `labour/` (+ `multiple_imputation/`) written in place; `output/impute_v060/` kept.
- Untracked and not ours: `mappings/` (Monterrey's `pimus2019_to_tasha.yaml`, the user's), `reports/.gitignore`,
  `scripts/rural_agebs_clipped.py`. Leave them out of commits.
- The auto-mode permission check blocks writing `data/` and moving or deleting `output/` roots: hand the user the
  `!` commands (copy the bundle, append or replace its line in `src/eodgdl/data/registry.txt`).
- `uv run pytest`: 182 passed.

## Still open for the TASHA file (after the pipeline)

At defaults: `EmploymentZone` / `SchoolZone` "0" (about 4,700 workers without a work trip, 3,286 reported plus the
drawn ones; 2,965 students without a school trip, mostly the Easter weeks 3-16 April 2023), `DwellingType`, `License`,
`TransitPass`, `FreeParking`. Pending user decisions: whether a worker without a work trip keeps "0" or gets an imputed
workplace (a stage-1 task, conditioned on the drawn giro); what License and TransitPass should mean and whether
external data exist. The draw-vs-arg-max choice was defaulted to one seeded draw (`--draw 0`), configurable; the user
did not object. Proposals for each are in `docs/handoff_impute_engine.md` ("TASHA gaps").

## A starting prompt

> Implement the staged imputation pipeline described in docs/handoff_pipeline.md: read it, docs/impute.md and the
> session-9 entries of docs/handoff_impute_engine.md, then propose the pipeline spec and the aggregation's place before
> writing code. Prove parity with the aggregation off before turning it on.
