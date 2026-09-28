# Handoff: `eodgdl.impute`, a categorical imputation engine that retires the giro notebook

Written 2026-09-28 on branch `giro-model`, for a new session. Read this file whole, then `CLAUDE.md`, then
`src/eodgdl/giro/` (≈1,000 lines) before changing anything. Every claim about the downstream repository
`../informal-jobs-model` comes from a read-only analysis made that day; re-check the ones you rely on (file:line
references below are into that repo unless stated).

## 1. Goal

Move the giro model's workflow out of `notebooks/giro_model.ipynb` into the package, as a **generic engine for
categorical imputation models** that can:

- **load the winning model and score** (what users and downstream code do; seconds);
- **retrain** a task end to end: grouped-CV selection → held-out evaluation → final fit → bundle + evaluation
  artifacts (what the notebook does now; ~20 min for giro);
- **compare** feature specifications cheaply while iterating (fixed hyperparameters, same folds, paired fold
  differences; seconds to a minute);
- **chain tasks**: a downstream categorical model consumes an upstream task's *probabilities* (not hard labels),
  marginalizing over the upstream classes.

All targets now and planned are categorical (user, 2026-09-28): no regression support is needed. Ordinal targets
are handled as categorical plus an optional expected score (see §5.7).

## 2. State of the branch when this was written

`giro-model` at `b05e8e5` = merge of `main` (v0.2.0, 55 commits) into the branch, tests green (116). On top of it,
**uncommitted** (check `git status`; commit them first, see §8 phase 0):

| change | files | why |
|---|---|---|
| Destination crosswalk rebuilt on the zone system | `giro/features.py`, `giro/__init__.py`, `giro/config.yaml`, `tests/test_giro.py`, README, pyproject comment | `main` retired `eodgdl.load_imeplan_agebs` (IMEPLAN's AGEB table). A destination code *is* a zone-system unit: 13-char urban AGEB or 9-char rural AGEB (`giro.zone_units(tables)` over `reweight.zoning.assign_units`). DENUE establishments count to their urban AGEB, else to the rural AGEB DENUE itself places them in (agrees 99.89% with the zoning's rural AGEB; also keeps establishments in unpopulated places, e.g. `14097059A`: 217 establishments, 373 workers). `destino_ambito` level `localidad_rural` → `ageb_rural`. Effect: 1,737 rural-destination workers; before, 1,121 were `desconocido` and 616 had a rural AGEB key joined to *locality* codes; now 1,590 carry their AGEB's mix. Urban destinations unchanged (old table was the identity for them). A code outside the zone system now raises. `add_destination_features(od, urban, rural, …)` changed signature. |
| Parallel tuning + staged boosting | `giro/model.py` (`tune_models`, `_fit_fold`, `_cv_tasks`, `_fold_metrics`) | Every (configuration, fold) fit runs in its own single-threaded joblib worker; a HistGradientBoosting grid over `max_iter` is fitted once per fold to the largest value and scored at each stage (`staged_predict_proba`). Verified identical to the old serial loop (fold log losses to 1e-16, same selection) and 12× faster on a small grid. |
| Notebook fixes | `notebooks/giro_model.ipynb` | Cell 20 used `giro` as a loop variable, shadowing the module (every later `giro.` call failed); cell 10's fit count text. |
| **Refit** | `data/od_giro_hybrid_model.joblib`, `src/eodgdl/data/registry.txt` | See §3. |

The merge also moved mxcensus from 0.1.0 to **v0.2.0** (the branch's own pin, `[tool.uv.sources]`); reweight tests
pass under it. informal-jobs-model runs mxcensus **0.4.0** with an `override-dependencies` because of this pin —
bump eodgdl to v0.4.0 (phase 0) so it can drop the override.

## 3. The refit (2026-09-28)

Refit with the notebook as it stands (full grid, random forest kept), on `load_eod()` of v0.2.0 and the zone-system
crosswalk. Bundle installed at `data/od_giro_hybrid_model.joblib`, sha256 `227afc71…` in `registry.txt`
(metadata: eodgdl 0.2.0, sklearn 1.9.0, DENUE 202211). The scored workers and the sensitivity scenarios are kept
as local references in `outputs/reference/od_giro_imputed.parquet` and `…_sensitivity.parquet` (gitignored).

- 26,913 OD workers; 17,429 with a reported giro (13,944 train / 3,485 held-out, by household); 9,484 imputed.
- Both arms again select **GradientBoosting** (lr 0.05, 50 iterations, 31 leaves, l2 1.0), as the previous bundle.
  CV log loss with education: GB 0.910, RF 0.925, LR 0.940; without: GB 0.926, RF 0.933 (+0.0070 ± 0.0040, just
  outside one SE), LR 0.955.
- Held-out, weighted:

  | arm | log loss (old → new) | accuracy (old → new) |
  |---|---|---|
  | with education | 0.8824 → 0.8863 | 64.3% → 63.5% (95% CI 61.0–65.8%) |
  | without education | 0.9078 → 0.9089 | 62.0% → 62.0% |

  31.2% / 29.4% below the weighted-marginal baseline (1.288). Within noise of the old bundle, which was trained on
  a pre-release eodgdl (chain rounds before 16–17, IMEPLAN crosswalk), so the change is not the crosswalk's alone.
- Imputed shares (weighted, %): comercio 24.7, servicio 35.7, educacion 1.3, industria 32.5, gobierno 5.9 (known:
  33.4 / 34.8 / 1.6 / 26.2 / 4.1). Shift-weighted within ~1 pp; delta factor for gobierno 0.692.
- 222 imputed rows have an `ocupacion` level absent from training (Desempleado, Estudiante, Hogar, Jubilado…;
  workers by `trabajo_semana_pasada` who describe themselves otherwise) and are marginalized over it.

**Where the 20 minutes go** (per-cell times from the executed notebook): features 29 s; tuning 85 s + 79 s (was
the bulk before the parallel tuner); evaluation 6 s + 6 s; **final refit + scoring 495 s; covariate-shift refit +
scoring 522 s**. The fits are cheap; **scoring is the bottleneck**: `predict_proba_marginalizing` recurses feature
by feature and, for rows with an unsupported value, calls the full pipeline's `predict_proba` once per supported
level on a small slice. Scoring the 9,484 imputed workers made **16,619 `predict_proba` calls** (314,806 rows in
all; only 535 workers need any marginalization, mostly `destino_trabajo+modo_trabajo` for workers without a work
trip), while one call on all 9,484 rows takes 0.04 s. Expanding every row's combinations into one frame, with
weights, and calling `predict_proba` once per model gives identical probabilities in seconds. Do this in phase 1
(§8); it is what makes "score in seconds" true.

## 4. What the downstream repositories showed

### 4.1 `../informal-jobs-model` (ijm): the real consumer

A uv package `informal_jobs_model` that depends on `eodgdl[giro]` pinned by **git branch `giro-model`, locked at
`9aa68d0`** (pre-merge; it still uses `load_imeplan_agebs` and the old bundle). No tests. Two chains:

1. **Informality** (package code + notebooks 01–05):
   - S1/S2: pooled ENOE 2022t1–2023t4 Jalisco via mxcensus (`fac_tri/8`, cross-quarter household key built from the
     *private* `mxcensus.enoe._DWELLING_KEY_SPEC` / `_level_key`), OD workers from `eodgdl.load_eod()`; harmonized to
     common levels (`harmonize_enoe_od_dataframes.py`, inline dicts + `assert_mapping_covers` fail-on-unmapped;
     only `mappings/sector.yaml` is YAML).
   - M1: **giro** scored from eodgdl (`giro.load_model`, `build_worker_features`, `impute_giro(…,
     destination_models=…)`, plus `impute_under_covariate_shift` and `adjust_imputed_share` for sensitivity), then
     collapsed 5→4 sectors by summing `prob_giro_*` through `sector.yaml:od_giro` (comercio→comercio;
     servicio, educacion→servicios_transporte; industria→manufactura_construccion;
     gobierno→gobierno_otro_agricultura); observed rows one-hot.
   - M2: **informality** trained on ENOE (binary; features `genero, ocupacion, edad_num, escolaridad, municipio,
     estado_civil, parentesco, tamano_viv_cat, sector, lugar_trabajo`; robust arm drops `escolaridad`), same
     families and one-SE rule as giro but a serial tuner; evaluation adds ECE/slope/intercept, isotonic calibration
     (not shipped), PSU-grouped CV, shift reweighting to the OD missing-education profile, an ENOE–OD gap
     decomposition. Scoring enumerates the 4 sectors: P(I|x) = Σ_s P(s|x) · P(I|x,s), writing
     `prob_informal_given_sector_<s>` etc.
   - M2a: **place-of-work auxiliary model** P(lugar_trabajo | x, s), fixed HGB, marginalized inside each sector
     scenario.
2. **NSE** (notebooks 06–08, all inline, none of the machinery above):
   - M3 head-of-household education: OD-trained, 7 AMAI classes, expected score.
   - M4 AMAI components (workers 14+, bathrooms, bedrooms) × with/without income: **ENIGH 2022 national**-trained,
     expected scores (368 MB bundle).
   - M5 NSE: sum of expected scores cut into a hard level, then a deterministic AGEB rank calibration. Not a model.

**Duplicated machinery.** ijm's `common.py` is a near-copy of `eodgdl/giro/_ml.py`, drifted both ways: ijm has the
**newer marginalization semantics** (`predict_proba_marginalizing`: `level_subsets` only narrows the averaging;
auxiliary level models carry `training_level_shares_` and marginalize their own inputs), an isotonic wrapper and
grouped robustness CV; eodgdl has the parallel/staged tuner. The engine must take the best of both (§5.5).

**ijm code that eodgdl changes will break** (keep this list for the ijm migration):
`giro.build_worker_features` + `giro.impute_giro(model_with_education, model_without_education, …,
destination_models=…)` call sequence (nb04 c2; README §7.1 documents `giro.impute`); bundle keys
`model_with_education`, `features_with_education`, `destination_models`, `metadata.selected`;
`impute_under_covariate_shift`, `adjust_imputed_share`, `calculate_model_usage`,
`calculate_probabilistic_distribution`, `calculate_distribution(giro_column=…)`, `validate_probability_rows`;
`giro.OUTPUT_COLUMNS`, `GIRO_CLASSES`, columns `prob_giro_<slug>`, `giro_fue_imputado`, `giro_model_used`,
`giro_marginalized_features`, `giro_desconocido`; the worker set and work-trip destination duplicated in ijm's
`config/od.yaml` and `compute_od_work_trip_destination` (its asserts at `harmonize…py:442,449` catch a drift in
the worker set, not in the destination); pickled references to `eodgdl.giro._ml.prepare_model_features` inside
bundles.

### 4.2 `../informal_jobs`: the 2023–24 Monterrey prototype

ENOE 2019-T4 Nuevo León + the **Monterrey** OD survey 2019 (not Guadalajara); nothing ports directly. Lessons it
teaches, all avoided in the design: hard labels passed between models (its informality rate was 29% by hard labels
vs 37% by mean probability); no survey weights; CV not grouped by household; balanced accuracy (a hard-label
metric) as the selector; value maps inline in code, drifting between its R and Python versions. Worth keeping: model
families and grids as data; a Bayesian practical-equivalence (ROPE, `baycomp`) rule as an optional alternative to
one-SE; SHAP for explanation.

## 5. Design

### 5.1 Layout

```
src/eodgdl/impute/                 the engine: pandas + scikit-learn only, knows no survey vocabulary
  spec.py        TaskSpec: parse + validate a task YAML; spec hash
  sources.py     registry of source builders (name → callable returning a frame + its key/weight/group columns)
  features.py    registry of feature builders; on-disk cache keyed by (builder name, builder version, input versions)
  levels.py      category-level contract: declared levels + missing label; assert_known_levels
  harmonize.py   YAML value maps with fail-on-unmapped and allowed_unmapped (vocabulary-agnostic; ijm uses it)
  models.py      families/grids from the spec → sklearn pipelines (prepare → preprocess → classifier)
  select.py      grouped CV (parallel, staged boosting), one-SE within/across families, fold tables, compare()
  marginalize.py predict_proba_marginalizing (ijm semantics), auxiliary level models with training shares
  arms.py        hybrid dispatch: N feature arms chosen per row by covariate missingness
  chain.py       depends_on: upstream probability transforms (many-to-one collapse), scenario marginalization
  evaluate.py    held-out metrics, group bootstrap, multiclass reliability/ECE, marginal baseline, shift checks
  bundle.py      fit/save/load; metadata; load-time checks
  run.py         run(tasks) in dependency order; retrain(task); outputs
src/eodgdl/impute/tasks/giro.yaml  giro's spec (absorbs giro/config.yaml)
src/eodgdl/giro/                   thin wrapper: keeps impute(), load_model(), OUTPUT_COLUMNS, GIRO_CLASSES,
                                   build_worker_features, column names; giro-only feature builders stay here
                                   (worker frame, work-trip destination, DENUE mix)
src/eodgdl/giro/_ml.py             shim re-exporting from eodgdl.impute so existing bundles still unpickle
reports/giro_model.qmd             replaces the notebook: reads a retrain's artifacts, trains nothing
```

`eodgdl.impute` hosts only **EOD-native tasks** (giro now; head-of-household education is a candidate, question
in §9). ENOE/ENIGH-trained tasks stay in ijm as task specs it registers and runs with this engine: eodgdl must
never import ijm, and the giro rationale ("knows nothing about ENOE") holds. The `impute` extra needs only
scikit-learn/joblib/pyyaml; mxcensus only for builders that need DENUE or the census (giro's).

### 5.2 Task spec (giro, sketch)

```yaml
task: giro
source: eodgdl.workers              # registered builder: load_eod() → OD workers (employed categories)
keys: [folio_vivienda, folio_habitante]
target:
  column: giro_empresa
  classes: {Comercio: comercio, Servicio: servicio, Educación: educacion, Industria: industria,
            Gobierno/sector público: gobierno}     # label → slug; order is the output order
weight: ponderador                  # normalized to mean 1 inside the engine
group: [folio_vivienda]             # CV grouping (composite keys allowed: ENOE's cross-quarter household)
features:                           # builders produce columns; arms pick from them
  builders: [giro.person, giro.dwelling, giro.work_trip, giro.destination_denue]
  numeric: [edad, dest_establecimientos_log, dest_share_grandes, dest_share_*]
levels: {destino_ambito: [ageb_urbana, ageb_rural, aeropuerto, fuera_zm, desconocido], …}   # + schema-derived
missing_label: no_especificado
arms:                               # first arm whose `requires` are observed scores the row
  with_education: {features: [...], requires: [escolaridad]}
  without_education: {features: [...]}
auxiliary:                          # P(feature | x) models used to marginalize a missing feature
  destino_trabajo: {features: arm features minus the destination block}
selection:
  cv: {splits: 5, seed: 42, test_fold: 0}
  metric: weighted_log_loss
  rule: one_se                      # optional: rope (baycomp), later
  families:
    LogisticRegression: {params: {C: [0.1, 1.0, 10.0]}}
    RandomForest: {n_estimators: 500, params: {max_leaf_nodes: [25, 50, 100], max_features: [sqrt, 0.7], min_samples_leaf: [1, 5, 10]}}
    GradientBoosting: {native_categoricals: true, params: {max_iter: [50, 100, 200, 400], learning_rate: [0.05, 0.1], max_leaf_nodes: [15, 31], l2_regularization: [0.0, 1.0]}}
evaluation:
  bootstrap: 500
  calibration_bins: 10
  shift: {profile: [sexo_nacimiento, edad, municipio, estado_civil, parentesco, personas_en_vivienda, ocupacion, trabajo_semana_pasada],
          from: observed, to: unobserved}
  delta: {class: gobierno}
outputs: {prefix: giro}             # prob_giro_<slug>, giro_final, giro_fue_imputado, …
```

The random forest stays in the grid (user, 2026-09-28), although it lost by more than one SE in both arms of the
previous fit; revisit if retrain time matters.

A downstream spec (ijm's informality, for shape only):

```yaml
task: informality
source: ijm.enoe_workers            # training source
score_source: ijm.od_workers        # scoring source (differs: target absent in the OD)
depends_on:
  giro:                             # upstream task (eodgdl's)
    as: sector                      # the feature it fills
    transform: {comercio: comercio, servicio: servicios_transporte, educacion: servicios_transporte,
                industria: manufactura_construccion, gobierno: gobierno_otro_agricultura}
auxiliary:
  lugar_trabajo: {features: [..., sector]}   # conditioned on the upstream scenario
```

### 5.3 Modes and CLI

| mode | API | CLI | cost |
|---|---|---|---|
| score | `impute.run(["giro"], tables)`; `giro.impute()` | `eodgdl impute run giro --out …` | seconds |
| compare | `impute.compare(spec, candidates={"name": spec_or_features}, seeds=(42,))` | `eodgdl impute compare giro --spec candidate.yaml` | ~10 fits/candidate |
| retrain | `impute.retrain(spec)` | `eodgdl impute retrain giro --out output/impute/giro` | full grid |

**compare** (fast iteration): hyperparameters fixed at the published bundle's winner (read from its metadata), the
same grouped folds on the training households only (the held-out fold is never touched), each candidate's fold log
losses paired against the baseline's; reports mean difference ± SE (one-SE semantics) and, with several seeds,
repeated-CV means. Features come from the cache, so a candidate that only changes the feature list costs one CV of
one configuration. A feature experiment is a spec edit (or a new registered builder named in the spec); the
winning spec is what gets committed, then `retrain` reselects hyperparameters and writes the bundle.

### 5.4 Outputs (keep giro's names; generalize with the prefix)

`prob_<p>_<slug>` (rows sum to 1, observed rows one-hot), `<p>_observado`, `<p>_imputado`, `<p>_final` (convenience
arg-max), `<p>_fue_imputado`, `<p>_model_used` (arm), `<p>_prediction_confidence`, `<p>_marginalized_features`;
optional seeded categorical draws; optional expected score from a score map (ordinal targets; flag as plug-in when
fed downstream). Scenario outputs (shift, delta) share the schema plus `scenario`. `validate_probability_rows` runs
on every output.

### 5.5 Machinery to port (and from where)

- **Marginalization: take ijm's `common.py` versions**, not eodgdl's: `predict_proba_marginalizing` with
  `averaging_shares` semantics and `level_subsets`; auxiliary `fit_level_model` attaching
  `training_level_shares_` and marginalizing its own inputs. eodgdl's auxiliary destination models lack the shares
  (unsupported predictor levels go through an all-zero one-hot block) — a bug fixed by the port.
- **Tuner: take eodgdl's** parallel/staged `tune_models` (this branch). Keep `select_one_se`, `complexity_key`,
  `FAMILY_COMPLEXITY`, `fold_table` (identical in both repos).
- **Evaluation:** eodgdl's `evaluate_model`, `test_metrics_with_uncertainty`, `bootstrap_by_group`,
  `marginal_log_loss`, `reweight_to_target_profile`; ijm's isotonic wrapper (optional, with a ship toggle) and
  grouped robustness CV; calibration tables made **multiclass** (one-vs-rest per class) — both repos only have the
  binary form.
- **Auxiliary models become tunable** (today fixed HGB hyperparameters in both repos) and live in the task bundle.

### 5.6 Bundles

One joblib per task: fitted arms, auxiliary models, category levels, class map, and metadata = spec hash, feature
builder versions + level-contract hash, eodgdl version, data versions (survey file sha256s from the registry, DENUE
release), winner params and fold losses (for `compare`), test metrics, sklearn version, random state, upstream
bundle hashes (for chained tasks). On load: refuse on sklearn version mismatch, on a level-contract mismatch (the
`localidad_rural`→`ageb_rural` class of error surfaces here, not inside a pickled encoder), and when an upstream
bundle hash differs. Pickled callables live at stable module paths (`eodgdl.impute.*`); `eodgdl.giro._ml` stays as
a re-export shim for bundles already in the wild. Bundles ship under `data/` with a sha256 in `registry.txt` and a
release tag (`REF`), like the other data files: **the current `v0.2.0` tag has no giro bundle**, so
`giro.load_model()` only works with `$EODGDL_DATA_DIR` pointing at a clone until the next tag.

### 5.7 Chaining

`depends_on` names an upstream task, the feature it fills, and a probability transform (many-to-one collapse by
summation). Scoring enumerates the upstream classes as scenarios: P(y|x) = Σ_s P(s|x) · P(y|x,s), observed upstream
values one-hot; auxiliary models that condition on the upstream feature are marginalized inside each scenario;
per-scenario conditionals `prob_<y>_given_<s>` are written (ijm uses them). Ordinal upstreams may pass an expected
score instead (plug-in, flagged). Retraining an upstream task invalidates downstream bundles through the recorded
upstream hash.

### 5.8 Feature cache

Parquet under the eodgdl cache dir (`platformdirs`), keyed by builder name + builder version + input versions
(eodgdl version and the sha256 of the survey files, DENUE release). The chain rules move trips into and out of
`motivo_viaje == "Trabajar"`, which moves `destino_trabajo`; an unversioned cache would go stale silently.
`--refresh` rebuilds. `build_worker_features` took ~1 min (DENUE + census for the zone system); everything after
it reads the cache.

## 6. Known defects to fix during the migration

1. `giro.impute_under_covariate_shift` (`src/eodgdl/giro/model.py`) calls `impute_giro` **without
   `destination_models`**: in the shift scenario, workers without a work trip are marginalized by global shares
   instead of P(destination | x). ijm reports this scenario as a headline (nb04 c4, nb05). Verified in code.
2. eodgdl's auxiliary destination models have no `training_level_shares_` (verified: `fit_level_model` in
   `giro/_ml.py` never attaches them). Fixed by porting ijm's version (§5.5).
3. `adjust_imputed_share` clips at 1, so the target share is not reached exactly when the clip binds (minor).
4. Scoring is ~8 minutes per call of `impute_giro` because of per-slice `predict_proba` calls in the
   marginalization recursion (§3); batch them. The bootstrap and the metrics are cheap (6 s per arm).

## 7. Division of responsibilities

| component | home |
|---|---|
| engine (§5.1), harmonization loader, level contract, marginalization, tuner, evaluation, bundles, cache, CLI | eodgdl.impute |
| EOD-native tasks (giro + destination auxiliary; maybe head-of-household education) and their feature builders | eodgdl (task YAML + builders) |
| ENOE/ENIGH↔OD harmonization maps, informality + place-of-work, AMAI components, NSE scoring + AGEB calibration, gap decomposition | ijm, as task specs on the engine |
| pooled ENOE loader with a public cross-quarter household key; ENIGH household assembly | mxcensus (replaces ijm's private `_level_key` import) |

## 8. Migration plan

Each phase ends with the full suite green (`uv run pytest`) and is committed separately. Parity tests compare
against reference outputs saved from the current bundle (phase 0), so behaviour changes are always deliberate.

0. **Land the current work.** Commit §2's changes and the refit bundle (already installed, sha256 already in
   `registry.txt`; ask the user before committing). Bump
   `[tool.uv.sources] mxcensus` to `v0.4.0`, `uv lock`, run the suite (reweight + giro). Reference outputs are
   already saved in `outputs/reference/` (gitignored, local; from the refit); add a small committed fixture (a few
   hundred workers' features and probabilities, including marginalized ones) for the parity test.
1. **Engine core without behaviour change.** Create `eodgdl.impute` (levels, harmonize, models, select,
   marginalize, arms, evaluate); move the generic code out of `giro/_ml.py` and `giro/model.py`; `giro/_ml.py`
   becomes a shim. Port ijm's marginalization semantics **and** the auxiliary shares — this changes giro's
   probabilities for workers with unsupported levels or no work trip, so the parity test must isolate it: first
   move code and batch the marginalization's `predict_proba` calls (§3; parity exact, minutes → seconds), then
   port semantics (parity except the documented rows). Tests: synthetic unit tests
   for marginalization, one-SE, staged-vs-unstaged tuning equality (the check in §2), harmonize fail-on-unmapped.
2. **Specs, sources, features, cache.** `TaskSpec` + `tasks/giro.yaml` (absorbs `giro/config.yaml`); giro's
   builders registered; cache. `giro.build_worker_features` becomes a wrapper; output identical.
3. **Bundles v2.** New bundle format and load-time checks; a converter from the current bundle dict so `score` works
   before any retrain; `giro.load_model()` returns the new object but keeps the old dict keys readable for one
   release (ijm reads them).
4. **Modes.** `run`, `retrain`, `compare` + CLI `eodgdl impute …`. `retrain giro` must reproduce the notebook's
   selection and metrics on the same data (it is the same code); `compare` baseline from bundle metadata.
5. **Report, retire the notebook.** `retrain` writes evaluation artifacts (metrics, fold tables, reliability,
   profiles, shift/delta scenarios, distributions) under `--out`; `reports/giro_model.qmd` reads them with
   `assert`s on the structural claims (as the other reports do). Delete `notebooks/giro_model.ipynb`. Fix §6.1
   here (shift scenario with auxiliary models).
6. **Chaining.** `depends_on`, transforms, scenario marginalization, per-scenario conditionals, expected scores,
   draws. Tested in eodgdl on a synthetic two-task chain; ijm's informality is the real case, migrated in ijm.
7. **Release.** Bump version, `REF` to the new tag, bundle in `registry.txt`, tag and push (see CLAUDE.md "Data
   access"). Write ijm's migration notes from §4.1's break list; ijm re-locks to the tag (not the branch) and
   drops its mxcensus override.

## 9. Open questions for the user

1. Should head-of-household education (ijm nb06, OD-trained, 7 AMAI classes) move into eodgdl as the second
   EOD-native task? It would exercise arms/expected scores inside eodgdl, and ijm's version has issues (person
   weight for a dwelling-level target; `handle_unknown="ignore"` + mode imputation).
2. Keep one-SE as the only selection rule, or add the ROPE (baycomp) option now?
3. Ship isotonic-calibrated models when they win on held-out calibration, or keep them diagnostic only (ijm:
   `SHIP_CALIBRATED=False`)?
4. mxcensus work (public ENOE household key, pooled loader, ENIGH assembly): in scope for this refactor, or a
   separate task in mxcensus?
5. Should `retrain` always run the shift and delta scenarios, or only with `--full-evaluation`? (Cheap once
   scoring is batched.)
6. `giro-model` merges into `main` when? ijm currently tracks the branch by name; a release tag is the safer pin.

## 10. Practical notes

- In this shell `cp`, `rm` and `mv` are aliased to interactive versions and hang a non-interactive command waiting
  for `y/n`: use `command cp -f`, `command rm -f`.
- Run anything that reads the survey with `EODGDL_DATA_DIR=$PWD/data` to use the in-repo files (the mirror tag
  lacks the giro bundle).
- Executing the notebook headless: copy it to a scratch directory **with a `data/` subdirectory** (its last cell
  writes `ROOT/data/od_giro_hybrid_model.joblib`), then
  `uv run jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=-1 --output executed.ipynb <copy>`.
- The reference analyses behind §4 were saved (notebook sources extracted) under the originating session's
  scratchpad, which a new session cannot read; re-run a read-only pass over ijm if a detail matters.
