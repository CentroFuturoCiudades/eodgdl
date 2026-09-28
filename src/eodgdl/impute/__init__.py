"""Categorical imputation engine: every imputation model on the survey (giro today; informality, place of work,
head-of-household education and the AMAI components as they move here, see ``docs/handoff_impute_engine.md``).

Optional extra ``eodgdl[giro]`` (scikit-learn, joblib). The engine knows no survey vocabulary: the numeric features,
declared category levels, missing label, classes and column names come from the caller (a task).

- :mod:`~eodgdl.impute.spec`: task specs (``impute/tasks/<task>.yaml``) and their hashes
- :mod:`~eodgdl.impute.sources`, :mod:`~eodgdl.impute.features`: source frames, feature builders, the feature cache
- :mod:`~eodgdl.impute.bundle`: fitted bundles, their load-time checks, the legacy giro bundle
- :mod:`~eodgdl.impute.run`: a task's modes: ``score_task``, ``retrain`` (+ ``write_retrain``), ``compare``
- :mod:`~eodgdl.impute.chain`, :mod:`~eodgdl.impute.derive`: chains of tasks, parallel or sequential
  (``enumerate`` / ``draws`` / ``expected``), and the derive steps run per completion
- :mod:`~eodgdl.impute.levels`: the category-level contract, feature preparation, weight and probability checks
- :mod:`~eodgdl.impute.harmonize`: value maps that fail on an unmapped code
- :mod:`~eodgdl.impute.models`: candidate pipelines per family, the auxiliary level model
- :mod:`~eodgdl.impute.select`: grouped CV (parallel, staged boosting) and the one-SE rule
- :mod:`~eodgdl.impute.marginalize`: batched marginalization of levels without training support
- :mod:`~eodgdl.impute.arms`: hybrid scoring by feature arms and the task's output columns
- :mod:`~eodgdl.impute.evaluate`: held-out metrics, bootstrap, calibration, shift reweighting, delta adjustment
"""
