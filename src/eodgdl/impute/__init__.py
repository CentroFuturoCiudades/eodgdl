"""Categorical imputation engine: every imputation model on the survey (giro, informality, the head of household's
education, the AMAI components), see ``docs/handoff_impute_engine.md``.

Optional extra ``eodgdl[giro]`` (scikit-learn, joblib). The engine knows no survey vocabulary: the numeric features,
declared category levels, missing label, classes and column names come from the caller (a task).

The workflow: a **task** (``impute/tasks/<task>.yaml``) is trained (``run.retrain`` + ``run.write_retrain``, written to
``<root>/<task>/``), compared (``run.compare``) and scored (``run.score_task``); a **chain** (``impute/chains/<chain>.yaml``)
scores its tasks in sequence (``chain.run_chain``) and is diagnosed (``diagnostics.evaluate_chain``). Every mode takes
its bundles from ``bundle.load_bundle``: ``retrained=<root>`` reads a retrain's, else the installed model file (``models/``).

- :mod:`~eodgdl.impute.spec`: task specs and their hashes, the YAML reader
- :mod:`~eodgdl.impute.sources`, :mod:`~eodgdl.impute.features`: source frames, feature builders, the feature cache
- :mod:`~eodgdl.impute.harmonize`: value maps that fail on an unmapped code, registered as feature builders
- :mod:`~eodgdl.impute.bundle`: fitted bundles, where they are read from, their load-time checks, bootstrap refits
- :mod:`~eodgdl.impute.run`: a task's modes: ``score_task`` / ``score_frame`` / ``predict_rows``, ``retrain``
  (+ ``write_retrain``), ``compare``
- :mod:`~eodgdl.impute.chain`, :mod:`~eodgdl.impute.derive`: chains of tasks, parallel or sequential
  (``enumerate`` / ``draws`` / ``expected``), and the derive steps run per completion
- :mod:`~eodgdl.impute.diagnostics`: a chain's diagnostics (``eodgdl impute evaluate``)
- :mod:`~eodgdl.impute.levels`: the category-level contract, feature preparation, weight and probability checks
- :mod:`~eodgdl.impute.models`: candidate pipelines per family, the auxiliary level model
- :mod:`~eodgdl.impute.select`: grouped CV (parallel, staged boosting) and the one-SE rule
- :mod:`~eodgdl.impute.marginalize`: batched marginalization of levels without training support
- :mod:`~eodgdl.impute.arms`: hybrid scoring by feature arms and the task's output columns
- :mod:`~eodgdl.impute.evaluate`: held-out metrics of a probability matrix, bootstrap, calibration, shift reweighting,
  delta adjustment
"""
