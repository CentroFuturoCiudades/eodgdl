"""Compatibility shim: giro bundles pickled before :mod:`eodgdl.impute` existed (eodgdl 0.2 and earlier) reference
``eodgdl.giro._ml.prepare_model_features`` as their first pipeline step; this keeps them loading. The modelling
machinery is :mod:`eodgdl.impute`."""

from eodgdl.impute.levels import prepare_features

from ._config import NO_ESPECIFICADO, NUMERIC_FEATURES


def prepare_model_features(dataframe, features):
    """Select the model features: numeric columns coerced, categorical columns as clean strings with a missing label."""
    return prepare_features(dataframe, features, NUMERIC_FEATURES, NO_ESPECIFICADO)
