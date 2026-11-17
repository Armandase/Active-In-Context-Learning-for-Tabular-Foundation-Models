"""Classifiers compared in the paper (Section IV-C), with their fixed hyperparameters."""

from __future__ import annotations

import numpy as np
from sklearn.semi_supervised import LabelSpreading

from .learner import default_estimator


class LabelSpreadingBaseline(LabelSpreading):
    """LabelSpreading predicting uniform probabilities, instead of NaN, for samples whose neighbours no label reaches."""

    def predict_proba(self, X):
        with np.errstate(divide="ignore", invalid="ignore"):
            proba = super().predict_proba(X)
        proba[~np.isfinite(proba).all(axis=1)] = 1 / proba.shape[1]
        return proba


def catboost(random_state=None):
    """CatBoost refit from scratch at each round (CatBoost-Margin baseline).

    The paper also sets early_stopping_rounds=50 but does not say on which
    validation set; none is used here.
    """
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
    )


def xgboost(random_state=None):
    """XGBoost refit from scratch at each round (XGBoost-Margin baseline).

    The paper also sets early_stopping_rounds=50 but does not say on which
    validation set; none is used here.
    """
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, random_state=random_state)


def label_spreading(random_state=None):
    """Semi-supervised Label Spreading with a 7-nearest-neighbour kernel and clamping factor 0.2."""
    return LabelSpreadingBaseline(kernel="knn", n_neighbors=7, alpha=0.2)


def tabpfn(random_state=None):
    """TabPFN with n_estimators=32, as for all Tab-AICL strategies."""
    return default_estimator(random_state)


ESTIMATORS = {
    "tabpfn": tabpfn,
    "catboost": catboost,
    "xgboost": xgboost,
    "label_spreading": label_spreading,
}

# method name of the paper -> (estimator, acquisition strategy)
METHODS = {
    "TabPFN-Coreset": ("tabpfn", "coreset"),
    "TabPFN-Hybrid": ("tabpfn", "hybrid"),
    "TabPFN-Proxy-Hybrid": ("tabpfn", "proxy_hybrid"),
    "TabPFN-Margin": ("tabpfn", "margin"),
    "TabPFN-Random": ("tabpfn", "random"),
    "CatBoost-Margin": ("catboost", "margin"),
    "XGBoost-Margin": ("xgboost", "margin"),
    "LabelSpreading-Random": ("label_spreading", "random"),
}
