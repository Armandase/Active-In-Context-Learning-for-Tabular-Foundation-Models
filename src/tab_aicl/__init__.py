"""Unofficial implementation of Tab-AICL, from *Active In-Context Learning for
Tabular Foundation Models* (Treerath & Pittorino, 2026, arXiv:2603.27385)."""

from .learner import ActiveLearner, unlabeled_mask
from .preprocessing import infer_categorical_features, make_preprocessor
from .simulation import initial_context, normalized_aulc, run_active_learning
from .strategies import (
    STRATEGIES,
    CoresetSampling,
    HybridSampling,
    MarginSampling,
    ProxyHybridSampling,
    QueryStrategy,
    RandomSampling,
    entropy,
    margin,
)

__version__ = "0.1.0"

__all__ = [
    "ActiveLearner",
    "CoresetSampling",
    "HybridSampling",
    "MarginSampling",
    "ProxyHybridSampling",
    "QueryStrategy",
    "RandomSampling",
    "STRATEGIES",
    "entropy",
    "infer_categorical_features",
    "initial_context",
    "make_preprocessor",
    "margin",
    "normalized_aulc",
    "run_active_learning",
    "unlabeled_mask",
]
