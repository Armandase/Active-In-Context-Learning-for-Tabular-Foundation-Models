"""Simulated labelling loop and metrics of the paper (Section IV-A and IV-B)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.utils import check_random_state

from .learner import ActiveLearner


def normalized_aulc(n_labeled, scores) -> float:
    """Area under the learning curve divided by the labelling budget (Eq. 2).

    It is the mean score over the acquisition, with the trapezoidal rule
    between the evaluated budgets.
    """
    n_labeled, scores = np.asarray(n_labeled, dtype=float), np.asarray(scores, dtype=float)
    if n_labeled[-1] == n_labeled[0]:
        return float(scores[0])
    areas = (scores[1:] + scores[:-1]) / 2 * np.diff(n_labeled)
    return float(areas.sum() / (n_labeled[-1] - n_labeled[0]))


def initial_context(y, random_state=None) -> np.ndarray:
    """Positions of one random sample per class, the initial labelled context L0 of the paper."""
    y = np.asarray(y)
    rng = check_random_state(random_state)
    return np.array([rng.choice(np.flatnonzero(y == label)) for label in np.unique(y)])


def run_active_learning(
    X_pool,
    y_pool,
    X_test,
    y_test,
    strategy="hybrid",
    estimator=None,
    batch_size: int = 10,
    budget: int = 100,
    max_pool_fraction: float = 0.5,
    preprocessor="auto",
    categorical_features="auto",
    random_state=None,
) -> pd.DataFrame:
    """Simulate the labelling loop of the paper with y_pool as the oracle.

    Starts from one random sample per class, queries batch_size samples per
    round until ``budget`` labels (or ``max_pool_fraction`` of a small pool),
    and evaluates on the test set after each round. Returns one row per round
    with ``n_labeled``, Cohen's ``kappa``, ``roc_auc`` (one-vs-rest, macro)
    and the ``seconds`` spent querying and refitting to reach this context.

    With the same integer ``random_state``, every strategy starts from the
    same initial context, as in the paired comparisons of the paper.
    """
    y_pool, y_test = np.asarray(y_pool), np.asarray(y_test)
    rng = check_random_state(random_state)
    classes = np.unique(y_pool)
    initial = initial_context(y_pool, rng)
    budget = max(len(initial), min(budget, int(max_pool_fraction * len(y_pool))))

    y_partial = np.full(len(y_pool), None, dtype=object)
    y_partial[initial] = y_pool[initial]
    start = time.perf_counter()
    learner = ActiveLearner(estimator, strategy, batch_size, preprocessor, categorical_features, random_state=rng)
    learner.fit(X_pool, y_partial)
    seconds = time.perf_counter() - start

    rows = []
    while True:
        proba = learner.predict_proba(X_test)
        kappa = cohen_kappa_score(y_test, learner.classes_[proba.argmax(axis=1)])
        rows.append({"n_labeled": learner.n_labeled_, "kappa": kappa, "roc_auc": roc_auc(y_test, proba, classes), "seconds": seconds})
        if learner.n_labeled_ >= budget:
            break
        start = time.perf_counter()
        indices = learner.query(min(batch_size, budget - learner.n_labeled_))
        learner.label(indices, y_pool[indices])
        seconds = time.perf_counter() - start
    return pd.DataFrame(rows)


def roc_auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """ROC AUC, one-vs-rest macro-averaged beyond two classes; NaN when it is undefined.

    The columns of ``proba`` follow ``classes``.
    """
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true == classes[1], proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
    except ValueError:  # a class is missing from the test set
        return float("nan")
