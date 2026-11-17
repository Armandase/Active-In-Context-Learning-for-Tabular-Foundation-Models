"""End-to-end checks with the real TabPFN model: pytest -m slow."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from tab_aicl import STRATEGIES, ActiveLearner, initial_context

pytest.importorskip("tabpfn")
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_tabpfn_round_from_one_sample_per_class(strategy):
    X, y = load_iris(return_X_y=True, as_frame=True)
    y = np.asarray(y)
    y_partial = np.full(len(y), -1)
    start = initial_context(y, random_state=0)
    y_partial[start] = y[start]

    learner = ActiveLearner(strategy=strategy, batch_size=5, random_state=0).fit(X, y_partial)
    indices = learner.query()
    assert len(set(indices)) == 5 and not set(indices) & set(start)
    learner.label(indices, y[indices])
    assert learner.n_labeled_ == 8
    assert learner.predict_proba(X).shape == (150, 3)
