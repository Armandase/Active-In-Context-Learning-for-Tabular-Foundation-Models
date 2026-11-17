import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from tab_aicl import initial_context, normalized_aulc, run_active_learning


def test_normalized_aulc():
    assert normalized_aulc([3, 13, 23], [0.4, 0.4, 0.4]) == pytest.approx(0.4)
    assert normalized_aulc([0, 50, 100], [0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert normalized_aulc([2, 10, 100], [0.0, 1.0, 1.0]) == pytest.approx((4 + 90) / 98)
    assert normalized_aulc([5], [0.7]) == pytest.approx(0.7)


def test_initial_context_has_one_sample_per_class():
    y = np.array([0, 0, 1, 2, 2, 2])
    labeled = initial_context(y, random_state=0)
    assert sorted(y[labeled]) == [0, 1, 2]


def data(n=300, n_classes=3):
    X, y = make_classification(n, n_features=6, n_informative=4, n_classes=n_classes, random_state=0)
    return X[:200], y[:200], X[200:], y[200:]


def test_learning_curve_steps():
    curve = run_active_learning(*data(), strategy="margin", estimator=LogisticRegression(), batch_size=10, budget=45, random_state=0)
    assert curve["n_labeled"].tolist() == [3, 13, 23, 33, 43, 45]
    assert list(curve.columns) == ["n_labeled", "kappa", "roc_auc", "seconds"]
    assert curve["roc_auc"].between(0, 1).all()


def test_small_pools_stop_at_half():
    X_pool, y_pool, X_test, y_test = data()
    curve = run_active_learning(X_pool[:40], y_pool[:40], X_test, y_test, strategy="random", estimator=LogisticRegression(), random_state=0)
    assert curve["n_labeled"].iloc[-1] == 20


def test_strategies_share_the_initial_context():
    first_scores = {
        strategy: run_active_learning(*data(), strategy=strategy, estimator=LogisticRegression(), budget=13, random_state=7)["kappa"].iloc[0]
        for strategy in ("random", "hybrid", "coreset")
    }
    assert len(set(first_scores.values())) == 1
