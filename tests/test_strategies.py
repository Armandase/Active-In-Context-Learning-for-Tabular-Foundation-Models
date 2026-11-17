import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tab_aicl.strategies import (
    STRATEGIES,
    CoresetSampling,
    HybridSampling,
    ProxyHybridSampling,
    entropy,
    kmeans_representatives,
    make_strategy,
    margin,
)


class RecordingClassifier:
    """Returns fixed probabilities per pool row and records how many rows it scored."""

    def __init__(self, proba_of_row):
        self.proba_of_row = proba_of_row
        self.scored = []

    def predict_proba(self, X):
        self.scored.append(len(X))
        return self.proba_of_row(X)


def blobs(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = np.r_[rng.randn(n // 2, 2) - 2, rng.randn(n - n // 2, 2) + 2].astype(np.float32)
    y = np.r_[np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)]
    return X, y


def test_entropy_and_margin():
    np.testing.assert_allclose(entropy(np.array([[0.5, 0.5], [1.0, 0.0]])), [np.log(2), 0.0], atol=1e-9)
    np.testing.assert_allclose(margin(np.array([[0.1, 0.6, 0.3], [0.5, 0.5, 0.0]])), [0.3, 0.0])


def test_kmeans_representatives_are_unique_even_with_duplicates():
    X = np.zeros((6, 2))
    selected = kmeans_representatives(X, 4, np.random.RandomState(0))
    assert len(set(selected)) == 4


def test_coreset_picks_the_farthest_sample_first():
    X_labeled = np.array([[0.0, 0.0]])
    X_pool = np.array([[1.0, 0.0], [10.0, 0.0], [-3.0, 0.0]])
    selected = CoresetSampling().select(None, X_labeled, None, X_pool, 2, np.random.RandomState(0))
    assert list(selected) == [1, 2]


def test_coreset_never_picks_twice_on_duplicated_pool():
    X_pool = np.zeros((5, 2))
    selected = CoresetSampling().select(None, np.zeros((1, 2)), None, X_pool, 5, np.random.RandomState(0))
    assert sorted(selected) == [0, 1, 2, 3, 4]


def test_hybrid_queries_among_the_most_entropic_half():
    X_pool, _ = blobs(100)
    # rows 0..49 are confidently classified, 50..99 are uncertain
    classifier = RecordingClassifier(lambda X: np.where((np.arange(len(X)) < 50)[:, None], [0.99, 0.01], [0.5, 0.5]))
    selected = HybridSampling().select(classifier, None, None, X_pool, 10, np.random.RandomState(0))
    assert len(set(selected)) == 10
    assert (selected >= 50).all()


def test_hybrid_candidates_are_at_least_two_batches():
    X_pool, _ = blobs(30)
    classifier = RecordingClassifier(lambda X: np.tile([0.5, 0.5], (len(X), 1)))
    selected = HybridSampling().select(classifier, None, None, X_pool, 12, np.random.RandomState(0))
    assert len(set(selected)) == 12  # N_cand = min(30, max(24, 15)) = 24 candidates for 12 picks


@pytest.mark.parametrize(("n_pool", "expected_shortlist"), [(1000, 200), (10_000, 500), (100_000, 2000), (150, 150)])
def test_proxy_hybrid_scores_only_the_clamped_shortlist(n_pool, expected_shortlist):
    rng = np.random.RandomState(0)
    X_pool = rng.randn(n_pool, 3).astype(np.float32)
    X_labeled = rng.randn(6, 3).astype(np.float32)
    y_labeled = np.array([0, 1, 0, 1, 0, 1])
    classifier = RecordingClassifier(lambda X: np.tile([0.5, 0.5], (len(X), 1)))
    selected = ProxyHybridSampling().select(classifier, X_labeled, y_labeled, X_pool, 10, rng)
    assert classifier.scored == [expected_shortlist]
    assert len(set(selected)) == 10


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_returns_distinct_positions_of_the_pool(name):
    X, y = blobs(200)
    labeled = np.r_[0, 1, 150, 151]
    pool = np.setdiff1d(np.arange(len(X)), labeled)
    estimator = LogisticRegression().fit(X[labeled], y[labeled])
    selected = make_strategy(name).select(estimator, X[labeled], y[labeled], X[pool], 15, np.random.RandomState(0))
    assert len(selected) == 15
    assert len(set(selected)) == 15
    assert ((0 <= selected) & (selected < len(pool))).all()


def test_make_strategy():
    instance = ProxyHybridSampling(filter_ratio=0.1)
    assert make_strategy(instance) is instance
    assert isinstance(make_strategy("coreset"), CoresetSampling)
    with pytest.raises(ValueError, match="Unknown strategy"):
        make_strategy("bald")
