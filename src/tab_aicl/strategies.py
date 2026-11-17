"""Acquisition rules of Tab-AICL (Section III-C and Algorithms 1-4 of the paper)."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

EPSILON = 1e-12  # keeps the logarithm of a zero probability finite


def entropy(proba: np.ndarray) -> np.ndarray:
    """Predictive entropy H(x) = -sum_k p_k log(p_k + eps) of each row of class probabilities."""
    return -(proba * np.log(proba + EPSILON)).sum(axis=1)


def margin(proba: np.ndarray) -> np.ndarray:
    """Gap p_(1) - p_(2) between the two most probable classes of each row, small when the model hesitates."""
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k highest scores, ties broken by position."""
    return np.argsort(-scores, kind="stable")[:k]


def kmeans_representatives(X: np.ndarray, batch_size: int, random_state: np.random.RandomState) -> np.ndarray:
    """Cluster X into batch_size groups and return the position of the sample nearest to each centroid.

    Each sample is returned at most once (unique selection of Algorithm 2, line 8).
    """
    k = min(batch_size, len(X))
    with warnings.catch_warnings():
        # duplicated rows can leave fewer distinct clusters than k, the unique selection still returns k samples
        warnings.simplefilter("ignore", ConvergenceWarning)
        centers = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X).cluster_centers_
    distances = pairwise_distances(centers, X)
    selected: list[int] = []
    for row in distances:
        row[selected] = np.inf  # a sample already taken by a previous centroid is skipped
        selected.append(int(row.argmin()))
    return np.array(selected, dtype=int)


class QueryStrategy(BaseEstimator):
    """Acquisition rule: picks the unlabelled samples worth labelling.

    Subclass it and implement ``select`` to add a rule. ``needs_model`` tells
    whether the rule reads the predictions of the classifier.
    """

    needs_model = True

    def select(
        self,
        estimator,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_pool: np.ndarray,
        batch_size: int,
        random_state: np.random.RandomState,
    ) -> np.ndarray:
        """Return the positions in X_pool of batch_size distinct samples to label.

        ``estimator`` is fitted on (X_labeled, y_labeled), ``y_labeled`` is
        encoded as 0..n_classes-1 and all the features are preprocessed.
        ``batch_size`` never exceeds ``len(X_pool)``.
        """
        raise NotImplementedError


class RandomSampling(QueryStrategy):
    """Uniform sampling without replacement, the passive baseline (TabPFN-Random)."""

    needs_model = False

    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        return random_state.choice(len(X_pool), size=batch_size, replace=False)


class MarginSampling(QueryStrategy):
    """Uncertainty (Algorithm 1): the samples with the smallest margin between the top two classes."""

    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        return top_k(-margin(estimator.predict_proba(X_pool)), batch_size)


class CoresetSampling(QueryStrategy):
    """Diversity (Algorithm 4): greedy k-center in the preprocessed feature space.

    Each pick is the sample farthest from the labelled samples and from the
    samples already picked. Only the distance of each pool sample to its
    nearest center is kept, so the memory is O(|pool|).
    """

    needs_model = False

    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        _, nearest = pairwise_distances_argmin_min(X_pool, X_labeled)
        nearest = nearest.astype(np.float64)
        selected = []
        for _ in range(batch_size):
            pick = int(nearest.argmax())
            selected.append(pick)
            nearest = np.minimum(nearest, np.linalg.norm(X_pool - X_pool[pick], axis=1))
            nearest[pick] = -np.inf  # never picked twice, even when the pool has duplicated rows
        return np.array(selected, dtype=int)


class HybridSampling(QueryStrategy):
    """Uncertainty then diversity (Algorithm 2): k-means over the most entropic part of the pool.

    The candidates are the ``candidate_fraction`` most entropic samples of the
    pool, and at least ``min_candidate_batches`` times the batch size:
    N_cand = min(|U|, max(2B, floor(|U| / 2))) with the defaults.
    """

    def __init__(self, candidate_fraction: float = 0.5, min_candidate_batches: int = 2):
        self.candidate_fraction = candidate_fraction
        self.min_candidate_batches = min_candidate_batches

    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        n_pool = len(X_pool)
        n_candidates = min(n_pool, max(self.min_candidate_batches * batch_size, int(self.candidate_fraction * n_pool)))
        candidates = top_k(entropy(estimator.predict_proba(X_pool)), n_candidates)
        return candidates[kmeans_representatives(X_pool[candidates], batch_size, random_state)]


class ProxyHybridSampling(QueryStrategy):
    """Hybrid sampling on a shortlist made by a cheap proxy classifier (Algorithm 3).

    The proxy (a balanced logistic regression by default) is fitted on the
    labelled samples and keeps the ``filter_ratio`` most entropic samples of
    the pool, clamped to [min_shortlist, max_shortlist] (Eq. 1). The
    classifier then only scores this shortlist, and k-means runs over its
    ``diversity_batches`` x batch_size most entropic samples. This divides
    the number of classifier evaluations by about 1 / filter_ratio.
    """

    def __init__(
        self,
        proxy=None,
        filter_ratio: float = 0.05,
        min_shortlist: int = 200,
        max_shortlist: int = 2000,
        diversity_batches: int = 3,
    ):
        self.proxy = proxy
        self.filter_ratio = filter_ratio
        self.min_shortlist = min_shortlist
        self.max_shortlist = max_shortlist
        self.diversity_batches = diversity_batches

    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        n_pool = len(X_pool)
        proxy = LogisticRegression(class_weight="balanced", max_iter=1000) if self.proxy is None else clone(self.proxy)
        proxy.fit(X_labeled, y_labeled)
        n_shortlist = min(n_pool, max(self.min_shortlist, min(self.max_shortlist, int(self.filter_ratio * n_pool))))
        shortlist = top_k(entropy(proxy.predict_proba(X_pool)), n_shortlist)

        n_diverse = min(self.diversity_batches * batch_size, len(shortlist))
        final = shortlist[top_k(entropy(estimator.predict_proba(X_pool[shortlist])), n_diverse)]
        return final[kmeans_representatives(X_pool[final], batch_size, random_state)]


STRATEGIES: dict[str, type[QueryStrategy]] = {
    "random": RandomSampling,
    "margin": MarginSampling,
    "coreset": CoresetSampling,
    "hybrid": HybridSampling,
    "proxy_hybrid": ProxyHybridSampling,
}


def make_strategy(strategy, registry: dict[str, type[QueryStrategy]] = STRATEGIES) -> QueryStrategy:
    """Strategy of a ``registry`` name, or the instance itself."""
    if not isinstance(strategy, str):
        return strategy
    if strategy not in registry:
        raise ValueError(f"Unknown strategy {strategy!r}, choose among {sorted(registry)}.")
    return registry[strategy]()
