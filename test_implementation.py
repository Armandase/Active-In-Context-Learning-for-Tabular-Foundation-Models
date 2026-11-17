from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, pairwise_distances, pairwise_distances_argmin_min, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

EPSILON = 1e-12  # keeps the logarithm of a zero probability finite


def entropy(proba: np.ndarray) -> np.ndarray:
    """Predictive entropy of each row of class probabilities."""
    return -(proba * np.log(proba + EPSILON)).sum(axis=1)


def margin(proba: np.ndarray) -> np.ndarray:
    """Gap between the two most probable classes of each row, small when the model hesitates."""
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k highest scores."""
    return np.argsort(-scores, kind="stable")[:k]


def kmeans_representatives(X: np.ndarray, batch_size: int, random_state: np.random.RandomState) -> np.ndarray:
    """Cluster X into batch_size groups and return the position of the sample nearest to each centroid."""
    k = min(batch_size, len(X))
    centers = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X).cluster_centers_
    distances = pairwise_distances(centers, X)
    selected = []
    for row in distances:
        # unique selection: a sample already taken by a previous centroid is skipped
        row[selected] = np.inf
        selected.append(int(row.argmin()))
    return np.array(selected)


class QueryStrategy(BaseEstimator):
    """Acquisition rule: picks the unlabelled samples worth labelling.

    Subclass it and implement ``select`` to add a rule. ``needs_model`` tells
    whether the rule reads the predictions of the classifier.
    """

    needs_model = True

    def select(self, estimator, X_labeled: np.ndarray, y_labeled: np.ndarray, X_pool: np.ndarray, batch_size: int, random_state: np.random.RandomState) -> np.ndarray:
        """Return the positions in X_pool of batch_size samples to label.

        estimator is fitted on (X_labeled, y_labeled), all the features are preprocessed.
        """
        raise NotImplementedError


class RandomSampling(QueryStrategy):
    """Uniform sampling without replacement, the passive baseline."""

    needs_model = False

    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        return random_state.choice(len(X_pool), size=batch_size, replace=False)


class MarginSampling(QueryStrategy):
    """Uncertainty: the samples with the smallest margin between the top two classes."""

    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        return top_k(-margin(estimator.predict_proba(X_pool)), batch_size)


class CoresetSampling(QueryStrategy):
    """Diversity: greedy k-center, each pick is the sample farthest from the labelled and already picked ones."""

    needs_model = False

    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        # only the distance of each pool sample to its nearest center is kept: O(pool) memory
        _, nearest = pairwise_distances_argmin_min(X_pool, X_labeled)
        selected = []
        for _ in range(batch_size):
            pick = int(nearest.argmax())
            selected.append(pick)
            nearest = np.minimum(nearest, np.linalg.norm(X_pool - X_pool[pick], axis=1))
        return np.array(selected)


class HybridSampling(QueryStrategy):
    """Uncertainty then diversity: k-means over the most entropic part of the pool.

    The candidates are the ``candidate_fraction`` most entropic samples of the
    pool, at least ``min_candidate_batches`` times the batch size.
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
    """Hybrid sampling on a shortlist made by a cheap proxy classifier.

    The proxy (a balanced logistic regression by default) is fitted on the
    labelled samples and keeps the ``filter_ratio`` most entropic samples of
    the pool, clamped to [min_shortlist, max_shortlist]. The classifier then
    only scores this shortlist, and k-means runs over its
    ``diversity_batches`` x batch_size most entropic samples.
    """

    def __init__(self, proxy=None, filter_ratio: float = 0.05, min_shortlist: int = 200, max_shortlist: int = 2000, diversity_batches: int = 3):
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


STRATEGIES = {
    "random": RandomSampling,
    "margin": MarginSampling,
    "coreset": CoresetSampling,
    "hybrid": HybridSampling,
    "proxy_hybrid": ProxyHybridSampling,
}


def make_strategy(strategy, registry: dict[str, type] = STRATEGIES) -> QueryStrategy:
    """Strategy of a ``registry`` name, or the instance itself."""
    if not isinstance(strategy, str):
        return strategy
    if strategy not in registry:
        raise ValueError(f"Unknown strategy {strategy!r}, choose among {sorted(registry)}.")
    return registry[strategy]()


def make_preprocessor(X, categorical_features="auto", max_categories: int = 20) -> ColumnTransformer:
    """Build the preprocessing of the paper, to fit on the whole pool.

    Numerical features are mean-imputed and standardised, categorical ones
    are mode-imputed and ordinal-encoded. With ``"auto"``, the categorical
    features are the non-numeric columns and the integer-valued columns with
    fewer than ``max_categories`` distinct values; otherwise pass their
    positions.
    """
    X = pd.DataFrame(X)
    if isinstance(categorical_features, str) and categorical_features == "auto":
        categorical = []
        for position in range(X.shape[1]):
            column = X.iloc[:, position]
            if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
                categorical.append(position)
            else:
                values = column.dropna()
                if values.nunique() < max_categories and (values % 1 == 0).all():
                    categorical.append(position)
    else:
        categorical = list(categorical_features)
    numerical = [position for position in range(X.shape[1]) if position not in set(categorical)]
    return ColumnTransformer(
        [
            ("numerical", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), numerical),
            ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), categorical),
        ]
    )


def fit_preprocessor(preprocessor, X: pd.DataFrame):
    """``preprocessor`` fitted on X: ``make_preprocessor`` with "auto", a clone of a given transformer, None to keep X as is."""
    if preprocessor is None:
        return None
    preprocessor = make_preprocessor(X) if isinstance(preprocessor, str) else clone(preprocessor)
    return preprocessor.fit(X)


def transform(preprocessor, X: pd.DataFrame) -> np.ndarray:
    """X transformed by the fitted ``preprocessor`` (as is if None), as float32 for the distances."""
    return np.asarray(X.to_numpy() if preprocessor is None else preprocessor.transform(X), dtype=np.float32)


def unlabeled_mask(y) -> np.ndarray:
    """True where y holds no label: -1, NaN or None."""
    y = pd.Series(np.asarray(y, dtype=object))
    return (y.isna() | y.map(lambda label: not isinstance(label, str) and label == -1)).to_numpy(dtype=bool)


class ActiveLearner(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Classifier that chooses which samples of a pool to label next.

    Parameters
    ----------
    estimator : classifier with ``predict_proba``, default TabPFNClassifier(n_estimators=32)
        Refit on the labelled samples after each batch. The labels it receives
        are always encoded as 0..n_classes-1.
    strategy : str or QueryStrategy, default "hybrid"
        One of ``STRATEGIES`` or a strategy instance, e.g. ``ProxyHybridSampling(filter_ratio=0.1)``.
    batch_size : int, default 10
        Number of samples returned by ``query``.
    preprocessor : "auto", None or transformer, default "auto"
        Fitted on the whole pool, labelled or not. ``"auto"`` is
        ``make_preprocessor``; None expects numeric features already on
        comparable scales (the diversity steps use Euclidean distances).
    random_state : int, RandomState or None

    Attributes
    ----------
    X_ : preprocessed pool.
    y_ : labels of the pool, with -1/NaN/None where unknown.
    labeled_mask_ : boolean mask of the labelled samples of the pool.
    n_labeled_ : number of labelled samples.
    classes_ : classes seen among the labelled samples.
    estimator_ : classifier fitted on the labelled samples (once two classes are labelled).
    """

    def __init__(self, estimator=None, strategy="hybrid", batch_size: int = 10, preprocessor="auto", random_state=None):
        self.estimator = estimator
        self.strategy = strategy
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.random_state = random_state

    def fit(self, X, y) -> "ActiveLearner":
        """Store the pool X and its partial labels y (-1, NaN or None where unlabelled)."""
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=object).copy()
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} samples and y has {len(y)}.")
        self.random_state_ = check_random_state(self.random_state)
        self.strategy_ = make_strategy(self.strategy)
        self.preprocessor_ = fit_preprocessor(self.preprocessor, X)
        self.X_ = transform(self.preprocessor_, X)
        self.y_ = y
        self.labeled_mask_ = ~unlabeled_mask(y)
        self._refit()
        return self

    def query(self, batch_size: int | None = None) -> np.ndarray:
        """Return the indices, in the pool given to ``fit``, of the next samples to label."""
        check_is_fitted(self, "X_")
        pool = np.flatnonzero(~self.labeled_mask_)
        batch_size = min(self.batch_size if batch_size is None else batch_size, len(pool))
        if batch_size == 0:
            return pool
        strategy = self.strategy_
        if self.n_labeled_ == 0:
            strategy = RandomSampling()
        elif len(self.classes_) < 2 and strategy.needs_model:
            warnings.warn("A single class is labelled, so the classifier cannot be fitted: querying with coreset sampling until a second class is found.", stacklevel=2)
            strategy = CoresetSampling()
        estimator = getattr(self, "estimator_", None)
        positions = strategy.select(estimator, self.X_[self.labeled_mask_], self._y_encoded, self.X_[pool], batch_size, self.random_state_)
        return pool[positions]

    def label(self, indices, y) -> "ActiveLearner":
        """Record the labels y of the pool samples at indices, then refit the classifier."""
        check_is_fitted(self, "X_")
        indices = np.atleast_1d(indices)
        y = np.atleast_1d(np.asarray(y, dtype=object))
        if len(indices) != len(y):
            raise ValueError(f"Got {len(indices)} indices and {len(y)} labels.")
        if unlabeled_mask(y).any():
            raise ValueError("The labels given to label() cannot be -1, NaN or None.")
        self.y_[indices] = y
        self.labeled_mask_[indices] = True
        self._refit()
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, one column per class of ``classes_``."""
        check_is_fitted(self, "X_")
        if not hasattr(self, "estimator_"):
            raise ValueError("At least two classes must be labelled before predicting.")
        return self.estimator_.predict_proba(transform(self.preprocessor_, pd.DataFrame(X)))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _make_estimator(self):
        if self.estimator is not None:
            return clone(self.estimator)
        from tabpfn import TabPFNClassifier  # imported here so that other classifiers work without tabpfn

        return TabPFNClassifier(n_estimators=32, random_state=self.random_state_.randint(2**31 - 1))

    def _refit(self) -> None:
        """Fit the classifier on the labelled samples; with TabPFN this only replaces the context."""
        self.n_labeled_ = int(self.labeled_mask_.sum())
        # the labels are encoded so that any classifier accepts a subset of the classes (XGBoost wants 0..K-1)
        encoder = LabelEncoder().fit(np.array(self.y_[self.labeled_mask_].tolist()))
        self.classes_ = encoder.classes_
        self._y_encoded = encoder.transform(np.array(self.y_[self.labeled_mask_].tolist())) if self.n_labeled_ else np.array([], dtype=int)
        if len(self.classes_) >= 2:
            self.estimator_ = self._make_estimator().fit(self.X_[self.labeled_mask_], self._y_encoded)
        elif hasattr(self, "estimator_"):
            del self.estimator_


def normalized_aulc(n_labeled, scores) -> float:
    """Area under the learning curve divided by the labelling budget: the mean score over the acquisition."""
    n_labeled, scores = np.asarray(n_labeled, dtype=float), np.asarray(scores, dtype=float)
    return float(np.trapezoid(scores, n_labeled) / (n_labeled[-1] - n_labeled[0]))


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
    random_state=None,
) -> pd.DataFrame:
    """Simulate the labelling loop of the paper with y_pool as the oracle.

    Starts from one random sample per class, queries batch_size samples per
    round until ``budget`` labels (or ``max_pool_fraction`` of a small pool),
    and evaluates on the test set after each round. Returns one row per round
    with ``n_labeled``, Cohen's ``kappa`` and ``roc_auc`` (one-vs-rest, macro).
    """
    y_pool, y_test = np.asarray(y_pool), np.asarray(y_test)
    rng = check_random_state(random_state)
    classes = np.unique(y_pool)
    initial = [rng.choice(np.flatnonzero(y_pool == label)) for label in classes]
    budget = max(len(initial), min(budget, int(max_pool_fraction * len(y_pool))))

    y_partial = np.full(len(y_pool), None, dtype=object)
    y_partial[initial] = y_pool[initial]
    learner = ActiveLearner(estimator, strategy, batch_size, preprocessor, random_state=rng).fit(X_pool, y_partial)

    rows = []
    while True:
        proba = learner.predict_proba(X_test)
        kappa = cohen_kappa_score(y_test, learner.classes_[proba.argmax(axis=1)])
        rows.append({"n_labeled": learner.n_labeled_, "kappa": kappa, "roc_auc": _roc_auc(y_test, proba, classes)})
        if learner.n_labeled_ >= budget:
            break
        indices = learner.query(min(batch_size, budget - learner.n_labeled_))
        learner.label(indices, y_pool[indices])
    return pd.DataFrame(rows)


def _roc_auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """ROC AUC, one-vs-rest macro-averaged beyond two classes; NaN when it is undefined."""
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true == classes[1], proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
    except ValueError:  # a class is missing from the test set
        return float("nan")
