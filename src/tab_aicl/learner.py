"""Pool-based active learner: the labelled samples are the context of an in-context classifier."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .preprocessing import as_frame, fit_preprocessor, transform
from .strategies import CoresetSampling, QueryStrategy, RandomSampling, make_strategy

# fitted on the whole pool, with -1 for the unlabelled samples
SEMI_SUPERVISED = (LabelPropagation, LabelSpreading, SelfTrainingClassifier)


def unlabeled_mask(y) -> np.ndarray:
    """True where y holds no label: -1, NaN or None."""
    y = pd.Series(np.asarray(y, dtype=object))
    return (y.isna() | y.map(lambda label: not isinstance(label, str) and label == -1)).to_numpy(dtype=bool)


def default_estimator(random_state=None):
    """TabPFN with the configuration of the paper (n_estimators=32)."""
    from tabpfn import TabPFNClassifier  # imported here so that other classifiers work without tabpfn

    return TabPFNClassifier(n_estimators=32, random_state=random_state)


class ActiveLearner(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Classifier that chooses which samples of a pool to label next.

    Parameters
    ----------
    estimator : classifier with ``predict_proba``, default TabPFNClassifier(n_estimators=32)
        Refit on the labelled samples after each batch; with TabPFN this only
        replaces the context. The labels it receives are always encoded as
        0..n_classes-1. Estimators of ``sklearn.semi_supervised`` are fitted
        on the whole pool, with -1 for the unlabelled samples.
    strategy : str or QueryStrategy, default "hybrid"
        One of ``STRATEGIES`` ("random", "margin", "coreset", "hybrid",
        "proxy_hybrid") or a strategy instance, e.g.
        ``ProxyHybridSampling(filter_ratio=0.1)``.
    batch_size : int, default 10
        Number of samples returned by ``query``.
    preprocessor : "auto", None or transformer, default "auto"
        Fitted on the whole pool, labelled or not, before any query.
        ``"auto"`` is ``make_preprocessor``; None expects numeric features
        already on comparable scales (the diversity steps use Euclidean
        distances).
    categorical_features : "auto" or list of int, default "auto"
        Positions of the categorical columns, used by ``preprocessor="auto"``.
        ``"auto"`` infers them, see ``infer_categorical_features``.
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

    def __init__(
        self,
        estimator=None,
        strategy="hybrid",
        batch_size: int = 10,
        preprocessor="auto",
        categorical_features="auto",
        random_state=None,
    ):
        self.estimator = estimator
        self.strategy = strategy
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.categorical_features = categorical_features
        self.random_state = random_state

    def fit(self, X, y) -> ActiveLearner:
        """Store the pool X and its partial labels y (-1, NaN or None where unlabelled)."""
        X = as_frame(X)
        y = np.asarray(y, dtype=object).copy()
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} samples and y has {len(y)}.")
        self.random_state_ = check_random_state(self.random_state)
        self.strategy_: QueryStrategy = make_strategy(self.strategy)
        self.preprocessor_ = fit_preprocessor(self.preprocessor, X, self.categorical_features)
        self.X_ = transform(self.preprocessor_, X)
        self.y_ = y
        self.labeled_mask_ = ~unlabeled_mask(y)
        self._refit()
        return self

    def query(self, batch_size: int | None = None) -> np.ndarray:
        """Return the positions, in the pool given to ``fit``, of the next samples to label.

        Without any label the samples are drawn at random; with a single
        labelled class the classifier cannot be fitted, so the strategies that
        need it fall back to coreset sampling.
        """
        check_is_fitted(self, "X_")
        pool = np.flatnonzero(~self.labeled_mask_)
        batch_size = min(self.batch_size if batch_size is None else batch_size, len(pool))
        if batch_size <= 0:
            return np.array([], dtype=int)
        strategy = self.strategy_
        if self.n_labeled_ == 0:
            strategy = RandomSampling()
        elif len(self.classes_) < 2 and strategy.needs_model:
            warnings.warn(
                "A single class is labelled, so the classifier cannot be fitted: querying with coreset sampling until a second class is found.",
                stacklevel=2,
            )
            strategy = CoresetSampling()
        estimator = getattr(self, "estimator_", None)
        positions = strategy.select(estimator, self.X_[self.labeled_mask_], self._y_encoded, self.X_[pool], batch_size, self.random_state_)
        return pool[positions]

    def label(self, indices, y) -> ActiveLearner:
        """Record the labels y of the pool samples at ``indices``, then refit the classifier."""
        check_is_fitted(self, "X_")
        indices = np.atleast_1d(np.asarray(indices, dtype=int))
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
        return self.estimator_.predict_proba(transform(self.preprocessor_, as_frame(X)))

    def predict(self, X) -> np.ndarray:
        """Most probable class of each sample."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _make_estimator(self):
        if self.estimator is not None:
            return clone(self.estimator)
        return default_estimator(random_state=self.random_state_.randint(2**31 - 1))

    def _refit(self) -> None:
        """Fit the classifier on the labelled samples; with TabPFN this only replaces the context."""
        self.n_labeled_ = int(self.labeled_mask_.sum())
        labels = np.array(self.y_[self.labeled_mask_].tolist())
        # the labels are encoded so that any classifier accepts a subset of the classes (XGBoost wants 0..K-1)
        encoder = LabelEncoder().fit(labels)
        self.classes_ = encoder.classes_
        self._y_encoded = encoder.transform(labels) if self.n_labeled_ else np.array([], dtype=int)
        if len(self.classes_) < 2:
            if hasattr(self, "estimator_"):
                del self.estimator_
            return
        estimator = self._make_estimator()
        if isinstance(estimator, SEMI_SUPERVISED):
            y_pool = np.full(len(self.X_), -1, dtype=int)
            y_pool[self.labeled_mask_] = self._y_encoded
            self.estimator_ = estimator.fit(self.X_, y_pool)
        else:
            self.estimator_ = estimator.fit(self.X_[self.labeled_mask_], self._y_encoded)
