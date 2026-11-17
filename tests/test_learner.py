import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading

from tab_aicl import ActiveLearner, unlabeled_mask
from tab_aicl.baselines import label_spreading


def pool(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            "x0": np.r_[rng.randn(n // 2) - 2, rng.randn(n - n // 2) + 2],
            "x1": rng.randn(n),
            "color": rng.choice(["red", "green", "blue"], n),
        }
    )
    y = np.array(["cat"] * (n // 2) + ["dog"] * (n - n // 2), dtype=object)
    return X, y


def partial(y, labeled):
    y_partial = np.full(len(y), None, dtype=object)
    y_partial[labeled] = y[labeled]
    return y_partial


def test_unlabeled_mask():
    assert unlabeled_mask([0, -1, np.nan, None, "a", "-1", 2.0]).tolist() == [False, True, True, True, False, False, False]


@pytest.mark.parametrize("strategy", ["random", "margin", "coreset", "hybrid", "proxy_hybrid"])
def test_labelling_loop(strategy):
    X, y = pool()
    learner = ActiveLearner(LogisticRegression(), strategy=strategy, batch_size=7, random_state=0)
    learner.fit(X, partial(y, [0, 119]))
    seen = {0, 119}
    for _ in range(3):
        indices = learner.query()
        assert len(indices) == 7 and len(set(indices)) == 7
        assert not seen & set(indices)
        seen |= set(indices)
        learner.label(indices, y[indices])
    assert learner.n_labeled_ == 23
    assert list(learner.classes_) == ["cat", "dog"]
    assert learner.predict_proba(X).shape == (120, 2)
    assert set(learner.predict(X)) <= {"cat", "dog"}
    assert learner.score(X, y) > 0.9


def test_queries_are_reproducible():
    X, y = pool()
    queries = []
    for _ in range(2):
        learner = ActiveLearner(LogisticRegression(), strategy="hybrid", random_state=3).fit(X, partial(y, [0, 119]))
        queries.append(learner.query())
    np.testing.assert_array_equal(*queries)


def test_without_labels_the_first_query_is_random():
    X, y = pool()
    learner = ActiveLearner(LogisticRegression(), strategy="margin", batch_size=5, random_state=0).fit(X, np.full(len(y), -1))
    assert learner.n_labeled_ == 0 and len(learner.classes_) == 0
    assert len(set(learner.query())) == 5
    with pytest.raises(ValueError, match="two classes"):
        learner.predict(X)


def test_a_single_class_falls_back_to_coreset():
    X, y = pool()
    learner = ActiveLearner(LogisticRegression(), strategy="hybrid", batch_size=5, random_state=0).fit(X, partial(y, [0]))
    assert not hasattr(learner, "estimator_")
    with pytest.warns(UserWarning, match="coreset"):
        indices = learner.query()
    assert len(set(indices)) == 5


def test_the_last_query_returns_what_remains_of_the_pool():
    X, y = pool(20)
    learner = ActiveLearner(LogisticRegression(), strategy="coreset", batch_size=8, random_state=0).fit(X, partial(y, np.arange(15)))
    assert sorted(learner.query()) == [15, 16, 17, 18, 19]
    learner.label(np.arange(15, 20), y[15:])
    assert len(learner.query()) == 0


def test_invalid_labels_are_rejected():
    X, y = pool()
    learner = ActiveLearner(LogisticRegression(), random_state=0).fit(X, partial(y, [0, 119]))
    with pytest.raises(ValueError, match="cannot be -1"):
        learner.label([1, 2], ["cat", None])
    with pytest.raises(ValueError, match="2 indices"):
        learner.label([1, 2], ["cat"])


def test_semi_supervised_estimators_see_the_whole_pool():
    X, y = pool()
    estimator = label_spreading()  # sklearn's LabelSpreading would predict NaN where no label reaches
    assert isinstance(estimator, LabelSpreading)
    learner = ActiveLearner(estimator, strategy="random", random_state=0).fit(X, partial(y, [0, 119]))
    assert learner.estimator_.transduction_.shape == (120,)
    proba = learner.predict_proba(X)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_mixed_column_names():
    X, y = pool()
    X.columns = [0, 1, "color"]
    learner = ActiveLearner(LogisticRegression(), random_state=0).fit(X, partial(y, [0, 119]))
    assert learner.predict(X.head(2)).shape == (2,)


def test_numeric_pool_without_preprocessing():
    X, y = pool()
    X = X[["x0", "x1"]].to_numpy()
    learner = ActiveLearner(LogisticRegression(), strategy="coreset", preprocessor=None, random_state=0).fit(X, partial(y, [0, 119]))
    np.testing.assert_allclose(learner.X_, X.astype(np.float32))
    assert len(learner.query()) == 10
