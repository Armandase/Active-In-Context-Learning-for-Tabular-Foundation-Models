import numpy as np
import pandas as pd

from tab_aicl.preprocessing import infer_categorical_features, make_preprocessor


def frame():
    return pd.DataFrame(
        {
            "real": [0.5, 1.5, np.nan, 3.0, 2.5, 1.0],
            "small_int": [1, 2, 2, 3, 1, 2],
            "large_int": range(100, 106),
            "text": ["a", None, "b", "a", "a", "c"],
            "flag": [True, False, True, True, False, True],
        }
    )


def test_infer_categorical_features():
    X = frame()
    X["large_int"] = np.arange(0, 600, 100)  # 6 unique values, still integer-valued
    assert infer_categorical_features(X) == [1, 2, 3, 4]
    assert infer_categorical_features(X, max_categories=5) == [1, 3, 4]


def test_numerical_features_are_imputed_and_standardised():
    X = frame()
    Z = make_preprocessor(X, categorical_features=[3, 4]).fit_transform(X)
    numerical = Z[:, :3]
    assert not np.isnan(Z).any()
    np.testing.assert_allclose(numerical.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(numerical.std(axis=0), 1, atol=1e-12)


def test_missing_categories_are_imputed_with_the_mode():
    X = frame()
    Z = make_preprocessor(X, categorical_features=[3]).fit_transform(X)
    text = Z[:, -1]
    assert text[1] == text[0]  # None -> "a", the most frequent category
    assert len(set(text)) == 3


def test_pandas_missing_values_are_supported():
    X = pd.DataFrame(
        {
            "string": pd.array(["a", pd.NA, "b", "a"], dtype="string"),
            "nullable_int": pd.array([1, pd.NA, 3, 1000], dtype="Int64"),
            "category": pd.Categorical(["x", None, "y", "x"]),
        }
    )
    Z = make_preprocessor(X).fit_transform(X)
    assert Z.shape == (4, 3)
    assert not np.isnan(Z).any()


def test_unseen_categories_are_encoded_as_minus_one():
    X = frame()
    preprocessor = make_preprocessor(X, categorical_features=[3]).fit(X)
    X_new = X.head(1).assign(text="unseen")
    assert preprocessor.transform(X_new)[0, -1] == -1
