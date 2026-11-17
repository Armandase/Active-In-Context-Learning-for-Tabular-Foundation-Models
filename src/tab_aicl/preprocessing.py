"""Preprocessing pipeline of the paper (Section III-B), fitted once on the whole pool."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler


def as_frame(X) -> pd.DataFrame:
    """X as a DataFrame; mixed column names (e.g. ints and strings, which scikit-learn rejects) become strings."""
    X = pd.DataFrame(X)
    if len({type(name) for name in X.columns}) > 1:
        X = X.set_axis(X.columns.astype(str), axis=1)
    return X


def infer_categorical_features(X, max_categories: int = 20) -> list[int]:
    """Positions of the categorical columns when no metadata tells them.

    Non-numeric and boolean columns are categorical, as are the integer-valued
    columns with fewer than ``max_categories`` distinct values.
    """
    X = pd.DataFrame(X)
    categorical = []
    for position in range(X.shape[1]):
        column = X.iloc[:, position]
        if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
            categorical.append(position)
            continue
        values = column.dropna()
        if values.nunique() < max_categories and (values % 1 == 0).all():
            categorical.append(position)
    return categorical


def _as_float(X) -> pd.DataFrame:
    """Numerical columns as float, so that pandas' missing values (pd.NA) become NaN."""
    return pd.DataFrame(X).astype(np.float64)


def _as_object(X) -> pd.DataFrame:
    """Categorical columns as objects with NaN for every missing value (None, pd.NA, NaN)."""
    X = pd.DataFrame(X).astype(object)
    return X.where(X.notna(), np.nan)


def make_preprocessor(X, categorical_features="auto", max_categories: int = 20) -> ColumnTransformer:
    """Build the preprocessing of the paper, to fit on the whole pool.

    Numerical features are mean-imputed and standardised, categorical ones
    are mode-imputed and ordinal-encoded (unseen categories become -1).
    ``categorical_features`` is ``"auto"`` (see ``infer_categorical_features``)
    or the positions of the categorical columns, e.g. from dataset metadata.
    """
    X = pd.DataFrame(X)
    if isinstance(categorical_features, str) and categorical_features == "auto":
        categorical = infer_categorical_features(X, max_categories)
    else:
        categorical = sorted(int(position) for position in categorical_features)
    numerical = [position for position in range(X.shape[1]) if position not in set(categorical)]
    return ColumnTransformer(
        [
            ("numerical", make_pipeline(FunctionTransformer(_as_float), SimpleImputer(strategy="mean"), StandardScaler()), numerical),
            (
                "categorical",
                make_pipeline(
                    FunctionTransformer(_as_object),
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                categorical,
            ),
        ]
    )


def fit_preprocessor(preprocessor, X: pd.DataFrame, categorical_features="auto"):
    """``preprocessor`` fitted on X.

    ``"auto"`` builds ``make_preprocessor(X, categorical_features)``, a
    transformer is cloned, None keeps X as is.
    """
    if preprocessor is None:
        return None
    if isinstance(preprocessor, str):
        if preprocessor != "auto":
            raise ValueError(f"preprocessor must be 'auto', None or a transformer, got {preprocessor!r}.")
        preprocessor = make_preprocessor(X, categorical_features)
    else:
        preprocessor = clone(preprocessor)
    return preprocessor.fit(X)


def transform(preprocessor, X: pd.DataFrame) -> np.ndarray:
    """X transformed by the fitted ``preprocessor`` (as is if None), as float32."""
    return np.asarray(X.to_numpy() if preprocessor is None else preprocessor.transform(X), dtype=np.float32)
