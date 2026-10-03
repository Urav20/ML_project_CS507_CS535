"""Preprocessing. Every learned step lives inside a Pipeline, so it is fitted
on training data only (no leakage)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category code by how common it was in the TRAINING data.
    Used for the high-cardinality geo_level_2/3 ids. Unseen codes -> 0."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X.iloc[:, i].value_counts(normalize=True).to_dict()
                      for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        cols = [X.iloc[:, i].map(self.maps_[i]).fillna(0.0).to_numpy(dtype=float)
                for i in range(len(self.maps_))]
        return np.column_stack(cols)


def _ohe():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_distance_preprocessor():
    """For KNN and Neural Network (scale-sensitive models)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUM_COLS),
            ("cat", _ohe(), config.CAT_COLS + ["geo_level_1_id"]),
            ("geo_freq", Pipeline([("freq", FrequencyEncoder()),
                                   ("scale", StandardScaler())]),
             ["geo_level_2_id", "geo_level_3_id"]),
            ("bin", "passthrough", config.BIN_COLS),
        ],
        sparse_threshold=0,
    )


def build_tree_preprocessor():
    """For Random Forest: no scaling needed. One-hot only for the 8 text columns."""
    return ColumnTransformer(
        [
            ("cat", _ohe(), config.CAT_COLS),
            ("rest", "passthrough",
             config.GEO_COLS + config.NUM_COLS + config.BIN_COLS),
        ],
        sparse_threshold=0,
    )
