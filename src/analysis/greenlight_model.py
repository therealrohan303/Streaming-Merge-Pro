"""Stacked greenlight predictor wrapper.

Lives in an importable module so the pickle can round-trip between the training
script (`scripts/11_train_greenlight_models.py`) and the Streamlit loader
(`load_greenlight_model` in src/data/loaders.py).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


class GreenlightStackedModel:
    """Blend of HistGradientBoostingRegressor (structured + SVD features) and
    a Ridge regressor on the SVD subspace alone.

    The Ridge component sharpens the response to the concept description, so
    different concept text actually moves the prediction at inference time.
    """

    def __init__(self) -> None:
        self.gbm_: Optional[HistGradientBoostingRegressor] = None
        self.ridge_: Optional[Ridge] = None
        self.vectorizer_ = None
        self.svd_: Optional[TruncatedSVD] = None
        self.feature_names_: list[str] = []
        self.svd_cols_: list[str] = []
        self.tone_keys_: list[str] = []
        self.genre_pairs_: list[tuple[str, str]] = []
        self.genre_names_: list[str] = []
        self.content_type_: str = "Movie"
        self.cv_rmse_: Optional[float] = None
        self.cv_rmse_std_: Optional[float] = None
        self.baseline_rmse_: Optional[float] = None
        self.global_mean_: Optional[float] = None
        self.training_size_: int = 0
        self.genre_percentiles_: dict = {}
        self.ridge_blend_: float = 0.20

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names_]
        pred = self.gbm_.predict(X)
        if self.ridge_ is not None and self.svd_cols_:
            pred_ridge = self.ridge_.predict(X[self.svd_cols_])
            pred = (1 - self.ridge_blend_) * pred + self.ridge_blend_ * pred_ridge
        return pred
