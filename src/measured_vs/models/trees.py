from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


@dataclass
class TreeProfileModel:
    model_type: str
    feature_columns: list[str]
    categorical_features: list[str]
    n_estimators: int
    max_features: str
    min_samples_leaf: int
    random_state: int = 42

    def __post_init__(self):
        self.imputer_ = SimpleImputer(strategy="median")
        self.category_maps_: dict[str, dict[str, int]] = {}
        self.model_ = None

    def _transform(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        work = X[self.feature_columns].copy()
        for col in self.categorical_features:
            series = work[col].astype(object).where(work[col].notna(), "__MISSING__").astype(str)
            if fit:
                values = list(pd.Index(series).unique())
                self.category_maps_[col] = {v: i for i, v in enumerate(values)}
            mapping = self.category_maps_.get(col, {})
            work[col] = series.map(mapping).fillna(-1).astype(float)
        if fit:
            return self.imputer_.fit_transform(work)
        return self.imputer_.transform(work)

    def _build_model(self):
        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=-1,
                random_state=self.random_state,
            )
        if self.model_type == "et":
            return ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=-1,
                random_state=self.random_state,
            )
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y_log_vs: np.ndarray, sample_weight: np.ndarray | None = None):
        Xt = self._transform(X, fit=True)
        self.model_ = self._build_model()
        self.model_.fit(Xt, y_log_vs, sample_weight=sample_weight)
        return self

    def predict_log_vs(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Tree model not fitted.")
        Xt = self._transform(X, fit=False)
        return self.model_.predict(Xt)

    def predict_vs(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_vs(X))

    def feature_importance(self) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Tree model not fitted.")
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model_.feature_importances_,
        }).sort_values("importance", ascending=False)
