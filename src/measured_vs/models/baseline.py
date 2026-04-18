from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor

from measured_vs.data.features import build_empirical_design


@dataclass
class EmpiricalBaselineModel:
    alpha: float = 1e-4
    max_iter: int = 8000

    def __post_init__(self):
        self.numeric_columns_ = [
            "log_qt_mpa",
            "log_fs_mpa",
            "log_sigma_eff_kpa",
            "log_depth_m",
            "rf_pct",
            "bq",
            "z_below_gwl_m",
        ]
        self.categorical_columns_ = ["geo_age"]
        self.pipeline_ = None

    def fit(self, X: pd.DataFrame, y_log_vs: np.ndarray):
        design = build_empirical_design(X)
        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.numeric_columns_,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.categorical_columns_,
                ),
            ]
        )
        self.pipeline_ = Pipeline(
            steps=[
                ("preprocess", pre),
                ("regressor", HuberRegressor(alpha=self.alpha, max_iter=self.max_iter)),
            ]
        )
        self.pipeline_.fit(design, y_log_vs)
        return self

    def predict_log_vs(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Empirical baseline has not been fitted yet.")
        design = build_empirical_design(X)
        return self.pipeline_.predict(design)

    def predict_vs(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_vs(X))
