from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WeightedStackModel:
    empirical_model: object
    rf_model: object
    et_model: object
    weights: dict[str, float]

    def fit(self, X: pd.DataFrame, y_log_vs: np.ndarray):
        self.empirical_model.fit(X, y_log_vs)
        self.rf_model.fit(X, y_log_vs)
        self.et_model.fit(X, y_log_vs)
        return self

    def predict_log_components(self, X: pd.DataFrame) -> pd.DataFrame:
        pred = pd.DataFrame(index=X.index)
        pred["pred_log_empirical"] = self.empirical_model.predict_log_vs(X)
        pred["pred_log_rf"] = self.rf_model.predict_log_vs(X)
        pred["pred_log_et"] = self.et_model.predict_log_vs(X)
        pred["pred_log_weighted_stack"] = (
            self.weights["empirical"] * pred["pred_log_empirical"]
            + self.weights["rf"] * pred["pred_log_rf"]
            + self.weights["et"] * pred["pred_log_et"]
        )
        return pred

    def predict_log_vs(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.predict_log_components(X)
        return pred["pred_log_weighted_stack"].to_numpy()

    def predict_vs(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_vs(X))
