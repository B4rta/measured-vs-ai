from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SpecialistWeightedStackModel:
    empirical_model: object
    rf_model: object
    et_model: object | None
    base_weights: dict[str, float]
    tertiary_model: object | None = None
    scpt_model: object | None = None
    high_vs_model: object | None = None
    specialist_weights: dict[str, float] | None = None
    high_vs_threshold_mps: float = 350.0
    tertiary_label: str = "Tertiary"
    scpt_label: str = "SCPT"

    def _blend(self, base: np.ndarray, specialist: np.ndarray, weight: float) -> np.ndarray:
        return (1.0 - weight) * base + weight * specialist

    def predict_log_components(self, X: pd.DataFrame) -> pd.DataFrame:
        pred = pd.DataFrame(index=X.index)
        pred["pred_log_empirical"] = self.empirical_model.predict_log_vs(X)
        pred["pred_log_rf"] = self.rf_model.predict_log_vs(X)
        if self.et_model is not None:
            pred["pred_log_et"] = self.et_model.predict_log_vs(X)
        else:
            pred["pred_log_et"] = np.nan

        pred["pred_log_base_stack"] = (
            self.base_weights.get("empirical", 0.0) * pred["pred_log_empirical"]
            + self.base_weights.get("rf", 0.0) * pred["pred_log_rf"]
            + self.base_weights.get("et", 0.0) * pred["pred_log_et"].fillna(0.0)
        )
        pred["pred_log_specialist_weighted_stack"] = pred["pred_log_base_stack"].to_numpy()

        spec_w = self.specialist_weights or {"tertiary": 0.0, "scpt": 0.0, "high_vs": 0.0}
        pred["pred_log_tertiary_specialist"] = np.nan
        pred["pred_log_scpt_specialist"] = np.nan
        pred["pred_log_high_vs_specialist"] = np.nan
        pred["used_tertiary_specialist"] = False
        pred["used_scpt_specialist"] = False
        pred["used_high_vs_specialist"] = False

        if self.tertiary_model is not None and spec_w.get("tertiary", 0.0) > 0:
            mask = X["geo_age"].astype(str).eq(self.tertiary_label).to_numpy()
            if mask.any():
                ter_log = self.tertiary_model.predict_log_vs(X.loc[mask])
                pred.loc[mask, "pred_log_tertiary_specialist"] = ter_log
                pred.loc[mask, "pred_log_specialist_weighted_stack"] = self._blend(
                    pred.loc[mask, "pred_log_specialist_weighted_stack"].to_numpy(),
                    ter_log,
                    spec_w["tertiary"],
                )
                pred.loc[mask, "used_tertiary_specialist"] = True

        if self.scpt_model is not None and spec_w.get("scpt", 0.0) > 0 and "test_method" in X.columns:
            mask = X["test_method"].astype(str).eq(self.scpt_label).to_numpy()
            if mask.any():
                scpt_log = self.scpt_model.predict_log_vs(X.loc[mask])
                pred.loc[mask, "pred_log_scpt_specialist"] = scpt_log
                pred.loc[mask, "pred_log_specialist_weighted_stack"] = self._blend(
                    pred.loc[mask, "pred_log_specialist_weighted_stack"].to_numpy(),
                    scpt_log,
                    spec_w["scpt"],
                )
                pred.loc[mask, "used_scpt_specialist"] = True

        if self.high_vs_model is not None and spec_w.get("high_vs", 0.0) > 0:
            gate = np.exp(pred["pred_log_base_stack"].to_numpy()) >= float(self.high_vs_threshold_mps)
            if gate.any():
                high_log = self.high_vs_model.predict_log_vs(X.loc[gate])
                pred.loc[gate, "pred_log_high_vs_specialist"] = high_log
                pred.loc[gate, "pred_log_specialist_weighted_stack"] = self._blend(
                    pred.loc[gate, "pred_log_specialist_weighted_stack"].to_numpy(),
                    high_log,
                    spec_w["high_vs"],
                )
                pred.loc[gate, "used_high_vs_specialist"] = True

        return pred

    def predict_log_vs(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_log_components(X)["pred_log_specialist_weighted_stack"].to_numpy()

    def predict_vs(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_vs(X))
