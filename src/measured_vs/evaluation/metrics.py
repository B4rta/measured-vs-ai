from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics_vs(y_true_mps: np.ndarray, y_pred_mps: np.ndarray) -> dict:
    y_true_mps = np.asarray(y_true_mps, dtype=float)
    y_pred_mps = np.asarray(y_pred_mps, dtype=float)
    return {
        "rmse_mps": float(np.sqrt(mean_squared_error(y_true_mps, y_pred_mps))),
        "mae_mps": float(mean_absolute_error(y_true_mps, y_pred_mps)),
        "r2": float(r2_score(y_true_mps, y_pred_mps)) if len(y_true_mps) > 1 else float('nan'),
        "bias_mps": float(np.mean(y_pred_mps - y_true_mps)),
    }
