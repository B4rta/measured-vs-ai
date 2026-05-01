from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalResult:
    nominal_coverage: float
    alpha: float
    quantile_mps: float
    empirical_coverage: float
    mean_width_mps: float
    median_width_mps: float
    n: int


def conformal_abs_quantile(abs_errors: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal absolute residual quantile.

    Uses the standard conservative quantile level ceil((n + 1) * (1-alpha)) / n.
    This is intentionally model-agnostic: it can be applied to OOF predictions from
    any estimator in this repository.
    """
    errors = np.asarray(abs_errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        raise ValueError("No finite residuals available for conformal calibration.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    n = errors.size
    q_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(errors, q_level, method="higher"))


def add_absolute_conformal_intervals(
    df: pd.DataFrame,
    *,
    y_true_col: str = "vs_meas_mps",
    y_pred_col: str = "pred_vs_mps_final",
    alphas: Iterable[float] = (0.10, 0.20),
    prefix: str = "pred_vs_mps_final",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add symmetric conformal intervals around Vs predictions.

    Parameters
    ----------
    df:
        DataFrame containing measured and predicted Vs values in m/s.
    y_true_col, y_pred_col:
        Column names for true and predicted Vs.
    alphas:
        Miscoverage levels. alpha=0.10 corresponds to a nominal 90% interval.
    prefix:
        Prefix used for the created lower/upper columns.

    Returns
    -------
    out_df, summary_df
        Updated predictions and interval summary table.
    """
    out = df.copy()
    y_true = out[y_true_col].to_numpy(dtype=float)
    y_pred = out[y_pred_col].to_numpy(dtype=float)
    abs_err = np.abs(y_true - y_pred)
    rows = []
    for alpha in alphas:
        nominal = 1.0 - float(alpha)
        pct = int(round(nominal * 100))
        q = conformal_abs_quantile(abs_err, alpha=float(alpha))
        lower = np.maximum(0.0, y_pred - q)
        upper = y_pred + q
        out[f"{prefix}_lower_{pct}"] = lower
        out[f"{prefix}_upper_{pct}"] = upper
        out[f"{prefix}_half_width_{pct}"] = q
        covered = (y_true >= lower) & (y_true <= upper)
        rows.append(
            {
                "nominal_coverage": nominal,
                "alpha": float(alpha),
                "quantile_mps": q,
                "empirical_coverage": float(np.mean(covered)),
                "mean_width_mps": float(np.mean(upper - lower)),
                "median_width_mps": float(np.median(upper - lower)),
                "n": int(np.isfinite(abs_err).sum()),
            }
        )
    return out, pd.DataFrame(rows)


def conformal_subset_summary(
    df: pd.DataFrame,
    *,
    y_true_col: str = "vs_meas_mps",
    prefix: str = "pred_vs_mps_final",
    nominal_pct: int = 90,
) -> pd.DataFrame:
    """Coverage summary for common manuscript subsets."""
    lower_col = f"{prefix}_lower_{nominal_pct}"
    upper_col = f"{prefix}_upper_{nominal_pct}"
    if lower_col not in df.columns or upper_col not in df.columns:
        raise ValueError(f"Missing interval columns {lower_col}/{upper_col}")
    masks = {
        "all": np.ones(len(df), dtype=bool),
        "Quaternary": df.get("geo_age", pd.Series(index=df.index, dtype=object)).astype(str).eq("Quaternary").to_numpy(),
        "Tertiary": df.get("geo_age", pd.Series(index=df.index, dtype=object)).astype(str).eq("Tertiary").to_numpy(),
        "MASW": df.get("test_method", pd.Series(index=df.index, dtype=object)).astype(str).eq("MASW").to_numpy(),
        "SCPT": df.get("test_method", pd.Series(index=df.index, dtype=object)).astype(str).eq("SCPT").to_numpy(),
        "vs_ge_400": df[y_true_col].to_numpy(dtype=float) >= 400.0,
        "vs_lt_400": df[y_true_col].to_numpy(dtype=float) < 400.0,
    }
    rows = []
    y_true = df[y_true_col].to_numpy(dtype=float)
    lower = df[lower_col].to_numpy(dtype=float)
    upper = df[upper_col].to_numpy(dtype=float)
    width = upper - lower
    for subset, mask in masks.items():
        if mask.sum() == 0:
            continue
        covered = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
        rows.append(
            {
                "subset": subset,
                "nominal_pct": nominal_pct,
                "n": int(mask.sum()),
                "empirical_coverage": float(np.mean(covered)),
                "mean_width_mps": float(np.mean(width[mask])),
                "median_width_mps": float(np.median(width[mask])),
            }
        )
    return pd.DataFrame(rows)
