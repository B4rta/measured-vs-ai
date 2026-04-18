from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_CAT = "__MISSING__"


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_log(series: pd.Series, eps: float = 1e-8) -> pd.Series:
    values = safe_numeric(series)
    return np.log(np.clip(values, eps, None))


def build_empirical_design(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_qt_mpa"] = safe_log(df["qt_mpa"])
    out["log_fs_mpa"] = safe_log(df["fs_mpa"])
    out["log_sigma_eff_kpa"] = safe_log(df["sigma_eff_kpa"])
    out["log_depth_m"] = safe_log(df["z_mid_m"])
    out["rf_pct"] = safe_numeric(df["rf_pct"])
    out["bq"] = safe_numeric(df["bq"])
    out["z_below_gwl_m"] = safe_numeric(df["z_below_gwl_m"])
    out["geo_age"] = df["geo_age"].astype(object).where(df["geo_age"].notna(), MISSING_CAT).astype(str)
    return out


def engineer_profile_features(
    df: pd.DataFrame,
    *,
    include_test_method_for_trees: bool = True,
    profile_windows: list[int] | tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    out = df.copy()
    if "geo_age" in out.columns:
        out["geo_age"] = out["geo_age"].astype(object).where(out["geo_age"].notna(), MISSING_CAT).astype(str)
    if "test_method" in out.columns:
        out["test_method"] = out["test_method"].astype(object).where(out["test_method"].notna(), MISSING_CAT).astype(str)

    out = out.sort_values(["group_cpt", "z_mid_m"]).copy()

    for col in ["qc_mpa", "qt_mpa", "fs_mpa", "u2_mpa", "rf_pct", "sigma_eff_kpa", "sigma_v_kpa", "z_mid_m", "gwl_m"]:
        base = safe_numeric(out[col])
        if col in {"u2_mpa", "rf_pct"}:
            base = base + 1e-6
        out[f"log_{col}"] = safe_log(base)

    out["qt_over_sigma_eff"] = safe_numeric(out["qt_mpa"]) * 1000.0 / np.clip(safe_numeric(out["sigma_eff_kpa"]), 1e-8, None)
    out["qc_over_sigma_eff"] = safe_numeric(out["qc_mpa"]) * 1000.0 / np.clip(safe_numeric(out["sigma_eff_kpa"]), 1e-8, None)
    out["fs_over_qt"] = safe_numeric(out["fs_mpa"]) / np.clip(safe_numeric(out["qt_mpa"]), 1e-8, None)
    out["depth_over_gwl"] = safe_numeric(out["z_mid_m"]) / np.clip(safe_numeric(out["gwl_m"]), 1e-8, None)
    out["below_gwl_ratio"] = safe_numeric(out["z_below_gwl_m"]) / np.clip(safe_numeric(out["z_mid_m"]), 1e-8, None)

    grp = out.groupby("group_cpt", group_keys=False)
    out["depth_rank_pct"] = grp["z_mid_m"].rank(pct=True)
    out["z_rel_minmax"] = (
        (safe_numeric(out["z_mid_m"]) - grp["z_mid_m"].transform("min"))
        / np.clip(grp["z_mid_m"].transform("max") - grp["z_mid_m"].transform("min"), 1e-8, None)
    )

    base_cols = ["log_qc_mpa", "log_qt_mpa", "log_fs_mpa", "log_rf_pct", "log_sigma_eff_kpa"]
    for base in base_cols:
        out[f"{base}_diff1"] = grp[base].diff()
        for window in profile_windows:
            out[f"{base}_roll{window}"] = grp[base].transform(lambda s, w=window: s.rolling(w, center=True, min_periods=1).mean())
        first_roll = profile_windows[0]
        out[f"{base}_local_resid_w{first_roll}"] = out[base] - out[f"{base}_roll{first_roll}"]

    out["age_method"] = out["geo_age"].astype(str) + "__" + out["test_method"].astype(str)
    if not include_test_method_for_trees:
        out["test_method"] = MISSING_CAT
        out["age_method"] = out["geo_age"].astype(str) + "__" + MISSING_CAT

    return out.reset_index(drop=True)


def tree_feature_columns(df: pd.DataFrame, include_test_method_for_trees: bool = True) -> tuple[list[str], list[str]]:
    cat_cols = ["geo_age"]
    if include_test_method_for_trees:
        cat_cols += ["test_method", "age_method"]

    excluded = {
        "project", "cpt_id", "vs_meas_mps", "log_vs_meas", "group_project", "group_cpt",
    }
    feature_cols = [c for c in df.columns if c not in excluded]
    return feature_cols, cat_cols


def compute_sample_weights(
    df: pd.DataFrame,
    *,
    high_vs_anchor_mps: float = 320.0,
    high_vs_span_mps: float = 180.0,
    high_vs_boost: float = 0.5,
    tertiary_boost: float = 0.2,
    scpt_boost: float = 0.3,
) -> np.ndarray:
    y = safe_numeric(df["vs_meas_mps"]).to_numpy()
    w = np.ones(len(df), dtype=float)
    if high_vs_boost > 0:
        w += high_vs_boost * np.clip((y - high_vs_anchor_mps) / max(high_vs_span_mps, 1e-8), 0.0, 1.5)
    if tertiary_boost > 0 and "geo_age" in df.columns:
        w += tertiary_boost * df["geo_age"].astype(str).eq("Tertiary").to_numpy(dtype=float)
    if scpt_boost > 0 and "test_method" in df.columns:
        w += scpt_boost * df["test_method"].astype(str).eq("SCPT").to_numpy(dtype=float)
    return w
