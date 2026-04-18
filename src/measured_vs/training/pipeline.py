from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import itertools
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from measured_vs.utils.config import load_config
from measured_vs.data.io import load_cleaned_data
from measured_vs.data.features import engineer_profile_features, tree_feature_columns, compute_sample_weights
from measured_vs.evaluation.metrics import regression_metrics_vs
from measured_vs.models.baseline import EmpiricalBaselineModel
from measured_vs.models.trees import TreeProfileModel
from measured_vs.models.specialist_stack import SpecialistWeightedStackModel


BASE_COMPONENTS = {
    "empirical_baseline": "pred_log_empirical",
    "random_forest_profile": "pred_log_rf",
    "extra_trees_profile": "pred_log_et",
    "base_weighted_stack": "pred_log_base_stack",
    "specialist_weighted_stack": "pred_log_specialist_weighted_stack",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fit_tree_model(config: dict, feature_columns: list[str], categorical_features: list[str], *, model_type: str, n_estimators: int, min_samples_leaf: int, random_state: int, train_df: pd.DataFrame, y_log: np.ndarray, sample_weight: np.ndarray | None):
    return TreeProfileModel(
        model_type=model_type,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        n_estimators=int(n_estimators),
        max_features=str(config["forest"]["max_features"]),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
    ).fit(train_df, y_log, sample_weight=sample_weight)


def _summary_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = oof["vs_meas_mps"].to_numpy()
    for name, col in BASE_COMPONENTS.items():
        if col in oof.columns and oof[col].notna().any():
            rows.append({"model": name, **regression_metrics_vs(y_true, np.exp(oof[col].to_numpy()))})
    return pd.DataFrame(rows).sort_values("rmse_mps")


def _compute_subset_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "all": np.ones(len(oof), dtype=bool),
        "Quaternary": oof["geo_age"].astype(str).eq("Quaternary").to_numpy(),
        "Tertiary": oof["geo_age"].astype(str).eq("Tertiary").to_numpy(),
        "MASW": oof["test_method"].astype(str).eq("MASW").to_numpy(),
        "SCPT": oof["test_method"].astype(str).eq("SCPT").to_numpy(),
        "vs_ge_400": oof["vs_meas_mps"].to_numpy() >= 400.0,
        "vs_lt_400": oof["vs_meas_mps"].to_numpy() < 400.0,
    }
    rows = []
    for subset, mask in masks.items():
        if mask.sum() == 0:
            continue
        y = oof.loc[mask, "vs_meas_mps"].to_numpy()
        for name, col in [("base_weighted_stack", "pred_log_base_stack"), ("specialist_weighted_stack", "pred_log_specialist_weighted_stack")]:
            if col not in oof.columns or not oof.loc[mask, col].notna().any():
                continue
            rows.append({"subset": subset, "model": name, **regression_metrics_vs(y, np.exp(oof.loc[mask, col].to_numpy()))})
    return pd.DataFrame(rows)


def _weight_grid(step: float):
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w_emp in grid:
        for w_rf in grid:
            w_et = 1.0 - w_emp - w_rf
            if w_et < -1e-9:
                continue
            if w_et < 0:
                w_et = 0.0
            yield {"empirical": float(w_emp), "rf": float(w_rf), "et": float(w_et)}


def _search_best_base_weights(oof: pd.DataFrame, step: float) -> tuple[dict[str, float], dict]:
    y_true = oof["vs_meas_mps"].to_numpy()
    best_weights = {"empirical": 1/3, "rf": 1/3, "et": 1/3}
    best_metrics = None
    best_rmse = np.inf
    for weights in _weight_grid(step):
        pred_log = (
            weights["empirical"] * oof["pred_log_empirical"].to_numpy()
            + weights["rf"] * oof["pred_log_rf"].to_numpy()
            + weights["et"] * oof["pred_log_et"].to_numpy()
        )
        metrics = regression_metrics_vs(y_true, np.exp(pred_log))
        if metrics["rmse_mps"] < best_rmse:
            best_rmse = metrics["rmse_mps"]
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def _specialist_grid(step: float, max_weight: float):
    grid = np.arange(0.0, max_weight + 1e-9, step)
    for w_ter, w_scpt, w_high in itertools.product(grid, grid, grid):
        yield {"tertiary": float(w_ter), "scpt": float(w_scpt), "high_vs": float(w_high)}


def _apply_specialist_blend(oof: pd.DataFrame, specialist_weights: dict[str, float]) -> np.ndarray:
    final_log = oof["pred_log_base_stack"].to_numpy().copy()

    w_ter = specialist_weights.get("tertiary", 0.0)
    if w_ter > 0 and "pred_log_tertiary_specialist" in oof.columns:
        mask = oof["eligible_tertiary_specialist"].fillna(False).to_numpy(dtype=bool) & oof["pred_log_tertiary_specialist"].notna().to_numpy()
        final_log[mask] = (1.0 - w_ter) * final_log[mask] + w_ter * oof.loc[mask, "pred_log_tertiary_specialist"].to_numpy()

    w_scpt = specialist_weights.get("scpt", 0.0)
    if w_scpt > 0 and "pred_log_scpt_specialist" in oof.columns:
        mask = oof["eligible_scpt_specialist"].fillna(False).to_numpy(dtype=bool) & oof["pred_log_scpt_specialist"].notna().to_numpy()
        final_log[mask] = (1.0 - w_scpt) * final_log[mask] + w_scpt * oof.loc[mask, "pred_log_scpt_specialist"].to_numpy()

    w_high = specialist_weights.get("high_vs", 0.0)
    if w_high > 0 and "pred_log_high_vs_specialist" in oof.columns:
        mask = oof["eligible_high_vs_specialist"].fillna(False).to_numpy(dtype=bool) & oof["pred_log_high_vs_specialist"].notna().to_numpy()
        final_log[mask] = (1.0 - w_high) * final_log[mask] + w_high * oof.loc[mask, "pred_log_high_vs_specialist"].to_numpy()

    return final_log


def _search_best_specialist_weights(oof: pd.DataFrame, step: float, max_weight: float) -> tuple[dict[str, float], dict]:
    y_true = oof["vs_meas_mps"].to_numpy()
    best_weights = {"tertiary": 0.0, "scpt": 0.0, "high_vs": 0.0}
    best_metrics = regression_metrics_vs(y_true, np.exp(oof["pred_log_base_stack"].to_numpy()))
    best_rmse = best_metrics["rmse_mps"]
    for weights in _specialist_grid(step, max_weight):
        pred_log = _apply_specialist_blend(oof, weights)
        metrics = regression_metrics_vs(y_true, np.exp(pred_log))
        if metrics["rmse_mps"] < best_rmse:
            best_rmse = metrics["rmse_mps"]
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def _make_weight_vector(config: dict, train_df: pd.DataFrame) -> np.ndarray | None:
    if not bool(config["weights"]["enable"]):
        return None
    return compute_sample_weights(
        train_df,
        high_vs_anchor_mps=float(config["weights"]["high_vs_anchor_mps"]),
        high_vs_span_mps=float(config["weights"]["high_vs_span_mps"]),
        high_vs_boost=float(config["weights"]["high_vs_boost"]),
        tertiary_boost=float(config["weights"]["tertiary_boost"]),
        scpt_boost=float(config["weights"]["scpt_boost"]),
    )


def run_training(config_path: str | Path = "configs/default.yaml"):
    config = load_config(config_path)
    labeled, unlabeled = load_cleaned_data(config["data"]["cleaned_dir"], prefer_parquet=bool(config["data"]["prefer_parquet"]))

    labeled = engineer_profile_features(
        labeled,
        include_test_method_for_trees=bool(config["features"]["include_test_method_for_trees"]),
        profile_windows=config["features"]["profile_windows"],
    )
    if unlabeled is not None and len(unlabeled) > 0:
        unlabeled = engineer_profile_features(
            unlabeled,
            include_test_method_for_trees=bool(config["features"]["include_test_method_for_trees"]),
            profile_windows=config["features"]["profile_windows"],
        )

    y_log = np.log(labeled["vs_meas_mps"].to_numpy())
    groups = labeled[config["splits"]["group_column"]].astype(str)
    splitter = GroupKFold(n_splits=int(config["splits"]["n_splits"]))
    tree_cols, tree_cat_cols = tree_feature_columns(
        labeled,
        include_test_method_for_trees=bool(config["features"]["include_test_method_for_trees"]),
    )

    oof = labeled[["project", "cpt_id", "group_project", "group_cpt", "geo_age", "test_method", "vs_meas_mps"]].copy()
    for col in [
        "pred_log_empirical", "pred_log_rf", "pred_log_et", "pred_log_base_stack",
        "pred_log_tertiary_specialist", "pred_log_scpt_specialist", "pred_log_high_vs_specialist",
        "pred_log_specialist_weighted_stack",
    ]:
        oof[col] = np.nan
    for col in ["eligible_tertiary_specialist", "eligible_scpt_specialist", "eligible_high_vs_specialist"]:
        oof[col] = False

    fold_rows = []

    # Pass 1: global base models
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(labeled, groups=groups), start=1):
        train_df = labeled.iloc[tr_idx].copy()
        valid_df = labeled.iloc[va_idx].copy()
        global_sw = _make_weight_vector(config, train_df)

        empirical = EmpiricalBaselineModel(
            alpha=float(config["baseline"]["alpha"]),
            max_iter=int(config["baseline"]["max_iter"]),
        ).fit(train_df, y_log[tr_idx])
        rf = _fit_tree_model(
            config,
            tree_cols,
            tree_cat_cols,
            model_type="rf",
            n_estimators=int(config["forest"]["n_estimators_rf"]),
            min_samples_leaf=int(config["forest"]["min_samples_leaf_rf"]),
            random_state=int(config["forest"]["random_state"]),
            train_df=train_df,
            y_log=y_log[tr_idx],
            sample_weight=global_sw,
        )
        et = _fit_tree_model(
            config,
            tree_cols,
            tree_cat_cols,
            model_type="et",
            n_estimators=int(config["forest"]["n_estimators_et"]),
            min_samples_leaf=int(config["forest"]["min_samples_leaf_et"]),
            random_state=int(config["forest"]["random_state"]) + 1,
            train_df=train_df,
            y_log=y_log[tr_idx],
            sample_weight=global_sw,
        )

        oof.loc[va_idx, "pred_log_empirical"] = empirical.predict_log_vs(valid_df)
        oof.loc[va_idx, "pred_log_rf"] = rf.predict_log_vs(valid_df)
        oof.loc[va_idx, "pred_log_et"] = et.predict_log_vs(valid_df)

        for name, col in [("empirical_baseline", "pred_log_empirical"), ("random_forest_profile", "pred_log_rf"), ("extra_trees_profile", "pred_log_et")]:
            metrics = regression_metrics_vs(valid_df["vs_meas_mps"].to_numpy(), np.exp(oof.loc[va_idx, col].to_numpy()))
            fold_rows.append({"fold": fold, "model": name, **metrics})

    base_weights, base_stack_metrics = _search_best_base_weights(oof, step=float(config["stack"]["base_weight_step"]))
    oof["pred_log_base_stack"] = (
        base_weights["empirical"] * oof["pred_log_empirical"]
        + base_weights["rf"] * oof["pred_log_rf"]
        + base_weights["et"] * oof["pred_log_et"]
    )

    # Pass 2: specialists
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(labeled, groups=groups), start=1):
        train_df = labeled.iloc[tr_idx].copy()
        valid_df = labeled.iloc[va_idx].copy()

        if bool(config["specialists"]["enable_tertiary"]):
            train_mask = train_df["geo_age"].astype(str).eq("Tertiary")
            valid_mask = valid_df["geo_age"].astype(str).eq("Tertiary")
            oof.loc[va_idx, "eligible_tertiary_specialist"] = valid_mask.to_numpy()
            if train_mask.sum() >= int(config["specialists"]["min_train_rows_tertiary"]) and valid_mask.any():
                spec_df = train_df.loc[train_mask].copy()
                spec_sw = _make_weight_vector(config, spec_df)
                ter_model = _fit_tree_model(
                    config,
                    tree_cols,
                    tree_cat_cols,
                    model_type=str(config["specialists"]["tertiary_model_type"]),
                    n_estimators=int(config["specialists"]["tertiary_n_estimators"]),
                    min_samples_leaf=int(config["specialists"]["tertiary_min_samples_leaf"]),
                    random_state=int(config["forest"]["random_state"]) + 10,
                    train_df=spec_df,
                    y_log=np.log(spec_df["vs_meas_mps"].to_numpy()),
                    sample_weight=spec_sw,
                )
                oof.loc[oof.index[va_idx][valid_mask.to_numpy()], "pred_log_tertiary_specialist"] = ter_model.predict_log_vs(valid_df.loc[valid_mask])

        if bool(config["specialists"]["enable_scpt"]) and bool(config["features"]["include_test_method_for_trees"]):
            train_mask = train_df["test_method"].astype(str).eq("SCPT")
            valid_mask = valid_df["test_method"].astype(str).eq("SCPT")
            oof.loc[va_idx, "eligible_scpt_specialist"] = valid_mask.to_numpy()
            if train_mask.sum() >= int(config["specialists"]["min_train_rows_scpt"]) and valid_mask.any():
                spec_df = train_df.loc[train_mask].copy()
                spec_sw = _make_weight_vector(config, spec_df)
                scpt_model = _fit_tree_model(
                    config,
                    tree_cols,
                    tree_cat_cols,
                    model_type=str(config["specialists"]["scpt_model_type"]),
                    n_estimators=int(config["specialists"]["scpt_n_estimators"]),
                    min_samples_leaf=int(config["specialists"]["scpt_min_samples_leaf"]),
                    random_state=int(config["forest"]["random_state"]) + 20,
                    train_df=spec_df,
                    y_log=np.log(spec_df["vs_meas_mps"].to_numpy()),
                    sample_weight=spec_sw,
                )
                oof.loc[oof.index[va_idx][valid_mask.to_numpy()], "pred_log_scpt_specialist"] = scpt_model.predict_log_vs(valid_df.loc[valid_mask])

        if bool(config["specialists"]["enable_high_vs"]):
            threshold = float(config["specialists"]["high_vs_threshold_mps"])
            train_mask = train_df["vs_meas_mps"].to_numpy() >= threshold
            valid_gate = np.exp(oof.loc[va_idx, "pred_log_base_stack"].to_numpy()) >= threshold
            oof.loc[va_idx, "eligible_high_vs_specialist"] = valid_gate
            if train_mask.sum() >= int(config["specialists"]["min_train_rows_high_vs"]) and valid_gate.any():
                spec_df = train_df.loc[train_mask].copy()
                spec_sw = _make_weight_vector(config, spec_df)
                high_model = _fit_tree_model(
                    config,
                    tree_cols,
                    tree_cat_cols,
                    model_type=str(config["specialists"]["high_model_type"]),
                    n_estimators=int(config["specialists"]["high_n_estimators"]),
                    min_samples_leaf=int(config["specialists"]["high_min_samples_leaf"]),
                    random_state=int(config["forest"]["random_state"]) + 30,
                    train_df=spec_df,
                    y_log=np.log(spec_df["vs_meas_mps"].to_numpy()),
                    sample_weight=spec_sw,
                )
                oof.loc[oof.index[va_idx][valid_gate], "pred_log_high_vs_specialist"] = high_model.predict_log_vs(valid_df.loc[valid_gate])

    specialist_weights, specialist_metrics = _search_best_specialist_weights(
        oof,
        step=float(config["specialists"]["blend_weight_step"]),
        max_weight=float(config["specialists"]["max_blend_weight"]),
    )
    oof["pred_log_specialist_weighted_stack"] = _apply_specialist_blend(oof, specialist_weights)

    for name, col in [("base_weighted_stack", "pred_log_base_stack"), ("specialist_weighted_stack", "pred_log_specialist_weighted_stack")]:
        metrics = regression_metrics_vs(oof["vs_meas_mps"].to_numpy(), np.exp(oof[col].to_numpy()))
        fold_rows.append({"fold": 0, "model": name, **metrics})

    summary_df = _summary_from_oof(oof)
    subset_metrics_df = _compute_subset_metrics(oof)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["outputs"]["root_dir"]) / f"{timestamp}_{config['experiment_name']}"
    _ensure_dir(run_dir)
    _ensure_dir(run_dir / "models")
    _ensure_dir(run_dir / "reports")
    _ensure_dir(run_dir / "predictions")

    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    global_sw = _make_weight_vector(config, labeled)
    empirical_full = EmpiricalBaselineModel(
        alpha=float(config["baseline"]["alpha"]),
        max_iter=int(config["baseline"]["max_iter"]),
    ).fit(labeled, y_log)
    rf_full = _fit_tree_model(
        config,
        tree_cols,
        tree_cat_cols,
        model_type="rf",
        n_estimators=int(config["forest"]["n_estimators_rf"]),
        min_samples_leaf=int(config["forest"]["min_samples_leaf_rf"]),
        random_state=int(config["forest"]["random_state"]),
        train_df=labeled,
        y_log=y_log,
        sample_weight=global_sw,
    )
    et_full = _fit_tree_model(
        config,
        tree_cols,
        tree_cat_cols,
        model_type="et",
        n_estimators=int(config["forest"]["n_estimators_et"]),
        min_samples_leaf=int(config["forest"]["min_samples_leaf_et"]),
        random_state=int(config["forest"]["random_state"]) + 1,
        train_df=labeled,
        y_log=y_log,
        sample_weight=global_sw,
    )

    tertiary_full = None
    if bool(config["specialists"]["enable_tertiary"]):
        mask = labeled["geo_age"].astype(str).eq("Tertiary")
        if mask.sum() >= int(config["specialists"]["min_train_rows_tertiary"]):
            spec_df = labeled.loc[mask].copy()
            tertiary_full = _fit_tree_model(
                config,
                tree_cols,
                tree_cat_cols,
                model_type=str(config["specialists"]["tertiary_model_type"]),
                n_estimators=int(config["specialists"]["tertiary_n_estimators"]),
                min_samples_leaf=int(config["specialists"]["tertiary_min_samples_leaf"]),
                random_state=int(config["forest"]["random_state"]) + 10,
                train_df=spec_df,
                y_log=np.log(spec_df["vs_meas_mps"].to_numpy()),
                sample_weight=_make_weight_vector(config, spec_df),
            )

    scpt_full = None
    if bool(config["specialists"]["enable_scpt"]) and bool(config["features"]["include_test_method_for_trees"]):
        mask = labeled["test_method"].astype(str).eq("SCPT")
        if mask.sum() >= int(config["specialists"]["min_train_rows_scpt"]):
            spec_df = labeled.loc[mask].copy()
            scpt_full = _fit_tree_model(
                config,
                tree_cols,
                tree_cat_cols,
                model_type=str(config["specialists"]["scpt_model_type"]),
                n_estimators=int(config["specialists"]["scpt_n_estimators"]),
                min_samples_leaf=int(config["specialists"]["scpt_min_samples_leaf"]),
                random_state=int(config["forest"]["random_state"]) + 20,
                train_df=spec_df,
                y_log=np.log(spec_df["vs_meas_mps"].to_numpy()),
                sample_weight=_make_weight_vector(config, spec_df),
            )

    high_full = None
    if bool(config["specialists"]["enable_high_vs"]):
        mask = labeled["vs_meas_mps"].to_numpy() >= float(config["specialists"]["high_vs_threshold_mps"])
        if mask.sum() >= int(config["specialists"]["min_train_rows_high_vs"]):
            spec_df = labeled.loc[mask].copy()
            high_full = _fit_tree_model(
                config,
                tree_cols,
                tree_cat_cols,
                model_type=str(config["specialists"]["high_model_type"]),
                n_estimators=int(config["specialists"]["high_n_estimators"]),
                min_samples_leaf=int(config["specialists"]["high_min_samples_leaf"]),
                random_state=int(config["forest"]["random_state"]) + 30,
                train_df=spec_df,
                y_log=np.log(spec_df["vs_meas_mps"].to_numpy()),
                sample_weight=_make_weight_vector(config, spec_df),
            )

    final_model = SpecialistWeightedStackModel(
        empirical_model=empirical_full,
        rf_model=rf_full,
        et_model=et_full,
        base_weights=base_weights,
        tertiary_model=tertiary_full,
        scpt_model=scpt_full,
        high_vs_model=high_full,
        specialist_weights=specialist_weights,
        high_vs_threshold_mps=float(config["specialists"]["high_vs_threshold_mps"]),
    )

    joblib.dump(empirical_full, run_dir / "models" / "empirical_baseline.joblib")
    joblib.dump(rf_full, run_dir / "models" / "random_forest_profile.joblib")
    joblib.dump(et_full, run_dir / "models" / "extra_trees_profile.joblib")
    if tertiary_full is not None:
        joblib.dump(tertiary_full, run_dir / "models" / "tertiary_specialist.joblib")
    if scpt_full is not None:
        joblib.dump(scpt_full, run_dir / "models" / "scpt_specialist.joblib")
    if high_full is not None:
        joblib.dump(high_full, run_dir / "models" / "high_vs_specialist.joblib")
    joblib.dump(final_model, run_dir / "models" / "specialist_weighted_stack.joblib")

    metadata = {
        "final_model_name": "specialist_weighted_stack",
        "tree_feature_columns": tree_cols,
        "tree_categorical_features": tree_cat_cols,
        "base_weights": base_weights,
        "specialist_weights": specialist_weights,
        "high_vs_threshold_mps": float(config["specialists"]["high_vs_threshold_mps"]),
        "weighting_enabled": bool(config["weights"]["enable"]),
    }
    with (run_dir / "models" / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    oof.to_csv(run_dir / "predictions" / "cv_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "reports" / "fold_metrics.csv", index=False)
    summary_df.to_csv(run_dir / "reports" / "summary_metrics.csv", index=False)
    subset_metrics_df.to_csv(run_dir / "reports" / "subset_metrics.csv", index=False)
    (run_dir / "reports" / "base_weights.json").write_text(json.dumps(base_weights, indent=2), encoding="utf-8")
    (run_dir / "reports" / "specialist_weights.json").write_text(json.dumps(specialist_weights, indent=2), encoding="utf-8")
    (run_dir / "reports" / "final_model.txt").write_text(
        f"specialist_weighted_stack\nbase_weights={base_weights}\nspecialist_weights={specialist_weights}\n",
        encoding="utf-8",
    )
    rf_full.feature_importance().to_csv(run_dir / "reports" / "feature_importance_random_forest_profile.csv", index=False)
    et_full.feature_importance().to_csv(run_dir / "reports" / "feature_importance_extra_trees_profile.csv", index=False)

    fitted = labeled.copy()
    fitted_pred = final_model.predict_log_components(labeled)
    fitted = pd.concat([fitted.reset_index(drop=True), fitted_pred.reset_index(drop=True)], axis=1)
    fitted["pred_vs_mps_final"] = np.exp(fitted["pred_log_specialist_weighted_stack"])
    fitted.to_csv(run_dir / "predictions" / "labeled_fitted_predictions.csv", index=False)

    if bool(config["inference"]["produce_unlabeled_predictions"]) and unlabeled is not None and len(unlabeled) > 0:
        pred = final_model.predict_log_components(unlabeled)
        unlabeled_out = pd.concat([unlabeled.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        unlabeled_out["pred_vs_mps_final"] = np.exp(unlabeled_out["pred_log_specialist_weighted_stack"])
        unlabeled_out.to_csv(run_dir / "predictions" / "unlabeled_predictions.csv", index=False)

    print(f"Training finished. Run directory: {run_dir}")
    return run_dir
