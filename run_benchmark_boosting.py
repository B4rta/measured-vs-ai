from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

from measured_vs.data.io import load_cleaned_data
from measured_vs.data.features import engineer_profile_features, tree_feature_columns, compute_sample_weights
from measured_vs.evaluation.metrics import regression_metrics_vs
from measured_vs.utils.config import load_config


def _encode_fit_transform(train_df, valid_df, feature_columns, categorical_features):
    tr = train_df[feature_columns].copy()
    va = valid_df[feature_columns].copy()
    maps = {}
    for col in categorical_features:
        trs = tr[col].astype(object).where(tr[col].notna(), "__MISSING__").astype(str)
        vas = va[col].astype(object).where(va[col].notna(), "__MISSING__").astype(str)
        vals = list(pd.Index(trs).unique())
        mp = {v: i for i, v in enumerate(vals)}
        tr[col] = trs.map(mp).fillna(-1).astype(float)
        va[col] = vas.map(mp).fillna(-1).astype(float)
        maps[col] = mp
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(tr), imp.transform(va)


def _sample_weight(config, train_df):
    weights = config.get("weights", {})
    if not bool(weights.get("enable", True)):
        return None
    return compute_sample_weights(
        train_df,
        high_vs_anchor_mps=float(weights.get("high_vs_anchor_mps", 320.0)),
        high_vs_span_mps=float(weights.get("high_vs_span_mps", 180.0)),
        high_vs_boost=float(weights.get("high_vs_boost", 0.5)),
        tertiary_boost=float(weights.get("tertiary_boost", 0.2)),
        scpt_boost=float(weights.get("scpt_boost", 0.3)),
    )


def _models(random_state: int):
    models = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=random_state,
        )
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=900,
            learning_rate=0.025,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception as exc:
        print(f"[INFO] XGBoost unavailable, skipping xgboost benchmark: {exc}")
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as exc:
        print(f"[INFO] LightGBM unavailable, skipping lightgbm benchmark: {exc}")
    return models


def main():
    parser = argparse.ArgumentParser(description="Grouped-CV gradient boosting benchmark for the CPT-Vs paper.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    labeled, _ = load_cleaned_data(config["data"]["cleaned_dir"], prefer_parquet=bool(config["data"].get("prefer_parquet", True)))
    labeled = engineer_profile_features(
        labeled,
        include_test_method_for_trees=bool(config["features"].get("include_test_method_for_trees", True)),
        profile_windows=config["features"].get("profile_windows", [3, 5]),
    )
    feature_columns, categorical_features = tree_feature_columns(
        labeled,
        include_test_method_for_trees=bool(config["features"].get("include_test_method_for_trees", True)),
    )
    groups = labeled[config["splits"].get("group_column", "group_project")].astype(str)
    n_splits = int(config["splits"].get("n_splits", 5))
    y_log = np.log(labeled["vs_meas_mps"].to_numpy())

    splitter = GroupKFold(n_splits=n_splits)
    oof = labeled[["project", "cpt_id", "group_project", "group_cpt", "geo_age", "test_method", "z_mid_m", "vs_meas_mps"]].copy()
    model_dict = _models(int(config["forest"].get("random_state", 42)))
    for name in model_dict:
        oof[f"pred_log_{name}"] = np.nan

    fold_rows = []
    for fold, (tr, va) in enumerate(splitter.split(labeled, groups=groups), start=1):
        train_df = labeled.iloc[tr].copy()
        valid_df = labeled.iloc[va].copy()
        Xtr, Xva = _encode_fit_transform(train_df, valid_df, feature_columns, categorical_features)
        sw = _sample_weight(config, train_df)
        for name, model in _models(int(config["forest"].get("random_state", 42)) + fold).items():
            model.fit(Xtr, y_log[tr], sample_weight=sw)
            pred_log = model.predict(Xva)
            oof.loc[oof.index[va], f"pred_log_{name}"] = pred_log
            metrics = regression_metrics_vs(valid_df["vs_meas_mps"].to_numpy(), np.exp(pred_log))
            fold_rows.append({"fold": fold, "model": name, **metrics})
            print(f"fold={fold} model={name} rmse={metrics['rmse_mps']:.3f}")

    summary_rows = []
    for name in model_dict:
        col = f"pred_log_{name}"
        if oof[col].notna().all():
            summary_rows.append({"model": name, **regression_metrics_vs(oof["vs_meas_mps"].to_numpy(), np.exp(oof[col].to_numpy()))})
    summary = pd.DataFrame(summary_rows).sort_values("rmse_mps")

    run_dir = Path(args.output_root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_boosting_benchmark"
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    oof.to_csv(run_dir / "predictions" / "boosting_cv_predictions.csv", index=False)
    summary.to_csv(run_dir / "reports" / "boosting_summary_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "reports" / "boosting_fold_metrics.csv", index=False)
    (run_dir / "reports" / "benchmark_config.json").write_text(json.dumps({"config": args.config, "models": list(model_dict)}, indent=2), encoding="utf-8")
    print(f"Benchmark finished. Run directory: {run_dir}")
    print(summary)


if __name__ == "__main__":
    main()
