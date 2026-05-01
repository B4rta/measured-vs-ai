from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measured_vs.data.io import load_cleaned_data
from measured_vs.data.features import engineer_profile_features


def main():
    parser = argparse.ArgumentParser(description="Create SHAP summary and dependency plots for RF/ET profile models.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", choices=["rf", "et"], default="rf")
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--dependence-features", nargs="*", default=["log_qt_mpa", "log_fs_mpa", "log_sigma_eff_kpa", "depth_rank_pct"])
    args = parser.parse_args()

    try:
        import shap
    except Exception as exc:
        raise SystemExit("SHAP is not installed. Run: python -m pip install shap") from exc

    run_dir = Path(args.run_dir)
    with (run_dir / "config_used.yaml").open("r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
    with (run_dir / "models" / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_file = "random_forest_profile.joblib" if args.model == "rf" else "extra_trees_profile.joblib"
    model = joblib.load(run_dir / "models" / model_file)

    labeled, _ = load_cleaned_data(config["data"]["cleaned_dir"], prefer_parquet=bool(config["data"].get("prefer_parquet", True)))
    df = engineer_profile_features(
        labeled,
        include_test_method_for_trees=bool(config["features"].get("include_test_method_for_trees", True)),
        profile_windows=config["features"].get("profile_windows", [3, 5]),
    )
    if len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42)

    X = model.transform_features(df)
    feature_names = model.encoded_feature_names()
    out_dir = run_dir / "figures" / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model.model_)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{args.model}.png", dpi=240, bbox_inches="tight")
    plt.savefig(out_dir / f"shap_summary_{args.model}.svg", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_{args.model}.png", dpi=240, bbox_inches="tight")
    plt.savefig(out_dir / f"shap_bar_{args.model}.svg", bbox_inches="tight")
    plt.close()

    for feat in args.dependence_features:
        if feat not in feature_names:
            print(f"[WARN] dependence feature not found, skipping: {feat}")
            continue
        idx = feature_names.index(feat)
        plt.figure()
        shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        safe = feat.replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"shap_dependence_{args.model}_{safe}.png", dpi=240, bbox_inches="tight")
        plt.savefig(out_dir / f"shap_dependence_{args.model}_{safe}.svg", bbox_inches="tight")
        plt.close()

    print(f"SHAP figures written to: {out_dir}")


if __name__ == "__main__":
    main()
