from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from measured_vs.data.io import load_cleaned_data
from measured_vs.data.features import engineer_profile_features


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def predict_from_run_dir(run_dir: Path, input_path: Path | None = None, output_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "models" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_name = metadata["final_model_name"]
    model_path = run_dir / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    model = joblib.load(model_path)

    import yaml
    with (run_dir / "config_used.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if input_path is None:
        _, unlabeled = load_cleaned_data(
            cleaned_dir=config["data"]["cleaned_dir"],
            prefer_parquet=bool(config["data"]["prefer_parquet"]),
        )
        if unlabeled is None or len(unlabeled) == 0:
            raise ValueError("No unlabeled cleaned data found. Pass --input-path explicitly.")
        raw_df = unlabeled.copy()
    else:
        raw_df = _read_any(Path(input_path)).copy()

    df = engineer_profile_features(
        raw_df,
        include_test_method_for_trees=bool(config["features"]["include_test_method_for_trees"]),
        profile_windows=config["features"]["profile_windows"],
    )

    pred = model.predict_log_components(df)
    out = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    out["pred_vs_mps_final"] = np.exp(out["pred_log_specialist_weighted_stack"])

    if output_path is None:
        output_path = run_dir / "predictions" / "unlabeled_predictions.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Prediction file written to: {output_path}")
    return output_path
