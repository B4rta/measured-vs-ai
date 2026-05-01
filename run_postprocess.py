from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import numpy as np
import pandas as pd

from measured_vs.evaluation.conformal import (
    add_absolute_conformal_intervals,
    conformal_subset_summary,
)


def find_latest_run_dir(outputs_dir: Path) -> Path:
    """Return newest outputs/<run>/ folder that contains predictions/cv_predictions.csv."""
    if not outputs_dir.exists():
        raise FileNotFoundError(
            f"Outputs directory does not exist: {outputs_dir}\n"
            "Run training first with: python run_train.py"
        )

    candidates = []
    for path in outputs_dir.iterdir():
        if path.is_dir() and (path / "predictions" / "cv_predictions.csv").exists():
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No valid training run found under: {outputs_dir}\n"
            "Expected: outputs/<RUN_NAME>/predictions/cv_predictions.csv"
        )

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def resolve_run_dir(run_dir_arg: str | None) -> Path:
    project_root = Path(__file__).resolve().parent
    if run_dir_arg is None:
        run_dir = find_latest_run_dir(project_root / "outputs")
        print("No --run-dir provided. Using latest finished run:")
        print(f"  {run_dir}")
        return run_dir
    run_dir = Path(run_dir_arg)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Add conformal prediction intervals to a finished training run.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory. If omitted, newest valid outputs/<RUN_NAME> folder is used automatically.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.10, 0.20],
        help="Miscoverage levels; 0.10 means nominal 90 percent interval.",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    pred_path = run_dir / "predictions" / "cv_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing OOF prediction file: {pred_path}")

    df = pd.read_csv(pred_path)
    if "pred_vs_mps_final" not in df.columns:
        if "pred_log_specialist_weighted_stack" not in df.columns:
            raise ValueError("Need either pred_vs_mps_final or pred_log_specialist_weighted_stack in cv_predictions.csv")
        df["pred_vs_mps_final"] = np.exp(df["pred_log_specialist_weighted_stack"])

    out, summary = add_absolute_conformal_intervals(
        df,
        y_true_col="vs_meas_mps",
        y_pred_col="pred_vs_mps_final",
        alphas=args.alphas,
        prefix="pred_vs_mps_final",
    )

    out_path = run_dir / "predictions" / "cv_predictions_with_conformal_intervals.csv"
    out.to_csv(out_path, index=False)

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "conformal_intervals.csv"
    summary.to_csv(summary_path, index=False)

    nominal_pct = int(round((1.0 - args.alphas[0]) * 100))
    subset = conformal_subset_summary(out, nominal_pct=nominal_pct)
    subset_path = report_dir / "conformal_subset_coverage.csv"
    subset.to_csv(subset_path, index=False)

    print("Conformal postprocess finished.")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {subset_path}")


if __name__ == "__main__":
    main()
