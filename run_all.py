from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import time

# -----------------------------------------------------------------------------
# VS Code friendly defaults
# -----------------------------------------------------------------------------
# You can simply press Run on this file in VS Code.
# For the full paper pipeline keep DEFAULT_CONFIG as configs/default.yaml.
# For a quick test change it to configs/smoke.yaml, or run from terminal:
#   python run_all.py --config configs/smoke.yaml --skip-shap --skip-benchmark --skip-sensitivity
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = "configs/default.yaml"
RUN_TRAINING = True
RUN_CONFORMAL_POSTPROCESS = True
RUN_PAPER_FIGURES = True
RUN_SHAP_RF = True
RUN_SHAP_ET = False
RUN_BOOSTING_BENCHMARK = True
RUN_SENSITIVITY = True
STOP_ON_OPTIONAL_FAILURE = False


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "pipeline_logs"


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run_command(name: str, cmd: list[str], *, required: bool = True) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{_timestamp()}_{name}.log"

    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("COMMAND:", " ".join(cmd))
    print(f"LOG: {log_file}")
    print("=" * 80)

    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"STEP: {name}\n")
        f.write("COMMAND: " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        return_code = proc.wait()
        f.write(f"\nRETURN_CODE: {return_code}\n")

    if return_code == 0:
        print(f"STEP FINISHED OK: {name}")
        return True

    message = f"STEP FAILED: {name} with return code {return_code}. See log: {log_file}"
    if required:
        raise RuntimeError(message)
    print("WARNING:", message)
    return False


def find_latest_run_dir(outputs_dir: Path) -> Path:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Missing outputs directory: {outputs_dir}")
    candidates = []
    for path in outputs_dir.iterdir():
        if path.is_dir() and (path / "predictions" / "cv_predictions.csv").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No finished run found under {outputs_dir}. Expected predictions/cv_predictions.csv"
        )
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete CPT-Vs paper pipeline from one VS Code friendly file.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Training config path.")
    parser.add_argument("--run-dir", default=None, help="Use an existing run directory and skip training unless --force-train is given.")
    parser.add_argument("--force-train", action="store_true", help="Train even if --run-dir is provided.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use latest finished run from outputs/.")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--shap-max-rows", type=int, default=600)
    args = parser.parse_args()

    config_path = str(Path(args.config))

    print("\nCOMPLETE CPT-Vs PAPER PIPELINE")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config:       {config_path}")

    should_train = RUN_TRAINING and not args.skip_train
    if args.run_dir and not args.force_train:
        should_train = False

    if should_train:
        run_command("01_train", [sys.executable, "run_train.py", "--config", config_path], required=True)
        run_dir = find_latest_run_dir(PROJECT_ROOT / "outputs")
    else:
        if args.run_dir:
            run_dir = Path(args.run_dir)
            if not run_dir.is_absolute():
                run_dir = PROJECT_ROOT / run_dir
        else:
            run_dir = find_latest_run_dir(PROJECT_ROOT / "outputs")

    print("\nACTIVE RUN DIRECTORY:")
    print(f"  {run_dir}")

    if RUN_CONFORMAL_POSTPROCESS and not args.skip_postprocess:
        run_command("02_conformal_postprocess", [sys.executable, "run_postprocess.py", "--run-dir", str(run_dir)], required=True)

    if RUN_PAPER_FIGURES and not args.skip_figures:
        run_command("03_paper_figures", [sys.executable, "scripts/make_paper_figures.py", "--run-dir", str(run_dir)], required=True)

    if (RUN_SHAP_RF or RUN_SHAP_ET) and not args.skip_shap:
        if RUN_SHAP_RF:
            run_command(
                "04_shap_rf",
                [sys.executable, "run_shap.py", "--run-dir", str(run_dir), "--model", "rf", "--max-rows", str(args.shap_max_rows)],
                required=STOP_ON_OPTIONAL_FAILURE,
            )
        if RUN_SHAP_ET:
            run_command(
                "04_shap_et",
                [sys.executable, "run_shap.py", "--run-dir", str(run_dir), "--model", "et", "--max-rows", str(args.shap_max_rows)],
                required=STOP_ON_OPTIONAL_FAILURE,
            )

    if RUN_BOOSTING_BENCHMARK and not args.skip_benchmark:
        run_command(
            "05_boosting_benchmark",
            [sys.executable, "run_benchmark_boosting.py", "--config", config_path],
            required=STOP_ON_OPTIONAL_FAILURE,
        )

    if RUN_SENSITIVITY and not args.skip_sensitivity:
        run_command(
            "06_sensitivity",
            [sys.executable, "run_sensitivity.py", "--config", config_path],
            required=STOP_ON_OPTIONAL_FAILURE,
        )

    print("\n" + "=" * 80)
    print("ALL REQUESTED STEPS FINISHED")
    print(f"Run directory: {run_dir}")
    print(f"Logs:          {LOG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
