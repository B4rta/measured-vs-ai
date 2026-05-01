from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
import yaml

from measured_vs.training.pipeline import run_training
from measured_vs.utils.config import load_config


def write_temp_config(config: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Retrain small sensitivity grid for manuscript robustness checks.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Use smoke.yaml first; use default.yaml for final sensitivity if runtime is acceptable.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[325, 350, 375])
    parser.add_argument("--high-blend-max", nargs="+", type=float, default=[0.5, 0.65, 0.8])
    parser.add_argument("--output-root", default="outputs/sensitivity_configs")
    args = parser.parse_args()

    base = load_config(args.config)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for threshold in args.thresholds:
        for max_blend in args.high_blend_max:
            cfg = copy.deepcopy(base)
            cfg["experiment_name"] = f"sensitivity_thr{int(threshold)}_maxblend{str(max_blend).replace('.', 'p')}"
            cfg.setdefault("specialists", {})["high_vs_threshold_mps"] = float(threshold)
            cfg["specialists"]["max_blend_weight"] = float(max_blend)
            cfg_path = Path(args.output_root) / tag / f"thr{int(threshold)}_maxblend{max_blend}.yaml"
            write_temp_config(cfg, cfg_path)
            print(f"Running sensitivity config: threshold={threshold}, max_blend={max_blend}")
            run_dir = run_training(cfg_path)
            rows.append({"threshold": threshold, "max_blend_weight": max_blend, "run_dir": str(run_dir)})
    out = Path(args.output_root) / tag / "sensitivity_runs.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Sensitivity grid finished. Manifest: {out}")


if __name__ == "__main__":
    main()
