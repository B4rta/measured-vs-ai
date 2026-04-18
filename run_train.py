from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
from measured_vs.training.pipeline import run_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_training(config_path=args.config)
