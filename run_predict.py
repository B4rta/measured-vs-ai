from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
from pathlib import Path
from measured_vs.training.inference import predict_from_run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()
    predict_from_run_dir(
        run_dir=Path(args.run_dir),
        input_path=Path(args.input_path) if args.input_path else None,
        output_path=Path(args.output_path) if args.output_path else None,
    )
