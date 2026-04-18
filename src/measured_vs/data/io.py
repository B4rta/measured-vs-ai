from __future__ import annotations

from pathlib import Path
import pandas as pd


def _try_read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return None
    raise ValueError(f"Unsupported table format: {path}")


def load_cleaned_data(cleaned_dir: str | Path, prefer_parquet: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    cleaned_dir = Path(cleaned_dir)
    labeled_candidates = [cleaned_dir / "cpt_vs_labeled.parquet", cleaned_dir / "cpt_vs_labeled.csv"] if prefer_parquet else [cleaned_dir / "cpt_vs_labeled.csv", cleaned_dir / "cpt_vs_labeled.parquet"]
    unlabeled_candidates = [cleaned_dir / "cpt_vs_unlabeled.parquet", cleaned_dir / "cpt_vs_unlabeled.csv"] if prefer_parquet else [cleaned_dir / "cpt_vs_unlabeled.csv", cleaned_dir / "cpt_vs_unlabeled.parquet"]

    labeled = None
    for p in labeled_candidates:
        if p.exists():
            labeled = _try_read_table(p)
            if labeled is not None:
                break
    if labeled is None:
        raise FileNotFoundError(f"No readable labeled cleaned file found in {cleaned_dir}")

    unlabeled = None
    for p in unlabeled_candidates:
        if p.exists():
            unlabeled = _try_read_table(p)
            if unlabeled is not None:
                break
    return labeled, unlabeled
