from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


def greedy_group_kfold(groups: pd.Series, n_splits: int) -> list[list[str]]:
    """Mirror the unshuffled GroupKFold size-balancing logic for audit tables."""
    sizes = groups.astype(str).value_counts(sort=True)
    fold_sizes = [0] * n_splits
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    for group, size in sizes.items():
        fold_idx = min(range(n_splits), key=lambda i: fold_sizes[i])
        fold_groups[fold_idx].append(str(group))
        fold_sizes[fold_idx] += int(size)
    return fold_groups


def summarize_projects(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("group_project")
        .agg(
            n_rows=("vs_meas_mps", "size"),
            n_cpt=("group_cpt", "nunique"),
            n_masw=("test_method", lambda s: int((s.astype(str) == "MASW").sum())),
            n_scpt=("test_method", lambda s: int((s.astype(str) == "SCPT").sum())),
            n_quaternary=("geo_age", lambda s: int((s.astype(str) == "Quaternary").sum())),
            n_tertiary=("geo_age", lambda s: int((s.astype(str) == "Tertiary").sum())),
            n_vs_ge_400=("vs_meas_mps", lambda s: int((s >= 400.0).sum())),
            vs_min_mps=("vs_meas_mps", "min"),
            vs_median_mps=("vs_meas_mps", "median"),
            vs_max_mps=("vs_meas_mps", "max"),
        )
        .reset_index()
        .sort_values("n_rows", ascending=False)
    )


def summarize_folds(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    rows = []
    for fold, projects in enumerate(greedy_group_kfold(df["group_project"], n_splits), start=1):
        sub = df[df["group_project"].astype(str).isin(projects)]
        rows.append(
            {
                "fold": fold,
                "n_rows": len(sub),
                "n_projects": sub["group_project"].nunique(),
                "n_cpt": sub["group_cpt"].nunique(),
                "n_masw": int((sub["test_method"].astype(str) == "MASW").sum()),
                "n_scpt": int((sub["test_method"].astype(str) == "SCPT").sum()),
                "n_quaternary": int((sub["geo_age"].astype(str) == "Quaternary").sum()),
                "n_tertiary": int((sub["geo_age"].astype(str) == "Tertiary").sum()),
                "n_vs_ge_400": int((sub["vs_meas_mps"] >= 400.0).sum()),
                "projects": "; ".join(sorted(projects)),
            }
        )
    return pd.DataFrame(rows)


def hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "unit": "depth_interval_records",
                "count": len(df),
                "interpretation": "Model rows; not statistically independent observations.",
            },
            {
                "unit": "cpt_soundings",
                "count": df["group_cpt"].nunique(),
                "interpretation": "Nested vertical profiles used for profile-aware features.",
            },
            {
                "unit": "projects_sites",
                "count": df["group_project"].nunique(),
                "interpretation": "Independent validation groups for project-level splitting.",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write dataset hierarchy and grouped-CV audit tables.")
    parser.add_argument("--input", default="data/cleaned/cpt_vs_labeled.csv")
    parser.add_argument("--out-dir", default="docs/results_snapshot")
    parser.add_argument("--splits", type=int, nargs="+", default=[5, 15])
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hierarchy(df).to_csv(out_dir / "data_hierarchy.csv", index=False)
    summarize_projects(df).to_csv(out_dir / "project_composition.csv", index=False)
    for n_splits in args.splits:
        summarize_folds(df, n_splits).to_csv(out_dir / f"fold_composition_{n_splits}fold.csv", index=False)

    print(f"Validation audit tables written to: {out_dir}")


if __name__ == "__main__":
    main()
