from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "predictions" / "cv_predictions_with_conformal_intervals.csv",
        run_dir / "predictions" / "cv_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "pred_vs_mps_final" not in df.columns and "pred_log_specialist_weighted_stack" in df.columns:
                df["pred_vs_mps_final"] = np.exp(df["pred_log_specialist_weighted_stack"])
            return df
    raise FileNotFoundError(f"No CV prediction file found under {run_dir}/predictions")


def scatter(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for (age, method), g in df.groupby(["geo_age", "test_method"]):
        marker = "o" if str(method) == "MASW" else "s"
        ax.scatter(g["vs_meas_mps"], g["pred_vs_mps_final"], s=22, marker=marker, alpha=0.65, label=f"{age}, {method}")
    mn = 0
    mx = max(df["vs_meas_mps"].max(), df["pred_vs_mps_final"].max()) * 1.05
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel("Measured Vs (m/s)")
    ax.set_ylabel("Predicted Vs (m/s)")
    ax.set_title("Measured vs predicted Vs, grouped-CV OOF predictions")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    save(fig, out_dir, "fig_measured_vs_predicted_by_age_method")


def residual_trend(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    df["residual"] = df["pred_vs_mps_final"] - df["vs_meas_mps"]
    bins = np.linspace(df["vs_meas_mps"].min(), df["vs_meas_mps"].max(), 11)
    df["bin"] = pd.cut(df["vs_meas_mps"], bins=bins, include_lowest=True)
    rows = []
    for b, g in df.groupby("bin", observed=False):
        if len(g) == 0:
            continue
        rows.append({
            "vs_mid": g["vs_meas_mps"].mean(),
            "median": g["residual"].median(),
            "q25": g["residual"].quantile(0.25),
            "q75": g["residual"].quantile(0.75),
        })
    agg = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.plot(agg["vs_mid"], agg["median"], marker="o", lw=1.6, label="Median residual")
    ax.fill_between(agg["vs_mid"], agg["q25"], agg["q75"], alpha=0.25, label="IQR")
    ax.set_xlabel("Measured Vs bin center (m/s)")
    ax.set_ylabel("Residual: predicted - measured (m/s)")
    ax.set_title("Residual trend with IQR envelope")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out_dir, "fig_residual_trend_iqr")


def depth_profile(df: pd.DataFrame, out_dir: Path, cpt_id: str | None):
    if "z_mid_m" not in df.columns:
        raise ValueError("z_mid_m missing from cv_predictions.csv. Rerun training with the updated repository.")
    work = df.copy()
    if cpt_id is None:
        # Choose a representative profile with many rows and both measured/predicted spread.
        counts = work.groupby("group_cpt").size().sort_values(ascending=False)
        cpt_id = str(counts.index[0])
    g = work[work["group_cpt"].astype(str).eq(str(cpt_id))].sort_values("z_mid_m")
    if len(g) == 0:
        raise ValueError(f"No rows for group_cpt={cpt_id}")
    fig, ax = plt.subplots(figsize=(5.2, 7.0))
    ax.plot(g["vs_meas_mps"], g["z_mid_m"], marker="o", lw=1.4, label="Measured Vs")
    ax.plot(g["pred_vs_mps_final"], g["z_mid_m"], marker="s", lw=1.4, label="Predicted Vs")
    if "pred_vs_mps_final_lower_90" in g.columns and "pred_vs_mps_final_upper_90" in g.columns:
        ax.fill_betweenx(g["z_mid_m"], g["pred_vs_mps_final_lower_90"], g["pred_vs_mps_final_upper_90"], alpha=0.18, label="90% conformal interval")
    ax.invert_yaxis()
    ax.set_xlabel("Vs (m/s)")
    ax.set_ylabel("Depth z_mid (m)")
    ax.set_title(f"Depth profile prediction: {cpt_id}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out_dir, f"fig_depth_profile_{str(cpt_id).replace('/', '_')}")


def main():
    parser = argparse.ArgumentParser(description="Create manuscript figures from a finished run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cpt-id", default=None, help="Optional group_cpt identifier for depth-profile figure.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures" / "paper"
    df = load_predictions(run_dir)
    scatter(df, out_dir)
    residual_trend(df, out_dir)
    depth_profile(df, out_dir, args.cpt_id)
    print(f"Figures written to: {out_dir}")


if __name__ == "__main__":
    main()
