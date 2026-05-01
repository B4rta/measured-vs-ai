# measured-vs-ai

Final CPT-to-measured Vs modelling repository for the Budapest CPT--Vs paper.

The core model is a **profile-aware specialist ensemble** evaluated with grouped site-level validation. This version also contains paper-extension analyses requested during internal review: conformal prediction intervals, SHAP explainability, boosting benchmarks, hyperparameter sensitivity runs, and depth-profile figures.

## Core result snapshot

Fixed reference metrics from the current best run are stored in `docs/results_snapshot/`.

- **specialist_weighted_stack**: RMSE **59.32**, MAE **42.15**, R2 **0.619**, bias **-3.36**
- **empirical_baseline**: RMSE **64.98**, MAE **46.53**, R2 **0.543**, bias **-16.50**

The result should be described carefully: the specialist ensemble improves over the empirical baseline under the same grouped validation protocol, especially for SCPT and high-Vs cases. It is not uniformly better in every subset.

## Code and data availability

The code, cleaned modelling tables, configuration files, generated figures, trained-model artifacts, benchmark outputs, and reproducibility scripts supporting the paper are available in this public GitHub repository:

https://github.com/B4rta/measured-vs-ai

Large generated artifacts are stored with Git LFS. Install Git LFS before cloning if the full model and output files are required.

The manuscript text and submitted DOCX/PDF files are intentionally not included in this repository.

## Repository structure

- `data/cleaned/` - cleaned labeled/unlabeled modelling tables
- `src/measured_vs/` - core package
- `configs/default.yaml` - recommended final config
- `configs/paper_reproduction_v4_best.yaml` - configuration matching the best recorded v4 run family
- `configs/deployable.yaml` - variant without `test_method`
- `configs/cv10.yaml` - 10-fold grouped-CV robustness config
- `configs/lopo15.yaml` - leave-one-project-out-like 15-fold robustness config
- `docs/results_snapshot/` - fixed metrics from the current manuscript snapshot
- `scripts/make_paper_figures.py` - publication figure generator
- `run_postprocess.py` - conformal prediction intervals from OOF predictions
- `run_shap.py` - SHAP plots from trained RF/ET models
- `run_benchmark_boosting.py` - HistGradientBoosting/XGBoost/LightGBM grouped-CV benchmark
- `run_sensitivity.py` - retraining-based sensitivity grid for selected hyperparameters

## Install

Core training environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional paper-extension tools:

```bash
python -m pip install -r requirements-extensions.txt
```

If `xgboost` or `lightgbm` is not available for your local Python version, the benchmark script will skip the missing library and still run the scikit-learn `HistGradientBoostingRegressor` baseline.

## Main training

```bash
python run_train.py --config configs/default.yaml
```

Paper reproduction run:

```bash
python run_train.py --config configs/paper_reproduction_v4_best.yaml
```

Robustness CV checks:

```bash
python run_train.py --config configs/cv10.yaml
python run_train.py --config configs/lopo15.yaml
```

Smoke test:

```bash
python run_train_smoke.py
```

## After training: paper extension analyses

Assume your run directory is `outputs/<RUN_NAME>`.

### 1) Conformal prediction intervals

```bash
python run_postprocess.py --run-dir outputs/<RUN_NAME>
```

This creates:

- `predictions/cv_predictions_with_conformal_intervals.csv`
- `reports/conformal_intervals.csv`
- `reports/conformal_subset_coverage.csv`

Use these for the uncertainty-quantification section. The implementation is model-agnostic and calibrated from grouped-CV OOF residuals.

### 2) SHAP explainability

```bash
python run_shap.py --run-dir outputs/<RUN_NAME> --model rf
python run_shap.py --run-dir outputs/<RUN_NAME> --model et
```

Outputs are written to:

- `figures/shap/`

Use the SHAP beeswarm/bar plots and 2--3 dependency plots in the feature interpretation section.

### 3) Boosting benchmark

```bash
python run_benchmark_boosting.py --config configs/default.yaml
```

This runs a grouped-CV benchmark for:

- scikit-learn HistGradientBoostingRegressor
- XGBoost, if installed
- LightGBM, if installed

Outputs are written to a new `outputs/<timestamp>_boosting_benchmark/` folder.

### 4) Hyperparameter sensitivity

Start with a cheaper smoke sensitivity grid:

```bash
python run_sensitivity.py --config configs/smoke.yaml
```

For final paper numbers, run a smaller but full configuration grid, for example:

```bash
python run_sensitivity.py --config configs/default.yaml --thresholds 325 350 375 --high-blend-max 0.50 0.65 0.80
```

This retrains the model for each setting and writes a manifest under `outputs/sensitivity_configs/`.

### 5) Paper figures and depth-profile plot

```bash
python scripts/make_paper_figures.py --run-dir outputs/<RUN_NAME>
```

This creates:

- measured vs predicted scatter, split by geological age and seismic method
- residual trend with IQR envelope
- one depth-profile prediction figure

If you want a specific CPT/profile:

```bash
python scripts/make_paper_figures.py --run-dir outputs/<RUN_NAME> --cpt-id <GROUP_CPT_ID>
```

## Manuscript support materials

This repository contains the computational materials needed to support the paper, including conformal prediction interval outputs, SHAP figures, boosting benchmark results, sensitivity runs, and depth-profile figures. The manuscript itself is kept outside the repository.

## One-file full pipeline for VS Code

The easiest way to run the complete paper pipeline is now:

```bash
python run_all.py
```

This single file runs, in order:

1. training with `configs/default.yaml`,
2. conformal prediction postprocessing,
3. paper figure generation,
4. SHAP analysis for the random forest model,
5. boosting benchmark,
6. hyperparameter sensitivity analysis.

For a quick test:

```bash
python run_all.py --config configs/smoke.yaml --skip-shap --skip-benchmark --skip-sensitivity
```

For using an already finished run without retraining:

```bash
python run_all.py --skip-train
```

or explicitly:

```bash
python run_all.py --run-dir outputs/<RUN_NAME>
```

Runtime logs are written locally to `pipeline_logs/`, which is ignored by Git.

In VS Code, use the included launch configuration: **Run full paper pipeline**.
