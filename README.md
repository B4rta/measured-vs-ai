# measured-vs-ai-final-repo

Final paper-ready repository for **CPT → measured Vs** modeling from the cleaned dataset.

This is the stabilized version of the workflow that actually outperformed the empirical baseline under **grouped site-level validation**.

## Final chosen modeling strategy

The repository uses a **profile-aware ensemble with specialist corrections**:

1. **Empirical baseline**
   - a regularized linear baseline on physically meaningful transformed CPT variables
2. **Profile-aware global tree models**
   - Random Forest
   - Extra Trees
3. **Weighted base stack**
   - learned from grouped out-of-fold predictions
4. **Specialist corrections**
   - **SCPT specialist**
   - **high-Vs specialist**
   - Tertiary specialist kept as an optional config only, because in the current best run its learned weight was `0.0`

## Best achieved grouped-CV result

Using the best v4 run snapshot stored in `docs/results_snapshot/`:

- **specialist_weighted_stack**: RMSE **59.32**, MAE **42.15**, R² **0.619**, bias **-3.36**
- **empirical_baseline**: RMSE **64.98**, MAE **46.53**, R² **0.543**, bias **-16.50**

So the final approach improved RMSE by about **5.66 m/s** over the empirical baseline, while also strongly reducing the overall underestimation bias.

### Where it helped most

Compared with the base stack, the specialist layer improved especially on:

- **SCPT**
- **high-Vs cases**
- **Tertiary conditions** indirectly through the global + specialist combination

See `docs/results_snapshot/subset_metrics_best_run.csv`.

## Repository structure

- `data/cleaned/` — cleaned labeled and unlabeled tables
- `src/measured_vs/` — source code
- `configs/default.yaml` — final recommended training config
- `configs/paper_reproduction_v4_best.yaml` — exact config of the best recorded v4 run family
- `configs/deployable.yaml` — safer deployment config without `test_method`
- `configs/smoke.yaml` — quick pipeline check
- `docs/results_snapshot/` — fixed reference metrics from the current best run
- `outputs/` — generated training outputs

## Recommended configs

### 1) Final recommended config

`configs/default.yaml`

Use this first. It keeps the winning v4 logic, but disables the tertiary specialist by default because the best run assigned it zero blend weight.

### 2) Paper reproduction config

`configs/paper_reproduction_v4_best.yaml`

This mirrors the exact specialist setup of the best recorded v4 run, including the tertiary specialist being available during weight search.

### 3) Deployment-oriented config

`configs/deployable.yaml`

Use this when `test_method` is unavailable or should not be relied on.

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Default training:

```bash
python run_train.py
```

Paper reproduction run:

```bash
python run_train.py --config configs/paper_reproduction_v4_best.yaml
```

Deployment-oriented run:

```bash
python run_train.py --config configs/deployable.yaml
```

Smoke test:

```bash
python run_train_smoke.py
```

Inference from an already trained run directory:

```bash
python run_predict.py --run-dir outputs/<run_folder>
```

Custom input table prediction:

```bash
python run_predict.py --run-dir outputs/<run_folder> --input-path path/to/table.csv --output-path path/to/preds.csv
```

## Notes for the manuscript

This repository supports a paper narrative like:

> A profile-aware ensemble with specialist corrections improves CPT-based measured-Vs estimation over empirical correlations under grouped site-level validation, with the strongest gains in SCPT and high-Vs regimes.

It does **not** support the stronger claim that ML is better everywhere. In the best run, the `Vs < 400 m/s` subset remained slightly worse than the specialist-focused model’s gains in the harder regimes.
