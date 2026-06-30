# Reviewer-response support map

This repository supports a revised manuscript that is intentionally more modest about novelty and more explicit about validation limits.

## Data scale and independence

- The labelled table has 1,304 depth-interval rows, but these are nested within 80 CPT/CPTu soundings from 15 Budapest projects.
- The rows should be described as depth-level paired intervals, not as 1,304 independent observations.
- Snapshot files: `docs/results_snapshot/data_hierarchy.csv` and `docs/results_snapshot/project_composition.csv`.

## Project-level validation

- The main manuscript result uses grouped validation by `group_project`.
- The five-fold split is leakage-reducing but not perfectly balanced because project sizes range from 18 to 389 rows.
- In the current unshuffled GroupKFold-style allocation, Puskas forms a single large validation fold.
- Snapshot files: `docs/results_snapshot/fold_composition_5fold.csv` and `docs/results_snapshot/fold_composition_15fold.csv`.
- Reproducibility script: `scripts/audit_validation_design.py`.

## Algorithmic novelty and performance claim

- The revised manuscript should not claim a novel Random Forest algorithm.
- The defensible contribution is a transparent local benchmark workflow: leakage control, profile-context features, project-level validation, specialist error analysis, uncertainty intervals, SHAP explainability, and open reproducibility assets.
- The performance gain is modest in global RMSE: 59.32 m/s versus 64.98 m/s for the empirical baseline under the same grouped validation protocol.
- The larger practical benefit is reduced mean bias, especially for stiff/high-Vs intervals.

## MASW, SCPT, and soil classification

- MASW and SCPT records are deliberately reported as separate diagnostic subsets because they have different resolution and fidelity.
- The dataset does not include a consistent interval-level USCS label. The Quaternary/Tertiary grouping follows the source Budapest dataset and should be framed as a geological-stratigraphic proxy, not a replacement for USCS.
- A future external dataset with USCS and laboratory/visual classification should be used for broader physical applicability tests.

## Explainability

- The repository already contains SHAP generation support in `run_shap.py`.
- SHAP figures from the current full run are stored under `outputs/20260429_102226_specialist_profile_ensemble_final/figures/shap/`.
- The revised paper should include or directly discuss the SHAP result, not only impurity-based feature importance.
