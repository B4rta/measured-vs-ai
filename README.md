# CPT-based shear-wave velocity prediction in Budapest

Code and data for a project-level study of shear-wave velocity prediction from cone penetration tests. The database contains 1,304 paired depth intervals from 80 soundings at 15 Budapest projects. The independent validation unit is the project.

The September 2026 revision replaces the original pooled stacking scores with nested validation, checks the source workbook, and evaluates prediction intervals on separate projects. The primary configuration uses CPT variables, groundwater information and geological age. It excludes the seismic measurement label and SCPT routing.

## Results

Five outer project folds, three inner project folds, seed 42:

| Model | RMSE (m/s) | MAE (m/s) | R² | Bias (m/s) |
|---|---:|---:|---:|---:|
| Robust regression baseline | 62.11 | 44.83 | 0.582 | −14.83 |
| Random Forest | 63.29 | 45.25 | 0.566 | −2.22 |
| Extra Trees | 62.47 | 44.55 | 0.578 | −5.07 |
| Base stack | 61.34 | 44.12 | 0.593 | −9.72 |
| Specialist stack without modality | 60.56 | 43.82 | 0.603 | −8.55 |

The pooled RMSE reduction is 2.50%. Equal weighting of projects gives mean RMSEs of 64.17 and 64.13 m/s for the baseline and specialist stack. The stack improves 8 of 15 projects; the paired mean difference is −0.04 m/s, with a descriptive project-bootstrap interval of −2.86 to 2.74 m/s. These results do not establish consistent superiority on new projects.

Seeds 42, 43 and 44 give primary RMSEs of 60.563037, 60.520698 and 60.699858 m/s. This fixed-fold comparison measures forest randomness, not regional transfer uncertainty. A separate retrospective configuration includes the seismic label and SCPT routing; its seed-42 RMSE is 59.47 m/s and is not the engineering deployment claim.

The former 59.32 m/s headline followed selection of blend weights on the same out-of-fold targets used for reporting. The former 90.18% and 80.14% coverages were calibration-set summaries. Historical outputs remain for traceability but are not independent validation results.

## Reproduce the revision

Use Python 3.12:

```bash
python -m pip install -r requirements-revision.txt
python -m unittest discover -s tests -v
python run_nested_validation.py --seeds 42 43 44 --modalities deployable retrospective
python run_project_intervals.py
python scripts/summarize_revision.py
```

Results go to `results/revised_stress/`, `results/revised_intervals/` and `results/revision_summary/`. The optional source audit requires a local copy supplied by the database owner:

```bash
python scripts/audit_source_workbook.py "/path/to/Data Table (Full FINAL).xlsx"
```

Audit summaries and the derived modelling table are already in `results/source_audit/`. The private workbook is not redistributed. The summary script uses this checked table for representative profiles.

All reported model scores use untouched outer projects. Base weights, specialist weights, thresholds and caps are selected inside each outer training set. The full grid and selected weights are saved per fold; predictions carry original row IDs. Do not choose a preferred seed from these outputs.

## Data and preprocessing

The archived CSV in `data/cleaned/` is unchanged. Revised entry points reconstruct CPT-only unit weights and stress proxies with `src/measured_vs/data/stress.py`. The source spreadsheet uses base-10 logarithms in its CPT-only unit-weight formula; the reconstruction matches that column for every record. The overburden proxy uses soil thickness above the interval, split at groundwater. It is a homogeneous-column approximation, not a measured stress profile.

The workbook's other density and stress formulas include measured Vs. Those columns are excluded. Vs, log(Vs), project IDs and sounding IDs are never predictors. Imputation, scaling and encoding are fitted within training partitions. Profile descriptors use only the same sounding's CPT data and require its completed interval profile.

The legacy `MASW` label denotes combined surface-wave methods. The published source describes 899 MASW and 261 tomography records, plus 144 SCPT records. The modelling table does not distinguish individual tomography rows. Revised figures call this category “surface-wave”.

## Prediction intervals

The separate random-project experiment has 3 fitting, 9 calibration and 3 test projects per split. All selection uses fitting projects only. Maximum absolute residuals, one per calibration project, give ranks 8 and 9 for 80% and 90% simultaneous project coverage under project exchangeability.

Held-out whole-project coverages are 12/15 and 15/15. Mean clipped interval widths are about 547 and 623 m/s. This small database cannot support both a large fitting set and precise project-level calibration. These bands use different models from the point-validation experiment and must not be attached to its predictions or to a full-data refit.

## Where to look

- `run_nested_validation.py`: point validation and inner sensitivity grid.
- `run_project_intervals.py`: separate calibration and assessment.
- `results/revision_summary/`: paired project, subset, calibration and influence summaries.
- `results/revised_stress/*/selection.json`: project allocation and selected weights.
- `docs/revision_notes.md`: changes, assumptions and remaining source limitations.
- `docs/figures/`: regenerated figures and attributed source map.

The old `run_train.py`, `run_all.py`, `run_postprocess.py` and YAML configurations reproduce the exploratory workflow, not the revised validation protocol. Old serialized models use different preprocessing and must not be mixed with revised stress variables.

## Sources

Mahler et al. (2026), *Regional Calibration of CPT Correlations for Shear Wave Velocity in Budapest Soils*, [doi:10.1007/s10706-026-03863-7](https://doi.org/10.1007/s10706-026-03863-7), describes the database. Related public data are on [Zenodo](https://doi.org/10.5281/zenodo.14970266). The manuscript cites a fixed Git commit. Submission documents are maintained separately.

The map reproduces Figure 1 of Mahler et al. under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); map data © OpenStreetMap contributors. JPEG conversion leaves the mapped content unchanged. Other revision figures are generated from observations and outer predictions.

