# Validation evidence

The former pooled OOF stack and conformal scores are superseded. See [revision_notes.md](revision_notes.md).

| Question | Evidence |
|---|---|
| Outer projects excluded from weight/gate selection | run_nested_validation.py; selection.json; inner_sensitivity.csv |
| Deployment without seismic modality | deployable results; label-invariance regression test |
| Separate calibration and coverage assessment | run_project_intervals.py; revised_intervals/splits.json |
| Project imbalance and paired performance | project_metrics.csv; revision_summary macro, paired and omission diagnostics |
| Random-seed stability | revision_summary/seed_metrics.csv |
| Data cleaning trail | source_audit/audit.json; 1,304 retained, zero removed |
| Remaining high-Vs bias | velocity_bins.csv; Figure 5 |
| Stress assumptions and profile/component sensitivity | review_round2.py; results/review_round2/; second_review.md |
| Fixed published empirical comparisons | empirical_benchmarks.py; results/review_round2/empirical_benchmark_metrics.csv |
| Influence after deleting a training project | followup_validation.py; results/followup_validation/influence/; combined_project_deletion.csv |
| Physical windows, context resampling and simulated omissions | results/followup_validation/*/predictions.csv and metrics.csv |
| Geographically separate public observations | public_external_validation.py; results/public_external/; followup_validation.md |
| External extrapolation and high-Vs failure | input_support_flags.csv; external_velocity_metrics.csv; largest_regression_predictions.csv |
| Source-specific acquisition and engineering validation gaps | followup_validation.md, Other sources and remaining gaps |
| Repository consistency and reproducibility | audit_repository.py; results/repository_audit.json; cache_signature.json files |

Page and line references belong to the separate manuscript response document.

