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

Page and line references belong to the separate manuscript response document.

