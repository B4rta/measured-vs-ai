# Repository audit, 2 November 2026

The audit covers all versioned files and the saved September and November
validation outputs. The machine-readable result is
[repository_audit.json](../results/repository_audit.json). Reproduce it with
`python scripts/audit_repository.py`; it checks saved outputs without refitting.

- Python, JSON, YAML, TOML, CSV and Parquet files parse successfully.
- Local Markdown links resolve, image files are readable and Git LFS payloads
  are present. Historical joblib models are retained without loading them.
- The CSV and Parquet modelling tables agree.
- Pooled, project and fold metrics agree with predictions for all six reference
  runs. Project allocations match the runner's GroupKFold splitter.
- Eight second-review scenarios, four fixed empirical comparators, 60 fixed-test
  deletion refits and five profile perturbations retain their reported scores.
- The separate interval experiment has disjoint 3/9/3 project allocations and
  reproduces 12/15 and 15/15 whole-project coverage at the two nominal levels.
- The age-free companion's Budapest scores, 2,229 external intervals, 50
  soundings, per-sounding scores and velocity subsets agree with saved predictions.

The audit corrected source row IDs in the eight second-review scenario tables.
All other cells in those tables, including every target and prediction, were
preserved byte for byte. No manuscript result changes.

Additional maintenance updates the VS Code launch configuration, labels legacy
configurations and results, uses the actual splitter in the fold-audit helper,
and fixes positional bias calculation and undefined single-row R² reporting.
Prediction caches now reject changes to inputs, analysis code, parameters or
the modelling environment. Their signatures were registered after checking the
existing outputs; they are not presented as original fit-time records.

The audit does not constitute a new model fit or independent scientific review.
The external-data limitations and missing acquisition records documented in
[followup_validation.md](followup_validation.md) remain applicable.
