# September 2026 analysis revision

## Evaluation

Five outer project folds contain three inner project folds. Inner predictions select base weights (0.05 grid), specialist weights (0.05 grid), thresholds (325, 350, 375 m/s) and caps (0.50, 0.65, 0.80). Numerical ties prefer a smaller cap and then a smaller correction. Each tree has 300 estimators, fixed before the revised runs; this is not an exact rerun of the historical 500/700-tree model. Global RF/ET minimum leaves are 2, and tertiary/SCPT/high-Vs specialist minimum leaves are 2/1/1. Minimum specialist sizes are 60/40/80. The primary setting excludes modality predictors, the SCPT specialist and its sample-weight boost.

Historical architecture decisions were informed by this database. Nesting protects the current fitted weights and gates but cannot undo prior researcher adaptation. Independent regional validation is still needed.

## Source audit

The workbook has 1,304 actual records on CPT_SCPT_MASW, despite a formatted extent of 6,057 rows. Raw inputs, identifiers, targets and modality labels match the archived CSV. All rows were retained, with no missing raw inputs or complete duplicates. Fourteen u2 values are nonpositive; their raw signed values are preserved. The log transform floors its shifted argument at 1e-8.

Unit weights are reconstructed from the CPT-only column S, not the target-dependent Q/R/T or original stress columns. The formula uses log10 for depth, fs in kPa and qt in kPa. All 1,304 values match source column S. The reconstructed range is 15.85–24.98 kN/m³, compared with 21.58–42.58 in the archived cleaned table. The published expression and workbook differ in the depth term; the revision explicitly implements the workbook expression.

The overburden proxy is gamma_unsat*min(z,gwl) + gamma_sat*max(z-gwl,0); effective stress subtracts hydrostatic pressure. This assumes a homogeneous column with local unit weight, not layer-by-layer integration. gamma_unsat = gamma_sat − 1 kN/m³ is an approximation. No measured Vs enters the reconstruction.

## Acquisition information

The published source specifies a maximum 15 m horizontal pairing offset and seismic-layer averaging. Individual offsets, inversion software/settings, mode picks and inversion ensembles are absent from the workbook. It contains interval averages, not the 2 cm field CPT streams or seismic traces. Representative figures are labelled accordingly.

The source publication reports 261 tomography records. The workbook labels every non-SCPT record MASW, so revised reports call them combined surface-wave records. No row-level tomography assignment is inferred. The published map is reproduced with attribution; it does not supply numeric coordinates for spatial validation.

## Interpretation

The paired project-average improvement is essentially zero, with a bootstrap interval spanning zero. The omission analysis removes already held-out errors and does not refit models. Seeds share a fixed outer partition. High-velocity observations remain strongly underpredicted.

Separate random-project conformal splits reserve nine calibration projects and three test projects, leaving three fitting projects. Maximum project residuals target conservative simultaneous coverage; they are not locally calibrated row-level intervals. Project exchangeability and comparable profile sampling are assumptions, and fifteen test projects give limited empirical evidence.

The historical SHAP, boosting and sensitivity artifacts use a different analysis specification. They have not been reused to claim superiority of the revision.

