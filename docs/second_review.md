# Second review analyses

The November 2026 revision preserves the September reference analysis and adds
diagnostics requested during review. Equal-project mean RMSE is now the main
interpretive endpoint. This prioritization is retrospective, not an a priori
analysis plan. The reference stack and robust regression have means of 64.13
and 64.17 m/s; no consistent cross-project advantage has been demonstrated.

Run `python scripts/review_round2.py` and
`python scripts/empirical_benchmarks.py`. Results are in
`results/review_round2/`. Refits repeat inner selection with seed 42 and 300
trees. Existing completed scenarios are skipped; remove a scenario's result
directory deliberately before recomputing that scenario after a code change.

The stress checks compare the workbook's log-depth term with the linear-depth
term printed in Mahler et al. (2026), Eq. (9), unsaturated offsets of 0 and
2 kN/m³, and trapezoidal integration between CPT-derived midpoint densities.
Integration extends the shallowest estimate to ground level. None of these
proxies is independently measured stress. The published linear-depth scenario
has stack mean-project/pooled RMSE 64.83/61.23 m/s; the integrated proxy gives
63.56/59.09 m/s. These are sensitivity results, not a selection of a new model.

Window and no-profile refits assess feature dependence. The no-profile variant
removes relative depth, first differences, rolling means and local residuals.
The one-interval window still retains differences and relative depth. LOPO
refits the complete workflow for 15 outer projects, giving mean-project/pooled
RMSE 63.42/60.29 m/s. It changes the partition and training size; it is neither
external validation nor a full fixed-test delete-one-training-project analysis.

Fixed Robertson (2009), Mayne (2006), Hegazy and Mayne (1995), and Piratheepan and Andrus (2002)
correlations are implemented as tabulated in Mahler et al., Table 4. Coefficients
are not fitted here. They use subsets of the available CPT variables, with kPa
and metre input conventions. The latter relationship's Holocene development
domain differs from the complete Budapest age range. These are representative
comparators, not an exhaustive ranking or an independent test of correlations
previously developed on this database.

Routing reports distinguish threshold eligibility from a nonzero applied
correction: 174/215 versus 143/215 at measured Vs ≥400 m/s, and 38/42 versus
37/42 at ≥500 m/s. Measured targets define diagnostic subsets only. The marginal
extrapolation check compares qc, fs, u2, midpoint depth and effective stress
against each outer training range; it does not test joint feature-space support.

Two archived records have nonpositive interval thicknesses. Their bounds are
listed in `interval_bound_flags.csv`. No boundaries or targets are repaired by
guesswork. Midpoint-based reference validation retains them, while positive
thickness summaries exclude them. Thickness-dependent engineering calculations
need verified source corrections first.

For 12/15 and 15/15 completely covered projects, independent-Bernoulli 95%
Clopper–Pearson reference bounds are 51.9–95.7% and 78.2–100%. Overlap in fitting
and calibration sets means these are descriptive references, not calibrated
regional confidence intervals. Bands remain too wide for design discrimination.

All refits still reuse 15 Budapest projects. No independent projects, original
seismic traces, inversion metadata, measured layer densities or end-use
engineering validation have been added. Transfer to other settlements along
the Danube is a hypothesis requiring local validation.

Source: [Mahler et al. (2026)](https://doi.org/10.1007/s10706-026-03863-7).

The Robertson (2009) net-cone-resistance equation is checked against the author's paper (doi:10.1139/T09-065), rather than substituting Qtn directly into the velocity equation. It uses Pa = 100 kPa, the reference CPT-only stress proxies and two-pass SBTn normalization starting with n = 1. No local coefficients are fitted. Its mean-project/pooled RMSE is 62.05/58.35 m/s, lower than the stack on both endpoints. This result precludes a claim of ensemble superiority over the established comparator.
