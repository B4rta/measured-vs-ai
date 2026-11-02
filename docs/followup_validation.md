# Follow-up validation on 2 November 2026

This follow-up adds 60 fixed-test training-project deletion refits, five controlled
profile perturbations and a frozen assessment of an age-free companion on public
seismic CPTu records. The age-aware primary model, its reference predictions and
its uncertainty experiment are unchanged. No replacement is selected from these checks.

## Fixed-test influence

For each original outer fold, remove each training project in turn, repeat all
inner selection and fitting, and predict the unchanged outer test projects.
Stack fold RMSE changes range from −8.12 to +9.38 m/s across 60 refits. Combining
predictions over the same 1,304 rows for each deleted project gives mean-project
RMSE 62.97–66.24 m/s and pooled RMSE 59.64–62.79 m/s. The fold where that project
is itself held out already excludes it from training and retains its reference
prediction. Overlapping fits do not provide 60 independent replications.

## Profile perturbations

All scenarios repeat inner selection and fitting with the original project
allocation, seed 42 and 300 trees. Scores concern the specialist stack, in m/s.

| Scenario | Rows | Mean project RMSE | Pooled RMSE | Reference pooled RMSE on same rows |
|---|---:|---:|---:|---:|
| Physical 3 m and 5 m windows | 1,304 | 64.32 | 60.43 | 60.56 |
| Context on interpolated 0.5 m grid | 1,304 | 64.97 | 61.57 | 60.56 |
| Context on interpolated 2 m grid | 1,304 | 64.22 | 60.74 | 60.56 |
| Omit every fifth interval | 1,072 | 65.82 | 61.21 | 61.15 |
| Omit every third interval | 895 | 63.66 | 60.27 | 60.67 |

Physical windows average logged CPT quantities at midpoints within 1.5 or 2.5 m
of each midpoint; original first differences remain. Resampling interpolates
existing interval CPT inputs, rebuilds context and maps context back to the
original scoring rows, retaining raw predictors and measured targets there.
Duplicate midpoint inputs are averaged only for interpolation. No Vs values
are interpolated. Periodic omissions occur in both training and test profiles
before feature construction, independently of Vs and residuals. Scores concern
retained rows only. These simulations cannot recover lost within-interval
structure, validate actual field dropouts, predict missing depths or establish
performance on original 2 cm logs.

## Frozen public-data assessment

The [Premstaller publication](https://doi.org/10.1016/j.dib.2020.106618) and
[university archive](https://www.tugraz.at/fileadmin/user_upload/Institute/IBG/Datenbank/Database_CPT_PremstallerGeotechnik.zip)
provide the source records. Archive SHA-256:
`4fb340af34321676d10e7502f78b2059c99250ff25b26a2fcb55b603399c3ddd`.
The assessed subset has 2,229 usable intervals from 50 seismic CPTu soundings
in five basins/valleys. Project identities are anonymized: 50 soundings are
not 50 independent projects. We have not reprocessed original seismic waveforms.

Because per-interval geological age is absent, both Budapest and external inputs
use a constant age category, removing that predictor, the Tertiary specialist
and the Tertiary sample boost. The remaining model family, high-Vs specialist,
search, seed and tree count follow the reference. This companion has its own
five-fold Budapest assessment and one final fit on all 15 Budapest projects.
External targets enter neither fitting, preprocessing estimates nor selection.
The test was designed during review, not preregistered. No external result
is used to select a replacement model or change exclusions.

Each positive Vs depth is treated as an interval bottom; the preceding positive
Vs depth is the top. The first uses the published nominal 0.5 m spacing. This is
an explicit support assumption rather than authenticated acquisition geometry.
Raw inputs are averaged over (top, bottom]; source qt is retained, fs and u2
converted from kPa to MPa, and Rf computed as 100 times mean fs divided by mean qc.
Groundwater is median z−u0/9.81 where source u0 is positive, clamped at zero:
a source hydrostatic assumption, not a verified field water level. Reference
CPT-only stress proxies are rebuilt. Nonpositive Vs, nonadvancing depths and
intervals lacking finite positive mean qc/qt/fs or finite mean u2 are flagged.
No retained target or extreme prediction is trimmed.

| Component | Budapest mean project RMSE | Budapest pooled RMSE | External mean sounding RMSE | External pooled RMSE |
|---|---:|---:|---:|---:|
| Huber regression | 64.17 | 62.05 | 5,740.43 | 36,863.43 |
| Random Forest | 67.35 | 63.61 | 87.28 | 99.27 |
| Extra Trees | 67.20 | 63.04 | 91.26 | 103.50 |
| Base stack | 64.84 | 61.65 | 187.58 | 675.47 |
| Specialist stack | 64.43 | 60.93 | 92.42 | 112.58 |

RMSEs are in m/s. Equal-sounding and equal-project means use different units
and are not equivalent. External specialist-stack R² is −0.169. RF gives the
lowest external pooled error among these components; observing this does not
select a replacement after seeing the external targets.

Of 2,229 intervals, 1,588 are outside at least one checked Budapest marginal
range (qc, qt, fs, u2, Rf, Bq, effective stress, depth or groundwater). The largest
regression prediction is approximately 1.70 million m/s at AT_619, midpoint
3.25 m: mean qc 0.0032 MPa, mean qt 0.0122 MPa and Rf 324%, versus maximum
Budapest Rf 11.07%. The next is approximately 393,000 m/s at Rf 274.75%.
These expose unbounded log-regression extrapolation and are not plausible
velocities. Four rows have missing Bq from nonpositive net resistance and
use training-fitted imputation. Range flags are descriptive, never score filters.
Source anomalies and support assumptions limit interpretation but do not justify
hiding failure. At measured Vs ≥500 m/s, stack RMSE is 528.24 m/s and bias
−453.09 m/s over 43 rows from 19 soundings.

The test supplies geographically separate observations for a companion variant.
It does not validate the unchanged age-aware primary model, in-domain transfer
to Budapest or Danube settlements, or the project uncertainty bands. It provides
no evidence of reliable high-Vs transfer.

## Other sources and remaining gaps

The downloaded [North Sea data](https://doi.org/10.5281/zenodo.10986519) contain
2,905 paired rows, 143 soundings and seven projects, with only five Vs ≥500 m/s
rows. They are screened rather than scored as complete-profile onshore validation:
age labels, raw profile support and offshore hydrostatics require separate treatment.
The downloaded [TU Graz laboratory data](https://doi.org/10.5281/zenodo.21793782)
report bulk and dry unit weights. Dry weight is not unsaturated bulk weight,
and bulk weights are not explicitly saturated. They cannot verify the original
saturated-minus-unsaturated assumption or recover Budapest layer densities.
See `results/public_external/data_screening.json` for columns and checksums.

Original Budapest acquisition and processing records, CPT–seismic distances,
measured layer densities and saturation, independent local project identities
and complete engineering case histories remain unavailable. Downloads cannot
replace source-specific records. No Vs,30, G0 or site-response accuracy is validated.

## Reproduction and attribution

```bash
python scripts/followup_validation.py
python scripts/public_external_validation.py /path/to/Database_CPT_PremstallerGeotechnik.zip
python scripts/audit_followup.py
python -m unittest discover -s tests -v
```

Outputs: `results/followup_validation/` and `results/public_external/`. Scripts
resume from prediction caches; use a fresh output tree after changing inputs,
code or environment. Manifests, predictions, selections, per-sounding errors,
velocity subsets and range flags are retained. Raw archives are separate.
External derived tables adapt Oberhollenzer S, Premstaller M, Marte R,
Tschuchnigg F, Erharter GH, Marcher T, *Cone penetration test dataset Premstaller
Geotechnik*, Data in Brief 34 (2021), 106618, doi:10.1016/j.dib.2020.106618.
The source publication is CC BY 4.0; changes comprise interval aggregation,
unit conversion, explicit support assumptions and added model predictions.
