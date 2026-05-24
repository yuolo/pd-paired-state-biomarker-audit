# Symptom-axis analysis plan

The Boundary-3 symptom-specific axis analysis (`analysis/4_symptom_axis/`) followed a
fixed testing procedure, defined before the clinical-label modelling, to guard against
selective reporting at this cohort size.

The plan fixed, in advance:
- the unit of analysis (subject x hemisphere, up to 36 rows);
- primary targets (contralateral akinesia-rigidity and tremor responses) with the
  ipsilateral side as a built-in control;
- the five physiologically defined transition components;
- a tremor effective-sample audit and a component collinearity audit (|rho| >= 0.70);
- FDR correction across the full family (4 symptoms x 6 components x 2 lateralities);
- ridge regression (alpha = 1.0, no tuning), LOSO with both hemispheres of the held-out
  subject removed together, a permutation null preserving subject clusters, and an
  honest negative-result rule;
- the MAGE-G geometry-distance constants, fixed before the OXF retrieval audit.

This protocol was not deposited in a public time-stamped registry. It is documented here
and in the manuscript Methods for transparency, and the negative symptom-prediction
result rests on the full-family FDR, LOSO, permutation null, and the honest
negative-result rule described above.
