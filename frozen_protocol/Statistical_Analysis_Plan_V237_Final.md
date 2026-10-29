# Statistical Analysis Plan (SAP)
## V2.3.7 Unified Four-Sampler Final Independent Benchmark

**SAP status:** Frozen before inspection of the final 30-seed benchmark results.

**Parent scientific protocol:** `2.3.7-unified-four-sampler1`

**Qualification data:** seeds 0-4, three budgets, four samplers (60 runs). These data were used only for protocol qualification and are excluded from all final confirmatory inference.

**Final independent data:** seeds 5-34 inclusive, three budgets, four samplers (360 runs).

## 1. Scientific question

Under an identical PINN formulation, physical-uniform residual measure, optimizer schedule, checkpoint rule, and shared optimization prefix, how do four collocation strategies compare:

- MC-R: uniform IID resampling;
- LHS-R: uniform Latin-hypercube resampling;
- RAD-IID: residual-adaptive IID sampling with generation-density correction;
- ALHS-R: residual-adaptive probability-space LHS with generation-density correction.

The final analysis is intended to distinguish the effect of residual adaptivity from the smaller incremental effect of probability-space stratification.

## 2. Frozen experiment design

For each seed and active collocation budget, one common warmup+transition prefix is created and reused exactly by all four methods. The shared model parameters, Adam optimizer state, RNG state, and transition point design must match within every seed-budget block.

Fixed active collocation budgets:

- `N_f = 250`
- `N_f = 500`
- `N_f = 1000`

Fixed final seeds:

- `5, 6, ..., 34` (30 paired seeds)

Total final runs:

`4 methods x 3 budgets x 30 seeds = 360 runs`.

No qualification seed (0-4) will be included in the final inferential dataset.

## 3. Frozen computational protocol

The following are fixed and may not be retuned after this SAP is frozen:

- PDE and physical parameters;
- network architecture;
- hard initial and both hard boundary constraints;
- boundary-layer-aware output transformation;
- target residual measure: physical-uniform;
- objective: `importance_uniform`;
- adaptive proposal: current-model `graded101`;
- uniform mixture parameter and adaptive proposal construction as implemented in the frozen source;
- Adam warmup/transition/joint schedule;
- collocation refresh interval;
- L-BFGS schedule and per-generation density weighting;
- checkpoint selection rule: unchanged 31 x 31 validation grid plus BC feasibility;
- FDM primary numerical reference and SG secondary numerical reference;
- resolved post-hoc residual audit;
- all primary, secondary, and diagnostic metrics defined below.

The frozen scientific source is identified by SHA-256 checksums in `FROZEN_PROTOCOL_MANIFEST_V237.json`.

## 4. Analysis population and technical validity

The confirmatory dataset is complete only when all 360 pre-specified runs are present.

A run is technically valid only if:

1. its method, budget, and seed belong to the final design;
2. it has a completed run record;
3. the selected checkpoint is BC-feasible;
4. all required primary/secondary metrics are finite;
5. within each budget-seed block, the four methods have identical frozen prefix identities for the pre-specified shared-prefix hash fields.

A failed or corrupted run is rerun with the same method, budget, seed, software environment, and frozen code. It is not replaced by a new seed. Runs are never removed because of poor accuracy or because they are statistical outliers.

If the complete 360-run dataset cannot be recovered without changing the frozen protocol, confirmatory inference will not be declared complete and the deviation will be reported.

## 5. Endpoints

### 5.1 Primary endpoint

`space_time_rel_l2`

This is the space-time relative L2 error of the PINN solution against the primary FDM reference. Lower is better.

This endpoint is the sole primary confirmatory outcome.

### 5.2 Secondary physics endpoint

`resolved_J`

This is the post-hoc resolved physical-domain integral of squared PDE residual. Lower is better.

It is analyzed with the same paired nonparametric framework as the primary endpoint, but in a separate secondary family. It cannot replace or override the primary endpoint conclusion.

### 5.3 Robustness reference outcome

`pinn_vs_sg_space_time_rel_l2`

This repeats the global field-error evaluation against the independent Scharfetter-Gummel reference. It is a robustness/sensitivity outcome and is not used to redefine the primary conclusion.

### 5.4 Diagnostic outcomes

The following are descriptive/explanatory only:

- `final_rel_l2`;
- `resolved_right_001_rms`;
- `resolved_right_0001_rms`;
- `independent_residual_rms`;
- runtime components;
- collocation coverage diagnostics;
- proposal construction cost.

No confirmatory superiority claim will be based solely on a diagnostic outcome.

## 6. Descriptive statistics

For every method-budget combination and every reported endpoint, report:

- `n`;
- median;
- first quartile (Q1);
- third quartile (Q3);
- interquartile range (IQR);
- minimum;
- maximum.

For paired method comparisons also report:

- median paired difference;
- number of seeds favoring each method;
- number of exact ties;
- matched rank-biserial correlation.

Plots will show individual paired seeds whenever practical; summary bars alone will not replace run-level dispersion.

## 7. Primary confirmatory inference

All confirmatory tests are two-sided with nominal alpha = 0.05.

### 7.1 Omnibus sampler effect

At each budget separately, apply a Friedman test across the four methods using seed as the paired block.

This produces three primary omnibus p-values (one per budget). Apply Holm adjustment across these three p-values.

A budget passes the omnibus gate when the Holm-adjusted Friedman p-value is < 0.05.

### 7.2 Pairwise sampler comparisons

At each budget, compute all six paired method contrasts using the two-sided Wilcoxon signed-rank test:

- MC-R vs LHS-R;
- MC-R vs RAD-IID;
- MC-R vs ALHS-R;
- LHS-R vs RAD-IID;
- LHS-R vs ALHS-R;
- RAD-IID vs ALHS-R.

Across all three budgets this yields 18 primary pairwise tests. Apply Holm adjustment jointly across all 18 pairwise p-values.

A pairwise result is interpreted as confirmatory only when:

1. the corresponding budget passed the Holm-adjusted Friedman omnibus gate; and
2. the pairwise Holm-adjusted p-value is < 0.05.

All pairwise results are still reported even when the omnibus gate is not passed, but are then descriptive only.

### 7.3 A priori contrasts of special scientific interest

The following contrasts are highlighted because they directly answer the study questions, but they receive no special multiplicity exemption:

- RAD-IID vs MC-R: effect of residual adaptivity over IID uniform sampling;
- ALHS-R vs MC-R: combined residual adaptivity plus stratification over IID uniform sampling;
- ALHS-R vs RAD-IID: incremental effect of probability-space stratification beyond residual adaptivity.

Failure to reject ALHS-R vs RAD-IID is not interpreted as equivalence. No equivalence margin has been pre-specified.

## 8. Effect size

For every paired Wilcoxon contrast, report the matched rank-biserial correlation computed from signed ranks of nonzero paired differences.

For a contrast labeled `A vs B`, the implementation defines the signed difference as `B - A`; therefore a positive rank-biserial value favors method A because lower endpoint values are better.

Also report the median of `A - B` so the direction remains transparent in original metric units.

## 9. Secondary resolved-J inference

Apply the same procedure to `resolved_J` as a separate secondary family:

1. three Friedman tests (one per budget), Holm-adjusted across the three budgets;
2. 18 paired Wilcoxon tests, Holm-adjusted jointly across all 18 secondary pairwise tests;
3. confirmatory secondary interpretation only when the corresponding secondary omnibus gate is passed.

The secondary family is not pooled with the primary family for multiplicity correction because it is explicitly labeled secondary and cannot rescue a nonsignificant primary endpoint.

## 10. Budget-effect analysis

Budget dependence is a secondary analysis performed separately within each sampler.

For each method, apply a Friedman test across `N_f = 250, 500, 1000` using seed as the paired block. Holm-adjust the four method-specific omnibus p-values.

For budget pairwise comparisons (250 vs 500, 250 vs 1000, 500 vs 1000), compute paired Wilcoxon tests. Across four methods this gives 12 comparisons; apply Holm adjustment across all 12.

Pairwise budget effects are interpreted inferentially only for methods whose Holm-adjusted budget omnibus test is significant.

The same budget-effect analysis may be repeated for `resolved_J` as secondary evidence.

## 11. Reference-solution robustness analysis

Repeat descriptive summaries for `pinn_vs_sg_space_time_rel_l2` and compare its qualitative method ranking with the primary FDM-based endpoint.

If the FDM and SG conclusions disagree materially, both results are reported and the discrepancy is investigated as a numerical-reference sensitivity issue. The primary endpoint remains the pre-specified FDM-based `space_time_rel_l2`.

## 12. Runtime and computational-cost interpretation

Runtime and proposal-evaluation costs are descriptive only.

MC-R/LHS-R and RAD-IID/ALHS-R do not have equal proposal-construction cost. Therefore:

- active collocation-point efficiency may be discussed;
- equal-compute efficiency will not be claimed unless supported by a separate pre-specified cost-normalized experiment;
- adaptive candidate-grid evaluations are reported separately from active collocation count.

## 13. Missing data, ties, and numerical edge cases

- No statistical imputation is permitted.
- Exact zero paired differences are excluded from signed-rank sums according to Wilcoxon `zero_method='wilcox'`.
- The frozen implementation uses SciPy's two-sided Wilcoxon signed-rank test with `correction=False` and `mode='auto'` under the frozen SciPy version.
- If all paired differences for a Wilcoxon contrast are zero, the p-value is set to 1 and the rank-biserial effect size to 0.
- If all treatments are exactly tied in every block of a Friedman comparison, the Friedman statistic is set to 0 and p-value to 1.
- Non-finite primary or secondary metrics trigger a technical-run audit rather than automatic exclusion.

## 14. Outlier policy

No accuracy result is excluded on statistical-outlier grounds. All technically valid runs remain in the primary dataset, including rare poor solutions.

Any sensitivity analysis that removes or winsorizes observations would be post hoc, explicitly labeled as such, and cannot replace the frozen analysis.

## 15. Stopping rule

There is no efficacy, futility, or significance-based early stopping. The final benchmark is evaluated only after all 360 pre-specified runs are complete and pass integrity checks.

The seed count will not be increased or decreased in response to observed p-values or effect sizes.

## 16. Software and reproducibility

The final training experiment must use the frozen qualification environment recorded in `FROZEN_PROTOCOL_MANIFEST_V237.json`.

The statistical analysis script `statistical_analysis_v237_final.py` is frozen before final results are inspected. It validates the final design, checks paired-prefix integrity, and writes all pre-specified summary/test tables.

Any change to scientific code, endpoints, multiplicity procedure, seed range, or analysis logic after the final experiment begins must be documented as a protocol deviation and cannot silently replace this SAP.

## 17. Planned manuscript language

The final manuscript will distinguish:

- protocol qualification (seeds 0-4), used to freeze the method and SAP;
- final independent confirmatory benchmark (seeds 5-34), used for inferential claims.

A method will not be called universally superior from one budget alone. Conclusions will be budget-specific and will distinguish the dominant effect of residual adaptivity from any incremental effect of probability-space stratification.
