# Consistency audit: Sections 2-3 against the frozen V2.3.7 manuscript

## Audit basis

The audit uses the final front matter, the frozen V2.3.7 protocol, the pre-specified Statistical Analysis Plan (SAP), and the final Sections 4-6. The objective is terminological and methodological consistency, not post-hoc retuning.

## Resolved inconsistencies

| Area | Earlier wording/problem | Audited replacement/status |
|---|---|---|
| Problem class | Legacy text referred to shuttlecock dynamics, angular trajectories, a second-order IVP, and initial angle/angular velocity. | Replaced throughout with the transient 1D convection--diffusion PDE, sinusoidal IC, and homogeneous Dirichlet BCs. |
| Numerical viewpoint | Sampling was described mainly as an ML design choice. | Reframed as part of a mesh-free strong-form collocation discretization, consistent with the Introduction and Discussion. |
| Hard constraints | Earlier wording described a quadratic transform for two ODE initial conditions. | Replaced with the frozen V2.3.2/V2.3.7 hard IC + both-BC trial field and the outflow-layer basis. |
| Boundary-layer scale | The role of the thin outflow structure was not consistently connected to the formulation. | Explicitly defines \(\delta=D/|c|=10^{-3}\), explains the boundary-layer-aware basis, and distinguishes it from the resolved post-hoc audit. |
| Continuous residual objective | Adaptive sampling was described without always separating point density from target measure. | Defines the physical-uniform \(J=\int_\Omega r^2\) and the generation-density-corrected empirical objective \(N_f^{-1}\sum r_i^2/q_i\). |
| Importance correction | Potential ambiguity between cell probability and continuous density. | States explicitly that \(q_{ij}=P_{ij}/A_{ij}\) is the physical continuous density used in \(1/q\) weighting. |
| Adaptive proposal | Older descriptions were generic residual-weighted/adaptive LHS. | Uses the frozen current-model `graded101` proposal, area-weighted cell masses, \(\eta=0.10\), 101x101 candidates, and distinct RAD-IID vs ALHS-R probability-space draws. |
| LHS interpretation | Could be read as physical-space stratification for the adaptive method. | Clarifies that ALHS-R is stratified in probability space after the residual-informed proposal; it does not guarantee uniform physical stratification. |
| Weight policy | Clipping/self-normalization were not always explicit. | States no clipping and no self-normalization; L-BFGS retains per-generation densities and aggregates duplicate occurrences. |
| Optimizer fairness | Historical adaptive runs did not share every cross-method prefix. | Final Methods now specifies one exact method-independent warmup+transition prefix for all four samplers within every budget-seed block. |
| Warmup | A nominal 100-epoch warmup could be misread as additional fitting under the hard formulation. | Clarifies that it is retained for lineage but is effectively inactive because hard IC/BC losses have zero gradients. |
| Checkpoint selection | Resolved \(J\) could be mistaken for a selection criterion. | Explicitly keeps the unchanged 31x31 BC-feasible selector and states that resolved \(J\) is post-hoc only. |
| Numerical references | FDM/SG could be mistaken for supervised targets. | States that both are evaluation-only references and are never used for training or checkpoint selection. |
| Historical development | Qualification stages risk being mixed with the final benchmark. | Historical V2.3.3--V2.3.6 results are retained only as diagnostic protocol-development evidence. |
| Final population | Qualification seeds could be conflated with final inference. | Explicitly separates seeds 0-4 (qualification) from unseen seeds 5-34 (final confirmatory dataset, 360 runs). |
| Endpoint hierarchy | Earlier text could allow residual diagnostics to substitute for solution error. | Aligns with the SAP: global space-time relative L2 is sole primary endpoint; resolved \(J\) is a separate secondary physics endpoint; SG and local/final-time metrics are robustness/diagnostic outcomes. |
| Statistical multiplicity | Generic nonparametric testing language was insufficient. | Records the frozen Friedman/Holm omnibus gates, 18 Holm-adjusted pairwise Wilcoxon tests, matched rank-biserial effects, and separate secondary family for \(J\). |
| Computational efficiency | Smaller active \(N_f\) could be read as equal wall-clock cost. | Explicitly reports 163,216 adaptive candidate residual evaluations/run and limits claims to active-collocation efficiency, not equal-compute superiority. |
| Sampler ranking | Earlier framing could imply ALHS was the proposed winner. | Related Work and Methods now pose residual density vs incremental probability-space stratification as separate questions; no method is required to win. |

## Cross-section terminology frozen in the master

- **MC-R**: uniform IID resampling.
- **LHS-R**: uniform probability-space Latin-hypercube resampling.
- **RAD-IID**: current-model residual-adaptive proposal sampled with IID probability coordinates, with ordinary \(1/q\) correction.
- **ALHS-R**: the same current-model residual-adaptive proposal sampled with LHS probability coordinates, with ordinary \(1/q\) correction.
- **Primary endpoint**: global space-time relative \(L_2\) error against the FDM reference.
- **Secondary physics endpoint**: resolved physical residual integral \(J\).
- **Robustness reference**: Scharfetter--Gummel global field error.
- **Active collocation budget**: \(N_f\in\{250,500,1000\}\); this is not an equal-compute budget.

## Items intentionally not changed

1. The 31x31 checkpoint selector remains unchanged even though the resolved audit shows boundary-layer under-resolution. This preserves the frozen protocol.
2. Sections 4-6 are copied from the previously approved final 30-seed versions and are not reinterpreted during this audit.
3. The final conclusions remain problem-specific; no universal sampler ranking or statistical equivalence claim is introduced.
4. Author names, affiliations, funding, disclosure details, and repository identifiers are left as submission-stage metadata because they were not supplied for this manuscript in the current workflow.
