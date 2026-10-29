# Training Protocol Audit — V2.3.7

## Changed intentionally

1. The four methods now branch from **one method-independent prefix per budget and seed**.
2. MC-R and LHS-R are brought into the same `importance_uniform` objective family. Because their generating density is uniform on the unit domain, their importance weights are identically one.
3. RAD-IID and ALHS-R use the V2.3.6 online graded101 current-model proposal with the same ordinary `1/q` correction.
4. Budgets 250, 500 and 1000 are supported in one runner.

## Held fixed

PINN formulation, physical parameters, architecture, staged curriculum, Adam hyperparameters, refresh interval, L-BFGS setup, boundary feasibility rule, grid31 checkpoint selection, FDM/SG references and resolved post-hoc diagnostics are held fixed.

## Why historical runs are not reused as final four-way comparisons

V2.3.6 paired RAD/ALHS online runs to their own historical shared prefixes. Those prefixes were controlled within each method but were not identical across RAD and ALHS for every seed. V2.3.7 fixes this by constructing one common prefix for all four methods. Therefore historical runs remain diagnostic evidence, not interchangeable rows of the new unified benchmark.

## Pilot decision rule

The 36-run pilot is accepted as a protocol check if:

- all shared-prefix hashes match across the four methods for every completed `(N_f,seed)`,
- no method uses FDM/SG for training or checkpoint selection,
- all selected checkpoints remain BC feasible,
- no non-finite metrics occur,
- the online adaptive proposal actually performs the expected candidate evaluations,
- the global, final-time and resolved residual metrics are inspectable without changing training rules after seeing outcomes.

No condition requires ALHS-R to win.
