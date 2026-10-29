# V2.3.7 — Unified Four-Sampler Protocol

V2.3.7 is the consolidation stage after the V2.3.5 objective ablation and V2.3.6 replay-to-online transfer. It does **not** change the PINN formulation again. Its purpose is to compare MC-R, LHS-R, RAD-IID and ALHS-R under one protocol.

## Core fairness rule

For every `(N_f, seed)` pair, warmup and transition are computed **once**. The exact model parameters, Adam state, RNG state and transition collocation set are saved as one common prefix. All four methods then restore that same prefix before joint training.

Thus the branch point is:

`one common warmup + transition -> {MC-R, LHS-R, RAD-IID, ALHS-R}`

The transition design is a uniform LHS and is method-independent.

## Unified objective

All four methods target the same physical-uniform mean-squared PDE residual. On the unit domain:

`L_PDE = mean( r(x,t)^2 / q_gen(x,t) )`

where `q_gen` is the generating density for each collocation point.

- MC-R: uniform IID, `q=1`, so the weight is exactly 1.
- LHS-R: uniform probability-space LHS, `q=1`, so the weight is exactly 1.
- RAD-IID: current-model graded101 proposal, IID probability coordinates, weight `1/q`.
- ALHS-R: the same current-model graded101 proposal, LHS probability coordinates, weight `1/q`.

No self-normalization and no weight clipping are used. For L-BFGS, each occurrence retains the density that generated it; duplicate-coordinate contributions are summed.

## PINN/training protocol retained

- PDE: `u_t + c u_x - D u_xx = 0`, `c=1`, `D=0.001`.
- Network: 2-64-64-64-1 tanh.
- V2.3.2 hard IC + hard left/right BC boundary-layer-aware output transform.
- Warmup 100 epochs; transition 900; joint 4000.
- Adam lr 1e-3; collocation refresh every 250 joint epochs.
- L-BFGS max 300 with union of the last four collocation sets.
- Checkpoint selection remains the unchanged grid31 + BC-feasible rule for this consolidation pilot.
- FDM and Scharfetter-Gummel remain evaluation references, never training targets.
- Resolved post-hoc residual audit remains part of every run.

## Presets

### Smoke

Four short runs, one per sampler, at `N_f=48`, seed 0. This verifies that all four branches share exactly the same prefix.

```bash
python convection_diffusion_unified_v237.py \
  --preset smoke \
  --output-root pde_v237_outputs \
  --device auto
```

### Pilot (recommended next step)

`4 methods x 3 budgets x 3 seeds = 36 runs`:

- budgets: 250, 500, 1000
- seeds: 0, 1, 2

```bash
python convection_diffusion_unified_v237.py \
  --preset pilot \
  --output-root pde_v237_outputs \
  --device auto
```

The runner supports incremental execution, e.g. one method at a time. Completed runs are skipped and the shared prefix is reused.

### Confirm

`4 x 3 x 5 = 60` runs. Do this only after the 36-run pilot is inspected and the protocol is frozen.

### Full

`4 x 3 x 30 = 360` runs. The preset exists for later use, but it is **not recommended yet**.

## Outputs

Each stage writes:

- `results_unified.csv`
- `summary.csv`
- `paired_differences.csv`
- `integrity.json`
- `manifest.json`
- `shared_prefix/nf{budget}_seed{seed}/common_transition.pt`
- per-run checkpoints, objective traces, sampling/PDF histories and resolved residual audits.

The report code asserts that all available methods for a `(budget, seed)` share the same checkpoint, model, Adam, RNG and transition-point hashes.

## Interpretation

The 36-run pilot is exploratory. Do not infer general sampler superiority from three seeds. Its job is to verify:

1. four-way protocol fairness,
2. stability across the three budgets,
3. whether adaptive methods retain about 1% global accuracy under the unified objective,
4. whether the known final-time and right-boundary residual limitations remain manageable,
5. whether a confirmatory/final experiment can be frozen without further algorithm changes.

Historical V2.3.3–V2.3.6 results remain valid ablation/diagnostic records but are not merged with V2.3.7 as if they were produced by this unified protocol.
