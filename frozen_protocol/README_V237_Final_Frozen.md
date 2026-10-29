# V2.3.7 Frozen Final Independent Benchmark

This package freezes the scientific V2.3.7 four-sampler protocol after the 60-run qualification stage (seeds 0-4).

## Final experiment

The final inferential dataset is deliberately independent of the qualification data:

- methods: MC-R, LHS-R, RAD-IID, ALHS-R;
- budgets: 250, 500, 1000 active collocation points;
- final seeds: 5-34 inclusive;
- total: 360 runs;
- qualification seeds 0-4 are excluded from final inference.

The directory `V237_Frozen_Core/` is an exact snapshot of the V2.3.7 scientific package used for the qualification runs. Do not edit files inside it.

## Environment

Use the same `shuttlecock-rad280` environment as the qualification experiment. The final launcher rejects an environment that differs from the frozen manifest.

## Preflight

From this package directory:

```bash
python verify_frozen_v237_final.py
python run_v237_final.py --dry-run
```

The first command verifies source and protocol-file checksums. The second shows the fixed final command without training.

## Run all 360 tasks

```bash
python run_v237_final.py --methods all
```

Default output root:

```text
pde_v237_final_independent/
```

The scientific core is invoked as:

```text
--preset full
--budgets 250,500,1000
--seeds 5-34
--device auto
--threads 8
```

These values are frozen by the launcher.

## Operational batching is allowed

If desired, run two method groups sequentially using the same output root:

```bash
python run_v237_final.py --methods mc,lhs
python run_v237_final.py --methods rad,alhs
```

This is only operational batching. It does not change the pre-specified 360-run design or the common-prefix protocol. Do not run two processes concurrently against the same output root.

## Resume behavior

The original V2.3.7 runner verifies completed tasks and skips them. Preserve partial output directories. Do not delete poor-performing runs and do not replace a failed run with a different seed.

## Recompute report only

```bash
python run_v237_final.py --report
```

## Statistical analysis

Do not run inferential analysis until the final integrity file reports 360/360 completed.

Then run:

```bash
python statistical_analysis_v237_final.py \
  --results pde_v237_final_independent \
  --output-dir v237_final_statistical_analysis
```

The script rejects qualification seeds 0-4, incomplete final designs, duplicate rows, BC-infeasible selections, non-finite endpoints, and shared-prefix mismatches.

The full pre-specified analysis is documented in `Statistical_Analysis_Plan_V237_Final.md`.

## What is frozen

No post-qualification retuning is permitted for the PINN formulation, objective, proposal, optimizer schedule, checkpoint selection, budgets, final seed count, primary endpoint, or multiplicity procedure.

If a scientific change becomes necessary, preserve this package and create a new protocol version rather than editing V2.3.7 in place.
