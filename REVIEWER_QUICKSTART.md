# Reviewer quick start

This repository is organized so that verification can be performed without launching the expensive 360-run benchmark.

## 1. Verify repository integrity

From the repository root:

```bash
python tools/verify_repo_integrity.py
```

This validates the frozen source-file SHA-256 values recorded in the protocol manifest. It intentionally does not require the original macOS/MPS runtime.

## 2. Inspect the frozen design

Read:

- `frozen_protocol/FROZEN_PROTOCOL_MANIFEST_V237.json`
- `frozen_protocol/Statistical_Analysis_Plan_V237_Final.md`
- `docs/PROVENANCE.md`

## 3. Inspect final statistical summaries

The available reported-analysis tables are in:

```text
results/statistical_summary/
```

These include primary and secondary Friedman tests, paired Wilcoxon contrasts, descriptive summaries, budget-effect analyses, and SG robustness summaries.

## 4. Reproduce the frozen launcher preflight

The original launcher is retained under `frozen_protocol/`. It was designed for the frozen macOS arm64/MPS environment and may reject a different runtime by design.

```bash
cd frozen_protocol
python verify_frozen_v237_final.py
python run_v237_final.py --dry-run
```

## 5. Full 360-run reproduction

Only use the frozen environment specified in the manifest. Then:

```bash
cd frozen_protocol
python run_v237_final.py --methods all
```

The fixed design is 4 methods x 3 budgets x 30 paired seeds = 360 runs. Qualification seeds 0-4 are excluded from final inference.

## 6. Re-run the pre-specified statistics

After a complete final output directory is available:

```bash
cd frozen_protocol
python statistical_analysis_v237_final.py \
  --results pde_v237_final_independent \
  --output-dir ../recomputed_statistical_analysis
```

The frozen analysis plan remains authoritative. Do not reinterpret nonsignificant RAD-IID vs ALHS-R comparisons as equivalence.
