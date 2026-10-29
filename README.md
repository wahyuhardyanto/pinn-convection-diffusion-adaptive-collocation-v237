# PINN Convection-Diffusion V2.3.7: Frozen Four-Sampler Benchmark

Reproducibility package for a transient 1D convection-diffusion PINN benchmark comparing four collocation strategies under a frozen, controlled protocol:

- **MC-R**: uniform IID resampling
- **LHS-R**: uniform Latin-hypercube resampling
- **RAD-IID**: residual-adaptive IID sampling with generation-density correction
- **ALHS-R**: residual-adaptive probability-space LHS with generation-density correction

The physical problem uses `c = 1` and `D = 0.001`. All methods share the same PINN formulation, physical-uniform residual target, optimizer schedule, checkpoint rule, and exact shared prefix within each budget-seed block. Adaptive methods use the frozen generation-density (`1/q`) correction.

## Frozen final design

| Item | Frozen value |
|---|---|
| Budgets | `N_f = 250, 500, 1000` |
| Final seeds | `5-34` (30 paired seeds) |
| Methods | MC-R, LHS-R, RAD-IID, ALHS-R |
| Total final runs | 360 |
| Primary endpoint | space-time relative L2 vs FDM (`space_time_rel_l2`) |
| Secondary endpoint | resolved physical residual integral (`resolved_J`) |
| Qualification seeds | `0-4`, excluded from final inference |

## Repository layout

```text
.
├── frozen_protocol/              # byte-preserved scientific protocol and code
│   ├── V237_Frozen_Core/         # core implementation, references, replay tapes
│   ├── FROZEN_PROTOCOL_MANIFEST_V237.json
│   ├── Statistical_Analysis_Plan_V237_Final.md
│   ├── run_v237_final.py
│   ├── statistical_analysis_v237_final.py
│   └── verify_frozen_v237_final.py
├── results/statistical_summary/  # available final statistical output tables
├── docs/                         # provenance and manuscript consistency audits
├── tools/verify_repo_integrity.py
├── REVIEWER_QUICKSTART.md
├── environment.yml
└── requirements-frozen.txt
```

`__pycache__` files from the archival package are intentionally excluded because they are interpreter-specific build artifacts, not scientific source. Source integrity is checked against `scientific_core_source_sha256` in the frozen manifest.

## Reviewer path (no expensive training)

```bash
python tools/verify_repo_integrity.py
```

Then inspect `REVIEWER_QUICKSTART.md` and the tables under `results/statistical_summary/`.

## Reproduction environment

The original frozen runtime recorded in the manifest is:

- Python 3.9.16
- PyTorch 2.8.0
- NumPy 1.25.2
- pandas 2.0.2
- SciPy 1.10.1
- macOS arm64, MPS, 8 threads

The experiment launcher intentionally enforces the frozen environment. `environment.yml` and `requirements-frozen.txt` are convenience files added for repository setup; the manifest remains authoritative.

## Full benchmark

From `frozen_protocol/` in the frozen runtime:

```bash
python verify_frozen_v237_final.py
python run_v237_final.py --dry-run
python run_v237_final.py --methods all
```

Operational batching by method group is permitted by the frozen protocol, but the scientific design must remain unchanged.

## Statistical analysis

The final analysis was pre-specified before inspection of the 30-seed final benchmark. It uses paired, two-sided nonparametric testing with Holm multiplicity adjustment and budget-specific omnibus gates. See `frozen_protocol/Statistical_Analysis_Plan_V237_Final.md`.

A nonsignificant ALHS-R vs RAD-IID contrast is **not** an equivalence claim; no equivalence margin was pre-specified.

## Data availability note

This repository package includes the frozen scientific source, reference arrays/replay materials, qualification summary, and final statistical summary tables available in the project files. The complete raw 360-run output directory was not present in the source package used to build this GitHub bundle. For maximum reproducibility, publish that raw output separately (recommended: archival data repository/DOI or a GitHub Release asset) and add its checksum and permanent link here.

## Citation and license before public release

Complete `CITATION.cff.template` with the final author list and repository/DOI information, rename it to `CITATION.cff`, and select an explicit software/data license. No license is inferred by this package.
