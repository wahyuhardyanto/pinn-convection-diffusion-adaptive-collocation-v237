# PINN Convection-Diffusion V2.3.7: Frozen Four-Sampler Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23794781.svg)](https://doi.org/10.5281/zenodo.23794781)

Reproducibility repository for a **transient 1D convection-diffusion physics-informed neural network (PINN) benchmark** comparing four collocation strategies under a frozen and controlled protocol:

- **MC-R** — uniform IID resampling;
- **LHS-R** — uniform Latin-hypercube resampling;
- **RAD-IID** — residual-adaptive IID sampling with generation-density correction;
- **ALHS-R** — residual-adaptive probability-space Latin-hypercube sampling with generation-density correction.

The physical problem uses `c = 1` and `D = 0.001`. All four methods use the same PINN formulation, physical-uniform residual target, optimizer schedule, checkpoint rule, and an exact shared optimization prefix within each budget-seed block. For the adaptive samplers, the frozen protocol applies generation-density (`1/q`) correction so that the training objective remains tied to the same physical-uniform residual measure.

> **Frozen scientific version:** `2.3.7-unified-four-sampler1`  
> **Freeze ID:** `V237-final-independent-30seed-freeze1`  
> **Freeze date:** 2026-10-22

---

## Scientific question

Under an identical PINN formulation and optimization protocol, how do uniform and residual-adaptive collocation strategies compare when the target residual measure is held fixed?

The benchmark is designed to separate:

1. the effect of **residual adaptivity** relative to uniform sampling; and
2. the smaller incremental effect of **probability-space stratification** in ALHS-R relative to RAD-IID.

The complete pre-specified analysis is given in [`frozen_protocol/Statistical_Analysis_Plan_V237_Final.md`](frozen_protocol/Statistical_Analysis_Plan_V237_Final.md).

---

## Frozen final design

| Item | Frozen value |
|---|---|
| PDE | transient 1D convection-diffusion |
| Physical parameters | `c = 1`, `D = 0.001` |
| Active collocation budgets | `N_f = 250, 500, 1000` |
| Final independent seeds | `5-34` (30 paired seeds) |
| Methods | MC-R, LHS-R, RAD-IID, ALHS-R |
| Total final runs | `4 x 3 x 30 = 360` |
| Primary endpoint | `space_time_rel_l2` against FDM |
| Secondary endpoint | `resolved_J` |
| Robustness reference | `pinn_vs_sg_space_time_rel_l2` |
| Qualification seeds | `0-4`, excluded from final inference |
| Confirmatory tests | paired, two-sided; nominal `alpha = 0.05` |
| Multiplicity control | Holm adjustment with budget-specific omnibus gates |

The authoritative frozen design and source hashes are stored in [`frozen_protocol/FROZEN_PROTOCOL_MANIFEST_V237.json`](frozen_protocol/FROZEN_PROTOCOL_MANIFEST_V237.json).

---

## Key findings

The final statistical conclusions are **budget-specific** and should not be interpreted as universal superiority of any sampler.

### Primary endpoint: global space-time field error

Lower `space_time_rel_l2` is better.

| Budget | Confirmatory interpretation |
|---|---|
| `N_f = 250` | RAD-IID improved over MC-R after multiplicity correction; ALHS-R vs MC-R did not meet the confirmatory threshold after Holm adjustment. |
| `N_f = 500` | RAD-IID and ALHS-R both improved over MC-R; ALHS-R also improved over LHS-R. |
| `N_f = 1000` | Neither RAD-IID nor ALHS-R was confirmatorily better than MC-R; both adaptive methods improved over LHS-R. |

Thus, the adaptive advantage for the **global field-error endpoint** is clearest at the lower and intermediate collocation budgets, while the adaptive methods do not show confirmatory superiority over MC-R at `N_f = 1000`.

### Secondary endpoint: resolved physical residual

For `resolved_J`, both adaptive methods produced lower values than MC-R and LHS-R across all three budgets under the pre-specified secondary testing family.

### RAD-IID versus ALHS-R

No confirmatory incremental benefit of ALHS-R over RAD-IID was detected for either the primary endpoint or `resolved_J`.

**This is not an equivalence claim.** No equivalence margin was pre-specified, and failure to reject the RAD-IID versus ALHS-R contrast must not be interpreted as proof that the two methods are equivalent.

The machine-readable final statistical summaries are available under [`results/statistical_summary/`](results/statistical_summary/).

---

## Reviewer quick start — no training required

A reviewer can verify that the scientific source in this repository matches the frozen manifest without installing PyTorch or reproducing the 360 training runs.

```bash
git clone https://github.com/wahyuhardyanto/pinn-convection-diffusion-adaptive-collocation-v237.git
cd pinn-convection-diffusion-adaptive-collocation-v237
python tools/verify_repo_integrity.py
```

Expected result:

```text
Repository integrity check: PASS (22 frozen source files verified)
```

This static verifier uses only the Python standard library. It checks byte-level SHA-256 values for the frozen scientific source against the manifest and **does not launch PINN training**.

For a guided review path, see [`REVIEWER_QUICKSTART.md`](REVIEWER_QUICKSTART.md).

---

## Three levels of reproducibility

### 1. Source-integrity verification

Portable and inexpensive:

```bash
python tools/verify_repo_integrity.py
```

This verifies that the frozen scientific source has not been silently modified.

### 2. Inspection of the reported statistical analysis

The repository includes the available final statistical output tables:

```text
results/statistical_summary/
```

These include:

- descriptive summaries;
- primary Friedman tests by budget;
- primary paired Wilcoxon contrasts;
- secondary `resolved_J` Friedman and Wilcoxon results;
- budget-effect analyses; and
- Scharfetter-Gummel robustness summaries.

The file [`results/statistical_summary/analysis_summary.json`](results/statistical_summary/analysis_summary.json) records a 360-row final analysis dataset and the SHA-256 hash of the raw final results file used by the frozen statistical analysis:

```text
56bc1cb6129658286b71fae3e6f2d554747d763bed418b58842430ea23d48410
```

### 3. Full 360-run training reproduction

Full reproduction should use the frozen runtime recorded in the manifest. From `frozen_protocol/`:

```bash
python verify_frozen_v237_final.py
python run_v237_final.py --dry-run
python run_v237_final.py --methods all
```

Operational batching by method group is allowed, but **scientific parameter overrides are not allowed**.

---

## Repository layout

```text
.
├── README.md
├── REVIEWER_QUICKSTART.md
├── frozen_protocol/
│   ├── FROZEN_PROTOCOL_MANIFEST_V237.json
│   ├── Statistical_Analysis_Plan_V237_Final.md
│   ├── README_V237_Final_Frozen.md
│   ├── run_v237_final.py
│   ├── statistical_analysis_v237_final.py
│   ├── verify_frozen_v237_final.py
│   ├── Qualification/
│   └── V237_Frozen_Core/
│       ├── convection_diffusion_unified_v237.py
│       ├── unified_context_v237.py
│       ├── objective_control_v235.py
│       ├── objective_trainer_v235.py
│       ├── shared_transition_v235.py
│       ├── frozen_references/
│       └── replay_tapes/
├── results/
│   └── statistical_summary/
├── docs/
│   ├── PROVENANCE.md
│   ├── Training_Protocol_Audit_V237.md
│   └── Consistency_Audit_Sections2_3.md
├── tools/
│   └── verify_repo_integrity.py
├── environment.yml
├── requirements-frozen.txt
└── .github/workflows/
    └── integrity-check.yml
```

Interpreter-specific `__pycache__` files from the archival research package are intentionally excluded from the public repository. They are not scientific source. Integrity verification is therefore based on `scientific_core_source_sha256` in the frozen manifest.

---

## Frozen runtime

The original final benchmark was run under the environment recorded in the frozen manifest:

| Component | Frozen value |
|---|---|
| Python | `3.9.16` |
| PyTorch | `2.8.0` |
| NumPy | `1.25.2` |
| pandas | `2.0.2` |
| SciPy | `1.10.1` |
| dtype | `float32` |
| Device | Apple MPS |
| Platform | macOS arm64 |
| Threads | `8` |

`environment.yml` and `requirements-frozen.txt` are convenience files added for repository setup. The frozen manifest remains authoritative if there is any discrepancy.

An optional convenience environment can be created with:

```bash
conda env create -f environment.yml
conda activate pinn-cd-v237-repro
```

The original full-reproduction verifier may reject a runtime that does not match the frozen environment; this is intentional.

---

## Statistical analysis

The Statistical Analysis Plan was frozen before inspection of the final 30-seed benchmark results.

The primary analysis uses:

1. a Friedman omnibus test across the four methods at each budget;
2. Holm adjustment across the three budget-level omnibus tests;
3. all six paired Wilcoxon signed-rank method contrasts at each budget; and
4. Holm adjustment jointly across the 18 primary pairwise tests.

A pairwise comparison is confirmatory only when the corresponding adjusted omnibus gate is passed and the Holm-adjusted pairwise p-value is below `0.05`.

`resolved_J` is analyzed as a **separate secondary family** and cannot override the primary endpoint conclusion.

To re-run the pre-specified statistics after a complete final output directory is available:

```bash
cd frozen_protocol
python statistical_analysis_v237_final.py \
  --results pde_v237_final_independent \
  --output-dir ../recomputed_statistical_analysis
```

---

## Frozen references and provenance

The repository retains the numerical-reference and provenance materials needed to audit the development chain, including:

- frozen FDM reference arrays;
- Scharfetter-Gummel secondary reference arrays;
- replay tapes used in earlier controlled development stages;
- protocol and training audits; and
- SHA-256 provenance records.

See:

- [`docs/PROVENANCE.md`](docs/PROVENANCE.md)
- [`docs/Training_Protocol_Audit_V237.md`](docs/Training_Protocol_Audit_V237.md)
- [`frozen_protocol/V237_Frozen_Core/frozen_references/`](frozen_protocol/V237_Frozen_Core/frozen_references/)

---

## Data availability and current repository scope

This repository currently includes:

- the frozen scientific source;
- the frozen final launcher and statistical-analysis script;
- the Statistical Analysis Plan;
- frozen FDM and Scharfetter-Gummel reference materials;
- replay/provenance materials;
- qualification summary material; and
- final statistical summary tables.

The **complete raw output directory for all 360 final training runs is not included in this GitHub repository at present**. The statistical summary records the SHA-256 hash of the raw 360-row results file used for analysis, so a subsequently archived copy can be checked against the exact analyzed dataset.

For archival reproducibility, the intended final release should deposit the complete raw final result set in a persistent research-data archive (for example, Zenodo or an institutional repository) and add its DOI and checksum here.

---

## Integrity policy

The directory `frozen_protocol/` represents the frozen scientific protocol. Scientific source, endpoint definitions, seed range, multiplicity procedure, or analysis logic should not be silently modified.

Repository-level documentation, citation metadata, archival links, and CI configuration may be added after the freeze, but such additions must remain distinguishable from the frozen scientific core.

The repository includes a GitHub Actions workflow that runs the static hash verification on repository updates. Its current status can be checked from the repository **Actions** tab.

---
## Citation

The frozen V2.3.7 reproducibility release is permanently archived on Zenodo:

**DOI:** `10.5281/zenodo.23794781`

Please cite the archived version when reproducibility of the reported benchmark is required. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

The associated journal article should also be cited once it is published.

---

## License

This repository uses separate licenses for software and non-software research materials.

- **Software and source code:** MIT License. See [`LICENSE`](LICENSE).
- **Project-authored data, numerical results, and documentation:** Creative Commons Attribution 4.0 International (CC BY 4.0). See [`LICENSE_DATA_DOCUMENTATION.md`](LICENSE_DATA_DOCUMENTATION.md).

Third-party materials, external datasets, and submitted or published journal materials remain subject to their own applicable terms and are not relicensed by this repository.

Reuse of the scientific code or data does not imply that a modified implementation represents the frozen V2.3.7 benchmark. Modified versions should be clearly identified as such.

---

## Reproducibility documents

For additional details:

- [`REVIEWER_QUICKSTART.md`](REVIEWER_QUICKSTART.md) — shortest reviewer verification path;
- [`frozen_protocol/Statistical_Analysis_Plan_V237_Final.md`](frozen_protocol/Statistical_Analysis_Plan_V237_Final.md) — pre-specified inferential plan;
- [`frozen_protocol/FROZEN_PROTOCOL_MANIFEST_V237.json`](frozen_protocol/FROZEN_PROTOCOL_MANIFEST_V237.json) — frozen design, runtime, and SHA-256 hashes;
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — repository provenance;
- [`docs/Training_Protocol_Audit_V237.md`](docs/Training_Protocol_Audit_V237.md) — training-protocol audit.
