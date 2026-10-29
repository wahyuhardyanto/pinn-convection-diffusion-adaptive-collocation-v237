# Provenance and freeze boundary

## Frozen scientific protocol

The contents under `frozen_protocol/` originate from the V2.3.7 frozen final independent benchmark package. Scientific files were copied without editing. The repository-facing documentation, CI workflow, environment helpers, and reviewer quick-start files were added afterward and are not part of the scientific protocol.

The frozen protocol identifies:

- parent scientific version: `2.3.7-unified-four-sampler1`;
- freeze date: 2026-10-22;
- final seeds: 5-34;
- active collocation budgets: 250, 500, and 1000;
- samplers: MC-R, LHS-R, RAD-IID, ALHS-R;
- final design: 360 runs;
- primary endpoint: `space_time_rel_l2` against FDM;
- secondary endpoint: `resolved_J`.

The protocol manifest is authoritative for hashes and runtime details.

## Repository additions

Files outside `frozen_protocol/` are packaging/documentation aids. They must not be interpreted as changes to the frozen experiment.

## Results included

`results/statistical_summary/` contains the available final statistical-analysis output tables. The complete per-run final output directory is not part of the source package used to construct this repository bundle. To enable full re-analysis from raw runs, publish that final output directory separately (for example as a Zenodo dataset or GitHub Release asset) and document its checksum.
