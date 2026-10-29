# GitHub public-release checklist

- [ ] Replace the repository title/URL placeholders if the final GitHub repository name changes.
- [ ] Complete `CITATION.cff.template`, rename it to `CITATION.cff`, and include the complete author list.
- [ ] Add the final manuscript citation/DOI once available.
- [ ] Select and add an explicit code/data license.
- [ ] Publish or archive the complete raw 360-run final output directory and record its SHA-256 / DOI.
- [ ] Confirm `python tools/verify_repo_integrity.py` passes before tagging a release.
- [ ] Tag the frozen public release (suggested: `v2.3.7`).
- [ ] Do not modify files under `frozen_protocol/V237_Frozen_Core/` after release; create a new version instead.
