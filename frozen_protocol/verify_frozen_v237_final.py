#!/usr/bin/env python3
"""Verify the frozen V2.3.7 scientific core and companion protocol files."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "V237_Frozen_Core"
MANIFEST = ROOT / "FROZEN_PROTOCOL_MANIFEST_V237.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    m = json.loads(MANIFEST.read_text())
    bad = []
    for name, expected in m["scientific_core_source_sha256"].items():
        p = CORE / name
        if not p.exists() or sha(p) != expected:
            bad.append(name)
    for rel, expected in m["companion_file_sha256"].items():
        p = ROOT / rel
        if not p.exists() or sha(p) != expected:
            bad.append(rel)
    if bad:
        raise RuntimeError("Frozen protocol verification FAILED: " + repr(bad))
    print("FROZEN PROTOCOL VERIFICATION: PASS")
    print("Scientific Python sources:", len(m["scientific_core_source_sha256"]))
    print("Final design: 4 methods x 3 budgets x 30 independent seeds = 360 runs")
    print("Seeds: 5-34; qualification seeds 0-4 excluded from final inference")


if __name__ == "__main__":
    main()
