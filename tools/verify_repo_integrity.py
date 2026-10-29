#!/usr/bin/env python3
"""Static source-integrity verifier for the public V2.3.7 repository package.

This checks byte-level SHA-256 values for the frozen scientific source files against
FROZEN_PROTOCOL_MANIFEST_V237.json. It does not enforce the original runtime; the
original verifier under frozen_protocol/ remains authoritative for full reproduction.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FP = ROOT / "frozen_protocol"
CORE = FP / "V237_Frozen_Core"
MANIFEST = FP / "FROZEN_PROTOCOL_MANIFEST_V237.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    mf = json.loads(MANIFEST.read_text())
    expected = mf.get("scientific_core_source_sha256", {})
    failures = []
    for rel, want in sorted(expected.items()):
        p = CORE / rel
        if not p.is_file():
            failures.append(f"MISSING {rel}")
            continue
        got = sha256(p)
        if got != want:
            failures.append(f"HASH MISMATCH {rel}: expected {want}, got {got}")

    companion = mf.get("companion_file_sha256", {})
    # Check only companion files included at frozen_protocol root.
    for rel, want in sorted(companion.items()):
        p = FP / rel
        if p.is_file():
            got = sha256(p)
            if got != want:
                failures.append(f"HASH MISMATCH {rel}: expected {want}, got {got}")

    if failures:
        print("Repository integrity check: FAILED")
        for x in failures:
            print(" -", x)
        return 1

    print(f"Repository integrity check: PASS ({len(expected)} frozen source files verified)")
    print("Note: archive-only __pycache__ artifacts are intentionally excluded from this public package.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
