#!/usr/bin/env python3
"""Frozen launcher for the independent V2.3.7 final benchmark.

This launcher does NOT modify the scientific V2.3.7 source. It verifies the
frozen source tree and qualification environment, then invokes the original
runner with preset=full, budgets=250,500,1000, and seeds=5-34.

Operational batching by method is allowed because all branches still use the
same frozen output root and the original V2.3.7 manifest records the complete
four-method design.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "V237_Frozen_Core"
MANIFEST = ROOT / "FROZEN_PROTOCOL_MANIFEST_V237.json"
CORE_RUNNER = CORE / "convection_diffusion_unified_v237.py"
DEFAULT_OUTPUT = ROOT / "pde_v237_final_independent"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    return json.loads(MANIFEST.read_text())


def verify_core(manifest):
    expected = manifest["scientific_core_source_sha256"]
    actual_names = sorted(p.name for p in CORE.glob("*.py"))
    if actual_names != sorted(expected):
        raise RuntimeError("Frozen scientific source inventory changed")
    bad = []
    for name, digest in expected.items():
        p = CORE / name
        got = sha256(p)
        if got != digest:
            bad.append((name, digest, got))
    if bad:
        raise RuntimeError("Frozen scientific source checksum mismatch: " + repr(bad))
    return True


def verify_companions(manifest):
    bad = []
    for rel, digest in manifest["companion_file_sha256"].items():
        p = ROOT / rel
        if not p.exists() or sha256(p) != digest:
            bad.append(rel)
    if bad:
        raise RuntimeError("Frozen SAP/companion checksum mismatch: " + repr(bad))
    return True


def current_runtime():
    import numpy as np
    import pandas as pd
    import scipy
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "system": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "dtype": "float32",
    }


def verify_runtime(manifest):
    expected = manifest["frozen_runtime"]
    actual = current_runtime()
    keys = ("python", "torch", "numpy", "pandas", "scipy", "system", "machine", "device", "dtype")
    diff = {k: {"expected": expected.get(k), "actual": actual.get(k)} for k in keys if expected.get(k) != actual.get(k)}
    if diff:
        raise RuntimeError("Frozen final benchmark requires the qualification environment exactly. Differences: " + json.dumps(diff, indent=2))
    return actual


def build_command(methods, output_root):
    return [
        sys.executable, str(CORE_RUNNER),
        "--preset", "full",
        "--output-root", str(Path(output_root).resolve()),
        "--methods", methods,
        "--budgets", "250,500,1000",
        "--seeds", "5-34",
        "--device", "auto",
        "--threads", "8",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", default="all", help="all or an operational subset, e.g. mc,lhs then rad,alhs")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dry-run", action="store_true", help="verify frozen source and print the fixed final command")
    ap.add_argument("--report", action="store_true", help="recompute reports only; does not train")
    args = ap.parse_args()

    manifest = load_manifest()
    verify_core(manifest)
    verify_companions(manifest)
    if args.dry_run:
        print("FROZEN CORE: PASS")
        print("FINAL DESIGN: methods=mc,lhs,rad,alhs; budgets=250,500,1000; seeds=5-34; n=360")
        print("COMMAND:", " ".join(build_command(args.methods, args.output_root)))
        return

    verify_runtime(manifest)
    if args.report:
        cmd = [sys.executable, str(CORE_RUNNER), "--preset", "report", "--output-root", str(args.output_root.resolve()), "--device", "auto", "--threads", "8"]
    else:
        cmd = build_command(args.methods, args.output_root)
    print("Frozen scientific core verified.", flush=True)
    print("Executing:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(CORE), check=True)


if __name__ == "__main__":
    main()
