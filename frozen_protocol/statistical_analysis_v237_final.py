#!/usr/bin/env python3
"""Pre-specified statistical analysis for the V2.3.7 final independent benchmark.

The analysis population is fixed at four methods x three budgets x seeds 5-34.
Qualification seeds 0-4 are rejected if present in the input dataset.
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

METHODS = ("mc", "lhs", "rad", "alhs")
BUDGETS = (250, 500, 1000)
SEEDS = tuple(range(5, 35))
PRIMARY = "space_time_rel_l2"
SECONDARY = "resolved_J"
ROBUSTNESS = "pinn_vs_sg_space_time_rel_l2"
DIAGNOSTICS = ("final_rel_l2", "resolved_right_001_rms", "resolved_right_0001_rms", "independent_residual_rms")
ROOT = Path(__file__).resolve().parent
MANIFEST_FILE = ROOT / "FROZEN_PROTOCOL_MANIFEST_V237.json"

PREFIX_KEYS = (
    "shared_checkpoint_sha256", "shared_model_sha256", "transition_state_sha256",
    "transition_optimizer_sha256", "transition_rng_sha256", "transition_points_sha256",
)
PAIRS = tuple(combinations(METHODS, 2))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def verify_analysis_freeze():
    manifest = json.loads(MANIFEST_FILE.read_text())
    for rel in ("Statistical_Analysis_Plan_V237_Final.md", "statistical_analysis_v237_final.py"):
        expected = manifest["companion_file_sha256"][rel]
        if sha256(ROOT / rel) != expected:
            raise RuntimeError("Frozen analysis checksum mismatch: " + rel)
    return manifest

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adj_sorted[rank] = running
    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adj_sorted[rank]
    return out


def rank_biserial_positive_favors_a(a, b):
    # Difference is B-A, so positive signed ranks mean A has the lower endpoint.
    d = np.asarray(b, float) - np.asarray(a, float)
    nz = d != 0
    d = d[nz]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    pos = float(ranks[d > 0].sum())
    neg = float(ranks[d < 0].sum())
    total = pos + neg
    return (pos - neg) / total if total else 0.0



def safe_friedman(arrays):
    arrays = [np.asarray(x, float) for x in arrays]
    stacked = np.column_stack(arrays)
    # scipy.stats.friedmanchisquare is undefined when every paired block is
    # exactly tied across all treatments; the natural null result is p=1.
    if np.all(stacked == stacked[:, [0]]):
        return 0.0, 1.0
    res = friedmanchisquare(*arrays)
    stat = float(res.statistic); p = float(res.pvalue)
    if not np.isfinite(stat) or not np.isfinite(p):
        raise RuntimeError("Friedman test returned a non-finite result outside the all-tied case")
    return stat, p

def paired_wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    if np.all(d == 0):
        return 0.0, 1.0
    res = wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
    return float(res.statistic), float(res.pvalue)


def descriptive(df, metrics):
    rows = []
    for metric in metrics:
        for (nf, method), g in df.groupby(["n_collocation", "method"], sort=True):
            v = g[metric].astype(float)
            q1 = float(v.quantile(0.25)); q3 = float(v.quantile(0.75))
            rows.append({
                "metric": metric, "n_collocation": int(nf), "method": method, "n": len(v),
                "median": float(v.median()), "q1": q1, "q3": q3, "iqr": q3-q1,
                "minimum": float(v.min()), "maximum": float(v.max()),
            })
    return pd.DataFrame(rows)


def validate(df):
    required = {"method", "n_collocation", "seed", "selected_bc_feasible", PRIMARY, SECONDARY, ROBUSTNESS, *DIAGNOSTICS, *PREFIX_KEYS}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise RuntimeError("Missing required columns: " + repr(missing_cols))
    if df.duplicated(["method", "n_collocation", "seed"]).any():
        raise RuntimeError("Duplicate method-budget-seed rows")
    expected = {(m, n, s) for m in METHODS for n in BUDGETS for s in SEEDS}
    got = set(zip(df.method.astype(str), df.n_collocation.astype(int), df.seed.astype(int)))
    extra = sorted(got - expected)
    miss = sorted(expected - got)
    if extra or miss:
        raise RuntimeError(f"Final design mismatch. Missing={len(miss)} Extra={len(extra)}; qualification seeds 0-4 must not be included")
    if len(df) != 360:
        raise RuntimeError(f"Expected 360 rows, found {len(df)}")
    bc_ok = df.selected_bc_feasible.astype(str).str.strip().str.lower().isin(("true", "1"))
    if not bc_ok.all():
        raise RuntimeError("BC-infeasible selected checkpoint in final dataset")
    for metric in (PRIMARY, SECONDARY, ROBUSTNESS, *DIAGNOSTICS):
        x = pd.to_numeric(df[metric], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all():
            raise RuntimeError("Non-finite values in " + metric)
    for (nf, seed), g in df.groupby(["n_collocation", "seed"]):
        if set(g.method) != set(METHODS) or len(g) != 4:
            raise RuntimeError(f"Incomplete four-method paired block nf={nf} seed={seed}")
        for key in PREFIX_KEYS:
            if g[key].astype(str).nunique() != 1:
                raise RuntimeError(f"Shared-prefix mismatch for {key} at nf={nf}, seed={seed}")
    return True


def method_inference(df, metric, family_name):
    omnibus = []
    pairrows = []
    for nf in BUDGETS:
        sub = df[df.n_collocation.astype(int) == nf]
        wide = sub.pivot(index="seed", columns="method", values=metric).loc[list(SEEDS), list(METHODS)]
        stat, p = safe_friedman([wide[m].to_numpy(float) for m in METHODS])
        omnibus.append({"family": family_name, "metric": metric, "n_collocation": nf, "n_blocks": len(wide), "friedman_statistic": float(stat), "p_raw": float(p)})
        for a, b in PAIRS:
            av = wide[a].to_numpy(float); bv = wide[b].to_numpy(float)
            wstat, wp = paired_wilcoxon(av, bv)
            d = av - bv
            pairrows.append({
                "family": family_name, "metric": metric, "n_collocation": nf,
                "method_a": a, "method_b": b, "n_pairs": len(av),
                "wilcoxon_statistic": wstat, "p_raw": wp,
                "median_a_minus_b": float(np.median(d)),
                "n_a_lower_better": int((av < bv).sum()), "n_b_lower_better": int((bv < av).sum()),
                "n_equal": int((av == bv).sum()),
                "rank_biserial_positive_favors_a": rank_biserial_positive_favors_a(av, bv),
            })
    omni = pd.DataFrame(omnibus)
    omni["p_holm_across_3_budgets"] = holm_adjust(omni.p_raw.to_numpy(float))
    omni["omnibus_gate_pass"] = omni["p_holm_across_3_budgets"] < 0.05
    pairs = pd.DataFrame(pairrows)
    pairs["p_holm_across_18_pairwise"] = holm_adjust(pairs.p_raw.to_numpy(float))
    gate = omni.set_index("n_collocation")["omnibus_gate_pass"].to_dict()
    pairs["omnibus_gate_pass"] = pairs.n_collocation.map(gate).astype(bool)
    pairs["confirmatory_significant"] = pairs.omnibus_gate_pass & (pairs.p_holm_across_18_pairwise < 0.05)
    return omni, pairs


def budget_inference(df, metric, family_name):
    omnibus = []
    pairs = []
    budget_pairs = tuple(combinations(BUDGETS, 2))
    for method in METHODS:
        sub = df[df.method == method]
        wide = sub.pivot(index="seed", columns="n_collocation", values=metric).loc[list(SEEDS), list(BUDGETS)]
        stat, p = safe_friedman([wide[n].to_numpy(float) for n in BUDGETS])
        omnibus.append({"family": family_name, "metric": metric, "method": method, "n_blocks": len(wide), "friedman_statistic": float(stat), "p_raw": float(p)})
        for a, b in budget_pairs:
            av = wide[a].to_numpy(float); bv = wide[b].to_numpy(float)
            wstat, wp = paired_wilcoxon(av, bv)
            pairs.append({
                "family": family_name, "metric": metric, "method": method,
                "budget_a": a, "budget_b": b, "n_pairs": len(av),
                "wilcoxon_statistic": wstat, "p_raw": wp,
                "median_a_minus_b": float(np.median(av-bv)),
                "n_a_lower_better": int((av < bv).sum()), "n_b_lower_better": int((bv < av).sum()),
                "n_equal": int((av == bv).sum()),
                "rank_biserial_positive_favors_a": rank_biserial_positive_favors_a(av, bv),
            })
    omni = pd.DataFrame(omnibus)
    omni["p_holm_across_4_methods"] = holm_adjust(omni.p_raw.to_numpy(float))
    omni["omnibus_gate_pass"] = omni.p_holm_across_4_methods < 0.05
    pairdf = pd.DataFrame(pairs)
    pairdf["p_holm_across_12_budget_pairs"] = holm_adjust(pairdf.p_raw.to_numpy(float))
    gate = omni.set_index("method")["omnibus_gate_pass"].to_dict()
    pairdf["omnibus_gate_pass"] = pairdf.method.map(gate).astype(bool)
    pairdf["secondary_significant"] = pairdf.omnibus_gate_pass & (pairdf.p_holm_across_12_budget_pairs < 0.05)
    return omni, pairdf


def resolve_results(path):
    path = Path(path)
    if path.is_file(): return path
    candidates = [path / "full" / "results_unified.csv", path / "results_unified.csv"]
    for p in candidates:
        if p.exists(): return p
    raise FileNotFoundError("Could not locate final results_unified.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path, help="final output root, full folder, or results_unified.csv")
    ap.add_argument("--output-dir", type=Path, default=Path("v237_final_statistical_analysis"))
    args = ap.parse_args()
    verify_analysis_freeze()
    result_file = resolve_results(args.results)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(result_file)
    validate(df)

    metrics = (PRIMARY, SECONDARY, ROBUSTNESS, *DIAGNOSTICS)
    descriptive(df, metrics).to_csv(out / "descriptive_all_metrics.csv", index=False)

    p_omni, p_pairs = method_inference(df, PRIMARY, "primary")
    p_omni.to_csv(out / "primary_friedman_by_budget.csv", index=False)
    p_pairs.to_csv(out / "primary_pairwise_wilcoxon.csv", index=False)

    s_omni, s_pairs = method_inference(df, SECONDARY, "secondary")
    s_omni.to_csv(out / "secondary_resolvedJ_friedman_by_budget.csv", index=False)
    s_pairs.to_csv(out / "secondary_resolvedJ_pairwise_wilcoxon.csv", index=False)

    b_omni, b_pairs = budget_inference(df, PRIMARY, "budget_primary")
    b_omni.to_csv(out / "budget_effect_primary_friedman.csv", index=False)
    b_pairs.to_csv(out / "budget_effect_primary_pairwise.csv", index=False)

    bj_omni, bj_pairs = budget_inference(df, SECONDARY, "budget_secondary")
    bj_omni.to_csv(out / "budget_effect_resolvedJ_friedman.csv", index=False)
    bj_pairs.to_csv(out / "budget_effect_resolvedJ_pairwise.csv", index=False)

    robustness = descriptive(df, (ROBUSTNESS,))
    robustness.to_csv(out / "robustness_sg_descriptive.csv", index=False)

    summary = {
        "analysis_plan": "V2.3.7 final SAP; independent seeds 5-34",
        "results_file": str(result_file.resolve()),
        "results_sha256": sha256(result_file),
        "n_rows": int(len(df)), "methods": list(METHODS), "budgets": list(BUDGETS), "seeds": list(SEEDS),
        "primary_endpoint": PRIMARY, "secondary_endpoint": SECONDARY,
        "alpha": 0.05, "tests_two_sided": True,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("FINAL ANALYSIS COMPLETE:", out)


if __name__ == "__main__":
    main()
