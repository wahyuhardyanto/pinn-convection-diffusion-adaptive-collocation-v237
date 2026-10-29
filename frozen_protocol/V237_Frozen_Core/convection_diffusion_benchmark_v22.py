#!/usr/bin/env python3
"""Convection-diffusion PINN V2.2: boundary-layer transform x enrichment ablation.

Based on verified V2 (unchanged FDM and validation) and V2.1 (BC-feasible
checkpoint selection and nested near-outflow diagnostics).

Four *diagnostic*, single-seed MC-R arms:
  standard_uniform  : V2.1 hard IC + soft BC, plain uniform MC collocation
  standard_enriched : unchanged output transform, common outflow enrichment
  bl_uniform        : right-BC exact exponential boundary correction, uniform MC
  bl_enriched       : exponential boundary correction and outflow enrichment

Both methods in each pair start with identically initialized raw neural networks;
within a transform class, the warmup parameter state and uniform RNG stream are
exactly matched. The FDM reference is never used for checkpoint selection.

IMPORTANT: Targeted points receive the same per-point loss as all other points,
so enrichment deliberately changes the implicit sampling-weighted training loss.
This is an ablation, not a publication-ready 4-sampler benchmark.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

import convection_diffusion_benchmark_v2 as core
import convection_diffusion_benchmark_v21 as v21

VERSION = "2.2.0"
BASE_V2_VERSION = "2.0.0"
BASE_V21_VERSION = "2.1.0"
VARIANTS = (
    "standard_uniform",
    "standard_enriched",
    "bl_uniform",
    "bl_enriched",
)


class BoundaryLayerPINN(core.PINN):
    """Hard IC, soft left BC; optional exponentially localized hard right BC.

    Correction B_R(x)=(exp(-(1-x)/delta)-exp(-1/delta))/(1-exp(-1/delta))
    has exactly B_R(0)=0 and B_R(1)=1 (modulo float precision).

    u_hat=sin(pi*x)+t*(N(x,t)-B_R(x)*N(1,t)).
    Standard mode calls the V2.1 hard_ic_soft_bc transform unchanged.
    """
    def __init__(self, layers: list, boundary_layer: bool, delta: float):
        super().__init__(layers, mode="hard_ic_soft_bc")
        if not (0 < delta < 0.5):
            raise ValueError("Boundary-layer width delta must be in (0,0.5)")
        self.boundary_layer = boundary_layer
        self.delta = float(delta)

    def right_basis(self, x: torch.Tensor) -> torch.Tensor:
        # x is in [0,1] in this application. expm1 avoids cancellation when
        # exp(-1/delta) is not negligible; for delta~0.001 it underflows safely.
        floor = math.exp(-1.0 / self.delta)
        return (torch.exp((x - 1.0) / self.delta) - floor) / (1.0 - floor)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        if not self.boundary_layer:
            return super().forward(xt)
        x, t = xt[:, 0:1], xt[:, 1:2]
        raw = self.net(xt)
        right_xt = torch.cat((torch.ones_like(x), t), dim=1)
        right_raw = self.net(right_xt)
        return torch.sin(torch.pi * x) + t * (raw - self.right_basis(x) * right_raw)


def make_config(preset: str) -> dict:
    cfg = v21.make_config(preset)
    cfg["w_bc"] = 10.0  # V2.1's best *global* diagnostic, not publication-optimal
    cfg["checkpoint_bc_max_tol"] = 0.02
    cfg["enrichment_fraction"] = 0.25
    cfg["enrich_dmin_factor"] = 0.01
    cfg["enrich_dmax_factor"] = 20.0
    cfg["bl_delta_factor"] = 1.0
    cfg["purpose"] = "2-by-2 boundary transform x enrichment ablation"
    return cfg


def variant_flags(variant: str) -> tuple:
    if variant not in VARIANTS:
        raise ValueError("Invalid variant " + str(variant))
    return (variant.startswith("bl_"), variant.endswith("_enriched"))


def focus_range(cfg: dict) -> tuple:
    physical_delta = cfg["D"] / abs(cfg["c"])
    lower = physical_delta * cfg["enrich_dmin_factor"]
    upper = min(physical_delta * cfg["enrich_dmax_factor"], 0.25)
    if not (0 < lower < upper < 1):
        raise ValueError("Invalid focus range; require 0 < d_min < d_max < 1")
    return float(lower), float(upper)


def sample_points(nf: int, global_rng: np.random.Generator,
                  layer_rng: np.random.Generator, cfg: dict,
                  enriched: bool) -> tuple:
    """Preserve SAME global RNG stream across all 4 diagnostic arms.

    Enriched arm replaces last n_focus uniform draws with independent focused
    draws; first n_global globally uniform points match control exactly.
    d=(1-x) is log-uniform on [d_min,d_max], time is uniform [0,1].
    """
    base = core.uniform_mc(nf, global_rng)  # draw nf on EVERY arm
    if not enriched:
        return base.astype(np.float32), 0
    n_focus = int(round(nf * cfg["enrichment_fraction"]))
    if not (1 <= n_focus < nf):
        raise ValueError("enrichment fraction must give 1..nf-1 focused points")
    d_min, d_max = focus_range(cfg)
    unit = layer_rng.random((n_focus, 2))
    distance = np.exp(np.log(d_min) + unit[:, 0] * np.log(d_max / d_min))
    focused = np.column_stack((1.0 - distance, unit[:, 1]))
    xy = np.vstack((base[:nf - n_focus], focused)).astype(np.float32)
    assert len(xy) == nf and np.isfinite(xy).all()
    assert np.all((xy >= 0) & (xy <= 1))
    return xy, n_focus


def sample_probe_stats(points: np.ndarray, delta: float) -> dict:
    distance = np.maximum(0.0, 1.0 - points[:, 0])
    return dict(n_points=len(points), within_delta=int(np.sum(distance < delta)),
                within_5delta=int(np.sum(distance < 5 * delta)),
                within_20delta=int(np.sum(distance < 20 * delta)),
                minimum_distance_to_outflow=float(distance.min()),
                median_distance_to_outflow=float(np.median(distance)))


def extra_diagnostics(folder: Path, model: BoundaryLayerPINN, cfg: dict,
                      device: torch.device, samples: np.ndarray,
                      all_samples: np.ndarray) -> dict:
    """Detailed PDE residual audits close to outflow, separate from selection.

    Importantly, core.fixed_validation/full_validation remain *unaltered*.
    This supplemental high-resolution grid is NEVER used to pick checkpoints.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    delta = cfg["D"] / abs(cfg["c"])
    d_min, d_max = focus_range(cfg)
    bands = [(0.90, 0.99), (0.99, 0.999), (0.999, 1.0)]
    tt = np.array([0.05, 0.10, 0.50, 0.75, 1.0], dtype=np.float64)
    rows = []
    xcoords = []
    valmaps = []
    for lo, hi in bands:
        xprobe = np.linspace(lo, hi, 65 if hi < 1.0 else 121, dtype=np.float64)
        xcoords.append(xprobe)
        xx, ts = np.meshgrid(xprobe, tt, indexing="xy")
        xy = np.column_stack((xx.ravel(), ts.ravel()))
        res = core.residual_numpy(model, xy, cfg, device).reshape(len(tt), -1)
        valmaps.append(res)
        for j, t in enumerate(tt):
            rows.append(dict(t=float(t), x_lo=lo, x_hi=hi,
                             residual_rms=float(np.sqrt(np.mean(res[j] ** 2))),
                             residual_max_abs=float(np.max(np.abs(res[j]))),
                             spatial_probe_points=len(xprobe)))
    residual_df = pd.DataFrame(rows)
    core.atomic_csv(folder / "outflow_residual_bands.csv", residual_df)
    # Sampling diagnostics include the point distribution across all refreshes.
    d = np.maximum(1e-10, 1.0 - all_samples[:, 0])
    thresholds = [delta, 2*delta, 5*delta, 10*delta, 20*delta]
    core.atomic_json(folder / "sampling_coverage.json", {
        "delta_physical":delta, "delta_transform":cfg["bl_delta_factor"]*delta,
        "focus_d_min": d_min, "focus_d_max":d_max,
        "collocation_sets":int(len(all_samples) // len(samples)),
        "last_set":sample_probe_stats(samples,delta),
        "all_sets":sample_probe_stats(all_samples,delta),
        "fractions_within_distance":{str(v):float(np.mean(d<=v)) for v in thresholds},
        "warning":"The focused samples are not importance-weight corrected."
    })
    fig, axes = plt.subplots(1, 2, figsize=(12,4.3), layout="constrained")
    axes[0].scatter(samples[:,0],samples[:,1],s=10,alpha=.6)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    axes[0].set_title("Last collocation set (Nf=%d)"%len(samples))
    axes[0].set_xlim(0,1);axes[0].set_ylim(0,1)
    axes[1].hist(np.log10(d),bins=45)
    axes[1].axvline(np.log10(delta),ls="--",c="black",label="D/c")
    axes[1].set_xlabel("log10(1-x)"); axes[1].set_ylabel("Count over all sets")
    axes[1].legend(fontsize=8)
    fig.savefig(folder/"collocation_coverage.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.7),layout="constrained")
    for a,b in bands:
        sub=residual_df[(residual_df.x_lo==a)&(residual_df.x_hi==b)]
        ax.plot(sub.t,sub.residual_rms,marker="o",label=f"x in [{a:g},{b:g}]")
    ax.set_yscale("log");ax.set_xlabel("t");ax.set_ylabel("RMS PDE residual")
    ax.set_title("Outflow PDE residual on diagnostic grids")
    ax.legend();ax.grid(alpha=.25)
    fig.savefig(folder/"outflow_residual_bands.png",dpi=160);plt.close(fig)
    np.savez_compressed(folder/"outflow_residual_arrays.npz",times=tt,
        x_90_99=xcoords[0],r_90_99=valmaps[0],
        x_99_999=xcoords[1],r_99_999=valmaps[1],
        x_999_1=xcoords[2],r_999_1=valmaps[2])
    return dict(
        residual_rms_outflow_999_1=float(np.sqrt(np.mean(valmaps[2]**2))),
        residual_max_outflow_999_1=float(np.max(np.abs(valmaps[2]))),
        collocation_within_delta_last=int(np.sum(1.0-samples[:,0]<delta)),
        collocation_within_delta_all=int(np.sum(1.0-all_samples[:,0]<delta)),
        outflow_diagnostic_files=["outflow_residual_bands.csv",
           "outflow_residual_bands.png","outflow_residual_arrays.npz",
           "collocation_coverage.png","sampling_coverage.json"])


def run_one(variant: str, cfg: dict, ref: dict, device: torch.device,
            diagdir: Path, checkpoint_policy: str) -> dict:
    seed=0
    nf=int(cfg["n_collocation"][0])
    use_bl, use_enrich=variant_flags(variant)
    t0=time.perf_counter()
    torch.manual_seed(seed)
    if device.type=="cuda": torch.cuda.manual_seed_all(seed)
    delta=cfg["D"] / abs(cfg["c"]) * cfg["bl_delta_factor"]
    model=BoundaryLayerPINN(cfg["layers"],use_bl,delta).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["adam_lr"])
    val=core.fixed_validation(cfg,seed,device)
    bc_grid=v21.make_boundary_grid(cfg["boundary_validation_times"],device)
    warm_bc_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,702]))
    warm_ic_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,701]))
    ic_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,992]))
    bc_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,993]))
    global_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,991]))
    focus_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,997]))
    history=[]
    candidates=[]
    states={}
    sample_hash=hashlib.sha256()
    samples_all=[]
    warm_start=time.perf_counter()

    for e in range(1,cfg["warmup_epochs"]+1):
        # Same number of warmup draws/optimization updates as V2.1.
        _ic,_target=core.sample_ic(cfg["n_ic"],warm_ic_rng,device)
        bc=core.sample_bc(cfg["n_bc"],warm_bc_rng,device)
        optimizer.zero_grad(set_to_none=True)
        penalty=torch.mean(model(bc)**2)
        objective=cfg["warmup_bc_weight"]*penalty
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError("Nonfinite warmup loss")
        objective.backward();optimizer.step()
        if e in (1,cfg["warmup_epochs"]) or e%cfg["diagnostics_every"]==0:
            history.append(dict(phase="warmup",global_epoch=e,main_epoch=0,
               pde_weight=0.0,bc_weight=cfg["warmup_bc_weight"],
               objective=float(objective.detach()),pde_loss=np.nan,
               ic_loss=0.0,bc_loss=float(penalty.detach()),
               val_full_loss=np.nan,val_bc_left_rmse=np.nan,
               val_bc_right_rmse=np.nan,val_bc_max_abs=np.nan))
    warm_s=time.perf_counter()-warm_start
    warm_hash=v21.state_hash(v21.model_checkpoint(model))
    recent=deque(maxlen=cfg["lbfgs_union_sets"])
    sample_s=0.0;validation_s=0.0;focus_count=0;set_count=0
    main_start=time.perf_counter()
    for e in range(1,cfg["main_epochs"]+1):
        if (e-1)%cfg["resample_every"]==0:
            ss=time.perf_counter()
            xy,nfocus=sample_points(nf,global_rng,focus_rng,cfg,use_enrich)
            sample_hash.update(xy.tobytes())
            samples_all.append(xy)
            recent.append(xy)
            residual_xt=torch.from_numpy(xy).to(device)
            focus_count=nfocus
            set_count+=1
            sample_s+=time.perf_counter()-ss
        ic,ic_y=core.sample_ic(cfg["n_ic"],ic_rng,device)
        bc=core.sample_bc(cfg["n_bc"],bc_rng,device)
        pde_w=cfg["w_res"]*min(1.0,e/cfg["physics_ramp_epochs"])
        optimizer.zero_grad(set_to_none=True)
        objective,pde,ic_loss,bc_loss=core.full_losses(
            model,residual_xt,ic,ic_y,bc,cfg,pde_w)
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError("Nonfinite Adam loss at %d"%e)
        objective.backward();optimizer.step()
        need_val=(e%cfg["validation_every"]==0 or e==cfg["main_epochs"])
        audited=None
        if need_val:
            vs=time.perf_counter()
            audited=v21.validation_audit(model,val,cfg,bc_grid,cfg["checkpoint_bc_max_tol"])
            validation_s+=time.perf_counter()-vs
            if e>=cfg["physics_ramp_epochs"]:
                cid="adam_%d"%e
                candidates.append(dict(checkpoint_id=cid,stage="adam",main_epoch=e,**audited))
                states[cid]=v21.model_checkpoint(model)
        if e==1 or e%cfg["diagnostics_every"]==0 or need_val:
            history.append(dict(phase="main",global_epoch=cfg["warmup_epochs"]+e,
                main_epoch=e,pde_weight=pde_w,bc_weight=cfg["w_bc"],
                objective=float(objective.detach()),pde_loss=float(pde.detach()),
                ic_loss=float(ic_loss.detach()),bc_loss=float(bc_loss.detach()),
                val_full_loss=audited["val_full_loss"] if audited else np.nan,
                val_bc_left_rmse=audited["bc_left_rmse"] if audited else np.nan,
                val_bc_right_rmse=audited["bc_right_rmse"] if audited else np.nan,
                val_bc_max_abs=audited["bc_max_abs"] if audited else np.nan))
    main_s=time.perf_counter()-main_start
    if not candidates:
        raise RuntimeError("No checkpoint after ramp")
    best_adam=min(candidates,key=lambda d:d["val_full_loss"])
    model.load_state_dict(states[best_adam["checkpoint_id"]])

    # V2.1 optimization stage: union of latest collocation sets, fixed IC/BC,
    # checkpoint decision on unchanged full-weight validation + dense BC grid.
    union=np.unique(np.concatenate(list(recent)),axis=0)
    x_union=torch.from_numpy(union).to(device)
    lb_ic,lb_y=core.sample_ic(cfg["n_ic"],
       np.random.default_rng(np.random.SeedSequence([seed,nf,995])),device)
    lb_bc=core.sample_bc(cfg["n_bc"],
       np.random.default_rng(np.random.SeedSequence([seed,nf,996])),device)
    lb_status="not_run";lb_error="";lb_calls=0;lb_s=0.0
    if cfg["lbfgs_max_iter"]>0:
        lb_optimizer=torch.optim.LBFGS(model.parameters(),lr=1.0,
                max_iter=cfg["lbfgs_max_iter"],history_size=50,
                line_search_fn="strong_wolfe")
        ls=time.perf_counter()
        def closure():
            nonlocal lb_calls
            lb_calls+=1
            lb_optimizer.zero_grad(set_to_none=True)
            loss=core.full_losses(model,x_union,lb_ic,lb_y,lb_bc,cfg,cfg["w_res"])[0]
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite L-BFGS loss")
            loss.backward()
            return loss
        try:
            lb_optimizer.step(closure)
            audited=v21.validation_audit(model,val,cfg,bc_grid,cfg["checkpoint_bc_max_tol"])
            candidates.append(dict(checkpoint_id="lbfgs",stage="lbfgs",
                    main_epoch=cfg["main_epochs"],**audited))
            states["lbfgs"]=v21.model_checkpoint(model)
            lb_status="succeeded"
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            lb_error=repr(exc)[:700]
            lb_status="failed_reverted"
            print("L-BFGS failed for",variant,"returning to Adam:",lb_error,flush=True)
        lb_s=time.perf_counter()-ls
    unrestricted=min(candidates,key=lambda d:d["val_full_loss"])
    feasible=[d for d in candidates if d["bc_feasible"]]
    if checkpoint_policy=="bc_feasible" and feasible:
        chosen=min(feasible,key=lambda d:d["val_full_loss"])
        reason="min_full_val_among_bc_feasible"
    elif checkpoint_policy=="bc_feasible":
        chosen=unrestricted
        reason="no_bc_feasible_checkpoint_fallback_to_min_full_val"
    else:
        chosen=unrestricted
        reason="min_full_validation_loss"
    model.load_state_dict(states[chosen["checkpoint_id"]])
    training_s=time.perf_counter()-t0
    es=time.perf_counter()
    metrics,pred=core.evaluate(model,ref,cfg,val,device)
    eval_s=time.perf_counter()-es
    ds=time.perf_counter()
    candidate_df=pd.DataFrame(candidates)
    details=v21.profiles_and_boundary_details(diagdir,model,cfg,ref,pred,history,
        candidate_df,chosen["checkpoint_id"],bc_grid,device)
    current_samples=np.concatenate(samples_all,axis=0)
    extra=extra_diagnostics(diagdir,model,cfg,device,samples_all[-1],current_samples)
    # Store checkpoint weights for accurate reuse and further visualization.
    torch.save(model.state_dict(),diagdir / "final_model_state.pt")
    diag_s=time.perf_counter()-ds
    if not all(np.isfinite(metrics[k]) for k in (
         "space_time_rel_l2","final_rel_l2","dense_physics_loss","ic_mse","bc_mse")):
        raise FloatingPointError("Nonfinite evaluation metric")
    return dict(protocol_version=VERSION,variant=variant,
        output_transform=("hard_ic_bl_right" if use_bl else "hard_ic_standard"),
        collocation_distribution=("uniform_plus_log_outflow" if use_enrich else "uniform"),
        method="mc",method_label="MC-R",n_collocation=nf,seed=seed,
        n_focus_per_set=focus_count,focus_fraction=cfg["enrichment_fraction"],
        delta_physical=cfg["D"]/abs(cfg["c"]),delta_transform=delta,
        focus_d_min=focus_range(cfg)[0],focus_d_max=focus_range(cfg)[1],
        main_bc_weight=cfg["w_bc"],warmup_bc_weight=cfg["warmup_bc_weight"],
        warmup_epochs=cfg["warmup_epochs"],main_epochs=cfg["main_epochs"],
        physics_ramp_epochs=cfg["physics_ramp_epochs"],
        w_res=cfg["w_res"],resample_every=cfg["resample_every"],
        collocation_sets=set_count,sampling_sha256=sample_hash.hexdigest(),
        warmup_state_sha256=warm_hash,lbfgs_union_count=len(union),
        lbfgs_closure_calls=lb_calls,lbfgs_status=lb_status,lbfgs_error=lb_error,
        selected_checkpoint=chosen["checkpoint_id"],selected_stage=chosen["stage"],
        selected_val_full_loss=chosen["val_full_loss"],
        selected_bc_max_abs=chosen["bc_max_abs"],
        selected_bc_feasible=chosen["bc_feasible"],
        checkpoint_selection_reason=reason,
        n_feasible_checkpoints=len(feasible),
        best_unrestricted_checkpoint=unrestricted["checkpoint_id"],
        best_adam_checkpoint=best_adam["checkpoint_id"],
        runtime_warmup_s=warm_s,runtime_adam_main_s=main_s,
        runtime_lbfgs_s=lb_s,runtime_sampling_s=sample_s,
        runtime_validation_s=validation_s,
        runtime_training_s=training_s,runtime_evaluation_s=eval_s,
        runtime_diagnostics_s=diag_s,diagnostic_dir=str(diagdir),
        **metrics,**details,**extra)


def compare(root: Path, preset: str, variants: tuple) -> None:
    data=[]
    for v in variants:
        path=root / ("PDE_v22_%s_%s"%(preset,v))/"results_raw.csv"
        if path.exists():
            data.append(pd.read_csv(path))
    if not data:
        return
    df=pd.concat(data,ignore_index=True)
    df=df.sort_values("variant").reset_index(drop=True)
    dest=root/("PDE_v22_%s_variant_comparison.csv"%preset)
    core.atomic_csv(dest,df)
    keys=df.variant.tolist()
    found=set(keys)
    integrity=dict(expected_variants=list(variants),found_variants=keys,
       duplicate_variants=len(keys)-len(found),
       missing_variants=sorted(set(variants)-found),
       complete=(len(keys)==len(found)==len(variants) and found==set(variants)),
       warning="Single-seed diagnostic only; no significance claims.")
    # Check shared inputs and sampling hashes for same sampler setting.
    if integrity["complete"] and len(variants)==4:
        values={r.variant:r for r in df.itertuples(index=False)}
        integrity["identical_warmup_within_transform"]=(
          values["standard_uniform"].warmup_state_sha256==values["standard_enriched"].warmup_state_sha256
          and values["bl_uniform"].warmup_state_sha256==values["bl_enriched"].warmup_state_sha256)
        integrity["identical_samples_within_distribution"]=(
          values["standard_uniform"].sampling_sha256==values["bl_uniform"].sampling_sha256
          and values["standard_enriched"].sampling_sha256==values["bl_enriched"].sampling_sha256)
        assert integrity["identical_warmup_within_transform"]
        assert integrity["identical_samples_within_distribution"]
    core.atomic_json(root / ("PDE_v22_%s_integrity.json"%preset),integrity)
    if not integrity["complete"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids=[v for v in VARIANTS if v in found]
    ordered=df.set_index("variant").loc[ids]
    fig,axs=plt.subplots(2,3,figsize=(14.2,8.3),layout="constrained")
    titlemap=[("space_time_rel_l2","Global relative L2"),
              ("final_rel_l2","Final-time relative L2"),
              ("bc_right_max_abs_densetime","Right BC max |u(1,t)|"),
              ("dense_physics_loss","Dense PDE loss"),
              ("residual_rms_outflow_999_1","RMS PDE residual x>=0.999"),
              ("runtime_training_s","Training time (s)")]
    for ax,(metric,title) in zip(axs.flat,titlemap):
        y=ordered[metric].to_numpy(dtype=float)
        ax.bar(np.arange(len(ids)),y)
        labels={"standard_uniform":"Std", "standard_enriched":"Enrich",
                "bl_uniform":"BL", "bl_enriched":"BL+Enrich"}
        ax.set_xticks(range(len(ids)),[labels[v] for v in ids],rotation=20)
        if metric in ("dense_physics_loss", "residual_rms_outflow_999_1") and np.all(y>0):
            ax.set_yscale("log")
        ax.set_title(title);ax.grid(alpha=.15,axis="y")
    fig.suptitle("V2.2 boundary-layer transform x enrichment: single seed only")
    fig.savefig(root/("PDE_v22_%s_comparison.png"%preset),dpi=145)
    plt.close(fig)
    print("Comparison saved:",dest,flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset",choices=["smoke","diagnostic"],default="smoke")
    parser.add_argument("--variants",default="all",
       help="all or comma-separated subset of standard_uniform,standard_enriched,bl_uniform,bl_enriched")
    parser.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
    parser.add_argument("--output-root",type=Path,default=Path("pde_v22_outputs"))
    parser.add_argument("--checkpoint-policy",choices=["bc_feasible","val_only"],default="bc_feasible")
    parser.add_argument("--bc-max-tol",type=float,default=0.02)
    parser.add_argument("--bc-weight",type=float,default=10.0)
    parser.add_argument("--enrichment-fraction",type=float,default=0.25)
    parser.add_argument("--enrich-dmin-factor",type=float,default=0.01)
    parser.add_argument("--enrich-dmax-factor",type=float,default=20.0)
    parser.add_argument("--bl-delta-factor",type=float,default=1.0)
    parser.add_argument("--D",type=float,default=0.001,help="Optional diffusion sensitivity; 0.001 is original")
    parser.add_argument("--warmup-epochs",type=int,default=None)
    parser.add_argument("--main-epochs",type=int,default=None)
    parser.add_argument("--ramp-epochs",type=int,default=None)
    parser.add_argument("--lbfgs-max-iter",type=int,default=None)
    args=parser.parse_args()
    if core.VERSION != BASE_V2_VERSION or v21.VERSION != BASE_V21_VERSION:
        raise RuntimeError("V2.2 needs core V2.0.0 and V2.1.0 unchanged")
    variants = (VARIANTS if args.variants=="all" else
        tuple(x.strip() for x in args.variants.split(",") if x.strip()))
    if not variants or len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS):
        parser.error("--variants must be unique, nonempty members of " + str(VARIANTS))
    cfg=make_config(args.preset)
    cfg["w_bc"]=args.bc_weight
    cfg["D"]=args.D
    cfg["enrichment_fraction"]=args.enrichment_fraction
    cfg["enrich_dmin_factor"]=args.enrich_dmin_factor
    cfg["enrich_dmax_factor"]=args.enrich_dmax_factor
    cfg["bl_delta_factor"]=args.bl_delta_factor
    cfg["checkpoint_bc_max_tol"]=args.bc_max_tol
    for key,arg in (("warmup_epochs",args.warmup_epochs),
                    ("main_epochs",args.main_epochs),
                    ("physics_ramp_epochs",args.ramp_epochs),
                    ("lbfgs_max_iter",args.lbfgs_max_iter)):
        if arg is not None:cfg[key]=arg
    core.validate_config(cfg)
    for name in ("D","w_bc","checkpoint_bc_max_tol","bl_delta_factor"):
        if not (np.isfinite(cfg[name]) and cfg[name]>0):
            parser.error(name+" must be strictly positive and finite")
    if not (0 < cfg["enrichment_fraction"] < 1):
        parser.error("--enrichment-fraction must lie between 0 and 1")
    focus_range(cfg)
    if cfg["D"] / abs(cfg["c"]) * cfg["bl_delta_factor"] >= 0.5:
        parser.error("Boundary layer delta too large")
    device=core.device_from_arg(args.device)
    output=args.output_root.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    hashes={"v22":v21.sha256_file(Path(__file__)),
        "v21":v21.sha256_file(Path(v21.__file__)),
        "v2":v21.sha256_file(Path(core.__file__))}
    print("PINN V2.2, device",device,"Torch",torch.__version__,
          "Python",sys.version.split()[0],flush=True)
    print("Preset",args.preset,"variants",variants,"BC",cfg["w_bc"],
          "delta",cfg["D"]/abs(cfg["c"])*cfg["bl_delta_factor"],flush=True)
    reference,refsig=core.get_reference(cfg,output)  # same reference solver V2.0
    core.sampling_preflight(cfg,device,output / "sampling_preflight.json")
    for variant in variants:
        folder=output / ("PDE_v22_%s_%s"%(args.preset,variant))
        folder.mkdir(parents=True,exist_ok=True)
        manifest=dict(version=VERSION,preset=args.preset,variant=variant,
           cfg=cfg,checkpoint_policy=args.checkpoint_policy,
           software=core.runtime_info(device),script_sha256=hashes,
           reference_signature=refsig,
           note="Fixed diagnostic MC-R, seed 0, not multi-seed publication evidence")
        mf=folder/"manifest.json"
        if mf.exists():
            if json.loads(mf.read_text(encoding="utf-8"))!=manifest:
                raise RuntimeError("Manifest conflict; choose NEW --output-root: "+str(folder))
        else:
            core.atomic_json(mf,manifest)
        raw=folder/"results_raw.csv"
        if raw.exists():
            d=pd.read_csv(raw)
            if len(d)!=1 or d.iloc[0].variant!=variant:
                raise RuntimeError("Existing unexpected run in "+str(raw))
            print("SKIP completed",variant,flush=True)
            continue
        diagdir=folder/"diagnostics"/("mc_nf%d_seed0"%cfg["n_collocation"][0])
        print("RUN",variant,flush=True)
        try:
            row=run_one(variant,cfg,reference,device,diagdir,args.checkpoint_policy)
            core.atomic_csv(raw,pd.DataFrame([row]))
            print("OK",variant,"global L2",row["space_time_rel_l2"],
                  "final L2",row["final_rel_l2"],
                  "BC right max",row["bc_right_max_abs_densetime"],flush=True)
            compare(output,args.preset,variants)
        except Exception:
            print(traceback.format_exc(),file=sys.stderr,flush=True)
            raise
    compare(output,args.preset,variants)
    print("V2.2 complete:",output,flush=True)


if __name__=="__main__":
    main()
