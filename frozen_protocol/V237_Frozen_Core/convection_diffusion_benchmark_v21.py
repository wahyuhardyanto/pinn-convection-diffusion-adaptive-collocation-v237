#!/usr/bin/env python3
"""Convection-diffusion PINN V2.1: controlled boundary-loss ablation.

Uses the unmodified V2 benchmark for the PDE, FDM, network parameterization,
and physics residual. Tests four main-stage BC weights for hard_ic_soft_bc:
  BC=1, 10, 50, 100; PDE=50; MC-R; Nf=500; seed=0 (diagnostic preset).
All variants use identical seed, samples, architecture and BC-only warmup
with weight=1, explicitly isolating the *main-stage* BC-weight change.

Additional diagnostics: BC left/right versus t, outflow-layer profiles on
nested scales, detailed error by spatial band, residual near x=1, full
checkpoint BC-feasibility audit and training traces.

Single-seed ablation only. Do NOT treat these runs as publication estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import convection_diffusion_benchmark_v2 as core

VERSION = "2.1.0"
BASE_REQUIRED_VERSION = "2.0.0"
BC_WEIGHTS = (1.0, 10.0, 50.0, 100.0)
MODE = "hard_ic_soft_bc"
METHOD = "mc"
DIAGNOSTIC_TIMES = (0.10, 0.50, 0.75, 1.00)
OUTFLOW_BANDS = ((0.0, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.0))


def make_config(preset: str) -> dict:
    if preset not in ("smoke", "diagnostic"):
        raise ValueError("V2.1 only implements smoke and diagnostic presets")
    cfg = core.preset_config(preset)
    cfg["methods"] = [METHOD]
    cfg["seeds"] = [0]
    cfg["n_collocation"] = [48] if preset == "smoke" else [500]
    cfg["w_res"] = 50.0
    cfg["w_ic"] = 1.0
    cfg["w_bc"] = 1.0 # baseline, overridden only in main training stage
    cfg["warmup_bc_weight"] = 1.0
    cfg["boundary_validation_times"] = 201 if preset == "smoke" else 401
    cfg["checkpoint_bc_max_tol"] = 0.02
    cfg["constraint_mode"] = MODE
    return cfg


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_label(w: float) -> str:
    return ("%03d" % int(w)) if w.is_integer() else str(w).replace(".", "p")


def model_checkpoint(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for k in sorted(state):
        digest.update(k.encode())
        digest.update(state[k].contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_boundary_grid(n: int, device: torch.device) -> tuple:
    t = np.linspace(0, 1, n, dtype=np.float32)
    left = np.stack((np.zeros_like(t), t), axis=1)
    right = np.stack((np.ones_like(t), t), axis=1)
    return t.astype(np.float64), torch.from_numpy(left).to(device), torch.from_numpy(right).to(device)


def boundary_eval(model: core.PINN, boundary_grid: tuple) -> dict:
    _, lxy, rxy = boundary_grid
    training_before = model.training
    model.eval()
    with torch.no_grad():
        l = model(lxy).detach().cpu().numpy().reshape(-1).astype(np.float64)
        r = model(rxy).detach().cpu().numpy().reshape(-1).astype(np.float64)
    if training_before:
        model.train()
    return dict(
        left_mse=float(np.mean(l*l)), right_mse=float(np.mean(r*r)),
        left_rmse=float(np.sqrt(np.mean(l*l))),
        right_rmse=float(np.sqrt(np.mean(r*r))),
        left_max_abs=float(np.max(np.abs(l))),
        right_max_abs=float(np.max(np.abs(r))),
        max_abs=float(max(np.max(np.abs(l)),np.max(np.abs(r)))),
    )


def validation_audit(model: core.PINN, fixed_val: tuple, cfg: dict,
                     boundary_grid: tuple, tol: float) -> dict:
    metrics = core.full_validation(model, fixed_val, cfg)
    bc = boundary_eval(model, boundary_grid)
    return dict(
        val_full_loss=metrics["total"],
        val_pde_loss=metrics["pde"],
        val_ic_loss=metrics["ic"],
        val_bc_loss=metrics["bc"],
        bc_left_rmse=bc["left_rmse"],
        bc_right_rmse=bc["right_rmse"],
        bc_left_max_abs=bc["left_max_abs"],
        bc_right_max_abs=bc["right_max_abs"],
        bc_max_abs=bc["max_abs"],
        bc_feasible=bool(bc["max_abs"] <= tol),
    )


def profiles_and_boundary_details(folder: Path, model: core.PINN, cfg: dict,
                                  ref: dict, pred: np.ndarray, history: list,
                                  candidates_df: pd.DataFrame, selected_id: str,
                                  boundary_grid: tuple, device: torch.device) -> dict:
    """Extend V2's 5 standard figures with detailed outflow/BC diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    v2_info = core.save_diagnostics(folder, model, cfg, ref, pred, history, device)
    t, lxy, rxy = boundary_grid
    bc_left = core.predict_numpy(model, lxy.detach().cpu().numpy(), device, cfg["evaluation_chunk"])
    bc_right = core.predict_numpy(model, rxy.detach().cpu().numpy(), device, cfg["evaluation_chunk"])
    boundary_trace = pd.DataFrame({"t":t, "u_left":bc_left,
                                  "u_right":bc_right,
                                  "abs_error_left":np.abs(bc_left),
                                  "abs_error_right":np.abs(bc_right)})
    core.atomic_csv(folder / "boundary_traces.csv", boundary_trace)
    fig,axs=plt.subplots(2,1,figsize=(10,6.2),sharex=True,layout="constrained")
    axs[0].plot(t, bc_left, label="PINN x=0", lw=1.5)
    axs[0].plot(t, bc_right, label="PINN x=1", lw=1.5)
    axs[0].axhline(0.0,color="black",lw=0.8,ls="--",label="BC target = 0")
    axs[0].set_ylabel("u at boundary");axs[0].legend();axs[0].grid(alpha=.2)
    axs[1].semilogy(t, np.maximum(np.abs(bc_left),1e-10),label="|u(0,t)|")
    axs[1].semilogy(t, np.maximum(np.abs(bc_right),1e-10),label="|u(1,t)|")
    axs[1].axhline(cfg["checkpoint_bc_max_tol"],ls=":",label="BC max tolerance")
    axs[1].set_xlabel("t");axs[1].set_ylabel("Absolute BC error (log)")
    axs[1].legend();axs[1].grid(alpha=.2)
    fig.suptitle("Boundary-condition diagnostics: left versus right")
    fig.savefig(folder / "boundary_condition_trace.png",dpi=155);plt.close(fig)

    x_ref=np.asarray(ref["x"]); times=np.asarray(ref["t"]); truth=np.asarray(ref["u"])
    xprobe=np.unique(np.r_[np.linspace(0,.9,91),
                           np.linspace(.9,.99,91),
                           np.linspace(.99,.999,91),
                           np.linspace(.999,1,121)]).astype(np.float64)
    profile_records=[]
    for ts in DIAGNOSTIC_TIMES:
        idx=int(np.argmin(abs(times-ts))); actual_t=float(times[idx])
        xy=np.column_stack((xprobe, np.full_like(xprobe,actual_t)))
        estimated=core.predict_numpy(model,xy,device,cfg["evaluation_chunk"])
        # Interpolation does not increase the resolution of the FDM reference.
        reference=np.interp(xprobe,x_ref,truth[idx])
        for x,yhat,yref in zip(xprobe,estimated,reference):
            profile_records.append(dict(t=actual_t,x=x,fdm_interpolated=yref,
                                        pinn=yhat,error=yhat-yref))
    detail_df=pd.DataFrame(profile_records)
    core.atomic_csv(folder/"outflow_probe_profiles.csv",detail_df)

    ranges=((0.9,1.0),(0.99,1.0),(0.999,1.0))
    fig,axs=plt.subplots(len(DIAGNOSTIC_TIMES),3,
                         figsize=(14,11.2),layout="constrained")
    for i,ts in enumerate(DIAGNOSTIC_TIMES):
        sub=detail_df[np.isclose(detail_df.t, float(times[np.argmin(abs(times-ts))]))]
        for j,(a,b) in enumerate(ranges):
            ax=axs[i,j]
            z=sub[sub.x.between(a,b)]
            ax.plot(z.x,z.fdm_interpolated,lw=1.6,label="FDM (interpolated)")
            ax.plot(z.x,z.pinn,lw=1.4,ls="--",label="PINN")
            ax.set_title(f"t={ts:g}, x in [{a:g},{b:g}]")
            ax.grid(alpha=.25)
            if j==0:ax.set_ylabel("u(x,t)")
            if i==len(DIAGNOSTIC_TIMES)-1:ax.set_xlabel("x")
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.suptitle("Nested outflow zooms; FDM subgrid positions interpolated",fontsize=13)
    fig.savefig(folder/"outflow_nested_zooms.png",dpi=155);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    for ax,ts in zip(axs,(.50,1.0)):
        sub=detail_df[np.isclose(detail_df.t, float(times[np.argmin(abs(times-ts))]))]
        z=sub[sub.x>=.99]
        ax.plot(z.x,z.error,label="PINN - FDM",lw=1.4)
        ax.axhline(0,color="black",ls="--",lw=.8)
        ax.set_xlabel("x");ax.set_ylabel("Error");ax.set_title("Outflow error t=%.2f"%ts)
        ax.grid(alpha=.25)
    fig.savefig(folder/"outflow_error_zoom.png",dpi=155);plt.close(fig)

    band_metrics=[]
    # Use original FDM mesh points so numerical accuracy is not implied below dx.
    for ts in DIAGNOSTIC_TIMES:
        i=int(np.argmin(abs(times-ts)));actual_t=float(times[i])
        for start,end in OUTFLOW_BANDS:
            mask=(x_ref>=start)&(x_ref<(end if end!=1.0 else end+1e-10))
            if mask.sum()==0:
                continue
            errors=pred[i,mask]-truth[i,mask]
            band_metrics.append(dict(t=actual_t,x_from=start,x_to=end,
                 fdm_points=int(mask.sum()),rmse=float(np.sqrt(np.mean(errors**2))),
                 mae=float(np.mean(np.abs(errors))),max_abs=float(np.max(np.abs(errors))),
                 relative_l2=float(np.linalg.norm(errors)/max(np.linalg.norm(truth[i,mask]),1e-15))))
    band_df=pd.DataFrame(band_metrics)
    core.atomic_csv(folder/"outflow_band_errors.csv",band_df)

    # Time-resolved outflow error on the 21 FDM snapshot times. The reference
    # is *spatially interpolated* from the original FDM mesh; this does not
    # imply that FDM features below the mesh spacing are resolved.
    xnear=np.unique(np.r_[np.linspace(.97,.99,51),
                            np.linspace(.99,.999,91),
                            np.linspace(.999,1,101)]).astype(np.float64)
    error_map=[]
    for i,ts in enumerate(times):
        xy=np.column_stack((xnear,np.full_like(xnear,float(ts))))
        yhat=core.predict_numpy(model,xy,device,cfg["evaluation_chunk"])
        yref=np.interp(xnear,x_ref,truth[i])
        error_map.append(yhat-yref)
    error_map=np.asarray(error_map)
    np.savez_compressed(folder/"outflow_error_snapshots.npz",
        t=times,x=xnear,error=error_map,
        reference_grid_dx=float(ref["dx"]))
    fig,axs=plt.subplots(1,2,figsize=(12.5,4.4),layout="constrained")
    colorlimit=max(1e-8,float(np.quantile(np.abs(error_map),.99)))
    for ax,limit in zip(axs,(.99,.999)):
        mask=xnear>=limit
        mesh=ax.pcolormesh(xnear[mask],times,error_map[:,mask],
             shading="auto",cmap="RdBu_r",vmin=-colorlimit,vmax=colorlimit)
        ax.set_title("PINN - FDM: x >= %.3f"%limit)
        ax.set_xlabel("x");ax.set_ylabel("t")
        fig.colorbar(mesh,ax=ax,shrink=.85)
    fig.suptitle("Outflow error; FDM spatial interpolation only")
    fig.savefig(folder/"outflow_error_heatmaps.png",dpi=155);plt.close(fig)

    candidates_df=candidates_df.copy()
    candidates_df["selected"]=(candidates_df["checkpoint_id"]==selected_id)
    core.atomic_csv(folder/"checkpoint_candidates.csv",candidates_df)
    fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
    is_feasible=candidates_df["bc_feasible"].astype(bool)
    for mask,name,marker in ((~is_feasible,"BC not feasible","x"),
                             (is_feasible,"BC feasible","o")):
        df=candidates_df[mask]
        if len(df):ax.scatter(df["bc_max_abs"],df["val_full_loss"],label=name,marker=marker)
    sel=candidates_df[candidates_df.selected]
    ax.scatter(sel.bc_max_abs,sel.val_full_loss,marker="*",s=170,label="Selected")
    ax.axvline(cfg["checkpoint_bc_max_tol"],ls="--",lw=1.0,label="BC threshold")
    ax.set_xlabel("Max absolute BC violation (both boundaries)")
    ax.set_ylabel("Full-weight validation loss")
    ax.set_yscale("log");ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(folder/"checkpoint_tradeoff.png",dpi=155);plt.close(fig)
    return dict(**v2_info,
         bc_left_rmse_densetime=float(np.sqrt(np.mean(bc_left**2))),
         bc_right_rmse_densetime=float(np.sqrt(np.mean(bc_right**2))),
         bc_left_max_abs_densetime=float(np.max(np.abs(bc_left))),
         bc_right_max_abs_densetime=float(np.max(np.abs(bc_right))),
         extra_diagnostic_files=["boundary_condition_trace.png",
           "outflow_nested_zooms.png","outflow_error_zoom.png",
           "checkpoint_tradeoff.png","outflow_probe_profiles.csv",
           "outflow_band_errors.csv","boundary_traces.csv",
           "checkpoint_candidates.csv","outflow_error_heatmaps.png",
           "outflow_error_snapshots.npz"])


def run_single(cfg: dict, weight: float, device: torch.device,
               ref: dict, diagnostic_folder: Path,
               checkpoint_policy: str) -> dict:
    seed=0
    nf=int(cfg["n_collocation"][0])
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    model=core.PINN(cfg["layers"],MODE).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["adam_lr"])
    val=core.fixed_validation(cfg,seed,device)
    bc_grid=make_boundary_grid(cfg["boundary_validation_times"],device)
    ic_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,992]))
    bc_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,993]))
    sample_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,991]))
    warm_bc_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,702]))
    warm_ic_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,701]))
    rows=[]
    checkpoint_records=[]
    checkpoint_states={}
    sampling_hash=hashlib.sha256()
    start_training=time.perf_counter()

    # STAGE A: EXACTLY SAME warmup for all four BC-weight variations.
    warm_start=time.perf_counter()
    for epoch in range(1,cfg["warmup_epochs"]+1):
        _ic, _icy=core.sample_ic(cfg["n_ic"],warm_ic_rng,device)
        bc=core.sample_bc(cfg["n_bc"],warm_bc_rng,device)
        optimizer.zero_grad(set_to_none=True)
        bc_loss=torch.mean(model(bc)**2)
        objective=cfg["warmup_bc_weight"]*bc_loss
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError("Nonfinite BC warmup loss")
        objective.backward();optimizer.step()
        if epoch in (1,cfg["warmup_epochs"]) or epoch%cfg["diagnostics_every"]==0:
            rows.append(dict(phase="warmup",global_epoch=epoch,main_epoch=0,
                  pde_weight=0.0,bc_weight=cfg["warmup_bc_weight"],
                  objective=float(objective.detach()),pde_loss=np.nan,
                  ic_loss=0.0,bc_loss=float(bc_loss.detach()),
                  val_full_loss=np.nan,val_bc_left_rmse=np.nan,
                  val_bc_right_rmse=np.nan,val_bc_max_abs=np.nan))
    warmup_seconds=time.perf_counter()-warm_start
    warmup_hash=state_hash(model_checkpoint(model))

    # STAGE B: main Adam; same samples and initialization, BC weight is varied.
    settings={**cfg,"w_bc":float(weight)}
    recent=deque(maxlen=cfg["lbfgs_union_sets"])
    sampling_seconds=0.0
    validation_seconds=0.0
    set_count=0
    main_start=time.perf_counter()
    for epoch in range(1,cfg["main_epochs"]+1):
        if (epoch-1)%cfg["resample_every"]==0:
            ss=time.perf_counter()
            positions=core.uniform_mc(nf,sample_rng).astype(np.float32)
            recent.append(positions)
            sampling_hash.update(positions.tobytes())
            residual_xt=torch.from_numpy(positions).to(device)
            set_count+=1
            sampling_seconds+=time.perf_counter()-ss
        ic,icy=core.sample_ic(cfg["n_ic"],ic_rng,device)
        bc=core.sample_bc(cfg["n_bc"],bc_rng,device)
        current_pde_weight=cfg["w_res"]*min(1.0,epoch/cfg["physics_ramp_epochs"])
        optimizer.zero_grad(set_to_none=True)
        objective,pde,ic_loss,bc_loss=core.full_losses(
             model,residual_xt,ic,icy,bc,settings,current_pde_weight)
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError("Nonfinite Adam loss at epoch %d"%epoch)
        objective.backward();optimizer.step()
        record_validation=(epoch%cfg["validation_every"]==0 or epoch==cfg["main_epochs"])
        audited=None
        if record_validation:
            vs=time.perf_counter()
            audited=validation_audit(model,val,settings,bc_grid,cfg["checkpoint_bc_max_tol"])
            validation_seconds+=time.perf_counter()-vs
            if epoch>=cfg["physics_ramp_epochs"]:
                identifier="adam_%d"%epoch
                checkpoint_records.append(dict(checkpoint_id=identifier,stage="adam",
                                                 main_epoch=epoch,**audited))
                checkpoint_states[identifier]=model_checkpoint(model)
        if epoch==1 or epoch%cfg["diagnostics_every"]==0 or record_validation:
            rows.append(dict(phase="main",global_epoch=cfg["warmup_epochs"]+epoch,
                  main_epoch=epoch,pde_weight=current_pde_weight,
                  bc_weight=weight,objective=float(objective.detach()),
                  pde_loss=float(pde.detach()),ic_loss=float(ic_loss.detach()),
                  bc_loss=float(bc_loss.detach()),
                  val_full_loss=audited["val_full_loss"] if audited else np.nan,
                  val_bc_left_rmse=audited["bc_left_rmse"] if audited else np.nan,
                  val_bc_right_rmse=audited["bc_right_rmse"] if audited else np.nan,
                  val_bc_max_abs=audited["bc_max_abs"] if audited else np.nan))
    adam_seconds=time.perf_counter()-main_start
    if not checkpoint_records:
        raise RuntimeError("No full-ramp Adam checkpoint available")
    adam_best=min((q for q in checkpoint_records if q["stage"]=="adam"),
                  key=lambda x:x["val_full_loss"])
    model.load_state_dict(checkpoint_states[adam_best["checkpoint_id"]])

    # STAGE C: identical L-BFGS protocol; start at best Adam full validation.
    union=np.unique(np.concatenate(list(recent),axis=0),axis=0)
    x_union=torch.from_numpy(union).to(device)
    lb_ic,lb_y=core.sample_ic(cfg["n_ic"],
         np.random.default_rng(np.random.SeedSequence([seed,nf,995])),device)
    lb_bc=core.sample_bc(cfg["n_bc"],
         np.random.default_rng(np.random.SeedSequence([seed,nf,996])),device)
    lb_calls=0;lb_seconds=0.0;lb_status="not_run";lb_error=""
    if cfg["lbfgs_max_iter"]>0:
        lb_optimizer=torch.optim.LBFGS(model.parameters(),lr=1.0,
          max_iter=cfg["lbfgs_max_iter"],history_size=50,line_search_fn="strong_wolfe")
        bs=time.perf_counter()
        def closure():
            nonlocal lb_calls
            lb_calls+=1
            lb_optimizer.zero_grad(set_to_none=True)
            obj=core.full_losses(model,x_union,lb_ic,lb_y,lb_bc,settings,cfg["w_res"])[0]
            if not bool(torch.isfinite(obj).item()):
                raise FloatingPointError("Nonfinite L-BFGS loss")
            obj.backward()
            return obj
        try:
            lb_optimizer.step(closure)
            audited=validation_audit(model,val,settings,bc_grid,cfg["checkpoint_bc_max_tol"])
            checkpoint_records.append(dict(checkpoint_id="lbfgs",stage="lbfgs",
                                           main_epoch=cfg["main_epochs"],**audited))
            checkpoint_states["lbfgs"]=model_checkpoint(model)
            lb_status="succeeded"
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            lb_status="failed_reverted"
            lb_error=repr(exc)[:700]
            print("L-BFGS failed, reverting to Adam:",lb_error,flush=True)
        lb_seconds=time.perf_counter()-bs
    candidates=pd.DataFrame(checkpoint_records)
    unrestricted=min(checkpoint_records,key=lambda x:x["val_full_loss"])
    feasible=[q for q in checkpoint_records if q["bc_feasible"]]
    # No FDM access in checkpoint selection, to keep evaluation independent.
    if checkpoint_policy=="bc_feasible" and feasible:
        chosen=min(feasible,key=lambda x:x["val_full_loss"])
        selection_reason="min_full_val_among_bc_feasible"
    elif checkpoint_policy=="bc_feasible":
        chosen=unrestricted
        selection_reason="no_bc_feasible_checkpoint_fallback_to_min_full_val"
    else:
        chosen=unrestricted
        selection_reason="min_full_validation_loss"
    model.load_state_dict(checkpoint_states[chosen["checkpoint_id"]])
    training_seconds=time.perf_counter()-start_training

    es=time.perf_counter()
    metrics,pred=core.evaluate(model,ref,settings,val,device)
    eval_seconds=time.perf_counter()-es
    ds=time.perf_counter()
    detail=profiles_and_boundary_details(diagnostic_folder,model,settings,ref,pred,
             rows,candidates,chosen["checkpoint_id"],bc_grid,device)
    diag_seconds=time.perf_counter()-ds
    if not all(np.isfinite(metrics[k]) for k in
       ("space_time_rel_l2","final_rel_l2","bc_mse","dense_physics_loss")):
        raise FloatingPointError("Nonfinite evaluation metric")
    return dict(protocol_version=VERSION,method=METHOD,method_label="MC-R",
        constraint_mode=MODE,n_collocation=nf,seed=seed,
        main_bc_weight=weight,warmup_bc_weight=cfg["warmup_bc_weight"],
        pde_weight=cfg["w_res"],warmup_epochs=cfg["warmup_epochs"],
        main_epochs=cfg["main_epochs"],physics_ramp_epochs=cfg["physics_ramp_epochs"],
        resample_every=cfg["resample_every"],resample_set_count=set_count,
        sampling_sha256=sampling_hash.hexdigest(),warmup_state_sha256=warmup_hash,
        n_lbfgs_points=len(union),lbfgs_closure_calls=lb_calls,
        lbfgs_status=lb_status,lbfgs_error=lb_error,
        checkpoint_policy=checkpoint_policy,checkpoint_bc_max_tol=cfg["checkpoint_bc_max_tol"],
        n_adam_checkpoints=int(sum(x["stage"]=="adam" for x in checkpoint_records)),
        n_feasible_checkpoints=len(feasible),selection_reason=selection_reason,
        selected_checkpoint=chosen["checkpoint_id"],
        selected_stage=chosen["stage"],
        selected_val_full_loss=chosen["val_full_loss"],
        selected_bc_max_abs=chosen["bc_max_abs"],
        selected_bc_feasible=chosen["bc_feasible"],
        best_unrestricted_checkpoint=unrestricted["checkpoint_id"],
        best_unrestricted_val_full_loss=unrestricted["val_full_loss"],
        best_unrestricted_bc_max_abs=unrestricted["bc_max_abs"],
        best_adam_checkpoint=adam_best["checkpoint_id"],
        best_adam_val_full_loss=adam_best["val_full_loss"],
        runtime_warmup_s=warmup_seconds,runtime_adam_main_s=adam_seconds,
        runtime_lbfgs_s=lb_seconds,runtime_sampling_s= sampling_seconds,
        runtime_validation_s=validation_seconds,runtime_training_s=training_seconds,
        runtime_evaluation_s=eval_seconds,runtime_diagnostics_s=diag_seconds,
        diagnostic_dir=str(diagnostic_folder),**metrics,**detail)


def make_comparison(root: Path, preset: str, weights: tuple) -> None:
    """Save descriptive comparison, integrity, and optional summary chart."""
    frames=[]
    for w in weights:
        path=root/("PDE_v21_%s_wbc%s"%(preset,weight_label(w)))/"results_raw.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    result=pd.concat(frames,ignore_index=True)
    result=result.sort_values("main_bc_weight").reset_index(drop=True)
    dest=root/("PDE_v21_%s_bc_weight_comparison.csv"%preset)
    core.atomic_csv(dest,result)
    keys=[float(v) for v in result.main_bc_weight]
    integrity=dict(expected_weights=list(weights),found_weights=keys,
          missing_weights=sorted(set(weights)-set(keys)),
          duplicate_weights=len(keys)-len(set(keys)),
          complete=(len(keys)==len(set(keys))==len(weights) and set(keys)==set(weights)),
          warning="One paired seed only; NO inferential statistical test.")
    core.atomic_json(root/("PDE_v21_%s_integrity.json"%preset),integrity)
    print("Summary saved:",dest,"complete:",integrity["complete"],flush=True)
    if not integrity["complete"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=result.main_bc_weight.to_numpy(dtype=float)
    fig,axs=plt.subplots(2,2,figsize=(11.5,8.3),layout="constrained")
    for ax,metric,title in ((axs[0,0],"space_time_rel_l2","Space-time relative L2"),
        (axs[0,1],"final_rel_l2","Final-time relative L2"),
        (axs[1,0],"bc_right_max_abs_densetime","Right BC max |u(1,t)|"),
        (axs[1,1],"dense_physics_loss","Dense PDE physics loss")):
        ax.plot(x,result[metric],marker="o",lw=1.5)
        ax.set_xscale("log");ax.set_xticks(x);ax.set_xticklabels([str(int(v)) for v in x])
        if metric in ("dense_physics_loss",): ax.set_yscale("log")
        ax.set_title(title);ax.set_xlabel("Main-stage BC weight");ax.grid(alpha=.3)
    fig.suptitle("BC-loss ablation: single seed, not a statistical comparison")
    fig.savefig(root/("PDE_v21_%s_comparison.png"%preset),dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset",choices=["smoke","diagnostic"],default="smoke")
    parser.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
    parser.add_argument("--output-root",type=Path,default=Path("pde_v21_outputs"))
    parser.add_argument("--bc-weights",default="1,10,50,100",
                        help="Subset of 1,10,50,100, e.g. 1,10")
    parser.add_argument("--checkpoint-policy",choices=["bc_feasible","val_only"],
                        default="bc_feasible")
    parser.add_argument("--bc-max-tol",type=float,default=0.02,
                        help="Feasibility: max boundary violation over dense-time grid")
    parser.add_argument("--warmup-epochs",type=int,default=None)
    parser.add_argument("--main-epochs",type=int,default=None)
    parser.add_argument("--ramp-epochs",type=int,default=None)
    parser.add_argument("--lbfgs-max-iter",type=int,default=None)
    args=parser.parse_args()
    if core.VERSION != BASE_REQUIRED_VERSION:
        raise RuntimeError("V2.1 requires V2 core version %s, received %s"%
                           (BASE_REQUIRED_VERSION,core.VERSION))
    weights=tuple(float(x.strip()) for x in args.bc_weights.split(",") if x.strip())
    if len(weights)!=len(set(weights)) or not weights or not set(weights)<=set(BC_WEIGHTS):
        parser.error("BC weights must be a unique non-empty subset of 1,10,50,100")
    if not (np.isfinite(args.bc_max_tol) and args.bc_max_tol>0):
        parser.error("--bc-max-tol must be finite and positive")
    cfg=make_config(args.preset)
    cfg["checkpoint_bc_max_tol"]=args.bc_max_tol
    for key,v in (("warmup_epochs",args.warmup_epochs),
                  ("main_epochs",args.main_epochs),
                  ("physics_ramp_epochs",args.ramp_epochs),
                  ("lbfgs_max_iter",args.lbfgs_max_iter)):
        if v is not None:cfg[key]=v
    core.validate_config(cfg)
    device=core.device_from_arg(args.device)
    root=args.output_root.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    sha_v21=sha256_file(Path(__file__))
    sha_v2=sha256_file(Path(core.__file__))
    print("Convection-diffusion V2.1 boundary-loss ablation",flush=True)
    print("Device",device,"Python",sys.version.split()[0],"Torch",torch.__version__,flush=True)
    print("Preset",args.preset,"weights",weights,"checkpoint",args.checkpoint_policy,flush=True)
    print("Hard IC/soft BC, MC-R, Nf",cfg["n_collocation"][0],"seed=0",flush=True)
    print("Same warmup BC weight for all runs:",cfg["warmup_bc_weight"],flush=True)
    ref,ref_sig=core.get_reference(cfg,root)
    core.sampling_preflight(cfg,device,root/"sampling_preflight.json")
    for w in weights:
        work=root/("PDE_v21_%s_wbc%s"%(args.preset,weight_label(w)))
        work.mkdir(parents=True,exist_ok=True)
        manifest=dict(protocol_version=VERSION,preset=args.preset,
            main_bc_weight=w,seed=0,method=METHOD,mode=MODE,
            config=cfg,checkpoint_policy=args.checkpoint_policy,
            runtime=core.runtime_info(device),script_v21_sha256=sha_v21,
            script_v2_sha256=sha_v2,reference_signature=ref_sig,
            note="Diagnostic of BC weights only; same warmup weight=1 for all runs.")
        manifest_file=work/"manifest.json"
        if manifest_file.exists():
            if json.loads(manifest_file.read_text(encoding="utf-8"))!=manifest:
                raise RuntimeError("Manifest mismatch in %s. Choose a NEW --output-root."%work)
        else:
            core.atomic_json(manifest_file,manifest)
        dest=work/"results_raw.csv"
        if dest.exists():
            df=pd.read_csv(dest)
            if len(df)!=1 or float(df.iloc[0].main_bc_weight)!=w:
                raise RuntimeError("Existing output has an unexpected run key: "+str(dest))
            print("SKIP completed BC weight",w,flush=True)
            continue
        diag_dir=work/"diagnostics"/("mc_nf%d_seed0" % int(cfg["n_collocation"][0]))
        print("RUN main BC weight:",w,flush=True)
        try:
            row=run_single(cfg,w,device,ref,diag_dir,args.checkpoint_policy)
            core.atomic_csv(dest,pd.DataFrame([row]))
            print("OK wBC=%g space-time L2=%.5f BC max=%.5f final L2=%.5f PDE=%.5g"%
                  (w,row["space_time_rel_l2"],row["selected_bc_max_abs"],
                   row["final_rel_l2"],row["dense_physics_loss"]),flush=True)
            make_comparison(root,args.preset,weights)
        except Exception:
            print(traceback.format_exc(),file=sys.stderr,flush=True)
            print("Completed runs retained. Rerun command to resume.",file=sys.stderr)
            raise
    make_comparison(root,args.preset,weights)
    print("V2.1 FINISHED:",root,flush=True)


if __name__=="__main__":
    main()
