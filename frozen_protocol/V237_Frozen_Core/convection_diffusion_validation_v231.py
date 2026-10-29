#!/usr/bin/env python3
"""PINN convection-diffusion V2.3.1 validation: V2.2 boundary-layer diagnostics + staged
training curriculum adapted from the team's V3.1 repository + independent
Scharfetter-Gummel (SG) secondary reference.

Scope: single-seed diagnosis (NOT publication four-sampler experiments).
Reference FDM and full-weight validation/checkpoint selection are inherited
unchanged from V2.0/V2.1.  SG is only for verification and never for training,
adaptation, checkpoint selection, or model tuning inside a run.

V3.1-like default: 100 BC-only Adam epochs, 900 linear PDE-weight ramp
and 4000 joint Adam epochs, weights (PDE,IC,BC)=(10,20,20).  Because V2.2
hard-enforces IC, the IC soft term is identically zero: actual objective is
w_pde*MSE(residual) + 20*MSE(BC).  The frozen 10:20:20 settings of the source
repo were for a different PINN configuration; this is an adaptation, not an
identical reimplementation of the original V3.1 experiment.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import splu

import convection_diffusion_benchmark_v2 as core
import convection_diffusion_benchmark_v21 as v21
import convection_diffusion_benchmark_v22 as v22

VERSION = "2.3.1"
METHODS = ("mc", "lhs", "rad", "alhs")
VARIANTS = v22.VARIANTS
SG_REFERENCE_VERSION = "sg-bernoulli-be-1.0"


def bernoulli(z):
    """Stable Bernoulli B(z)=z/(exp(z)-1) for SG exponentially fitted flux."""
    zz = np.asarray(z, dtype=np.float64)
    out = np.empty_like(zz)
    small = np.abs(zz) < 1e-4
    pos = zz > 50
    neg = zz < -50
    mid = ~(small | pos | neg)
    z0 = zz[small]
    out[small] = 1 - z0/2 + z0*z0/12 - z0**4/720
    out[pos] = zz[pos] * np.exp(-zz[pos]) / (-np.expm1(-zz[pos]))
    out[neg] = zz[neg] / np.expm1(zz[neg])
    out[mid] = zz[mid] / np.expm1(zz[mid])
    return float(out) if np.ndim(z) == 0 else out


def solve_sg_reference(cfg: dict) -> dict:
    """Backward Euler finite volume SG reference at SAME x/t/snapshot grid as FDM.

    J_{i+1/2}=D/dx*( B(-Pe)*u_i - B(Pe)*u_{i+1} ), Pe=c*dx/D.
    BC zero at x=0 and x=1; u(x,0)=sin(pi x).  No source terms.
    """
    nx,nt = int(cfg["reference_nx"]),int(cfg["reference_nt"])
    x0,x1 = cfg["x_range"]
    t0,t1 = cfg["t_range"]
    x = np.linspace(x0,x1,nx,dtype=np.float64)
    dx = (x1-x0)/(nx-1)
    dt = (t1-t0)/(nt-1)
    D,c = float(cfg["D"]),float(cfg["c"])
    if D <= 0:
        raise ValueError("Scharfetter-Gummel requires strictly positive diffusion D")
    pe = c*dx/D
    bm,bp = bernoulli(-pe),bernoulli(pe)
    fac = D*dt/(dx*dx)
    lower = -fac*bm
    upper = -fac*bp
    center = 1 + fac*(bm+bp)
    matrix = diags((np.full(nx-3,lower),np.full(nx-2,center),
                    np.full(nx-3,upper)), (-1,0,1), shape=(nx-2,nx-2),format="csc")
    lu = splu(matrix)
    u = np.sin(np.pi*(x-x0)/(x1-x0))
    u[[0,-1]]=0.0
    selected=np.linspace(0,nt-1,int(cfg["reference_snapshots"]),dtype=np.int64)
    wanted={int(s):int(j) for j,s in enumerate(selected)}
    snapshots=np.empty((len(selected),nx),dtype=np.float64)
    snapshots[0]=u
    for k in range(1,nt):
        u[1:-1]=lu.solve(u[1:-1])
        if k in wanted:
            snapshots[wanted[k]]=u
    return dict(x=x,t=t0+selected*dt,u=snapshots,dx=dx,dt=dt,
                cell_peclet=abs(pe),sg_peclet=pe)


def secondary_reference(cfg, fdm, root):
    """Secondary SG reference; cached by grid/physics. Does not touch FDM cache."""
    ref_keys=("c","D","x_range","t_range","reference_nx","reference_nt","reference_snapshots")
    signature=hashlib.sha256(json.dumps({k:cfg[k] for k in ref_keys},sort_keys=True).encode()).hexdigest()[:12]
    out=root / ("sg_secondary_" + signature + ".npz")
    meta=root / ("sg_secondary_" + signature + ".json")
    if not out.exists():
        started=time.perf_counter()
        sg=solve_sg_reference(cfg)
        np.savez_compressed(out,**sg)
        core.atomic_json(meta,{"solver":"Backward Euler + Scharfetter-Gummel flux",
             "signature":signature,"elapsed_s":time.perf_counter()-started,
             "reference_only":True,"operator":"J=c*u-D*u_x", "version":SG_REFERENCE_VERSION})
    with np.load(out) as z:
        sg={k:z[k] for k in z.files}
    if not np.array_equal(sg["x"],fdm["x"]) or not np.array_equal(sg["t"],fdm["t"]):
        raise RuntimeError("SG/FDM reference grids or snapshots do not match")
    error=np.asarray(sg["u"])-np.asarray(fdm["u"])
    boundary=(np.asarray(fdm["x"])>=.99)
    report={"verification":"secondary_reference_not_exact_solution",
        "fdm_primary_unchanged":True,"cell_peclet":float(fdm["cell_peclet"]),
        "grid_nx":len(fdm["x"]), "snapshot_count":len(fdm["t"]),
        "relative_l2_sg_minus_fdm":float(np.linalg.norm(error)/max(np.linalg.norm(fdm["u"]),1e-15)),
        "max_absolute_sg_minus_fdm":float(np.max(np.abs(error))),
        "outflow_x_ge_099_rmse":float(np.sqrt(np.mean(error[:,boundary]**2))),
        "sg_min":float(np.min(sg["u"])),"sg_max":float(np.max(sg["u"])),
        "fdm_min":float(np.min(fdm["u"])),"fdm_max":float(np.max(fdm["u"])),
        "warning":"Agreement of SG and FDM on the same mesh is not a grid-convergence proof."}
    core.atomic_json(root/("sg_vs_fdm_"+signature+".json"),report)
    make_sg_figures(root,signature,fdm,sg)
    return sg,report,signature


def make_sg_figures(root,signature,fdm,sg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,8.2),layout="constrained")
    xx=np.asarray(fdm["x"])
    t=np.asarray(fdm["t"])
    for k, target in enumerate((.1,.5,1.0)):
        i=int(np.argmin(abs(t-target)))
        axs[0,k].plot(xx,fdm["u"][i],label="FDM centered",lw=1.6)
        axs[0,k].plot(xx,sg["u"][i],ls="--",label="SG fitted",lw=1.2)
        axs[0,k].set_title(f"Reference at t={t[i]:.3f}")
        axs[0,k].set_xlabel("x");axs[0,k].grid(alpha=.22)
        mask=xx>=.99
        axs[1,k].plot(xx[mask],fdm["u"][i][mask],label="FDM centered",lw=1.6)
        axs[1,k].plot(xx[mask],sg["u"][i][mask],ls="--",label="SG fitted",lw=1.2)
        axs[1,k].set_xlabel("x near right boundary");axs[1,k].grid(alpha=.22)
        if k==0: axs[0,k].legend(fontsize=8)
    fig.suptitle("Reference-solver comparison on identical grids (not a convergence proof)")
    fig.savefig(root/("sg_vs_fdm_"+signature+".png"),dpi=145)
    plt.close(fig)


def make_config(preset: str, curriculum: str="v31_staged") -> dict:
    cfg=v22.make_config("smoke" if preset=="smoke" else "diagnostic")
    if preset not in ("smoke","diagnostic"):
        raise ValueError("V2.3 only permits smoke or diagnostic until PDE is validated")
    cfg["constraint_mode"]="hard_ic_soft_bc"
    if curriculum == "v31_staged":
        if preset=="smoke":
            cfg.update(warmup_epochs=8,transition_epochs=12,joint_epochs=18,
                       w_res=10.0,w_ic=20.0,w_bc=20.0,warmup_bc_weight=20.0,
                       lbfgs_max_iter=3)
        else:
            cfg.update(warmup_epochs=100,transition_epochs=900,joint_epochs=4000,
                       w_res=10.0,w_ic=20.0,w_bc=20.0,warmup_bc_weight=20.0,
                       lbfgs_max_iter=300)
    elif curriculum == "v22_matched_5000":
        # Controlled 5000-Adam-update comparator: similar total effort to V3.1.
        # This is NOT identical to historical V2.2's 6000 Adam updates.
        if preset=="smoke":
            cfg.update(warmup_epochs=8,transition_epochs=12,joint_epochs=18,
                       w_res=50.0,w_ic=1.0,w_bc=10.0,warmup_bc_weight=1.0,
                       lbfgs_max_iter=3)
        else:
            cfg.update(warmup_epochs=1000,transition_epochs=500,joint_epochs=3500,
                       w_res=50.0,w_ic=1.0,w_bc=10.0,warmup_bc_weight=1.0,
                       lbfgs_max_iter=300)
    else:
        raise ValueError("Unknown curriculum "+curriculum)
    cfg["physics_ramp_epochs"]=int(cfg["transition_epochs"])
    cfg["main_epochs"]=int(cfg["transition_epochs"]+cfg["joint_epochs"])
    cfg["curriculum"]=curriculum
    cfg["checkpoint_bc_max_tol"]=0.02
    cfg["resample_every"]=5 if preset=="smoke" else 250
    cfg["validation_every"]=5 if preset=="smoke" else 250
    cfg["independent_residual_points"]=128 if preset=="smoke" else 4096
    cfg["independent_residual_seed"]=1234567
    cfg["validation_fdm_policy"]="fdm_from_v2_unchanged"
    return cfg


def curriculum_weight(cfg,transition_epoch):
    n=int(cfg["transition_epochs"])
    return float(cfg["w_res"])*min(1.0,max(0.0,transition_epoch/n))


def method_points(nf,method,cfg,initial_rng,sampler_rng,focus_rng,
                  adaptive,model,phase,step,enriched):
    """Common initial LHS during transition, then method-specific joint sampling.

    Focused points replace global points (budget stays nf). Global MC/LHS and
    adaptive RAD-IID/ALHS-R apply only to the global subset.
    """
    if method not in METHODS:
        raise ValueError(method)
    n_focus=int(round(nf*cfg["enrichment_fraction"])) if enriched else 0
    n_global=nf-n_focus
    if phase=="transition":
        xy=core.uniform_lhs(nf,initial_rng)
        base=xy[:n_global]
    elif method=="mc":
        # full-draw prefix alignment between enriched vs uniform for MC.
        base=core.uniform_mc(nf,sampler_rng)[:n_global]
    elif method=="lhs":
        base=core.uniform_lhs(n_global,sampler_rng)
    else:
        if adaptive is None or adaptive.updates<1:
            raise RuntimeError("Residual adaptive model must be updated at each joint refresh")
        base=adaptive.sample(n_global,sampler_rng,method=="alhs")
    if not enriched:
        return np.asarray(base,dtype=np.float32),0
    dmin,dmax=v22.focus_range(cfg)
    z=focus_rng.random((n_focus,2))
    dist=np.exp(np.log(dmin)+z[:,0]*np.log(dmax/dmin))
    focused=np.column_stack((1.0-dist,z[:,1]))
    pts=np.vstack((base,focused)).astype(np.float32)
    if not (len(pts)==nf and np.isfinite(pts).all() and np.all((pts>=0)&(pts<=1))):
        raise RuntimeError("Nonfinite or wrong-size collocation set")
    return pts,n_focus


def independent_residual_audit(model,cfg,device,seed=1234567):
    rng=np.random.default_rng(seed)
    xy=core.uniform_lhs(cfg["independent_residual_points"],rng)
    residual=core.residual_numpy(model,xy,cfg,device)
    return {"rms":float(np.sqrt(np.mean(residual**2))),
            "p95_abs":float(np.quantile(np.abs(residual),.95)),
            "p99_abs":float(np.quantile(np.abs(residual),.99)),
            "max_abs":float(np.max(np.abs(residual))),
            "n_points":len(residual),"seed":seed}


def run_one(variant,method,cfg,ref,sg_ref,device,diagdir,checkpoint_policy,seed=0):
    # V2.3 training code retained; the seed is now an explicit run argument.
    nf=int(cfg["n_collocation"][0]);seed=int(seed)
    use_bl,enriched=v22.variant_flags(variant)
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    delta=cfg["D"]/abs(cfg["c"])*cfg["bl_delta_factor"]
    model=v22.BoundaryLayerPINN(cfg["layers"],use_bl,delta).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["adam_lr"])
    val=core.fixed_validation(cfg,seed,device)
    boundary_grid=v21.make_boundary_grid(cfg["boundary_validation_times"],device)
    warm_bc_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,702]))
    bc_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,993]))
    ic_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,992]))
    init_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,990]))
    global_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,991]))
    focus_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,997]))
    hist=[];states={};candidates=[]
    hash_all=hashlib.sha256()
    collocation_sets=[]
    t_start=time.perf_counter()
    warm_t=time.perf_counter()
    for e in range(1,cfg["warmup_epochs"]+1):
        bc=core.sample_bc(cfg["n_bc"],warm_bc_rng,device)
        optimizer.zero_grad(set_to_none=True)
        constraint=(model(bc)**2).mean()
        objective=cfg["warmup_bc_weight"]*constraint
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError("Nonfinite warmup objective")
        objective.backward();optimizer.step()
        if e==1 or e==cfg["warmup_epochs"] or e%cfg["diagnostics_every"]==0:
            hist.append(dict(phase="warmup",curriculum_stage="warmup",
                global_epoch=e,main_epoch=0,pde_weight=0.0,bc_weight=cfg["warmup_bc_weight"],
                objective=float(objective.detach()),pde_loss=np.nan,ic_loss=0.0,
                bc_loss=float(constraint.detach()),val_full_loss=np.nan))
    warm_s=time.perf_counter()-warm_t
    warm_hash=v21.state_hash(v21.model_checkpoint(model))
    recent=deque(maxlen=int(cfg["lbfgs_union_sets"]))
    sampler_seconds=0.0;validation_seconds=0.0
    focus_last=0
    sampler_updates=0
    if cfg["transition_epochs"]>0:
        pts,focus_last=method_points(nf,method,cfg,init_rng,global_rng,focus_rng,
                                    None,model,"transition",0,enriched)
        # This initial LHS design is frozen for all 900 transition epochs.
        hash_all.update(pts.tobytes());collocation_sets.append(pts)
        recent.append(pts)
        current=torch.from_numpy(pts).to(device)
    transition_start=time.perf_counter()
    for e in range(1,cfg["transition_epochs"]+1):
        ic,icy=core.sample_ic(cfg["n_ic"],ic_rng,device)
        bc=core.sample_bc(cfg["n_bc"],bc_rng,device)
        w=curriculum_weight(cfg,e)
        optimizer.zero_grad(set_to_none=True)
        loss,pde,ic_l,bc_l=core.full_losses(model,current,ic,icy,bc,cfg,w)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Nonfinite transition loss")
        loss.backward();optimizer.step()
        if e==1 or e==cfg["transition_epochs"] or e%cfg["diagnostics_every"]==0:
            hist.append(dict(phase="main",curriculum_stage="transition",
                global_epoch=cfg["warmup_epochs"]+e,main_epoch=e,
                pde_weight=w,bc_weight=cfg["w_bc"],objective=float(loss.detach()),
                pde_loss=float(pde.detach()),ic_loss=float(ic_l.detach()),
                bc_loss=float(bc_l.detach()),val_full_loss=np.nan))
    transition_s=time.perf_counter()-transition_start
    adaptive=None
    if method in ("rad","alhs"):
        adaptive=core.ResidualAdaptive2D(cfg["adapt_grid_side"],cfg["eta"],
                   cfg["c"],cfg["D"],cfg["adapt_eval_chunk"],device)
    joint_start=time.perf_counter()
    for e in range(1,cfg["joint_epochs"]+1):
        if (e-1)%cfg["resample_every"]==0:
            tic=time.perf_counter()
            if adaptive is not None:
                adaptive.update(model)
                sampler_updates+=1
            pts,focus_last=method_points(nf,method,cfg,init_rng,global_rng,focus_rng,
                                        adaptive,model,"joint",e,enriched)
            hash_all.update(pts.tobytes())
            collocation_sets.append(pts)
            recent.append(pts)
            current=torch.from_numpy(pts).to(device)
            sampler_seconds+=time.perf_counter()-tic
        ic,icy=core.sample_ic(cfg["n_ic"],ic_rng,device)
        bc=core.sample_bc(cfg["n_bc"],bc_rng,device)
        optimizer.zero_grad(set_to_none=True)
        loss,pde,ic_l,bc_l=core.full_losses(model,current,ic,icy,bc,cfg,cfg["w_res"])
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Nonfinite joint loss at epoch %d"%e)
        loss.backward();optimizer.step()
        inx=cfg["transition_epochs"]+e
        due=(e%cfg["validation_every"]==0 or e==cfg["joint_epochs"])
        audit=None
        if due:
            tic=time.perf_counter()
            audit=v21.validation_audit(model,val,cfg,boundary_grid,cfg["checkpoint_bc_max_tol"])
            validation_seconds+=time.perf_counter()-tic
            cid="adam_%d"%(cfg["warmup_epochs"]+inx)
            candidates.append(dict(checkpoint_id=cid,stage="adam",
                                   joint_epoch=e,main_epoch=inx,**audit))
            states[cid]=v21.model_checkpoint(model)
        if e==1 or due or e%cfg["diagnostics_every"]==0:
            hist.append(dict(phase="main",curriculum_stage="joint",
                global_epoch=cfg["warmup_epochs"]+inx,main_epoch=inx,
                pde_weight=cfg["w_res"],bc_weight=cfg["w_bc"],
                objective=float(loss.detach()),pde_loss=float(pde.detach()),
                ic_loss=float(ic_l.detach()),bc_loss=float(bc_l.detach()),
                val_full_loss=audit["val_full_loss"] if audit else np.nan))
    joint_s=time.perf_counter()-joint_start
    if not candidates:
        raise RuntimeError("No checkpoint available after full-weight joint training")
    best_adam=min(candidates,key=lambda q:q["val_full_loss"])
    model.load_state_dict(states[best_adam["checkpoint_id"]])
    # Same V2.2 final stage: fixed-data L-BFGS refinement + BC-feasible checkpoint.
    union=np.unique(np.concatenate(list(recent)),axis=0)
    lb_pts=torch.from_numpy(union).to(device)
    lb_ic,lb_y=core.sample_ic(cfg["n_ic"],np.random.default_rng(
        np.random.SeedSequence([seed,nf,995])),device)
    lb_bc=core.sample_bc(cfg["n_bc"],np.random.default_rng(
        np.random.SeedSequence([seed,nf,996])),device)
    lb_status="disabled";lb_calls=0;lb_seconds=0.0;lb_error=""
    if cfg["lbfgs_max_iter"]>0:
        lb=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=cfg["lbfgs_max_iter"],
                  history_size=50,line_search_fn="strong_wolfe")
        tic=time.perf_counter()
        def closure():
            nonlocal lb_calls
            lb_calls+=1
            lb.zero_grad(set_to_none=True)
            result=core.full_losses(model,lb_pts,lb_ic,lb_y,lb_bc,cfg,cfg["w_res"])[0]
            if not bool(torch.isfinite(result).item()):
                raise FloatingPointError("Nonfinite L-BFGS objective")
            result.backward();return result
        try:
            lb.step(closure)
            audit=v21.validation_audit(model,val,cfg,boundary_grid,cfg["checkpoint_bc_max_tol"])
            candidates.append(dict(checkpoint_id="lbfgs",stage="lbfgs",
                                   joint_epoch=cfg["joint_epochs"],
                                   main_epoch=cfg["main_epochs"],**audit))
            states["lbfgs"]=v21.model_checkpoint(model)
            lb_status="succeeded"
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            lb_error=repr(exc)[:600]
            lb_status="failed_reverted"
        lb_seconds=time.perf_counter()-tic
    best_unrestricted=min(candidates,key=lambda q:q["val_full_loss"])
    feasible=[q for q in candidates if q["bc_feasible"]]
    if checkpoint_policy=="bc_feasible" and feasible:
        chosen=min(feasible,key=lambda q:q["val_full_loss"])
        reason="minimum_full_val_among_bc_feasible"
    elif checkpoint_policy=="bc_feasible":
        chosen=best_unrestricted
        reason="no_bc_feasible_checkpoint_fallback"
    else:
        chosen=best_unrestricted
        reason="minimum_full_validation_loss"
    model.load_state_dict(states[chosen["checkpoint_id"]])
    train_sec=time.perf_counter()-t_start
    tic=time.perf_counter()
    metrics,pred=core.evaluate(model,ref,cfg,val,device)
    eval_seconds=time.perf_counter()-tic
    tic=time.perf_counter()
    diagdir.mkdir(parents=True,exist_ok=True)
    details=v21.profiles_and_boundary_details(diagdir,model,cfg,ref,pred,hist,
           pd.DataFrame(candidates),chosen["checkpoint_id"],boundary_grid,device)
    more=v22.extra_diagnostics(diagdir,model,cfg,device,
                               collocation_sets[-1],np.concatenate(collocation_sets))
    if sg_ref is not None:
        err_sg=pred-sg_ref["u"]
        sg_metrics={"pinn_vs_sg_space_time_rel_l2":float(
           np.linalg.norm(err_sg)/max(np.linalg.norm(sg_ref["u"]),1e-15)),
           "pinn_vs_sg_final_mse":float(np.mean(err_sg[-1]**2))}
        np.savez_compressed(diagdir/"pinn_vs_sg_snapshots.npz",x=sg_ref["x"],
            t=sg_ref["t"],sg=sg_ref["u"],pinn=pred,fdm=ref["u"])
    else:
        sg_metrics={}
    indep=independent_residual_audit(model,cfg,device,cfg["independent_residual_seed"])
    core.atomic_json(diagdir/"independent_residual_test.json",indep)
    torch.save(model.state_dict(),diagdir/"final_model_state.pt")
    diagnostics_seconds=time.perf_counter()-tic
    for k in ("space_time_rel_l2","final_rel_l2","dense_physics_loss","ic_mse","bc_mse"):
        if not np.isfinite(metrics[k]):
            raise FloatingPointError("Nonfinite evaluation metric: "+k)
    return dict(protocol_version=VERSION,variant=variant,method=method,
        method_label={"mc":"MC-R","lhs":"LHS-R","rad":"RAD-IID","alhs":"ALHS-R"}[method],
        seed=seed,n_collocation=nf,curriculum=cfg["curriculum"],
        warmup_epochs=cfg["warmup_epochs"],transition_epochs=cfg["transition_epochs"],
        joint_epochs=cfg["joint_epochs"],adam_total_epochs=cfg["warmup_epochs"]+cfg["main_epochs"],
        lambda_residual=cfg["w_res"],lambda_initial=cfg["w_ic"],
        lambda_boundary=cfg["w_bc"],soft_initial_active=False,
        boundary_layer_transform=use_bl,boundary_enrichment=enriched,
        n_focus_last=focus_last,number_collocation_sets=len(collocation_sets),
        number_joint_refreshes=len(collocation_sets)-1,
        adaptive_updates=sampler_updates,
        adaptive_candidate_evaluations=adaptive.eval_points if adaptive else 0,
        sampling_sha256=hash_all.hexdigest(),warmup_state_sha256=warm_hash,
        bc_max_tol=cfg["checkpoint_bc_max_tol"],
        selected_checkpoint=chosen["checkpoint_id"],
        selected_val_full_loss=chosen["val_full_loss"],
        selected_bc_max_abs=chosen["bc_max_abs"],
        selected_bc_feasible=bool(chosen["bc_feasible"]),
        checkpoint_selection_reason=reason,n_feasible_checkpoints=len(feasible),
        lbfgs_union_count=len(union),lbfgs_closure_calls=lb_calls,
        lbfgs_status=lb_status,lbfgs_error=lb_error,
        runtime_warmup_s=warm_s,runtime_transition_s=transition_s,
        runtime_joint_s=joint_s,runtime_lbfgs_s=lb_seconds,
        runtime_training_s=train_sec,runtime_sampling_s=sampler_seconds,
        runtime_validation_s=validation_seconds,
        runtime_evaluation_s=eval_seconds,runtime_diagnostics_s=diagnostics_seconds,
        independent_residual_rms=indep["rms"],
        independent_residual_p95_abs=indep["p95_abs"],
        independent_residual_p99_abs=indep["p99_abs"],
        independent_residual_max_abs=indep["max_abs"],
        **metrics,**details,**more,**sg_metrics)


def compare(root,preset,curriculum,sg_report):
    frames=[]
    for path in sorted(root.glob(f"PDE_v23_{preset}_{curriculum}_*/results_raw.csv")):
        frames.append(pd.read_csv(path))
    if not frames:return
    df=pd.concat(frames,ignore_index=True)
    core.atomic_csv(root/f"PDE_v23_{preset}_{curriculum}_comparison.csv",df)
    keys=[(r.variant,r.method,int(r.seed),int(r.n_collocation)) for r in df.itertuples(index=False)]
    if len(df)>1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,3,figsize=(15,8),layout="constrained")
        display=df.sort_values(["variant","method"]).reset_index(drop=True)
        labels=[r.variant.replace("standard_","std_").replace("enriched","focus")+"\n"+r.method
                for r in display.itertuples(index=False)]
        metrics=(("space_time_rel_l2","Space-time relative L2"),
                 ("final_rel_l2","Final-time relative L2"),
                 ("bc_right_max_abs_densetime","Right BC max abs"),
                 ("dense_physics_loss","Dense PDE loss"),
                 ("independent_residual_rms","Independent residual RMS"),
                 ("runtime_training_s","Training time (s)"))
        for ax,(col,title) in zip(axs.flat,metrics):
            y=display[col].to_numpy(dtype=float)
            ax.bar(range(len(display)),y)
            ax.set_xticks(range(len(display)),labels,rotation=50,ha="right",fontsize=7)
            if col in ("dense_physics_loss","independent_residual_rms") and np.all(y>0):
                ax.set_yscale("log")
            ax.set_title(title);ax.grid(alpha=.2,axis="y")
        fig.suptitle("V2.3 diagnostics: single-seed, not statistical evidence")
        fig.savefig(root/f"PDE_v23_{preset}_{curriculum}_comparison.png",dpi=145)
        plt.close(fig)
    return df,keys



# ---------------------------------------------------------------------------
# V2.3.1: independent numerical validation, never used for model selection.
# ---------------------------------------------------------------------------

OUTFLOW_X_BANDS = ((0.9800, 0.9900), (0.9900, 0.9990),
                   (0.9990, 0.9999), (0.9999, 1.0))


def valid_seed_list(text: str) -> list[int]:
    values = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, stop = part.split('-', 1)
            a, b = int(start), int(stop)
            if not (0 <= a <= b <= 100000):
                raise ValueError('Invalid seed range: ' + part)
            values.extend(range(a, b + 1))
        else:
            values.append(int(part))
    if not values or any(v < 0 or v > 100000 for v in values):
        raise ValueError('Seed list must be nonempty and nonnegative')
    if len(values) != len(set(values)):
        raise ValueError('Duplicate seed in --seeds')
    return values


def validation_config(preset: str) -> dict:
    if preset not in ('smoke', 'validation'):
        raise ValueError('Unknown preset')
    cfg = make_config('smoke' if preset == 'smoke' else 'diagnostic', 'v31_staged')
    cfg['purpose'] = 'V2.3.1 five-seed validation, no method hyperparameter tuning'
    cfg['n_collocation'] = [48 if preset == 'smoke' else 500]
    cfg['seeds'] = [0] if preset == 'smoke' else list(range(5))
    return cfg


def source_hashes() -> dict:
    files = {
        'v231': Path(__file__).resolve(),
        'v22': Path(v22.__file__).resolve(),
        'v21': Path(v21.__file__).resolve(),
        'v2': Path(core.__file__).resolve(),
    }
    # Explicitly keep the underlying V2.3 source in the package and manifest.
    import convection_diffusion_benchmark_v23 as original
    files['v23_base'] = Path(original.__file__).resolve()
    return {name: v21.sha256_file(path) for name, path in files.items()}


def load_frozen_model(cfg: dict, variant: str, device: torch.device,
                      model_file: Path) -> v22.BoundaryLayerPINN:
    if not model_file.exists():
        raise FileNotFoundError('Missing selected-model checkpoint: ' + str(model_file))
    use_bl, _ = v22.variant_flags(variant)
    delta = cfg['D'] / abs(cfg['c']) * cfg['bl_delta_factor']
    model = v22.BoundaryLayerPINN(cfg['layers'], use_bl, delta).to(device)
    try:
        state = torch.load(model_file, map_location=device, weights_only=True)
    except TypeError:  # Older PyTorch environments, not target environment.
        state = torch.load(model_file, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _area_rms(arr: np.ndarray, x: np.ndarray, times: np.ndarray) -> float:
    if len(x) < 2 or len(times) < 2:
        return float('nan')
    area = (float(x[-1])-float(x[0])) * (float(times[-1])-float(times[0]))
    if area <= 0:
        return float('nan')
    area_integral = np.trapz(np.trapz(np.asarray(arr)**2, x=x, axis=1),
                             x=times, axis=0)
    return float(np.sqrt(max(float(area_integral) / area, 0.0)))


def dense_outflow_residual_audit(model, cfg, device, target: Path,
                                  x_points: int = 181, t_points: int = 61) -> dict:
    """Post-training audit on a deterministic, log-spaced x near 1 grid.

    A separate metric is area-weighted, because unweighted mean on a log mesh
    overrepresents the thinnest spatial bands.  This audit NEVER picks a model.
    t=0 is omitted from primary summary to avoid conflating the IC/BC corner
    compatibility singularity with interior PDE-residual generalization.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target.mkdir(parents=True, exist_ok=True)
    if x_points < 70 or t_points < 10:
        raise ValueError('Residual audit grids must be sufficiently resolved')
    distances = np.geomspace(1e-5, 2e-2, x_points)
    x = np.sort(1.0 - distances).astype(np.float64)
    t = np.linspace(0.01, 1.0, t_points, dtype=np.float64)
    xx, tt = np.meshgrid(x, t)
    xy = np.column_stack((xx.ravel(), tt.ravel()))
    rr = core.residual_numpy(model, xy, cfg, device).reshape(len(t), len(x))
    if not np.isfinite(rr).all():
        raise FloatingPointError('Nonfinite residual in independent outflow audit')
    np.savez_compressed(target / 'dense_outflow_residual.npz',
                        x=x, t=t, residual=rr,
                        c=float(cfg['c']), D=float(cfg['D']),
                        zero_time_excluded=True)
    rows = []
    for left, right in OUTFLOW_X_BANDS:
        mask = (x >= left) & (x < right)
        if right == 1.0:
            mask = (x >= left) & (x <= right)
        selected = rr[:, mask]
        selected_x = x[mask]
        if selected_x.size < 2:
            raise RuntimeError(f'Residual grid too sparse in band {left}..{right}')
        arg = int(np.argmax(np.abs(selected)))
        it, ix = np.unravel_index(arg, selected.shape)
        rows.append({
            'x_lo': left, 'x_hi': right, 'n_x': len(selected_x),
            'n_t': len(t), 't_min': float(t[0]), 't_max': float(t[-1]),
            'rms_on_log_grid': float(np.sqrt(np.mean(selected**2))),
            'rms_area_weighted': _area_rms(selected, selected_x, t),
            'p95_abs_log_grid': float(np.quantile(np.abs(selected), 0.95)),
            'max_abs': float(np.max(np.abs(selected))),
            'x_at_max': float(selected_x[ix]), 't_at_max': float(t[it]),
        })
    band_df = pd.DataFrame(rows)
    core.atomic_csv(target / 'dense_outflow_residual_bands.csv', band_df)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.7), layout='constrained')
    abs_r = np.maximum(np.abs(rr), 1e-8)
    mesh = axs[0].pcolormesh(np.log10(1.0 - x), t, np.log10(abs_r),
                            shading='auto', cmap='viridis')
    axs[0].set_xlabel('log10(1-x)'); axs[0].set_ylabel('t')
    axs[0].set_title('log10 absolute PDE residual near right boundary')
    fig.colorbar(mesh, ax=axs[0], label='log10 |residual|')
    for i, ts in enumerate((0.05, 0.5, 1.0)):
        j = int(np.argmin(np.abs(t - ts)))
        axs[1].plot(1.0-x, abs_r[j], label=f't={t[j]:.3f}')
    axs[1].set_xscale('log'); axs[1].set_yscale('log')
    axs[1].set_xlabel('Distance from outflow, 1-x'); axs[1].set_ylabel('|PDE residual|')
    axs[1].set_title('Residual versus distance from outflow')
    axs[1].grid(alpha=.2); axs[1].legend()
    fig.savefig(target / 'dense_outflow_residual.png', dpi=155)
    plt.close(fig)
    worst = max(rows, key=lambda a: a['max_abs'])
    summary = dict(
        audit='independent_dense_log_distance_outflow',
        grid_x_points=int(x_points), grid_t_points=int(t_points),
        x_range=[float(x[0]), float(x[-1])], t_range=[float(t[0]),float(t[-1])],
        residual_global_rms_on_log_grid=float(np.sqrt(np.mean(rr**2))),
        residual_global_rms_area_weighted=_area_rms(rr,x,t),
        residual_max_abs=float(np.max(np.abs(rr))),
        residual_x_at_max=float(worst['x_at_max']),
        residual_t_at_max=float(worst['t_at_max']),
        residual_rms_area_999=float([r for r in rows if r['x_lo']==.999][0]['rms_area_weighted']),
        residual_rms_area_9999=float([r for r in rows if r['x_lo']==.9999][0]['rms_area_weighted']),
        corner_t0_excluded=True,
        training_or_checkpoint_use=False,
        interpretation='Integral RMS uses physical x,t weights, unlike log-grid pointwise RMS.',
        warning='Residual size alone does not bound solution error in this near-boundary IVP.')
    core.atomic_json(target / 'dense_outflow_residual_summary.json', summary)
    return summary


def reference_grid_levels(profile: str) -> list[tuple[int, int]]:
    if profile == 'smoke':
        return [(101, 201), (201, 401), (401, 801)]
    if profile == 'full':
        return [(1001, 25001), (2001, 50001), (4001, 100001)]
    raise ValueError('Unrecognized reference grid profile: '+profile)


def _metrics_for_pair(coarse, fine, left_name, right_name) -> dict:
    """Compare two nested grids at identical physical x and snapshot times."""
    x1 = np.asarray(coarse['x']); x2 = np.asarray(fine['x'])
    t1 = np.asarray(coarse['t']); t2 = np.asarray(fine['t'])
    if len(x2) != 2*(len(x1)-1)+1 or not np.allclose(x1,x2[::2],atol=2e-14,rtol=0):
        raise RuntimeError('Spatial grids are not nested 2-to-1')
    if not np.allclose(t1,t2,atol=2e-14,rtol=0):
        raise RuntimeError('Snapshot times do not match')
    ua=np.asarray(coarse['u'],dtype=np.float64)
    ub=np.asarray(fine['u'],dtype=np.float64)[:,::2]
    diff=ua-ub
    results={
      'coarse':left_name,'fine':right_name,
      'coarse_nx':len(x1),'fine_nx':len(x2),
      'coarse_dx':float(coarse['dx']), 'fine_dx':float(fine['dx']),
      'rel_l2_all_snapshots':float(np.linalg.norm(diff)/max(np.linalg.norm(ub),1e-15)),
      'abs_rmse_all_snapshots':float(np.sqrt(np.mean(diff**2))),
      'max_abs_all_snapshots':float(np.max(np.abs(diff))),
      'final_rmse':float(np.sqrt(np.mean(diff[-1]**2))),
      'final_relative_l2':float(np.linalg.norm(diff[-1])/max(np.linalg.norm(ub[-1]),1e-15)),
    }
    for edge,label in ((.99,'099'),(.999,'0999')):
        mask=x1>=edge
        results['n_coarse_x_ge_'+label]=int(mask.sum())
        results['rmse_x_ge_'+label]=float(np.sqrt(np.mean(diff[:,mask]**2)))
        results['max_abs_x_ge_'+label]=float(np.max(np.abs(diff[:,mask])))
    return results


def same_grid_agreement(a,b,nx,nt):
    if not np.allclose(a['x'],b['x']) or not np.allclose(a['t'],b['t']):
        raise RuntimeError('FDM and SG grid mismatch')
    diff=np.asarray(a['u'])-np.asarray(b['u'])
    x=np.asarray(a['x'])
    return dict(nx=nx,nt=nt,dx=float(a['dx']),dt=float(a['dt']),
        cell_peclet=float(a['cell_peclet']),
        rel_l2_fdm_vs_sg=float(np.linalg.norm(diff)/max(np.linalg.norm(a['u']),1e-15)),
        global_rmse=float(np.sqrt(np.mean(diff**2))),
        max_abs=float(np.max(np.abs(diff))),
        outflow_099_rmse=float(np.sqrt(np.mean(diff[:,x>=.99]**2))),
        outflow_0999_rmse=float(np.sqrt(np.mean(diff[:,x>=.999]**2))),
        solution_min_fdm=float(np.min(a['u'])), solution_max_fdm=float(np.max(a['u'])),
        solution_min_sg=float(np.min(b['u'])), solution_max_sg=float(np.max(b['u'])))


def relative_error_vs_reference(model,ref,cfg,device) -> dict:
    x=np.asarray(ref['x']); ts=np.asarray(ref['t']); true=np.asarray(ref['u'])
    pred=[]
    for t in ts:
        coords=np.column_stack((x,np.full_like(x,t)))
        pred.append(core.predict_numpy(model,coords,device,cfg['evaluation_chunk']))
    pred=np.asarray(pred)
    diff=pred-true
    results=dict(
        relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(true),1e-15)),
        final_relative_l2=float(np.linalg.norm(diff[-1])/max(np.linalg.norm(true[-1]),1e-15)),
        final_mse=float(np.mean(diff[-1]**2)),
        global_rmse=float(np.sqrt(np.mean(diff**2))),
        outflow_099_rmse=float(np.sqrt(np.mean(diff[:,x>=.99]**2))),
        outflow_0999_rmse=float(np.sqrt(np.mean(diff[:,x>=.999]**2))),
        outflow_0999_nx=int(np.sum(x>=.999)))
    return results


def _snapshot_summary(root: Path, df: pd.DataFrame, cfg: dict, expected: list) -> None:
    """Descriptive only: n=5 is not a robust basis for ranking strategies."""
    if df.empty:
        return
    metrics=('space_time_rel_l2','final_rel_l2','dense_physics_loss',
             'bc_max_abs_error','selected_bc_max_abs','residual_rms_area_999',
             'residual_rms_area_9999','runtime_training_s')
    rows=[]
    for (method, variant), part in df.groupby(['method','variant'],sort=False):
        row={'method':method,'variant':variant,'n_completed':len(part),
             'n_bc_feasible':int(part['selected_bc_feasible'].astype(bool).sum()),
             'n_both_1pct_and_bc_feasible':int((
                (part['space_time_rel_l2']<=.01)&part['selected_bc_feasible'].astype(bool)).sum()),
             'threshold_note':'Exploratory <=1% space-time relative L2, BC <=0.02; not publication acceptance criterion'}
        for col in metrics:
            z=part[col].astype(float).to_numpy()
            row[col+'_median']=float(np.median(z))
            row[col+'_q25']=float(np.quantile(z,.25))
            row[col+'_q75']=float(np.quantile(z,.75))
            row[col+'_min']=float(np.min(z))
            row[col+'_max']=float(np.max(z))
        rows.append(row)
    core.atomic_csv(root/'seed_summary.csv',pd.DataFrame(rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for (method, variant),part in df.groupby(['method','variant'],sort=False):
        label=f'{variant}/{method}'
        s=part.sort_values('seed')
        for ax,col in zip(axs,('space_time_rel_l2','final_rel_l2','residual_rms_area_999')):
            ax.plot(s['seed'],s[col],marker='o',label=label)
    for ax,col,title in zip(axs,
        ('space_time_rel_l2','final_rel_l2','residual_rms_area_999'),
        ('Space-time relative L2','Final-time relative L2','Area-weighted residual RMS, x>=.999')):
        ax.set_title(title);ax.set_xlabel('Seed')
        ax.set_yscale('log');ax.grid(alpha=.25)
    axs[0].legend(fontsize=8)
    fig.suptitle('V2.3.1 validation across seeds (descriptive, no significance testing)')
    fig.savefig(root/'seed_summary.png',dpi=155)
    plt.close(fig)
    keys={(str(r.variant),str(r.method),int(r.seed)) for r in df.itertuples(index=False)}
    core.atomic_json(root/'integrity.json',{
        'expected_runs':len(expected),'unique_keys':len(keys),'total_rows':len(df),
        'duplicate_keys':len(df)-len(keys),'missing':sorted([list(x) for x in set(expected)-keys]),
        'extra':sorted([list(x) for x in keys-set(expected)]),
        'complete':set(expected).issubset(keys) and len(df)==len(keys),
        'complete_for_requested_keys':set(expected).issubset(keys) and len(df)==len(keys),
        'five_seed_panels_complete':{str(method):all((variant,method,i) in keys for i in range(5)) for variant,method,_ in keys},
        'seed_count_requested':len(set(seed for _,_,seed in expected)),
        'note':'No causal sampling comparison or significance test is made from five seeds.'})


def run_validation(args):
    cfg=validation_config(args.preset)
    device=core.device_from_arg(args.device)
    root=args.output_root.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    dest=root/args.preset
    dest.mkdir(parents=True,exist_ok=True)
    ref_root=root/'references'
    ref_root.mkdir(parents=True,exist_ok=True)
    methods=METHODS if args.methods=='all' else tuple(k.strip() for k in args.methods.split(',') if k.strip())
    if not methods or len(set(methods))!=len(methods) or not set(methods)<=set(METHODS):
        raise ValueError('Invalid --methods, expected mc,lhs,rad,alhs or all')
    variant=args.variant
    if variant not in VARIANTS:
        raise ValueError('Invalid variant')
    seeds=valid_seed_list(args.seeds if args.seeds else ('0' if args.preset=='smoke' else '0-4'))
    config={
        'v231_version':VERSION,'base_model_training':'V2.3 copied unchanged except explicit seed argument',
        'cfg':cfg,'device':str(device),'runtime':core.runtime_info(device),
        'checkpoint_policy':'bc_feasible','variant':variant,
        'source_sha256':source_hashes(),
        'audit':{'x_points':args.audit_nx,'t_points':args.audit_nt,'t_min':0.01,
                 'x_min':0.98,'x_max':.99999,'training_or_checkpoint_use':False},
        'no_fdm_sg_in_checkpoint':True,
        'warning':'Seed-0 reproduction and 5-seed evaluation, not a 4-method performance benchmark.'
    }
    manifest_file=dest/'manifest.json'
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding='utf-8'))!=config:
            raise RuntimeError('Existing V2.3.1 validation manifest differs: use a NEW output root')
    else:
        core.atomic_json(manifest_file,config)
    print('V2.3.1',VERSION,'preset',args.preset,'variant',variant,'methods',methods,'seeds',seeds,flush=True)
    print('Device',device,'Torch',torch.__version__,'reference and SG:',flush=True)
    ref,_=core.get_reference(cfg,ref_root)
    sg,sg_report,_=secondary_reference(cfg,ref,ref_root)
    print('SG-FDM reference difference:',sg_report['relative_l2_sg_minus_fdm'],flush=True)
    # The sampler preflight is unchanged from V2.3 and is *not* training.
    core.sampling_preflight(cfg,device,dest/'sampling_preflight.json')
    for method in methods:
        for seed in seeds:
            run_path=dest/'runs'/f'{variant}_{method}_seed{seed:03d}'
            run_path.mkdir(parents=True,exist_ok=True)
            diagnostics=run_path/'diagnostics'/f'{method}_nf{cfg["n_collocation"][0]}_seed{seed}'
            row_path=run_path/'results_raw.csv'
            if row_path.exists():
                old=pd.read_csv(row_path)
                marker=diagnostics/'dense_outflow_residual_summary.json'
                if len(old)!=1 or old.iloc[0]['method']!=method or int(old.iloc[0]['seed'])!=seed:
                    raise RuntimeError('Invalid existing result at '+str(row_path))
                if not marker.exists() or not (diagnostics/'final_model_state.pt').exists():
                    raise RuntimeError('Incomplete existing run (diagnostics missing); inspect before resume: '+str(run_path))
                print('SKIP completed:',method,'seed',seed,flush=True)
                continue
            print('RUN:',method,'seed',seed,flush=True)
            # V2.3 training + identical checkpoint selection; seed passed explicitly.
            started=time.perf_counter()
            row=run_one(variant,method,cfg,ref,sg,device,diagnostics,'bc_feasible',seed=seed)
            trained_file=diagnostics/'final_model_state.pt'
            model=load_frozen_model(cfg,variant,device,trained_file)
            new=dense_outflow_residual_audit(model,cfg,device,diagnostics,
                 x_points=args.audit_nx,t_points=args.audit_nt)
            row.update({
                 'validation_version':VERSION,'run_preset':args.preset,
                 'audit_x_points':args.audit_nx,'audit_t_points':args.audit_nt,
                 'residual_rms_area_999':new['residual_rms_area_999'],
                 'residual_rms_area_9999':new['residual_rms_area_9999'],
                 'residual_rms_full_logmesh':new['residual_global_rms_on_log_grid'],
                 'residual_rms_full_area':new['residual_global_rms_area_weighted'],
                 'runtime_posttraining_total_s':time.perf_counter()-started-row['runtime_training_s'],
                 'validation_fdm_sg_used_for_training':False})
            core.atomic_csv(row_path,pd.DataFrame([row]))
            print('DONE:',method,seed,'error',row['space_time_rel_l2'],
                'BC feasible',row['selected_bc_feasible'],'elapsed',round(time.perf_counter()-started,1),flush=True)
    files=sorted((dest/'runs').glob('*/results_raw.csv'))
    df=pd.concat([pd.read_csv(file) for file in files],ignore_index=True) if files else pd.DataFrame()
    if not df.empty:
        # Disallow collecting extra keys from a different method/variant by accident.
        if not (df['variant']==variant).all():
            raise RuntimeError('Collected results do not match the target methods and variant')
        df=df.sort_values(['method','seed']).reset_index(drop=True)
        core.atomic_csv(dest/'results_combined.csv',df)
    expected=[(variant,method,seed) for method in methods for seed in seeds]
    _snapshot_summary(dest,df,cfg,expected)
    print('V2.3.1',args.preset,'finished:',len(df),'runs; see',dest/'integrity.json',flush=True)


def run_reference_grid(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    device=core.device_from_arg(args.device)
    root=args.output_root.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    report_dir=root/('grid_convergence_'+args.grid_profile)
    report_dir.mkdir(exist_ok=True)
    ref_cache=root/'references'
    ref_cache.mkdir(exist_ok=True)
    base=make_config('diagnostic','v31_staged')
    levels=reference_grid_levels(args.grid_profile)
    cfg_meta={k:base[k] for k in ('c','D','x_range','t_range','reference_snapshots')}
    manifest={'version':VERSION,'grid_profile':args.grid_profile,'physics':cfg_meta,
              'levels':levels,'solvers':['BE-centered-FDM','BE-Scharfetter-Gummel'],
              'source_sha256':source_hashes(),
              'fixed_snapshot_times':True,
              'warning':'Convergence differences approximate discretization errors but are not exact-solution errors.'}
    manifest_file=report_dir/'manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text(encoding='utf-8'))!=manifest:
        raise RuntimeError('Grid study configuration/code changed. Use a NEW output root.')
    if not manifest_file.exists():
        core.atomic_json(manifest_file,manifest)
    refs=[]; agreement=[]
    for nx,nt in levels:
        cfg=dict(base,reference_nx=nx,reference_nt=nt)
        print('GRID:',nx,'x',nt,flush=True)
        fdm,_=core.get_reference(cfg,ref_cache)
        sg,sg_rep,_=secondary_reference(cfg,fdm,ref_cache)
        refs.append((cfg,fdm,sg))
        agreement.append(same_grid_agreement(fdm,sg,nx,nt))
    core.atomic_csv(report_dir/'solver_agreement.csv',pd.DataFrame(agreement))
    comparisons=[]
    for solver in ('fdm','sg'):
        for i in range(len(refs)-1):
            a=refs[i][1 if solver=='fdm' else 2]
            b=refs[i+1][1 if solver=='fdm' else 2]
            row=_metrics_for_pair(a,b,
                f'{solver}_{levels[i][0]}_{levels[i][1]}',
                f'{solver}_{levels[i+1][0]}_{levels[i+1][1]}')
            row['solver']=solver;row['level_from']=i;row['level_to']=i+1
            comparisons.append(row)
    compare=pd.DataFrame(comparisons)
    for solver in ('fdm','sg'):
        index=compare.index[compare.solver==solver].tolist()
        if len(index)==2:
            a=float(compare.at[index[0],'rel_l2_all_snapshots'])
            b=float(compare.at[index[1],'rel_l2_all_snapshots'])
            order=float(np.log2(a/b)) if a>0 and b>0 else None
            compare.loc[index,'observed_refinement_order_from_pair_norms']=order
    core.atomic_csv(report_dir/'grid_refinement.csv',compare)
    charts=plt.figure(figsize=(12,4.8),layout='constrained')
    axs=charts.subplots(1,2)
    for solver in ('fdm','sg'):
        sub=compare[compare.solver==solver].sort_values('fine_dx',ascending=False)
        axs[0].loglog(sub.fine_dx,sub.rel_l2_all_snapshots,marker='o',label=solver)
        axs[1].loglog(sub.fine_dx,sub.rmse_x_ge_099,marker='o',label=solver)
    axs[0].set_title('Change under 2x refinement (shared coarse grid)')
    axs[1].set_title('Outflow x >= 0.99, mesh-to-mesh RMSE')
    for ax in axs:
        ax.invert_xaxis();ax.set_xlabel('Fine spatial dx');ax.grid(alpha=.25);ax.legend()
    axs[0].set_ylabel('Relative L2 change');axs[1].set_ylabel('RMSE')
    charts.savefig(report_dir/'grid_convergence.png',dpi=155)
    plt.close(charts)
    # Strictly post-hoc: reevaluate previously selected PINN checkpoints without retraining.
    model_dir=root/'validation'/'runs'
    models=list(model_dir.glob('*/results_raw.csv')) if model_dir.is_dir() else []
    reeval=[]
    if models:
        print('Re-evaluating',len(models),'frozen PINN checkpoints on three reference grids',flush=True)
        for file in models:
            stored=pd.read_csv(file)
            if len(stored)!=1:
                raise RuntimeError('Unexpected run format in '+str(file))
            meta=stored.iloc[0]
            variant=str(meta['variant']);method=str(meta['method']);seed=int(meta['seed'])
            diag= file.parent/'diagnostics'/f'{method}_nf{int(meta["n_collocation"])}_seed{seed}'
            model=load_frozen_model(base,variant,device,diag/'final_model_state.pt')
            for cfg,fdm,sg in refs:
                for ref_name,ref in [('FDM',fdm),('SG',sg)]:
                    extra=relative_error_vs_reference(model,ref,cfg,device)
                    reeval.append({'variant':variant,'method':method,'seed':seed,
                                   'ref':ref_name,'reference_nx':int(cfg['reference_nx']),
                                   'reference_nt':int(cfg['reference_nt']),
                                   'trained_checkpoint':str(meta['selected_checkpoint']),
                                   'checkpoint_frozen':True,'use_for_selection':False,**extra})
        core.atomic_csv(report_dir/'frozen_models_vs_grid_references.csv',pd.DataFrame(reeval))
    summary={'grid_profile':args.grid_profile,'grids':levels,
             'solver_agreement':agreement,
             'refinement_comparison_rows':comparisons,
             're_evaluated_frozen_models':len(models),
             'warning':'Do not select training configuration using finest-grid reference. Results are post-hoc validation only.'}
    core.atomic_json(report_dir/'grid_summary.json',summary)
    print('Grid convergence outputs:',report_dir,flush=True)


def main():
    parser=argparse.ArgumentParser(description='V2.3.1 independent multi-seed and reference-grid validation')
    parser.add_argument('--preset',choices=['smoke','validation','grid'],default='smoke')
    parser.add_argument('--variant',choices=VARIANTS,default='bl_uniform')
    parser.add_argument('--methods',default='mc',help='mc or comma list mc,lhs,rad,alhs, or all')
    parser.add_argument('--seeds',default=None,help='E.g. 0-4 (validation default) or 0,2,4')
    parser.add_argument('--grid-profile',choices=['smoke','full'],default='smoke')
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--output-root',type=Path,default=Path('pde_v231_outputs'))
    parser.add_argument('--audit-nx',type=int,default=181)
    parser.add_argument('--audit-nt',type=int,default=61)
    args=parser.parse_args()
    if args.preset=='grid':
        run_reference_grid(args)
    else:
        run_validation(args)

if __name__=='__main__':
    main()
