#!/usr/bin/env python3
"""PINN convection-diffusion V2.3: V2.2 boundary-layer diagnostics + staged
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

VERSION = "2.3.0"
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


def run_one(variant,method,cfg,ref,sg_ref,device,diagdir,checkpoint_policy):
    nf=int(cfg["n_collocation"][0]);seed=0
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset",choices=["smoke","diagnostic"],default="smoke")
    parser.add_argument("--variants",default="all")
    parser.add_argument("--samplers",default="mc",
        help="mc (default) or comma list, e.g. mc,lhs,rad,alhs, or all")
    parser.add_argument("--curriculum",choices=["v31_staged","v22_matched_5000"],default="v31_staged")
    parser.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
    parser.add_argument("--output-root",type=Path,default=Path("pde_v23_outputs"))
    parser.add_argument("--checkpoint-policy",choices=["bc_feasible","val_only"],default="bc_feasible")
    parser.add_argument("--bc-max-tol",type=float,default=.02)
    parser.add_argument("--D",type=float,default=.001)
    parser.add_argument("--bc-weight",type=float,default=None)
    parser.add_argument("--resample-every",type=int,default=None)
    parser.add_argument("--enrichment-fraction",type=float,default=.25)
    parser.add_argument("--warmup-epochs",type=int,default=None)
    parser.add_argument("--transition-epochs",type=int,default=None)
    parser.add_argument("--joint-epochs",type=int,default=None)
    parser.add_argument("--lbfgs-max-iter",type=int,default=None)
    parser.add_argument("--sg-skip",action="store_true",help="Skip secondary SG computation (testing only)")
    args=parser.parse_args()
    if not (core.VERSION=="2.0.0" and v21.VERSION=="2.1.0" and v22.VERSION=="2.2.0"):
        raise RuntimeError("V2.3 requires unchanged V2.0/V2.1/V2.2 modules")
    variants=VARIANTS if args.variants=="all" else tuple(x.strip() for x in args.variants.split(",") if x.strip())
    methods=METHODS if args.samplers=="all" else tuple(x.strip() for x in args.samplers.split(",") if x.strip())
    if not variants or len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS):
        parser.error("Invalid --variants")
    if not methods or len(set(methods))!=len(methods) or not set(methods)<=set(METHODS):
        parser.error("Invalid --samplers")
    cfg=make_config(args.preset,args.curriculum)
    cfg["D"]=args.D
    cfg["checkpoint_bc_max_tol"]=args.bc_max_tol
    cfg["enrichment_fraction"]=args.enrichment_fraction
    if args.bc_weight is not None: cfg["w_bc"]=args.bc_weight
    if args.resample_every is not None:cfg["resample_every"]=args.resample_every
    for name,val in (("warmup_epochs",args.warmup_epochs),
                     ("transition_epochs",args.transition_epochs),
                     ("joint_epochs",args.joint_epochs),
                     ("lbfgs_max_iter",args.lbfgs_max_iter)):
        if val is not None:cfg[name]=val
    cfg["main_epochs"]=int(cfg["transition_epochs"]+cfg["joint_epochs"])
    cfg["physics_ramp_epochs"]=int(cfg["transition_epochs"])
    if not (cfg["warmup_epochs"]>=0 and cfg["transition_epochs"]>=1 and cfg["joint_epochs"]>=1):
        parser.error("Invalid stage epoch counts")
    core.validate_config(cfg)
    if not (cfg["D"]>0 and cfg["c"]>0):parser.error("This BL test requires c>0 and D>0")
    if not (cfg["resample_every"]>0 and 0<cfg["enrichment_fraction"]<1):
        parser.error("Invalid resampling / enrichment settings")
    if cfg["D"]/cfg["c"]*cfg["bl_delta_factor"] >= .5:
        parser.error("D/c transform scale must be smaller than 0.5")
    device=core.device_from_arg(args.device)
    root=args.output_root.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    codes={"v23":v21.sha256_file(Path(__file__)),
           "v22":v21.sha256_file(Path(v22.__file__)),
           "v21":v21.sha256_file(Path(v21.__file__)),
           "v2":v21.sha256_file(Path(core.__file__))}
    print("V2.3",VERSION,"device",device,"Torch",torch.__version__,flush=True)
    print("Curriculum",cfg["curriculum"],"warmup/transition/joint",
          cfg["warmup_epochs"],cfg["transition_epochs"],cfg["joint_epochs"],
          "weights",cfg["w_res"],cfg["w_ic"],cfg["w_bc"],flush=True)
    ref,signature=core.get_reference(cfg,root)
    if args.sg_skip:
        sg=None;sg_report={"skipped":True};sg_sig="none"
    else:
        sg,sg_report,sg_sig=secondary_reference(cfg,ref,root)
        print("SG/FDM relative discrepancy",sg_report["relative_l2_sg_minus_fdm"],flush=True)
    core.sampling_preflight(cfg,device,root/"sampling_preflight.json")
    for variant in variants:
        for method in methods:
            name=f"PDE_v23_{args.preset}_{cfg['curriculum']}_{variant}_{method}"
            folder=root/name;folder.mkdir(parents=True,exist_ok=True)
            manifest=dict(version=VERSION,preset=args.preset,variant=variant,
                method=method,cfg=cfg,checkpoint_policy=args.checkpoint_policy,
                software=core.runtime_info(device),script_hashes=codes,
                reference_signature=signature,sg_secondary_signature=sg_sig,
                warning="This is a diagnostic, not a replicated 30-seed benchmark")
            mf=folder/"manifest.json"
            if mf.exists():
                if json.loads(mf.read_text(encoding="utf-8"))!=manifest:
                    raise RuntimeError("Existing output manifest conflicts; use NEW --output-root: "+str(folder))
            else:core.atomic_json(mf,manifest)
            raw=folder/"results_raw.csv"
            if raw.exists():
                old=pd.read_csv(raw)
                if len(old)!=1 or str(old.iloc[0].variant)!=variant or str(old.iloc[0].method)!=method:
                    raise RuntimeError("Existing completed run has incorrect key")
                print("SKIP completed",variant,method,flush=True)
                continue
            diagdir=folder/"diagnostics"/f"{method}_nf{cfg['n_collocation'][0]}_seed0"
            print("RUN",variant,method,flush=True)
            row=run_one(variant,method,cfg,ref,sg,device,diagdir,args.checkpoint_policy)
            core.atomic_csv(raw,pd.DataFrame([row]))
            print("DONE",variant,method,"relL2",row["space_time_rel_l2"],
                  "final",row["final_rel_l2"],"BC feasible",row["selected_bc_feasible"],flush=True)
    data=compare(root,args.preset,cfg["curriculum"],sg_report)
    if data:
        df,keys=data
        expected={(v,m,0,int(cfg["n_collocation"][0])) for v in variants for m in methods}
        core.atomic_json(root/f"PDE_v23_{args.preset}_{cfg['curriculum']}_integrity.json",
            {"expected_runs":len(expected),"found_unique":len(set(keys)),
             "duplicates":len(keys)-len(set(keys)),"missing":sorted([list(x) for x in expected-set(keys)]),
             "complete":set(keys)==expected and len(keys)==len(set(keys)),
             "sg_reference":sg_report,"comparison_file":str(root/f"PDE_v23_{args.preset}_{cfg['curriculum']}_comparison.csv")})
    print("V2.3 processing complete",flush=True)


if __name__=="__main__":
    main()
