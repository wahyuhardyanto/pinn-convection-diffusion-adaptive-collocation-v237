#!/usr/bin/env python3
"""PINN convection-diffusion benchmark V2: IC/BC-enforcement diagnostics.

Preserves the PDE, FDM reference and four samplers of benchmark V1, but adds:
  - soft_warmup:          unconstrained network, IC+BC pretraining;
  - hard_ic_soft_bc:      u=sin(pi*x)+t*N(x,t), BC pretraining;
  - hard_bc_soft_ic:      u=x*(1-x)*N(x,t), IC pretraining;
  - a smooth residual-weight ramp in the main Adam stage;
  - profiles, full-domain and near-outflow residual heatmaps, and training traces.

First run the compare-mode smoke test, then the 3-mode diagnostic on MC-R,
Nf=500, seed=0. Do NOT treat a diagnostic result as a publication benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.stats import friedmanchisquare, qmc, rankdata, wilcoxon
import torch
from torch import nn

VERSION = "2.0.0"
METHODS = ("mc", "lhs", "rad", "alhs")
METHOD_LABELS = {"mc": "MC-R", "lhs": "LHS-R", "rad": "RAD-IID", "alhs": "ALHS-R"}
MODES = ("soft_warmup", "hard_ic_soft_bc", "hard_bc_soft_ic")
MODE_DESCRIPTION = {
    "soft_warmup": "Soft initial and boundary conditions with IC/BC warmup",
    "hard_ic_soft_bc": "Exact initial condition with soft boundary conditions",
    "hard_bc_soft_ic": "Exact boundary conditions with soft initial condition",
}


def preset_config(preset: str) -> dict:
    cfg = dict(
        c=1.0, D=0.001, x_range=[0.0, 1.0], t_range=[0.0, 1.0],
        layers=[2, 64, 64, 64, 1],
        w_res=50.0, w_ic=1.0, w_bc=1.0,
        n_ic=500, n_bc=500,
        warmup_epochs=1000,
        main_epochs=5000,
        physics_ramp_epochs=500,
        adam_lr=1e-3,
        resample_every=250, validation_every=250,
        val_grid_side=31, lbfgs_max_iter=300,
        lbfgs_union_sets=4,
        adapt_grid_side=51, adapt_eval_chunk=512, eta=0.10,
        reference_nx=2001, reference_nt=50001, reference_snapshots=21,
        n_collocation=[250, 500, 1000], seeds=list(range(30)), methods=list(METHODS),
        diagnostics_every=100,
        evaluation_chunk=1024,
        pde="u_t+c*u_x-D*u_xx=0",
    )
    if preset == "smoke":
        cfg.update(
            warmup_epochs=12, main_epochs=25, physics_ramp_epochs=10,
            resample_every=5, validation_every=5,
            n_collocation=[48], seeds=[0], methods=["mc"],
            n_ic=32, n_bc=32, lbfgs_max_iter=3,
            val_grid_side=11, adapt_grid_side=15,
            reference_nx=201, reference_nt=801, reference_snapshots=11,
            diagnostics_every=5,
        )
    elif preset == "diagnostic":
        cfg.update(n_collocation=[500], seeds=[0], methods=["mc"])
    elif preset == "pilot":
        cfg.update(seeds=[0])
    elif preset != "full":
        raise ValueError("Unknown preset: " + str(preset))
    return cfg


def validate_config(cfg: dict) -> None:
    if not (0 < cfg["eta"] < 1):
        raise ValueError("eta must lie strictly between 0 and 1")
    if cfg["main_epochs"] <= 0 or cfg["warmup_epochs"] < 0:
        raise ValueError("Main epochs must be positive, warmup epochs nonnegative")
    if not (1 <= cfg["physics_ramp_epochs"] <= cfg["main_epochs"]):
        raise ValueError("physics_ramp_epochs must be between 1 and main_epochs")
    if cfg["resample_every"] <= 0 or cfg["validation_every"] <= 0:
        raise ValueError("resample_every and validation_every must be positive")
    if cfg["n_ic"] <= 0 or cfg["n_bc"] <= 0:
        raise ValueError("n_ic and n_bc must be positive")
    if cfg["w_res"] <= 0 or cfg["w_ic"] <= 0 or cfg["w_bc"] <= 0:
        raise ValueError("All loss weights must be positive")
    if cfg["reference_nt"] < cfg["reference_snapshots"]:
        raise ValueError("reference_nt must exceed snapshot count")


def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        arg = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(arg)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def runtime_info(device: torch.device) -> dict:
    return dict(python=sys.version.split()[0], torch=torch.__version__,
                numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                system=platform.platform(), machine=platform.machine(),
                device=str(device), dtype="float32")


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def solve_fdm_reference(cfg: dict) -> dict:
    """V1 backward Euler in time, centered advection/diffusion in space.

    Sparse LU solves the same tridiagonal stencil as the original notebook's
    TDMA implementation. Assumes homogeneous Dirichlet boundary conditions.
    """
    nx, nt = cfg["reference_nx"], cfg["reference_nt"]
    x0, x1 = cfg["x_range"]
    t0, t1 = cfg["t_range"]
    if nx < 4 or nt < 2:
        raise ValueError("FDM resolution too small")
    x = np.linspace(x0, x1, nx)
    dx, dt = (x1-x0)/(nx-1), (t1-t0)/(nt-1)
    c1, c2 = cfg["c"] * dt / (2*dx), cfg["D"] * dt / (dx*dx)
    low, center, high = -(c1+c2), 1+2*c2, c1-c2
    mat = diags([np.full(nx-3, low), np.full(nx-2, center), np.full(nx-3, high)],
                [-1, 0, 1], shape=(nx-2,nx-2),format="csc")
    lu = splu(mat)
    u = np.sin(np.pi*x)
    u[[0,-1]] = 0.0
    selected = np.linspace(0,nt-1,cfg["reference_snapshots"],dtype=np.int64)
    steps_to_index = {int(s):i for i,s in enumerate(selected)}
    snapshots = np.empty((len(selected),nx),dtype=np.float64)
    snapshots[0] = u
    for step in range(1,nt):
        u[1:-1] = lu.solve(u[1:-1])
        if step in steps_to_index:
            snapshots[steps_to_index[step]] = u
    t = t0+selected*dt
    return dict(x=x,t=t,u=snapshots,dx=dx,dt=dt,
                cell_peclet=(abs(cfg["c"])*dx/cfg["D"] if cfg["D"]>0 else np.inf))


def get_reference(cfg: dict, output_root: Path) -> tuple:
    """One immutable FDM reference shared by all three enforcement modes."""
    source = {k:cfg[k] for k in ("c","D","x_range","t_range","reference_nx",
                                  "reference_nt","reference_snapshots")}
    signature = hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()[:12]
    cache = output_root / ("fdm_reference_"+signature+".npz")
    if not cache.exists():
        print("Computing FDM reference (one cache across modes)...",flush=True)
        t0=time.perf_counter()
        ref = solve_fdm_reference(cfg)
        np.savez_compressed(cache, **ref)
        atomic_json(output_root/("fdm_reference_"+signature+".json"),
                    {"source":source,"elapsed_s":time.perf_counter()-t0,
                     "cell_peclet":float(ref["cell_peclet"]),
                     "warning":"Check grid convergence before publication."})
    with np.load(cache) as f:
        result={k:f[k] for k in f.files}
    if abs(float(result["cell_peclet"])) > 2:
        print("WARNING: coarse FDM grid has cell Peclet number above 2; "
              "smoke results are not for numerical conclusions.",flush=True)
    return result, signature


class PINN(nn.Module):
    """Three genuinely different IC/BC enforcement parameterizations."""
    def __init__(self, layers: list, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Invalid enforcement mode: "+str(mode))
        blocks=[]
        for i in range(len(layers)-1):
            blocks.append(nn.Linear(layers[i],layers[i+1]))
            if i < len(layers)-2:
                blocks.append(nn.Tanh())
        self.net=nn.Sequential(*blocks)
        self.mode=mode

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        raw=self.net(xt)
        if self.mode == "soft_warmup":
            return raw
        x,t=xt[:,0:1],xt[:,1:2]
        if self.mode == "hard_ic_soft_bc":
            return torch.sin(torch.pi*x)+t*raw
        # Hard BC. Network must learn IC sin(pi*x)/(x*(1-x)) near the ends.
        return x*(1.0-x)*raw


def pde_residual(model: PINN, xt: torch.Tensor, c: float, D: float) -> torch.Tensor:
    xt=xt.detach().requires_grad_(True)
    pred=model(xt)
    du=torch.autograd.grad(pred,xt,grad_outputs=torch.ones_like(pred),
                           create_graph=True)[0]
    dxx=torch.autograd.grad(du[:,0:1],xt,
                             grad_outputs=torch.ones_like(du[:,0:1]),
                             create_graph=True)[0][:,0:1]
    return du[:,1:2]+c*du[:,0:1]-D*dxx


def uniform_mc(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n,2))


def uniform_lhs(n: int, rng: np.random.Generator) -> np.ndarray:
    return qmc.LatinHypercube(d=2,seed=rng).random(n)


def sample_ic(n: int, rng: np.random.Generator, device:torch.device) -> tuple:
    x=rng.random((n,1)).astype(np.float32)
    xt=np.column_stack((x,np.zeros_like(x))).astype(np.float32)
    target=np.sin(np.pi*x).astype(np.float32)
    return torch.from_numpy(xt).to(device),torch.from_numpy(target).to(device)


def sample_bc(n: int, rng: np.random.Generator, device:torch.device) -> torch.Tensor:
    t=rng.random((n,1)).astype(np.float32)
    x=np.zeros((n,1),dtype=np.float32)
    x[n//2:]=1.0
    return torch.from_numpy(np.column_stack((x,t))).to(device)


def inv_cdf(u:np.ndarray,cdf:np.ndarray) -> tuple:
    index=np.minimum(np.searchsorted(cdf,u,side="right"),len(cdf)-1)
    low=np.where(index>0,cdf[np.maximum(0,index-1)],0.0)
    mass=np.maximum(cdf[index]-low,1e-16)
    return index,np.clip((u-low)/mass,0.0,1.0)


class ResidualAdaptive2D:
    """V1's piecewise constant distribution and inverse Rosenblatt sampling."""
    def __init__(self,side:int,eta:float,c:float,D:float,chunk:int,device:torch.device):
        self.side,self.eta,self.c,self.D,self.chunk,self.device=side,eta,c,D,chunk,device
        locations=(np.arange(side)+0.5)/side
        xx,tt=np.meshgrid(locations,locations,indexing="xy")
        self.candidates=np.column_stack((xx.ravel(),tt.ravel())).astype(np.float32)
        self.prob=np.full((side,side),1.0/(side*side),dtype=np.float64)
        self.updates=0
        self.eval_points=0
        self.eval_seconds=0.0

    def update(self,model:PINN) -> None:
        start=time.perf_counter()
        old=model.training
        model.eval()
        residuals=[]
        with torch.enable_grad():
            for i in range(0,len(self.candidates),self.chunk):
                xy=torch.from_numpy(self.candidates[i:i+self.chunk]).to(self.device)
                val=pde_residual(model,xy,self.c,self.D)
                residuals.append(np.abs(val.detach().cpu().numpy().reshape(-1)))
        if old:
            model.train()
        field=np.concatenate(residuals).reshape(self.side,self.side).astype(np.float64)
        q=self.eta+(1-self.eta)*field/(field.mean()+1e-12)
        if not np.isfinite(q).all() or q.sum()<=0:
            raise RuntimeError("Nonfinite adaptive proposal density")
        self.prob=q/q.sum()
        self.updates+=1
        self.eval_points+=len(self.candidates)
        self.eval_seconds+=time.perf_counter()-start

    def sample(self,n:int,rng:np.random.Generator,stratified:bool) -> np.ndarray:
        uv=uniform_lhs(n,rng) if stratified else uniform_mc(n,rng)
        # First transform marginal time, then conditional space at t-cell.
        p_t=self.prob.sum(axis=1)
        t_index,t_frac=inv_cdf(uv[:,0],np.cumsum(p_t))
        x_index=np.empty(n,dtype=np.int64)
        x_frac=np.empty(n,dtype=np.float64)
        for index in np.unique(t_index):
            mask=t_index==index
            p_x=self.prob[index,:]/p_t[index]
            x_index[mask],x_frac[mask]=inv_cdf(uv[mask,1],np.cumsum(p_x))
        x=(x_index+x_frac)/self.side
        t=(t_index+t_frac)/self.side
        return np.column_stack((x,t)).astype(np.float32)


def sampling_preflight(cfg:dict,device:torch.device,path:Path)->None:
    n=100
    test=ResidualAdaptive2D(23,cfg["eta"],cfg["c"],cfg["D"],256,device)
    xx,tt=np.meshgrid((np.arange(23)+0.5)/23,(np.arange(23)+0.5)/23,indexing="xy")
    q=1+9*np.exp(-((xx-.83)**2+(tt-.43)**2)/.015)
    test.prob=q/q.sum()
    iid=test.sample(n,np.random.default_rng(9),False)
    strat=test.sample(n,np.random.default_rng(9),True)
    uv=uniform_lhs(n,np.random.default_rng(12345))
    strata=[len(np.unique(np.floor(uv[:,i]*n).astype(int))) for i in range(2)]
    assert strata==[n,n]
    assert np.isclose(test.prob.sum(),1.0)
    assert np.isfinite(iid).all() and np.isfinite(strat).all()
    assert (iid>=0).all() and (iid<=1).all()
    assert (strat>=0).all() and (strat<=1).all()
    atomic_json(path,{"test_density":"synthetic, not PINN residual",
                      "density_integral_1":True,"lhs_probability_strata":strata,
                      "iid_mean_xt":iid.mean(axis=0).tolist(),
                      "alhs_mean_xt":strat.mean(axis=0).tolist()})


def fixed_validation(cfg:dict,seed:int,device:torch.device)->tuple:
    side=cfg["val_grid_side"]
    z=(np.arange(side)+.5)/side
    xx,tt=np.meshgrid(z,z,indexing="xy")
    xy=torch.from_numpy(np.column_stack((xx.ravel(),tt.ravel())).astype(np.float32)).to(device)
    ic,ic_y=sample_ic(201,np.random.default_rng(np.random.SeedSequence([seed,611])),device)
    bc=sample_bc(200,np.random.default_rng(np.random.SeedSequence([seed,612])),device)
    return xy,ic,ic_y,bc


def constraint_terms(model:PINN,ic:torch.Tensor,ic_y:torch.Tensor,
                     bc:torch.Tensor)->tuple:
    # Report only nontrivial penalties; the exact constraints are checked
    # independently in the final evaluation rather than treated as soft loss.
    zero=next(model.parameters()).new_zeros(())
    ic_loss=(torch.mean((model(ic)-ic_y)**2)
             if model.mode != "hard_ic_soft_bc" else zero)
    bc_loss=(torch.mean(model(bc)**2)
             if model.mode != "hard_bc_soft_ic" else zero)
    return ic_loss,bc_loss


def full_losses(model:PINN,res_xt:torch.Tensor,ic:torch.Tensor,ic_y:torch.Tensor,
                bc:torch.Tensor,cfg:dict,res_weight:float)->tuple:
    res=pde_residual(model,res_xt,cfg["c"],cfg["D"])
    pde_loss=torch.mean(res**2)
    ic_loss,bc_loss=constraint_terms(model,ic,ic_y,bc)
    total=res_weight*pde_loss+cfg["w_ic"]*ic_loss+cfg["w_bc"]*bc_loss
    return total,pde_loss,ic_loss,bc_loss


def full_validation(model:PINN,val:tuple,cfg:dict)->dict:
    old=model.training
    model.eval()
    with torch.enable_grad():
        parts=full_losses(model,*val,cfg,cfg["w_res"])
        values=[float(x.detach()) for x in parts]
    if old:
        model.train()
    if not all(np.isfinite(v) for v in values):
        raise FloatingPointError("Nonfinite validation losses")
    return dict(total=values[0],pde=values[1],ic=values[2],bc=values[3])


def predict_numpy(model:PINN,xy:np.ndarray,device:torch.device,chunk:int)->np.ndarray:
    results=[]
    old=model.training
    model.eval()
    with torch.no_grad():
        for i in range(0,len(xy),chunk):
            xt=torch.from_numpy(xy[i:i+chunk].astype(np.float32)).to(device)
            results.append(model(xt).detach().cpu().numpy().ravel())
    if old:
        model.train()
    return np.concatenate(results).astype(np.float64)


def residual_numpy(model:PINN,xy:np.ndarray,cfg:dict,device:torch.device)->np.ndarray:
    results=[]
    old=model.training
    model.eval()
    with torch.enable_grad():
        for i in range(0,len(xy),cfg["evaluation_chunk"]):
            xt=torch.from_numpy(xy[i:i+cfg["evaluation_chunk"]].astype(np.float32)).to(device)
            out=pde_residual(model,xt,cfg["c"],cfg["D"])
            results.append(out.detach().cpu().numpy().ravel())
    if old:
        model.train()
    return np.concatenate(results).astype(np.float64)


def evaluate(model:PINN,ref:dict,cfg:dict,val:tuple,device:torch.device)->tuple:
    x,t,U=ref["x"],ref["t"],ref["u"]
    pred=[]
    for ti in t:
        xy=np.column_stack((x,np.full_like(x,ti)))
        pred.append(predict_numpy(model,xy,device,cfg["evaluation_chunk"]))
    pred=np.array(pred)
    diff=pred-U
    edge=(x>=0.8)
    near_edge=(x>=0.98)
    dside=41
    z1=(np.arange(dside)+.31)/dside
    z2=(np.arange(dside)+.67)/dside
    xx,tt=np.meshgrid(z1,z2,indexing="xy")
    dense_xy=np.column_stack((xx.ravel(),tt.ravel()))
    rr=residual_numpy(model,dense_xy,cfg,device)
    valres=full_validation(model,val,cfg)
    final_norm=max(np.linalg.norm(U[-1]),1e-15)
    metrics=dict(
        space_time_rel_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(U),1e-15)),
        space_time_mse=float(np.mean(diff**2)),
        final_mse=float(np.mean(diff[-1]**2)),
        final_rmse=float(np.sqrt(np.mean(diff[-1]**2))),
        final_rel_l2=float(np.linalg.norm(diff[-1])/final_norm),
        final_zero_baseline_mse=float(np.mean(U[-1]**2)),
        boundary_layer_rmse_t1=float(np.sqrt(np.mean(diff[-1,edge]**2))),
        near_boundary_rmse_t1=float(np.sqrt(np.mean(diff[-1,near_edge]**2))),
        dense_physics_loss=float(np.mean(rr**2)),
        residual_rms_dense=float(np.sqrt(np.mean(rr**2))),
        ic_mse=float(np.mean((pred[0]-np.sin(np.pi*x))**2)),
        bc_mse=float(np.mean(pred[:,[0,-1]]**2)),
        ic_max_abs_error=float(np.max(np.abs(pred[0]-np.sin(np.pi*x)))),
        bc_max_abs_error=float(np.max(np.abs(pred[:,[0,-1]]))),
        final_physics_val_loss=valres["pde"],
        final_ic_val_loss=valres["ic"],
        final_bc_val_loss=valres["bc"],
    )
    return metrics,pred


def save_diagnostics(folder:Path,model:PINN,cfg:dict,ref:dict,
                     pred:np.ndarray,history:list,device:torch.device) -> dict:
    """Write numerical arrays and figures for physically interpretable diagnosis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True,exist_ok=True)
    x=np.asarray(ref["x"])
    times=np.asarray(ref["t"])
    truth=np.asarray(ref["u"])
    chosen=[0.0,0.1,0.5,0.75,1.0]
    frames=[int(np.argmin(abs(times-v))) for v in chosen]

    fig,axs=plt.subplots(2,3,figsize=(13,7.5),layout="constrained")
    for j,ix in enumerate(frames):
        ax=axs.flat[j]
        ax.plot(x,truth[ix],label="FDM reference",lw=1.9)
        ax.plot(x,pred[ix],label="PINN prediction",lw=1.6,ls="--")
        ax.set_title("t = %.3f"%times[ix])
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.grid(alpha=.25)
        if j==0:
            ax.legend(fontsize=8)
    axs.flat[-1].axis("off")
    fig.suptitle("Convection-diffusion: reference vs. PINN",fontsize=13)
    fig.savefig(folder/"solution_profiles.png",dpi=145)
    plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(12.5,3.6),layout="constrained")
    for ax,tx in zip(axs,[0.5,0.75,1.0]):
        ix=int(np.argmin(abs(times-tx)))
        mask=x>=.9
        ax.plot(x[mask],truth[ix,mask],label="FDM",lw=1.8)
        ax.plot(x[mask],pred[ix,mask],label="PINN",lw=1.5,ls="--")
        ax.set_title("Outflow region at t=%.2f"%times[ix])
        ax.set_xlabel("x")
        ax.grid(alpha=.25)
        if ax==axs[0]: ax.legend(fontsize=8)
    fig.savefig(folder/"outflow_zoom.png",dpi=145)
    plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12.5,4.3),layout="constrained")
    for ax,data,title in ((axs[0],pred,"PINN u(x,t)"),
                          (axs[1],pred-truth,"PINN - FDM error")):
        im=ax.pcolormesh(x,times,data,shading="auto",cmap="viridis")
        fig.colorbar(im,ax=ax,shrink=.85)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
    fig.savefig(folder/"solution_error_maps.png",dpi=145)
    plt.close(fig)

    # Nonuniform spatial grid probes the very thin near-outflow layer.
    xgrid=np.unique(np.r_[np.linspace(0,.9,91),np.linspace(.9,.99,48),
                            np.linspace(.99,1,70)]).astype(np.float64)
    tgrid=np.unique(np.r_[np.linspace(1e-4,.1,26),np.linspace(.1,1,69)]).astype(np.float64)
    xx,tt=np.meshgrid(xgrid,tgrid,indexing="xy")
    xy=np.column_stack((xx.ravel(),tt.ravel()))
    residual=residual_numpy(model,xy,cfg,device).reshape(len(tgrid),len(xgrid))
    pred_map=predict_numpy(model,xy,device,cfg["evaluation_chunk"]).reshape(residual.shape)
    np.savez_compressed(folder/"diagnostic_arrays.npz",x=x,t=times,fdm=truth,
                        pinn=pred,diag_x=xgrid,diag_t=tgrid,
                        pde_residual=residual,diag_pinn=pred_map)
    log_abs=np.log10(np.maximum(np.abs(residual),1e-9))
    fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout="constrained")
    for ax,mask,title in ((axs[0],xgrid>=0,"Full spatial domain"),
                          (axs[1],xgrid>=.95,"Outflow x >= 0.95")):
        mesh=ax.pcolormesh(xgrid[mask],tgrid,log_abs[:,mask],shading="auto",
                           cmap="magma",vmin=-6,vmax=max(-5,float(np.nanpercentile(log_abs,99))))
        fig.colorbar(mesh,ax=ax,label="log10 |PDE residual|",shrink=.8)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
    fig.savefig(folder/"residual_heatmaps.png",dpi=145)
    plt.close(fig)

    history_df=pd.DataFrame(history)
    history_df.to_csv(folder/"training_history.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(12.3,4.2),layout="constrained")
    for phase,ax in (("warmup",axs[0]),("main",axs[1])):
        df=history_df[history_df.phase==phase]
        if len(df):
            for key in ("objective","pde_loss","ic_loss","bc_loss"):
                values=df[key].to_numpy(dtype=float)
                if np.isfinite(values).any():
                    ax.semilogy(df["global_epoch"],np.maximum(values,1e-12),label=key)
        ax.set_title("%s training"%phase.capitalize())
        ax.set_xlabel("Total Adam epochs")
        ax.set_ylabel("Loss")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(folder/"training_loss_history.png",dpi=145)
    plt.close(fig)

    focus=(xgrid>=.98)
    return dict(residual_rms_diagnostic=float(np.sqrt(np.mean(residual**2))),
                residual_rms_near_outflow=float(np.sqrt(np.mean(residual[:,focus]**2))),
                diagnostic_grid_points=int(residual.size),
                diagnostic_files=["solution_profiles.png","outflow_zoom.png",
                                  "solution_error_maps.png","residual_heatmaps.png",
                                  "training_loss_history.png","diagnostic_arrays.npz",
                                  "training_history.csv"])


def train_one(method:str,nf:int,seed:int,mode:str,cfg:dict,device:torch.device,
              ref:dict,diagnostics_folder:Optional[Path]=None)->dict:
    # Identical Torch initialization across modes and samplers for this seed.
    torch.manual_seed(seed)
    if device.type=="cuda":
        torch.cuda.manual_seed_all(seed)
    model=PINN(cfg["layers"],mode).to(device)
    optim=torch.optim.Adam(model.parameters(),lr=cfg["adam_lr"])
    warm_ic_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,701]))
    warm_bc_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,702]))
    ic_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,992]))
    bc_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,993]))
    sampler_rng=np.random.default_rng(np.random.SeedSequence([seed,nf,991]))
    val=fixed_validation(cfg,seed,device)
    history=[]
    training_start=time.perf_counter()

    # A) Warmup: fit whichever IC/BC constraints are soft in this mode.
    start=time.perf_counter()
    for epoch in range(1,cfg["warmup_epochs"]+1):
        ic,ic_y=sample_ic(cfg["n_ic"],warm_ic_rng,device)
        bc=sample_bc(cfg["n_bc"],warm_bc_rng,device)
        optim.zero_grad(set_to_none=True)
        ic_loss,bc_loss=constraint_terms(model,ic,ic_y,bc)
        objective=cfg["w_ic"]*ic_loss+cfg["w_bc"]*bc_loss
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite warmup objective")
        objective.backward()
        optim.step()
        if epoch==1 or epoch==cfg["warmup_epochs"] or epoch%max(1,cfg["diagnostics_every"])==0:
            history.append(dict(phase="warmup",global_epoch=epoch,main_epoch=0,
                                pde_weight=0.0,objective=float(objective.detach()),
                                pde_loss=np.nan,ic_loss=float(ic_loss.detach()),
                                bc_loss=float(bc_loss.detach()),validation_full_loss=np.nan))
    warm_seconds=time.perf_counter()-start

    # B) Main stage: gradually introduce the full PDE residual weight.
    adaptive=(ResidualAdaptive2D(cfg["adapt_grid_side"],cfg["eta"],cfg["c"],cfg["D"],
                                 cfg["adapt_eval_chunk"],device)
              if method in ("rad","alhs") else None)
    recent=deque(maxlen=cfg["lbfgs_union_sets"])
    sets=0
    sample_seconds=0.0
    val_seconds=0.0
    best_val=np.inf
    best_epoch=None
    best_state=None
    start=time.perf_counter()
    for step in range(1,cfg["main_epochs"]+1):
        if (step-1)%cfg["resample_every"]==0:
            begin=time.perf_counter()
            if adaptive is not None and step>1:
                adaptive.update(model)
            if method=="mc":
                positions=uniform_mc(nf,sampler_rng).astype(np.float32)
            elif method=="lhs":
                positions=uniform_lhs(nf,sampler_rng).astype(np.float32)
            else:
                positions=adaptive.sample(nf,sampler_rng,method=="alhs")
            recent.append(positions)
            res_xt=torch.from_numpy(positions).to(device)
            sets+=1
            sample_seconds+=time.perf_counter()-begin
        ic,ic_y=sample_ic(cfg["n_ic"],ic_rng,device)
        bc=sample_bc(cfg["n_bc"],bc_rng,device)
        pde_weight=cfg["w_res"]*min(1.0,step/cfg["physics_ramp_epochs"])
        optim.zero_grad(set_to_none=True)
        objective,pde_loss,ic_loss,bc_loss=full_losses(
            model,res_xt,ic,ic_y,bc,cfg,pde_weight)
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite main Adam objective at epoch %d"%step)
        objective.backward()
        optim.step()
        needs_val=(step%cfg["validation_every"]==0 or step==cfg["main_epochs"])
        val_full=np.nan
        if needs_val:
            t0=time.perf_counter()
            checkpoint_loss=full_validation(model,val,cfg)
            val_seconds+=time.perf_counter()-t0
            val_full=checkpoint_loss["total"]
            # Avoid selecting an underweighted-ramp checkpoint as the best model.
            if step>=cfg["physics_ramp_epochs"] and val_full<best_val:
                best_val=val_full
                best_epoch=step
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if needs_val or step==1 or step%cfg["diagnostics_every"]==0:
            history.append(dict(phase="main",global_epoch=cfg["warmup_epochs"]+step,
                                main_epoch=step,pde_weight=pde_weight,
                                objective=float(objective.detach()),
                                pde_loss=float(pde_loss.detach()),
                                ic_loss=float(ic_loss.detach()),
                                bc_loss=float(bc_loss.detach()),
                                validation_full_loss=val_full))
    main_seconds=time.perf_counter()-start
    if best_state is None:
        raise RuntimeError("No valid checkpoint after the physics ramp")

    # C) Shared, full-weight L-BFGS refinement with checkpoint selection.
    model.load_state_dict(best_state)
    union=np.unique(np.concatenate(list(recent),axis=0),axis=0)
    x_union=torch.from_numpy(union).to(device)
    lb_ic,lb_icy=sample_ic(cfg["n_ic"],
                          np.random.default_rng(np.random.SeedSequence([seed,nf,995])),device)
    lb_bc=sample_bc(cfg["n_bc"],
                    np.random.default_rng(np.random.SeedSequence([seed,nf,996])),device)
    after_val=np.inf
    lb_calls=0
    lb_seconds=0.0
    lb_status="not_run"
    lb_error=""
    if cfg["lbfgs_max_iter"]>0:
        solver=torch.optim.LBFGS(model.parameters(),lr=1.0,
                                 max_iter=cfg["lbfgs_max_iter"],history_size=50,
                                 line_search_fn="strong_wolfe")
        start=time.perf_counter()
        def closure():
            nonlocal lb_calls
            lb_calls+=1
            solver.zero_grad(set_to_none=True)
            total=full_losses(model,x_union,lb_ic,lb_icy,lb_bc,cfg,cfg["w_res"])[0]
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite loss in L-BFGS closure")
            total.backward()
            return total
        try:
            solver.step(closure)
            after_val=full_validation(model,val,cfg)["total"]
            lb_status="succeeded"
        except (FloatingPointError,RuntimeError,ValueError) as exc:
            lb_status="failed_reverted"
            lb_error=repr(exc)[:700]
            print("WARNING L-BFGS failed; reverting to best Adam:",lb_error,flush=True)
        lb_seconds=time.perf_counter()-start
    if after_val<best_val:
        selected="lbfgs"
        selected_val=after_val
    else:
        model.load_state_dict(best_state)
        selected="adam"
        selected_val=best_val
    training_seconds=time.perf_counter()-training_start
    eval_start=time.perf_counter()
    metrics,pred=evaluate(model,ref,cfg,val,device)
    evaluation_seconds=time.perf_counter()-eval_start
    diag_start=time.perf_counter()
    detail={}
    if diagnostics_folder is not None:
        detail=save_diagnostics(diagnostics_folder,model,cfg,ref,pred,history,device)
    diagnostic_seconds=time.perf_counter()-diag_start
    if not all(np.isfinite(metrics[k]) for k in
               ("space_time_rel_l2","final_mse","dense_physics_loss","ic_mse","bc_mse")):
        raise FloatingPointError("Nonfinite evaluation metric")
    row=dict(method=method,method_label=METHOD_LABELS[method],
             constraint_mode=mode,n_collocation=nf,seed=seed,
             warmup_epochs=cfg["warmup_epochs"],main_epochs=cfg["main_epochs"],
             physics_ramp_epochs=cfg["physics_ramp_epochs"],
             pde_loss_weight=cfg["w_res"],
             resample_every=cfg["resample_every"],resample_set_count=sets,
             adaptive_updates=adaptive.updates if adaptive is not None else 0,
             adaptive_proposal_evaluations=adaptive.eval_points if adaptive is not None else 0,
             lbfgs_union_count=len(union),lbfgs_closure_calls=lb_calls,
             lbfgs_status=lb_status,lbfgs_error=lb_error,
             adam_best_epoch=best_epoch,adam_best_val_loss=best_val,
             lbfgs_val_loss=(float(after_val) if np.isfinite(after_val) else np.nan),
             selected_stage=selected,selected_val_loss=selected_val,
             runtime_warmup_s=warm_seconds,runtime_adam_main_s=main_seconds,
             runtime_lbfgs_s=lb_seconds,
             runtime_sampling_in_adam_s=sample_seconds,
             runtime_adaptive_density_s=(adaptive.eval_seconds if adaptive is not None else 0.0),
             runtime_validation_s=val_seconds,
             runtime_training_s=training_seconds,
             runtime_evaluation_s=evaluation_seconds,
             runtime_diagnostics_s=diagnostic_seconds,
             diagnostic_dir=(str(diagnostics_folder) if diagnostics_folder else ""),
             **metrics,**detail)
    return row


def write_summary(df:pd.DataFrame,folder:Path,cfg:dict)->None:
    if df.empty:
        return
    keys=["space_time_rel_l2","final_mse","final_rel_l2",
          "dense_physics_loss","ic_mse","bc_mse","runtime_training_s"]
    rows=[]
    for (method,nf),sub in df.groupby(["method","n_collocation"]):
        item=dict(method=method,n_collocation=int(nf),count=int(len(sub)))
        for name in keys:
            x=sub[name].to_numpy(dtype=float)
            item[name+"_median"]=float(np.median(x))
            item[name+"_q1"]=float(np.quantile(x,.25))
            item[name+"_q3"]=float(np.quantile(x,.75))
        rows.append(item)
    atomic_csv(folder/"summary.csv",pd.DataFrame(rows))
    expected={(m,n,s) for m in cfg["methods"] for n in cfg["n_collocation"]
              for s in cfg["seeds"]}
    got=[(str(row.method),int(row.n_collocation),int(row.seed))
         for row in df.itertuples(index=False)]
    missing=sorted(expected-set(got))
    atomic_json(folder/"integrity.json",dict(
        expected_runs=len(expected),actual_rows=len(df),unique_run_keys=len(set(got)),
        duplicate_key_rows=len(got)-len(set(got)),missing_run_keys=missing,
        complete=len(missing)==0 and len(df)==len(expected)==len(set(got))))
    # Pilot/diagnostic (1 seed) is descriptive only; never apply significance tests.
    stat_rows=[]
    for nf in cfg["n_collocation"]:
        sub=df[df.n_collocation==nf]
        for metric in ("space_time_rel_l2","final_mse"):
            if len(cfg["methods"])<3 or len(cfg["seeds"])<3:
                continue
            if len(sub)!=len(cfg["methods"])*len(cfg["seeds"]):
                continue
            wide=sub.pivot(index="seed",columns="method",values=metric).reindex(columns=cfg["methods"])
            if len(wide)!=len(cfg["seeds"]) or wide.isna().any().any():
                continue
            arrays=[wide[m].to_numpy() for m in cfg["methods"]]
            stat,p=friedmanchisquare(*arrays)
            stat_rows.append(dict(n_collocation=nf,metric=metric,test="Friedman",
                                  method_a="all",method_b="all",statistic=stat,p_raw=p,
                                  p_holm=np.nan,rrb=np.nan))
            pairs=[]
            for a,b in combinations(cfg["methods"],2):
                xa,xb=wide[a].to_numpy(),wide[b].to_numpy()
                w=wilcoxon(xa,xb,alternative="two-sided",zero_method="wilcox")
                delta=xa-xb
                mask=delta!=0
                ranks=rankdata(abs(delta[mask]))
                rrb=(float(np.sum(ranks[delta[mask]>0])-np.sum(ranks[delta[mask]<0]))/
                     max(float(np.sum(ranks)),1.0))
                pairs.append(dict(n_collocation=nf,metric=metric,test="Wilcoxon_Holm",
                                  method_a=a,method_b=b,statistic=float(w.statistic),
                                  p_raw=float(w.pvalue),rrb=rrb))
            order=np.argsort([e["p_raw"] for e in pairs]);last=0.0
            for i,k in enumerate(order):
                last=max(last,min(1.0,pairs[k]["p_raw"]*(len(order)-i)))
                pairs[k]["p_holm"]=last
            stat_rows.extend(pairs)
    atomic_csv(folder/"statistics.csv",pd.DataFrame(stat_rows,
               columns=["n_collocation","metric","test","method_a","method_b",
                        "statistic","p_raw","p_holm","rrb"]))


def run_mode(mode:str,cfg:dict,ref:dict,ref_signature:str,root:Path,
             device:torch.device,preset:str,diagnostics_policy:str,script_sha:str)->None:
    work=root/("PDE_v2_"+preset+"_"+mode)
    work.mkdir(parents=True,exist_ok=True)
    manifest=dict(protocol_version=VERSION,preset=preset,mode=mode,
                  mode_description=MODE_DESCRIPTION[mode],config=cfg,
                  runtime=runtime_info(device),script_sha256=script_sha,
                  reference_signature=ref_signature,
                  diagnostics_policy=diagnostics_policy,
                  source="V2 derived from user's uploaded convection-diffusion V1 notebook.",
                  warning="A single-seed diagnostic is not publication evidence.")
    mf=work/"manifest.json"
    if mf.exists():
        if json.loads(mf.read_text(encoding="utf-8"))!=manifest:
            raise RuntimeError("Existing manifest/runtime/script differs in "+str(work)+
                               "; choose a new --output-root to avoid mixing runs")
    else:
        atomic_json(mf,manifest)
    sampling_preflight(cfg,device,work/"sampling_preflight.json")
    raw_file=work/"results_raw.csv"
    results=pd.read_csv(raw_file) if raw_file.exists() else pd.DataFrame()
    key_set=set()
    if len(results):
        existing_keys=list(zip(results.method,
                               results.n_collocation.astype(int),results.seed.astype(int)))
        key_set=set(existing_keys)
        if len(existing_keys)!=len(key_set):
            raise RuntimeError("Duplicate run keys in resume data")
    expected=len(cfg["methods"])*len(cfg["n_collocation"])*len(cfg["seeds"])
    print("\nMODE:",mode,"runs:",expected,"folder:",work,flush=True)
    count_run=0
    for nf in cfg["n_collocation"]:
        for seed in cfg["seeds"]:
            for method in cfg["methods"]:
                key=(method,nf,seed)
                if key in key_set:
                    print("SKIP",key,flush=True)
                    continue
                count_run+=1
                do_diag=(diagnostics_policy=="all" or
                         (diagnostics_policy=="first" and count_run==1))
                diagdir=(work/"diagnostics"/f"{method}_nf{nf}_seed{seed}"
                         if do_diag else None)
                print("RUN %d/%d: %s, Nf=%d, seed=%d, mode=%s"%(
                    len(key_set)+1,expected,method,nf,seed,mode),flush=True)
                try:
                    row=train_one(method,nf,seed,mode,cfg,device,ref,diagdir)
                    results=pd.concat([results,pd.DataFrame([row])],ignore_index=True)
                    atomic_csv(raw_file,results)
                    key_set.add(key)
                    write_summary(results,work,cfg)
                    print("OK relative-L2=%.6f, IC=%.2e, BC=%.2e, PDE=%.3e, time=%.1fs"%(
                        row["space_time_rel_l2"],row["ic_mse"],row["bc_mse"],
                        row["dense_physics_loss"],row["runtime_training_s"]),flush=True)
                except Exception:
                    print(traceback.format_exc(),file=sys.stderr,flush=True)
                    print("Incomplete experiment is saved; rerun to resume.",
                          file=sys.stderr,flush=True)
                    raise
    write_summary(results,work,cfg)
    print("COMPLETE",mode,"rows:",len(results),"/",expected,flush=True)


def combine_enforcement_diagnostics(root:Path,preset:str)->None:
    rows=[]
    for mode in MODES:
        path=root/("PDE_v2_"+preset+"_"+mode)/"results_raw.csv"
        if path.exists():
            df=pd.read_csv(path)
            rows.append(df)
    if not rows:
        return
    joint=pd.concat(rows,ignore_index=True)
    fields=["constraint_mode","method","n_collocation","seed",
            "space_time_rel_l2","final_rel_l2","ic_mse","bc_mse",
            "dense_physics_loss","runtime_training_s","runtime_evaluation_s",
            "runtime_diagnostics_s","selected_stage","diagnostic_dir"]
    atomic_csv(root/("PDE_v2_"+preset+"_enforcement_comparison.csv"),joint[fields])
    print("Cross-mode descriptive file saved (not a statistical inference).",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preset",choices=["smoke","diagnostic","pilot","full"],default="smoke")
    parser.add_argument("--constraint-mode",choices=list(MODES)+["compare"],default="compare")
    parser.add_argument("--device",choices=["auto","mps","cuda","cpu"],default="auto")
    parser.add_argument("--output-root",type=Path,default=Path("pde_v2_outputs"))
    parser.add_argument("--warmup-epochs",type=int,default=None)
    parser.add_argument("--main-epochs",type=int,default=None)
    parser.add_argument("--ramp-epochs",type=int,default=None)
    parser.add_argument("--lbfgs-max-iter",type=int,default=None)
    parser.add_argument("--diagnostics",choices=["all","first","none"],default=None,
                        help="Default all for smoke/diagnostic, first for pilot/full")
    args=parser.parse_args()
    if args.preset in ("pilot","full") and args.constraint_mode=="compare":
        parser.error("For pilot/full, choose ONE --constraint-mode; compare is only for smoke/diagnostic")
    diagnostics_policy=(args.diagnostics if args.diagnostics else
                        ("all" if args.preset in ("smoke","diagnostic") else "first"))
    cfg=preset_config(args.preset)
    for key,value in (("warmup_epochs",args.warmup_epochs),
                      ("main_epochs",args.main_epochs),
                      ("physics_ramp_epochs",args.ramp_epochs),
                      ("lbfgs_max_iter",args.lbfgs_max_iter)):
        if value is not None:
            cfg[key]=value
    validate_config(cfg)
    device=device_from_arg(args.device)
    root=args.output_root.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print("PDE benchmark V2, device:",device,"precision: float32",flush=True)
    print("PyTorch:",torch.__version__,"Python:",sys.version.split()[0],flush=True)
    print("Warmup:",cfg["warmup_epochs"],"main Adam:",cfg["main_epochs"],
          "ramp:",cfg["physics_ramp_epochs"],flush=True)
    ref,ref_signature=get_reference(cfg,root)
    modes=MODES if args.constraint_mode=="compare" else [args.constraint_mode]
    for mode in modes:
        run_mode(mode,cfg,ref,ref_signature,root,device,args.preset,
                 diagnostics_policy,script_sha)
    if args.constraint_mode=="compare":
        combine_enforcement_diagnostics(root,args.preset)
    print("V2 FINISHED. Output root:",root,flush=True)


if __name__=="__main__":
    main()
