#!/usr/bin/env python3
"""V2.3.2: five paired seed output-transform ablation against frozen V2.3.1 MC-R.

Training loop is cloned from V2.3.1 run_one; only PINN constructor / variant
is changed. V2.3.1 source and baseline checkpoint selection are NOT edited.
FDM+SG are evaluation-only; baseline results included from the user's archive.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import convection_diffusion_benchmark_v2 as core
import convection_diffusion_benchmark_v21 as v21
import convection_diffusion_benchmark_v22 as v22
import convection_diffusion_benchmark_v23 as v23
import convection_diffusion_validation_v231 as v231

VERSION = '2.3.2'
CANDIDATE_VARIANT = 'hard_both_bl_uniform'
BASELINE_VARIANT = 'bl_uniform'
REQUIRED_SEEDS = (0,1,2,3,4)
BASELINE_FILENAME = 'V231_Baseline_5Seeds_Frozen.csv'

# Reuse EXACT same helper functions as frozen v2.3.1.
method_points = v231.method_points
curriculum_weight = v231.curriculum_weight
independent_residual_audit = v231.independent_residual_audit


class HardBothBoundaryLayerPINN(v22.BoundaryLayerPINN):
    """u=sin(pi*x)+t*x*(N(x,t)-B_R(x)*N(1,t)).

    Hard IC + both Dirichlet BC (to float precision), soft penalties remain
    in the unchanged base loss for audit but their gradients are exactly zero.
    Multiplying by x modifies the interior parameterization too: an ablation
    of the OUTPUT TRANSFORM rather than a mathematically isolated BC change.
    """
    def __init__(self, layers, delta):
        super().__init__(layers, boundary_layer=True, delta=delta)

    def forward(self, xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        raw = self.net(xt)
        right_xt = torch.cat((torch.ones_like(x), t), dim=1)
        right_raw = self.net(right_xt)
        return torch.sin(torch.pi*x) + t*x*(raw - self.right_basis(x)*right_raw)


def run_candidate(variant,method,cfg,ref,sg_ref,device,diagdir,checkpoint_policy,seed=0):
    # V2.3 training code retained; the seed is now an explicit run argument.
    nf=int(cfg["n_collocation"][0]);seed=int(seed)
    if variant != CANDIDATE_VARIANT or method != 'mc':
        raise ValueError('V2.3.2 ablation accepts only hard_both_bl_uniform / MC-R')
    use_bl,enriched=True,False
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    delta=cfg["D"]/abs(cfg["c"])*cfg["bl_delta_factor"]
    model=HardBothBoundaryLayerPINN(cfg["layers"],delta).to(device)
    initial_hash=v21.state_hash(v21.model_checkpoint(model))
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
        output_enforcement='hard_ic_hard_left_hard_right',
        initial_state_sha256=initial_hash,
        warmup_noop_expected=True,
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




def load_candidate(cfg, device, filename: Path):
    if not filename.is_file():
        raise FileNotFoundError(filename)
    delta=float(cfg['D'])/abs(float(cfg['c']))*float(cfg['bl_delta_factor'])
    model=HardBothBoundaryLayerPINN(cfg['layers'],delta).to(device)
    state=torch.load(filename,map_location=device,weights_only=True)
    model.load_state_dict(state,strict=True)
    model.eval()
    return model


def dense_inflow_residual_audit(model, cfg, device, folder, x_points=181, t_points=61):
    """Independent post-hoc autograd PDE audit focused near left boundary.

    Used ONLY for reporting, never for checkpoints, optimization or model choice.
    Complements the frozen V2.3.1 log-grid outflow audit.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    distances=np.geomspace(1e-5,2e-2,x_points)
    x=np.asarray(distances,dtype=np.float64)
    t=np.linspace(.01,1.0,t_points,dtype=np.float64)
    xx,tt=np.meshgrid(x,t)
    xy=np.column_stack((xx.ravel(),tt.ravel()))
    residual=core.residual_numpy(model,xy,cfg,device).reshape(len(t),len(x))
    if not np.isfinite(residual).all():
        raise FloatingPointError('Nonfinite inflow residual')
    folder=Path(folder)
    np.savez_compressed(folder/'dense_inflow_residual.npz',x=x,t=t,residual=residual,
                        corner_t0_excluded=True)
    rows=[]
    bands=((1e-5,1e-4),(1e-4,1e-3),(1e-3,1e-2),(1e-2,2e-2))
    for k,(lo,hi) in enumerate(bands):
        mask=(x>=lo)&(x<hi if k<len(bands)-1 else x<=hi)
        r=residual[:,mask]; xloc=x[mask]
        rows.append({'x_lo':lo,'x_hi':hi,'n_x':len(xloc),'n_t':len(t),
            'rms_on_log_grid':float(np.sqrt(np.mean(r*r))),
            'rms_area_weighted':v231._area_rms(r,xloc,t),
            'max_abs':float(np.max(np.abs(r))),
            'p95_abs':float(np.quantile(np.abs(r),.95))})
    core.atomic_csv(folder/'dense_inflow_residual_bands.csv',pd.DataFrame(rows))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    m=axes[0].pcolormesh(np.log10(x),t,np.log10(np.maximum(abs(residual),1e-9)),
                         shading='auto',cmap='viridis')
    fig.colorbar(m,ax=axes[0],label='log10 |PDE residual|')
    axes[0].set_xlabel('log10(x)');axes[0].set_ylabel('t');axes[0].set_title('Near inflow x=0')
    for target in (.05,.5,1.0):
        j=int(np.argmin(abs(t-target)))
        axes[1].plot(x,np.maximum(abs(residual[j]),1e-9),label=f't={t[j]:.3f}')
    axes[1].set_xscale('log');axes[1].set_yscale('log')
    axes[1].set_xlabel('x');axes[1].set_ylabel('|PDE residual|')
    axes[1].grid(alpha=.25);axes[1].legend()
    fig.suptitle('V2.3.2 independent inflow PDE residual audit')
    fig.savefig(folder/'dense_inflow_residual.png',dpi=145);plt.close(fig)
    summary={
        'audit':'independent_dense_log_inflow',
        'rms_area_x_le_0001':rows[0]['rms_area_weighted'],
        'rms_area_x_le_001':v231._area_rms(residual[:,x<=.001],x[x<=.001],t),
        'rms_area_full_log_inflow':v231._area_rms(residual,x,t),
        'rms_pointwise_logmesh':float(np.sqrt(np.mean(residual**2))),
        'max_abs':float(np.max(abs(residual))),
        'training_or_selection_use':False,'corner_t0_excluded':True,
        'x_points':x_points,'t_points':t_points,
        'warning':'Log-grid average overweights closest-to-boundary nodes; use area-weighted RMS for comparison.'}
    core.atomic_json(folder/'dense_inflow_residual_summary.json',summary)
    return summary


def code_provenance():
    paths={'v232':Path(__file__).resolve(),
           'v231':Path(v231.__file__).resolve(),
           'v23':Path(v23.__file__).resolve(),
           'v22':Path(v22.__file__).resolve(),
           'v21':Path(v21.__file__).resolve(),
           'v2':Path(core.__file__).resolve()}
    return {k:v21.sha256_file(v) for k,v in paths.items()}


def default_baseline_path():
    return Path(__file__).resolve().parent/BASELINE_FILENAME


def frozen_baseline(path=None,complete_required=True):
    filename=Path(path) if path is not None else default_baseline_path()
    d=pd.read_csv(filename)
    if not {'seed','variant','method','space_time_rel_l2','sampling_sha256'}.issubset(d.columns):
        raise RuntimeError('Baseline V2.3.1 missing required columns')
    if (not set(d.seed.astype(int)).issubset(set(REQUIRED_SEEDS))
            or not (d.variant=='bl_uniform').all() or not (d.method=='mc').all()
            or not (d.n_collocation==500).all()
            or not (d.curriculum=='v31_staged').all()
            or not (d.protocol_version.astype(str)=='2.3.1').all()
            or d.duplicated('seed').any()):
        raise RuntimeError('Baseline does not match frozen 5-seed V2.3.1 MC-R data')
    if complete_required and set(d.seed.astype(int))!=set(REQUIRED_SEEDS):
        raise RuntimeError('Baseline is not the complete seeds 0..4')
    return d.sort_values('seed').reset_index(drop=True), filename


def assert_candidate_matched(baseline, candidate_row, reference_cfg, is_smoke=False):
    if is_smoke:
        return
    seed=int(candidate_row['seed'])
    subset=baseline.loc[baseline.seed==seed]
    if len(subset)!=1:
        raise RuntimeError(f'No baseline available for seed {seed}')
    b=subset.iloc[0]
    for k in ('n_collocation','warmup_epochs','transition_epochs','joint_epochs'):
        if int(candidate_row[k])!=int(b[k]):
            raise RuntimeError(f'Protocol mismatch seed {seed}, {k}')
    for k in ('lambda_residual','lambda_initial','lambda_boundary'):
        if not np.isclose(float(candidate_row[k]),float(b[k]),rtol=0,atol=0):
            raise RuntimeError(f'Loss-weight mismatch seed {seed}, {k}')
    # Sampling streams may differ across NumPy/SciPy versions even with the same seeds.
    # Record that explicitly; never claim identical realizations unless hashes match.
    if candidate_row['sampling_sha256']!=b['sampling_sha256']:
        print(f'WARNING: seed {seed}: MC-R sampling SHA differs from frozen V2.3.1. '
              'Check NumPy/SciPy environment; same-seed comparison is NOT a common-samples pairing.',flush=True)
    if bool(candidate_row['validation_fdm_sg_used_for_training']):
        raise RuntimeError('Reference leakage flag invalid')
    if candidate_row['variant']!=CANDIDATE_VARIANT:
        raise RuntimeError('Unexpected candidate variant')


PAIR_METRICS=(
 ('space_time_rel_l2','Space-time relative L2'),
 ('final_rel_l2','Final-time relative L2'),
 ('pinn_vs_sg_space_time_rel_l2','Space-time L2 versus SG'),
 ('bc_left_max_abs_densetime','Maximum left BC error'),
 ('bc_right_max_abs_densetime','Maximum right BC error'),
 ('independent_residual_rms','Independent PDE residual RMS'),
 ('residual_rms_area_999','Outflow RMS x>=0.999'),
 ('residual_rms_area_9999','Outflow RMS x>=0.9999'),
 ('dense_physics_loss','Dense PDE loss'),
 ('runtime_training_s','Training seconds'),
)


def summarize_pairs(root,baseline,candidates,profile_dir=None):
    """Pure post-hoc descriptive comparison; candidate training is already done."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=root/'comparison'
    target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for _,r in candidates.sort_values('seed').iterrows():
        b=baseline.loc[baseline.seed==int(r.seed)]
        if len(b)!=1:
            raise RuntimeError('Missing baseline paired seed')
        b=b.iloc[0]
        assert_candidate_matched(baseline,r.to_dict(),{},is_smoke=False)
        row={'seed':int(r.seed),
             'baseline_version':'2.3.1','candidate_version':VERSION,
             'same_collocation_sha256':r.sampling_sha256==b.sampling_sha256,
             'bc_feasible_baseline':bool(b.selected_bc_feasible),
             'bc_feasible_candidate':bool(r.selected_bc_feasible),
             'checkpoint_baseline':str(b.selected_checkpoint),
             'checkpoint_candidate':str(r.selected_checkpoint)}
        for k,_ in PAIR_METRICS:
            old=float(b[k]);new=float(r[k])
            row[k+'_baseline']=old
            row[k+'_candidate']=new
            row[k+'_difference_candidate_minus_baseline']=new-old
            row[k+'_ratio_candidate_over_baseline']=(new/old if old!=0 else np.nan)
        rows.append(row)
    paired=pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)
    core.atomic_csv(target/'paired_by_seed.csv',paired)
    summary=[]
    for k,label in PAIR_METRICS:
        if paired.empty:continue
        b=paired[k+'_baseline'].to_numpy(float)
        a=paired[k+'_candidate'].to_numpy(float)
        summary.append({'metric':k,'description':label,'n_pairs':len(paired),
                'baseline_median':float(np.median(b)),'candidate_median':float(np.median(a)),
                'median_paired_difference':float(np.median(a-b)),
                'candidate_better_count':int((a<b).sum()),
                'equal_count':int((a==b).sum()),'baseline_better_count':int((a>b).sum()),
                'direction':'smaller is better','statistical_inference':'none; n<=5 exploratory'})
    core.atomic_csv(target/'paired_metric_summary.csv',pd.DataFrame(summary))
    if len(paired)>0:
        graph_metrics=(('space_time_rel_l2','Global rel L2'),('final_rel_l2','Final rel L2'),
                       ('bc_left_max_abs_densetime','Left BC max'),
                       ('independent_residual_rms','Independent RMS'),
                       ('residual_rms_area_999','Outflow local RMS'),
                       ('dense_physics_loss','Dense PDE loss'))
        fig,axes=plt.subplots(2,3,figsize=(14.8,8.2),layout='constrained')
        for ax,(k,title) in zip(axes.flat,graph_metrics):
            xs=paired.seed.to_numpy(int)
            old=paired[k+'_baseline'].to_numpy(float)
            new=paired[k+'_candidate'].to_numpy(float)
            ax.plot(xs,old,'o-',label='Frozen V2.3.1')
            ax.plot(xs,new,'s--',label='Hard-both V2.3.2')
            ax.set_title(title);ax.set_xlabel('Seed');ax.grid(alpha=.25)
            ax.set_xticks(xs)
            if (old>0).all() and (new>0).all():ax.set_yscale('log')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'Paired MC-R output-transform ablation; {len(paired)}/5 seeds (descriptive only)')
        fig.savefig(target/'paired_comparison.png',dpi=155)
        plt.close(fig)
        if profile_dir is not None:
            profile_dir=Path(profile_dir)
            for seed in paired.seed.to_numpy(int):
                candidate_file=(root/'validation'/'runs'/f'{CANDIDATE_VARIANT}_mc_seed{seed:03d}'/
                                'diagnostics'/f'mc_nf500_seed{seed}'/'diagnostic_arrays.npz')
                base_file=profile_dir/f'baseline_seed{seed}.npz'
                if not (candidate_file.exists() and base_file.exists()):continue
                with np.load(candidate_file) as c, np.load(base_file) as b:
                    x=np.asarray(c['x']);t=np.asarray(c['t'])
                    if not np.array_equal(x,b['x']) or not np.array_equal(t,b['t']):
                        raise RuntimeError('Baseline / candidate profile grids mismatch')
                    fig,axes=plt.subplots(2,3,figsize=(13.5,7.2),layout='constrained')
                    for j,tm in enumerate((.1,.5,1.)):
                        idx=int(np.argmin(abs(t-tm)))
                        for row in range(2):
                            ax=axes[row,j]
                            s=slice(None) if row==0 else x>=.99
                            ax.plot(x[s],c['fdm'][idx,s],label='FDM',lw=1.6)
                            ax.plot(x[s],b['pinn'][idx,s],label='V2.3.1 baseline',ls='--')
                            ax.plot(x[s],c['pinn'][idx,s],label='V2.3.2 candidate',ls=':')
                            ax.set_title(f't={t[idx]:.2f}'+(' outflow zoom' if row else ''))
                            ax.set_xlabel('x');ax.grid(alpha=.2)
                    axes[0,0].legend(fontsize=8)
                    fig.suptitle(f'Frozen V2.3.1 vs V2.3.2; paired seed {seed}')
                    fig.savefig(target/f'paired_profiles_seed{seed}.png',dpi=140)
                    plt.close(fig)
    complete=set(paired.seed.astype(int))==set(REQUIRED_SEEDS)
    report={'n_completed_candidate':len(paired),'n_expected':5,
            'paired_complete':complete,'missing_seeds':sorted(set(REQUIRED_SEEDS)-set(paired.seed)),
            'all_sampling_hashes_match':bool(paired.same_collocation_sha256.all()) if len(paired) else False,
            'both_bc_feasible_candidate':int(paired.bc_feasible_candidate.sum()),
            'both_bc_feasible_baseline':int(paired.bc_feasible_baseline.sum()),
            'sampling_pairing_status':('all_matched' if bool(paired.same_collocation_sha256.all())
                                      else 'WARNING_different_collocation_realizations'),
            'summary':'This is a descriptive paired ablation of output representations, not a sampling-method comparison.',
            'no_new_baseline_training':True}
    core.atomic_json(target/'paired_integrity.json',report)
    return report


def _config(preset):
    cfg=v231.validation_config('smoke' if preset=='smoke' else 'validation')
    cfg['purpose']='V2.3.2 hard-left-BC output-transform ablation against frozen V2.3.1'
    return cfg


def _read_manifest(root,cfg,device,baseline_file,audit_nx,audit_nt):
    manifest={'version':VERSION,'candidate_variant':CANDIDATE_VARIANT,
        'experiment_design':'Single-output-transform ablation versus frozen V2.3.1',
        'cfg':cfg,'device':str(device),'runtime':core.runtime_info(device),
        'source_sha256':code_provenance(),
        'baseline_csv_sha256':v21.sha256_file(baseline_file),
        'baseline_retraining':False,
        'checkpoint_policy':'bc_feasible',
        'audit_nx':audit_nx,'audit_nt':audit_nt,
        'changed_factor':'u=sin(pi*x)+t*x*(N(x,t)-B_R(x)*N(1,t))',
        'notes':['The x factor changes interior NN correction amplitude too.',
                 'Hard IC and both BC are satisfied to floating-point precision.',
                 'All FDM, SG, checkpoint selection and curriculum schedule retained.',
                 'Warmup BC-only gradients are zero for hard-both model.']}
    path=root/'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=manifest:
            raise RuntimeError('Different manifest found. Use another output root (no mixed experiments).')
    else:
        core.atomic_json(path,manifest)


def run_ablation(args):
    cfg=_config(args.preset)
    root=args.output_root.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    output=root/args.preset
    output.mkdir(parents=True,exist_ok=True)
    device=core.device_from_arg(args.device)
    baseline, baseline_file=frozen_baseline(args.baseline_csv)
    seeds=v231.valid_seed_list(args.seeds if args.seeds is not None else ('0' if args.preset=='smoke' else '0-4'))
    if not set(seeds).issubset(set(REQUIRED_SEEDS)):
        raise ValueError('Only paired seeds 0..4 are allowed for V2.3.2')
    _read_manifest(output,cfg,device,baseline_file,args.audit_nx,args.audit_nt)
    reference_root=root/'references'
    reference_root.mkdir(parents=True,exist_ok=True)
    print('V2.3.2',args.preset,'hard-both, MC-R, seeds',seeds,'device',device,flush=True)
    ref,_=core.get_reference(cfg,reference_root)
    sg,report,_=v231.secondary_reference(cfg,ref,reference_root)
    print('SG vs FDM:',report['relative_l2_sg_minus_fdm'],flush=True)
    core.sampling_preflight(cfg,device,output/'sampling_preflight.json')
    for seed in seeds:
        run=output/'runs'/f'{CANDIDATE_VARIANT}_mc_seed{seed:03d}'
        diag=run/'diagnostics'/f'mc_nf{cfg["n_collocation"][0]}_seed{seed}'
        row_file=run/'results_raw.csv'
        if row_file.exists():
            stored=pd.read_csv(row_file)
            if len(stored)!=1 or int(stored.iloc[0].seed)!=seed:
                raise RuntimeError(f'Unexpected preexisting run at {run}')
            for name in ('final_model_state.pt','dense_outflow_residual_summary.json',
                         'dense_inflow_residual_summary.json'):
                if not (diag/name).is_file():
                    raise RuntimeError(f'Incomplete stored run; do not silently overwrite: {diag/name}')
            print('SKIP complete candidate seed',seed,flush=True)
            continue
        print('RUN candidate seed',seed,flush=True)
        tic=time.perf_counter()
        row=run_candidate(CANDIDATE_VARIANT,'mc',cfg,ref,sg,device,diag,'bc_feasible',seed=seed)
        checkpoint=diag/'final_model_state.pt'
        model=load_candidate(cfg,device,checkpoint)
        out=v231.dense_outflow_residual_audit(model,cfg,device,diag,args.audit_nx,args.audit_nt)
        left=dense_inflow_residual_audit(model,cfg,device,diag,args.audit_nx,args.audit_nt)
        row.update({'validation_version':VERSION,'run_preset':args.preset,
            'audit_x_points':args.audit_nx,'audit_t_points':args.audit_nt,
            'residual_rms_area_999':out['residual_rms_area_999'],
            'residual_rms_area_9999':out['residual_rms_area_9999'],
            'residual_rms_full_logmesh':out['residual_global_rms_on_log_grid'],
            'residual_rms_full_area':out['residual_global_rms_area_weighted'],
            'residual_rms_inflow_x_le_0001':left['rms_area_x_le_0001'],
            'residual_rms_inflow_x_le_001':left['rms_area_x_le_001'],
            'warmup_actually_updates_parameters':row['initial_state_sha256']!=row['warmup_state_sha256'],
            'validation_fdm_sg_used_for_training':False,
            'runtime_posttraining_total_s':time.perf_counter()-tic-row['runtime_training_s']})
        if args.preset=='validation':
            assert_candidate_matched(baseline,row,cfg)
        run.mkdir(parents=True,exist_ok=True)
        core.atomic_csv(row_file,pd.DataFrame([row]))
        print('DONE seed',seed,'rel L2',row['space_time_rel_l2'],
              'left BC max',row['bc_left_max_abs_densetime'],
              'elapsed',round(time.perf_counter()-tic,1),'s',flush=True)
    outputs=sorted((output/'runs').glob('*/results_raw.csv'))
    combined=pd.concat([pd.read_csv(path) for path in outputs],ignore_index=True)
    if combined.duplicated(['variant','method','seed']).any():
        raise RuntimeError('Duplicate candidate results')
    combined=combined.sort_values('seed').reset_index(drop=True)
    core.atomic_csv(output/'results_combined.csv',combined)
    expected=[(CANDIDATE_VARIANT,'mc',seed) for seed in seeds]
    v231._snapshot_summary(output,combined,cfg,expected)
    if args.preset=='validation':
        report=summarize_pairs(root,baseline,combined,
            Path(__file__).resolve().parent/'V231_Baseline_Profiles')
        print('PAIRED ABLATION:',len(combined),'/5','complete',report['paired_complete'],flush=True)
    else:
        print('SMOKE ONLY: baseline n=500 not compared with smoke n=48',flush=True)
    print('Output:',output,flush=True)


def compare_only(args):
    root=args.output_root.expanduser().resolve()
    baseline,_=frozen_baseline(args.baseline_csv)
    path=root/'validation'/'results_combined.csv'
    if not path.exists():
        print('No candidate validation results yet. Run --preset validation first.',flush=True)
        return
    df=pd.read_csv(path)
    report=summarize_pairs(root,baseline,df,
        Path(__file__).resolve().parent/'V231_Baseline_Profiles')
    print('Paired comparison:',report,flush=True)


def run_candidate_grid(args):
    """Optional post-hoc frozen-model grid refinement; never retrains or selects."""
    root=args.output_root.expanduser().resolve()
    run_dir=root/'validation'/'runs'
    if not run_dir.exists():
        raise RuntimeError('Run candidate validation BEFORE grid re-evaluation')
    runs=sorted(run_dir.glob('*/results_raw.csv'))
    if not runs:
        raise RuntimeError('No frozen candidate checkpoints found')
    baseline,_=frozen_baseline(args.baseline_csv)
    output=root/('grid_convergence_'+args.grid_profile)
    output.mkdir(parents=True,exist_ok=True)
    (root/'references').mkdir(parents=True,exist_ok=True)
    device=core.device_from_arg(args.device)
    cfg=v231.make_config('diagnostic','v31_staged')
    grids=v231.reference_grid_levels(args.grid_profile)
    refs=[];agreement=[]
    for nx,nt in grids:
        this=dict(cfg,reference_nx=nx,reference_nt=nt)
        print('GRID',nx,nt,flush=True)
        fdm,_=core.get_reference(this,root/'references')
        sg,_rep,_sig=v231.secondary_reference(this,fdm,root/'references')
        refs.append((this,fdm,sg))
        agreement.append(v231.same_grid_agreement(fdm,sg,nx,nt))
    core.atomic_csv(output/'solver_agreement.csv',pd.DataFrame(agreement))
    mesh_rows=[]
    for solver in ('FDM','SG'):
        for j in range(len(refs)-1):
            i=1 if solver=='FDM' else 2
            rec=v231._metrics_for_pair(refs[j][i],refs[j+1][i],
                     f'{solver}_{grids[j][0]}',f'{solver}_{grids[j+1][0]}')
            rec['solver']=solver;mesh_rows.append(rec)
    core.atomic_csv(output/'grid_refinement.csv',pd.DataFrame(mesh_rows))
    rows=[]
    for record in runs:
        meta=pd.read_csv(record).iloc[0]
        seed=int(meta.seed)
        diag=record.parent/'diagnostics'/f'mc_nf500_seed{seed}'
        model=load_candidate(cfg,device,diag/'final_model_state.pt')
        for this,fdm,sg in refs:
            for label,reference in [('FDM',fdm),('SG',sg)]:
                rows.append({'seed':seed,'ref':label,'reference_nx':int(this['reference_nx']),
                   'reference_nt':int(this['reference_nt']),
                   'model_family':CANDIDATE_VARIANT,'checkpoint_frozen':True,
                   'reference_used_in_selection':False,
                   **v231.relative_error_vs_reference(model,reference,this,device)})
    df=pd.DataFrame(rows)
    core.atomic_csv(output/'candidate_frozen_vs_grid_references.csv',df)
    baseline_file=Path(__file__).resolve().parent/'V231_Baseline_Grid_Frozen.csv'
    if baseline_file.exists():
        b=pd.read_csv(baseline_file)
        merged=df.merge(b,on=['seed','ref','reference_nx','reference_nt'],
                        suffixes=('_candidate','_baseline'),validate='one_to_one')
        if not merged.empty:
            metric='relative_l2'
            merged['relative_l2_difference_candidate_minus_baseline']=(
                merged[metric+'_candidate']-merged[metric+'_baseline'])
            core.atomic_csv(output/'paired_frozen_models_vs_grid_references.csv',merged)
    core.atomic_json(output/'grid_status.json',{
        'no_training':True,'reference_for_checkpoint_selection':False,
        'number_candidate_models':len(runs),'grids':grids,
        'notes':'Same PDE FDM/SG grid as V2.3.1; fine-grid check is post-hoc only.'})
    print('Frozen grid comparison saved to',output,flush=True)


def main():
    parser=argparse.ArgumentParser(description='V2.3.2 hard-left BC ablation versus frozen 5-seed V2.3.1')
    parser.add_argument('--preset',choices=('smoke','validation','compare','grid'),default='smoke')
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    parser.add_argument('--seeds',default=None,help='0, 0-4 or subset (only 0..4)')
    parser.add_argument('--output-root',type=Path,default=Path('pde_v232_outputs'))
    parser.add_argument('--baseline-csv',type=Path,default=None)
    parser.add_argument('--audit-nx',type=int,default=181)
    parser.add_argument('--audit-nt',type=int,default=61)
    parser.add_argument('--grid-profile',choices=('smoke','full'),default='smoke')
    args=parser.parse_args()
    if args.preset in ('smoke','validation'):
        run_ablation(args)
    elif args.preset=='compare':
        compare_only(args)
    else:
        run_candidate_grid(args)


if __name__=='__main__':
    main()
