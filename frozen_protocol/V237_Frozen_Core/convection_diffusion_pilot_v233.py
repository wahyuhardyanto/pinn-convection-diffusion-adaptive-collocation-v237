#!/usr/bin/env python3
"""V2.3.3: controlled four-sampler 5-seed pilot and V2.3.1 sensitivity.

Training routine copied from V2.3.2 run_candidate with sampler restriction lifted
and read-only sample-coverage statistics appended. Optimizers, seeds, losses,
curriculum, checkpoint policy and PDE reference are UNCHANGED.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import sys

import numpy as np
import pandas as pd
import torch

import convection_diffusion_benchmark_v2 as core
import convection_diffusion_benchmark_v21 as v21
import convection_diffusion_benchmark_v22 as v22
import convection_diffusion_benchmark_v23 as v23
import convection_diffusion_validation_v231 as v231
import convection_diffusion_ablation_v232 as v232

VERSION = '2.3.3'
CANDIDATE_VARIANT = v232.CANDIDATE_VARIANT
SENSITIVITY_VARIANT = 'bl_uniform'
METHODS = tuple(core.METHODS)
SEEDS = (0,1,2,3,4)
HardBothBoundaryLayerPINN = v232.HardBothBoundaryLayerPINN
method_points = v231.method_points
curriculum_weight = v231.curriculum_weight
independent_residual_audit = v231.independent_residual_audit


def sampler_snapshot(pts, phase, epoch, method, adaptive, cfg):
    """Descriptive sampling audit; no extra RNG calls or network evaluation."""
    x=np.asarray(pts[:,0],dtype=np.float64)
    t=np.asarray(pts[:,1],dtype=np.float64)
    delta=float(cfg['D'])/abs(float(cfg['c']))
    result=dict(phase=phase,epoch=int(epoch),method=method,n_points=len(pts),
        x_mean=float(x.mean()),t_mean=float(t.mean()),
        n_x_ge_099=int((x>=.99).sum()),n_x_ge_0999=int((x>=.999).sum()),
        n_x_ge_09999=int((x>=.9999).sum()),
        n_in_delta=int(((1-x)<=delta).sum()),
        min_outflow_distance=float((1-x).min()))
    if adaptive is not None:
        p=np.asarray(adaptive.prob,dtype=float)
        result.update(adaptive_updates=adaptive.updates,
            adaptive_last_x_cell_probability=float(p[:,-1].sum()),
            adaptive_entropy_nats=float(-(p*np.log(np.maximum(p,1e-300))).sum()),
            candidate_x_max=float(adaptive.candidates[:,0].max()),
            candidate_dx=float(1/adaptive.side))
    return result


def run_candidate(variant,method,cfg,ref,sg_ref,device,diagdir,checkpoint_policy,seed=0):
    # V2.3 training code retained; the seed is now an explicit run argument.
    nf=int(cfg["n_collocation"][0]);seed=int(seed)
    if variant != CANDIDATE_VARIANT or method not in core.METHODS:
        raise ValueError('V2.3.3 main pilot requires hard_both_bl_uniform and a valid sampler')
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
    sampler_trace=[]
    if cfg["transition_epochs"]>0:
        pts,focus_last=method_points(nf,method,cfg,init_rng,global_rng,focus_rng,
                                    None,model,"transition",0,enriched)
        # This initial LHS design is frozen for all 900 transition epochs.
        hash_all.update(pts.tobytes());collocation_sets.append(pts)
        recent.append(pts)
        sampler_trace.append(sampler_snapshot(pts, phase='transition', epoch=0, method=method, adaptive=None, cfg=cfg))
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
            sampler_trace.append(sampler_snapshot(pts, phase='joint', epoch=e, method=method, adaptive=adaptive, cfg=cfg))
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
    core.atomic_csv(diagdir/'sampler_coverage_by_refresh.csv', pd.DataFrame(sampler_trace))
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
        sample_trace_updates=len(sampler_trace),
        last_sample_x_ge_0999=sampler_trace[-1]['n_x_ge_0999'],
        last_sample_x_ge_099=sampler_trace[-1]['n_x_ge_099'],
        sample_x_ge_0999_total=int(sum(q['n_x_ge_0999'] for q in sampler_trace)),
        adaptive_candidate_xmax=(float((cfg['adapt_grid_side']-.5)/cfg['adapt_grid_side'])
                                 if method in ('rad','alhs') else np.nan),
        adaptive_cell_width=(float(1/cfg['adapt_grid_side'])
                             if method in ('rad','alhs') else np.nan),
        **metrics,**details,**more,**sg_metrics)




# -------------------------- V2.3.3 pilot orchestration ------------------------
METRICS = (
    'space_time_rel_l2', 'final_rel_l2', 'pinn_vs_sg_space_time_rel_l2',
    'dense_physics_loss', 'independent_residual_rms',
    'residual_rms_area_999', 'residual_rms_area_9999',
    'selected_bc_max_abs', 'runtime_training_s', 'runtime_sampling_s',
)
METRIC_TITLES = {
    'space_time_rel_l2':'Space-time rel L2',
    'final_rel_l2':'Final-time rel L2',
    'residual_rms_area_999':'Outflow RMS x>=0.999',
    'independent_residual_rms':'Independent PDE RMS',
}


def local_path(name):
    return Path(__file__).resolve().parent / name


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    modules={'v233':__file__, 'v232':v232.__file__, 'v231':v231.__file__,
             'v23':v23.__file__, 'v22':v22.__file__, 'v21':v21.__file__,
             'v2':core.__file__}
    return {k:sha256_file(v) for k,v in modules.items()}


def check_frozen_rows(family,cfg):
    name='Frozen_V232_Main_MC5.csv' if family=='main' else 'Frozen_V231_Sensitivity_MC5.csv'
    file=local_path(name)
    if not file.exists():
        raise FileNotFoundError(f'Required frozen experiment data missing: {file}')
    provenance=json.loads(local_path('Frozen_Provenance_V233.json').read_text(encoding='utf-8'))[family]
    if sha256_file(file)!=provenance['raw_results_sha256']:
        raise RuntimeError('Frozen MC-R data were modified since extraction: '+str(file))
    d=pd.read_csv(file)
    expected_variant=CANDIDATE_VARIANT if family=='main' else SENSITIVITY_VARIANT
    expected_version='2.3.2' if family=='main' else '2.3.1'
    if not (len(d)==5 and set(d.seed.astype(int))==set(SEEDS)
            and (d.method=='mc').all() and (d.variant==expected_variant).all()
            and (d.protocol_version.astype(str)==expected_version).all()
            and not d.duplicated(['seed','method']).any()):
        raise RuntimeError('Unexpected frozen MC-R rows for '+family)
    for key,val in [
        ('n_collocation',cfg['n_collocation'][0]),('warmup_epochs',cfg['warmup_epochs']),
        ('transition_epochs',cfg['transition_epochs']),('joint_epochs',cfg['joint_epochs']),
        ('lambda_residual',cfg['w_res']),('lambda_initial',cfg['w_ic']),
        ('lambda_boundary',cfg['w_bc'])]:
        if not np.allclose(d[key].astype(float).to_numpy(),float(val),atol=0,rtol=0):
            raise RuntimeError('Frozen MC-R protocol mismatch: '+key)
    manifest_name=('Frozen_V232_manifest.json' if family=='main'
                   else 'Frozen_V231_manifest.json')
    manifest=json.loads(local_path(manifest_name).read_text(encoding='utf-8'))
    old=manifest['cfg']
    # Non-algorithmic fields such as description/purpose are intentionally excluded.
    fields=('c','D','layers','n_ic','n_bc','n_collocation','warmup_epochs',
            'transition_epochs','joint_epochs','adam_lr','w_res','w_ic','w_bc',
            'warmup_bc_weight','lbfgs_max_iter','lbfgs_union_sets','resample_every',
            'validation_every','val_grid_side','adapt_grid_side','eta',
            'reference_nx','reference_nt','bl_delta_factor','checkpoint_bc_max_tol')
    for field in fields:
        if old.get(field)!=cfg.get(field):
            raise RuntimeError('Frozen '+family+' metadata differs: '+field+
                ': '+repr(old.get(field))+' != '+repr(cfg.get(field)))
    d=d.copy()
    d['pilot_data_origin']='frozen_historical_'+expected_version
    d['pilot_is_new_training']=False
    return d,manifest


def pilot_config(preset):
    cfg=v231.validation_config('smoke' if preset=='smoke' else 'validation')
    cfg['purpose']='V2.3.3 paired five-seed four-sampler pilot'
    if preset=='pilot' and (cfg['n_collocation']!=[500] or cfg['resample_every']!=250):
        raise AssertionError('Unexpected V2.3.1 pilot protocol')
    return cfg


def read_only_environment_warning(saved,device):
    prior=saved.get('runtime',{})
    now=core.runtime_info(device)
    keys=('device','torch','numpy','scipy','pandas','python')
    discrepancy={key:{'frozen':prior.get(key),'current':now.get(key)}
                 for key in keys if prior.get(key)!=now.get(key)}
    return discrepancy


def create_or_check_manifest(folder,body):
    path=folder/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old!=body:
            raise RuntimeError('Pilot manifest mismatch. Do not mix code / configs /'
                               ' devices; use a NEW --output-root. Folder: '+str(folder))
    else:
        core.atomic_json(path,body)


def run_path_for(root,family,method,seed):
    variant=CANDIDATE_VARIANT if family=='main' else SENSITIVITY_VARIANT
    return root/f'{variant}_{method}_seed{seed:03d}'


def make_diagnostics_dirs(run,method,nf,seed):
    return run/'diagnostics'/f'{method}_nf{nf}_seed{seed}'


def _load_trained_model(family,cfg,device,filename):
    if family=='main':
        return v232.load_candidate(cfg,device,filename)
    return v231.load_frozen_model(cfg,SENSITIVITY_VARIANT,device,filename)


def run_new_job(family,method,seed,cfg,ref,sg,device,run_dir,audit_nx,audit_nt):
    diag=make_diagnostics_dirs(run_dir,method,cfg['n_collocation'][0],seed)
    if diag.exists() and list(diag.iterdir()):
        raise RuntimeError('Partial prior diagnostics without completed results. '
                           'Inspect or move this run directory before retry: '+str(diag))
    started=time.perf_counter()
    if family=='main':
        row=run_candidate(CANDIDATE_VARIANT,method,cfg,ref,sg,device,diag,'bc_feasible',seed=seed)
    else:
        row=v231.run_one(SENSITIVITY_VARIANT,method,cfg,ref,sg,device,diag,'bc_feasible',seed=seed)
    model=_load_trained_model(family,cfg,device,diag/'final_model_state.pt')
    out=v231.dense_outflow_residual_audit(model,cfg,device,diag,audit_nx,audit_nt)
    left=v232.dense_inflow_residual_audit(model,cfg,device,diag,audit_nx,audit_nt)
    row.update({
        'pilot_version':VERSION,'pilot_formulation':family,'pilot_data_origin':'trained_v233',
        'pilot_is_new_training':True,'audit_x_points':audit_nx,'audit_t_points':audit_nt,
        'residual_rms_area_999':out['residual_rms_area_999'],
        'residual_rms_area_9999':out['residual_rms_area_9999'],
        'residual_rms_full_logmesh':out['residual_global_rms_on_log_grid'],
        'residual_rms_full_area':out['residual_global_rms_area_weighted'],
        'residual_rms_inflow_x_le_0001':left['rms_area_x_le_0001'],
        'residual_rms_inflow_x_le_001':left['rms_area_x_le_001'],
        'validation_fdm_sg_used_for_training':False,
        'runtime_posttraining_total_s':time.perf_counter()-started-row['runtime_training_s'],
    })
    if row['n_collocation']!=cfg['n_collocation'][0] or row['seed']!=seed:
        raise RuntimeError('Wrong training outcome metadata')
    core.atomic_csv(run_dir/'results_raw.csv',pd.DataFrame([row]))
    return row


def load_new_rows(path,family,preset,cfg):
    variant=CANDIDATE_VARIANT if family=='main' else SENSITIVITY_VARIANT
    files=sorted(path.glob('*/results_raw.csv'))
    if not files:
        return pd.DataFrame()
    df=pd.concat((pd.read_csv(f) for f in files),ignore_index=True)
    if (df.duplicated(['method','seed']).any() or not (df.variant==variant).all()
            or not set(df.method).issubset(METHODS)
            or not set(df.seed.astype(int)).issubset(SEEDS)
            or not (df.n_collocation==cfg['n_collocation'][0]).all()):
        raise RuntimeError('Duplicate/unexpected pilot run rows; inspect '+str(path))
    for f in files:
        r=pd.read_csv(f).iloc[0]
        diag=make_diagnostics_dirs(f.parent,str(r.method),cfg['n_collocation'][0],int(r.seed))
        for name in ('final_model_state.pt','dense_outflow_residual_summary.json',
                     'dense_inflow_residual_summary.json'):
            if not (diag/name).is_file():
                raise RuntimeError('Result marked complete but diagnostic is missing: '+str(diag/name))
        if family=='main' and not (diag/'sampler_coverage_by_refresh.csv').exists():
            raise RuntimeError('Main pilot sampling audit missing in '+str(diag))
    if 'pilot_is_new_training' not in df:
        raise RuntimeError('Pilot provenance column missing')
    return df


def aggregate_data(family,folder,cfg,preset,include_frozen=True):
    new=load_new_rows(folder/'runs',family,preset,cfg)
    if preset=='pilot' and include_frozen:
        frozen,_=check_frozen_rows(family,cfg)
        if not new.empty and (new.method=='mc').any():
            # User explicitly requested rerun; historic MC is not also mixed in.
            all_rows=new
        else:
            all_rows=pd.concat([frozen,new],ignore_index=True,sort=False)
    else:
        all_rows=new
    if all_rows.empty:
        return all_rows
    if all_rows.duplicated(['method','seed']).any():
        raise RuntimeError('A method/seed appears more than once in '+str(folder))
    all_rows=all_rows.sort_values(['method','seed']).reset_index(drop=True)
    core.atomic_csv(folder/'results_combined.csv',all_rows)
    return all_rows


def _descriptive_stats(z):
    z=np.asarray(z,float)
    if len(z)==0:
        return dict(n=0)
    return dict(n=len(z),median=float(np.median(z)),mean=float(np.mean(z)),
        q25=float(np.quantile(z,.25)),q75=float(np.quantile(z,.75)),
        minimum=float(z.min()),maximum=float(z.max()),
        std_sample=float(np.std(z,ddof=1)) if len(z)>1 else np.nan)


def summarize_family(root,family,all_rows,preset):
    output=root/('main_pilot' if family=='main' else 'sensitivity_pilot')
    output.mkdir(parents=True,exist_ok=True)
    missing=[{'method':m,'seed':s} for m in METHODS for s in SEEDS
             if all_rows.empty or not ((all_rows.method==m)&(all_rows.seed==s)).any()]
    complete=len(missing)==0 and len(all_rows)==20
    report=dict(preset=preset,formulation=family,
        expected_runs=20 if preset=='pilot' else 4,
        expected_frozen_mc=5 if preset=='pilot' else 0,
        expected_new_jobs=15 if preset=='pilot' else 4,
        n_completed=len(all_rows),
        missing_pairs=missing if preset=='pilot' else [],
        pilot_complete=complete if preset=='pilot' else False,
        frozen_main_mc_reused=(family=='main' and preset=='pilot' and
             not all_rows.empty and 'frozen_historical_2.3.2' in set(all_rows.pilot_data_origin)),
        inference='exploratory 5-seed descriptive comparison; do not report p-values',
        shared_collocation_across_methods=False,
        paired_design='matched seed and budget; adaptive model-dependent sampling differs by method',
        adaptive_candidate_grid_side=51,
        adaptive_candidate_dx_over_delta=19.6078431372549,
        warning='51x51 residual-evaluation grid cannot resolve a physical layer of width D/c=0.001; pilot is diagnostic, not definitive.')
    core.atomic_json(output/'integrity.json',report)
    if all_rows.empty:
        return report
    rows=[]
    for method in METHODS:
        subset=all_rows[all_rows.method==method]
        if subset.empty:
            continue
        meta={'formulation':family,'method':method,'n_seeds':len(subset),
              'n_bc_feasible':int(subset.selected_bc_feasible.astype(bool).sum()),
              'n_rel_l2_le_001':int((subset.space_time_rel_l2<=.01).sum()),
              'is_complete_5_seeds':len(subset)==5}
        for metric in METRICS:
            if metric not in subset:
                continue
            for k,v in _descriptive_stats(subset[metric].astype(float).to_numpy()).items():
                meta[f'{metric}_{k}']=v
        rows.append(meta)
    core.atomic_csv(output/'method_summary.csv',pd.DataFrame(rows))
    differences=[]
    for left,right in [('mc','lhs'),('mc','rad'),('mc','alhs'),('rad','alhs')]:
        a=all_rows[all_rows.method==left].set_index('seed')
        b=all_rows[all_rows.method==right].set_index('seed')
        for seed in sorted(set(a.index)&set(b.index)):
            row={'formulation':family,'method_a':left,'method_b':right,'seed':int(seed),
                 'direction':'method_b minus method_a; negative favors method_b'}
            for metric in METRICS:
                if metric in a and metric in b:
                    row[f'delta_{metric}']=float(b.loc[seed,metric]-a.loc[seed,metric])
            differences.append(row)
    core.atomic_csv(output/'paired_differences.csv',pd.DataFrame(differences))
    rankings=[]
    for seed,group in all_rows.groupby('seed'):
        if len(group)!=4:
            continue
        if len(set(group.method))!=4:
            raise RuntimeError('Duplicate method in ranking group')
        for i,m in enumerate(group.sort_values('space_time_rel_l2').method,1):
            rankings.append({'seed':int(seed),'method':m,'rank_global_l2':i})
    if rankings:
        core.atomic_csv(output/'rank_by_seed.csv',pd.DataFrame(rankings))
    plot_pilot(output,all_rows,family)
    return report


def plot_pilot(output,df,family):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12.3,8.5),layout='constrained')
    for ax,(col,title) in zip(axes.flat,METRIC_TITLES.items()):
        for m in METHODS:
            block=df[df.method==m].sort_values('seed')
            if block.empty: continue
            ax.plot(block.seed,block[col],marker='o',label=m.upper())
        ax.set_xticks(SEEDS);ax.set_xlabel('Seed')
        if len(set(df.seed.astype(int)))==1:
            only=int(df.seed.iloc[0]);ax.set_xlim(only-.4,only+.4)
        ax.set_title(title);ax.grid(alpha=.25)
        if (df[col].astype(float)>0).all():ax.set_yscale('log')
    axes[0,0].legend(fontsize=9)
    fig.suptitle(f'V2.3.3 {family}: paired-seed pilot, descriptive only')
    fig.savefig(output/'method_comparison.png',dpi=155)
    plt.close(fig)


def compare_formulations(root,preset):
    if preset!='pilot':return
    a=root/'main_pilot'/'results_combined.csv'
    b=root/'sensitivity_pilot'/'results_combined.csv'
    if not(a.exists() and b.exists()):return
    x=pd.read_csv(a);y=pd.read_csv(b)
    merged=x.merge(y,on=['method','seed'],suffixes=('_main','_sensitivity'),
                   validate='one_to_one')
    if merged.empty:return
    target=root/'formulation_sensitivity'
    target.mkdir(parents=True,exist_ok=True)
    out=merged[['method','seed']].copy()
    for metric in METRICS:
        first=metric+'_main';second=metric+'_sensitivity'
        if first in merged and second in merged:
            out[metric+'_main']=merged[first]
            out[metric+'_sensitivity']=merged[second]
            out[metric+'_main_minus_sensitivity']=merged[first]-merged[second]
    core.atomic_csv(target/'paired_formulation_contrasts.csv',out)
    summaries=[]
    for m,subset in out.groupby('method'):
        item={'method':m,'n_pairs':len(subset),
              'formulations_both_complete':len(subset)==5}
        for metric in METRICS:
            col=metric+'_main_minus_sensitivity'
            if col in subset:
                item[metric+'_median_difference']=float(np.median(subset[col].astype(float)))
        summaries.append(item)
    core.atomic_csv(target/'formulation_sensitivity_summary.csv',pd.DataFrame(summaries))
    sensitivity_complete=(len(out)==20 and set(out.method)==set(METHODS))
    metadata={'common_method_seed_rows':len(out),
        'all_4_methods_5_seeds_present':sensitivity_complete,
        'status_note':('The cross-formulation four-sampler sensitivity has all 20 pairs.'
             if sensitivity_complete else 'Partial sensitivity: complete the missing V2.3.1 sampler runs before comparing sampler rankings across formulations.')}
    core.atomic_json(target/'sensitivity_status.json',metadata)


def _run_family(root,family,args,device,cfg,methods,seeds):
    folder=root/('main_pilot' if family=='main' else 'sensitivity_pilot') if args.preset=='pilot' else root/'smoke'/family
    folder.mkdir(parents=True,exist_ok=True)
    saved={}
    if args.preset=='pilot':
        _,saved=check_frozen_rows(family,cfg)
    manifest={'pilot_version':VERSION,'preset':args.preset,'formulation':family,
        'pde_config':cfg,'runtime':core.runtime_info(device),
        'source_sha256':source_hashes(),
        'frozen_sources_sha256':{
            'main':sha256_file(local_path('Frozen_V232_Main_MC5.csv')),
            'sensitivity':sha256_file(local_path('Frozen_V231_Sensitivity_MC5.csv'))},
        'frozen_mc_reuse':args.preset=='pilot' and not args.rerun_mc,
        'diagnostic_grid':{'nx':args.audit_nx,'nt':args.audit_nt},
        'checkpoint_policy':'bc_feasible',
        'note':'One common training protocol and Nf, 5 paired seeds; method-dependent adaptive trajectories.'}
    create_or_check_manifest(folder,manifest)
    if args.preset=='pilot':
        diff=read_only_environment_warning(saved,device)
        core.atomic_json(folder/'frozen_environment_comparison.json',{
            'discrepancies':diff,'runtime_matched':len(diff)==0,
            'frozen_torch':saved.get('runtime',{}).get('torch'),
            'note':'Mismatch does not invalidate frozen data, but is a confound to report.'})
        if diff:
            print('WARNING historical/current environment mismatch for',family,':',diff,flush=True)
    tasks=[(method,seed) for method in methods for seed in seeds
           if not (args.preset=='pilot' and method=='mc' and not args.rerun_mc)]
    todo=[]
    for method,seed in tasks:
        run=run_path_for(folder/'runs',family,method,seed)
        row=run/'results_raw.csv'
        if row.exists():
            # load_new_rows validates everything before summarization
            print('SKIP completed',family,method,'seed',seed,flush=True)
        else:
            todo.append((method,seed,run))
    print('V2.3.3',args.preset,family,'selected',len(tasks),'new jobs, remaining',len(todo),flush=True)
    if args.preset=='pilot':
        print('CAUTION: adaptive candidate cell width=1/51, while D/c=.001; sampling resolution limits apply.',flush=True)
    if todo:
        refs_root=root/'references';refs_root.mkdir(parents=True,exist_ok=True)
        ref,_=core.get_reference(cfg,refs_root)
        sg,report,_=v231.secondary_reference(cfg,ref,refs_root)
        print('FDM vs SG relative error:',report['relative_l2_sg_minus_fdm'],flush=True)
        core.sampling_preflight(cfg,device,folder/'sampling_preflight.json')
        for method,seed,run in todo:
            print('START',family,method,seed,flush=True)
            tic=time.perf_counter()
            row=run_new_job(family,method,seed,cfg,ref,sg,device,run,args.audit_nx,args.audit_nt)
            print('FINISH',family,method,seed,'global L2=',row['space_time_rel_l2'],
                'local RMS=',row['residual_rms_area_999'],
                'seconds=',round(time.perf_counter()-tic,1),flush=True)
    df=aggregate_data(family,folder,cfg,args.preset,include_frozen=not args.rerun_mc)
    report=summarize_family(root,family,df,args.preset) if args.preset=='pilot' else summarize_smoke(folder,df,methods,seeds)
    print('REPORT',family,report.get('n_completed'),'/20 if pilot, complete',report.get('pilot_complete'),flush=True)


def summarize_smoke(folder,df,methods,seeds):
    expected={(m,s) for m in methods for s in seeds}
    observed={(m,int(s)) for m,s in zip(df.method,df.seed)} if not df.empty else set()
    report={'preset':'smoke','n_completed':len(df),'expected_requested':len(expected),
            'complete_requested':expected.issubset(observed),
            'missing_requested':sorted([list(x) for x in expected-observed]),
            'note':'Smoke Nf48/short training is NOT compared with archived Nf500 pilot data.'}
    core.atomic_json(folder/'integrity.json',report)
    if not df.empty:
        plot_pilot(folder,df,'smoke')
    return report


def report_only(root,args):
    cfg=pilot_config('pilot')
    for family in ('main','sensitivity'):
        folder=root/('main_pilot' if family=='main' else 'sensitivity_pilot')
        if not folder.exists():
            if family=='main': folder.mkdir(parents=True,exist_ok=True)
            else:continue
        if not (folder/'manifest.json').exists():
            # Reconstruct summaries from frozen data without claiming training.
            df,_=check_frozen_rows(family,cfg)
            core.atomic_csv(folder/'results_combined.csv',df)
        else:
            frozen_enabled=json.loads((folder/'manifest.json').read_text())['frozen_mc_reuse']
            df=aggregate_data(family,folder,cfg,'pilot',include_frozen=frozen_enabled)
        rep=summarize_family(root,family,df,'pilot')
        print('Recomputed',family,'n',rep['n_completed'],flush=True)
    compare_formulations(root,'pilot')


def main():
    p=argparse.ArgumentParser(description='V2.3.3: 4 samplers x 5 seeds, V2.3.2 main + V2.3.1 sensitivity')
    p.add_argument('--preset',choices=['smoke','pilot','report'],default='smoke')
    p.add_argument('--formulations',choices=['main','sensitivity','both'],default='main',
                   help='main = V2.3.2 hard both; sensitivity = V2.3.1 soft left; both runs both.')
    p.add_argument('--methods',default='all',help='mc,lhs,rad,alhs or all')
    p.add_argument('--seeds',default=None,help='0-4 or a subset, default smoke=0 pilot=0-4')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--output-root',type=Path,default=Path('pde_v233_outputs'))
    p.add_argument('--audit-nx',type=int,default=181)
    p.add_argument('--audit-nt',type=int,default=61)
    p.add_argument('--rerun-mc',action='store_true',help='Ignore frozen MC5 and train MC anew. Use a new output root.')
    args=p.parse_args()
    root=args.output_root.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    if args.preset=='report':
        report_only(root,args);return
    methods=METHODS if args.methods=='all' else tuple(s.strip().lower() for s in args.methods.split(','))
    if not methods or len(set(methods))!=len(methods) or not set(methods).issubset(METHODS):
        raise ValueError('Methods must be distinct subset of mc,lhs,rad,alhs')
    seeds=v231.valid_seed_list(args.seeds if args.seeds is not None else ('0' if args.preset=='smoke' else '0-4'))
    if not set(seeds).issubset(SEEDS):
        raise ValueError('V2.3.3 seeds must be 0..4')
    if args.audit_nx<70 or args.audit_nt<10:
        raise ValueError('Audit grid requires at least 70 space x 10 time points.')
    cfg=pilot_config(args.preset)
    device=core.device_from_arg(args.device)
    for family in (('main','sensitivity') if args.formulations=='both' else (args.formulations,)):
        _run_family(root,family,args,device,cfg,methods,seeds)
    if args.preset=='pilot':
        compare_formulations(root,'pilot')
    print('ALL DONE. Results at',root,flush=True)


if __name__=='__main__':
    main()
