#!/usr/bin/env python3
"""V235 controlled objective trainer: generated from frozen V233 run_candidate.
Only training-loss routing, fixed replay points, and read-only trace hooks differ.
All original validation functions and optimizer construction remain unchanged.
"""
from __future__ import annotations
from collections import deque
import hashlib
import copy
import time
import numpy as np
import pandas as pd
import torch
from convection_diffusion_pilot_v233 import (
    core, v21, v22, CANDIDATE_VARIANT, HardBothBoundaryLayerPINN,
    curriculum_weight, independent_residual_audit, sampler_snapshot)
VERSION = '2.3.5-sharedstart1'

def run_objective_candidate(variant,method,cfg,ref,sg_ref,device,diagdir,checkpoint_policy,seed=0,objective_context=None):
    if objective_context is None: raise ValueError("Objective context is required")
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
    shared=objective_context.shared_transition
    prefix=shared.begin(model,optimizer,initial_hash) if shared is not None else None
    warm_t=time.perf_counter()
    if prefix is None:
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
    else:
        model.load_state_dict(prefix['warm_state'],strict=True)
        hist.extend(copy.deepcopy([r for r in prefix['history'] if r['phase']=='warmup']))
    warm_s=time.perf_counter()-warm_t
    warm_state=v21.model_checkpoint(model)
    warm_hash=v21.state_hash(warm_state)
    recent=deque(maxlen=int(cfg["lbfgs_union_sets"]))
    sampler_seconds=0.0;validation_seconds=0.0
    focus_last=0
    sampler_updates=0
    sampler_trace=[]
    if cfg["transition_epochs"]>0:
        pts,focus_last=objective_context.points(nf,method,cfg,init_rng,global_rng,focus_rng,
                                    None,model,"transition",0,enriched)
        # This initial LHS design is frozen for all 900 transition epochs.
        hash_all.update(pts.tobytes());collocation_sets.append(pts)
        recent.append(pts)
        sampler_trace.append(sampler_snapshot(pts, phase='transition', epoch=0, method=method, adaptive=None, cfg=cfg))
        current=torch.from_numpy(pts).to(device)
    transition_start=time.perf_counter()
    if prefix is None:
        for e in range(1,cfg["transition_epochs"]+1):
            ic,icy=core.sample_ic(cfg["n_ic"],ic_rng,device)
            bc=core.sample_bc(cfg["n_bc"],bc_rng,device)
            w=curriculum_weight(cfg,e)
            optimizer.zero_grad(set_to_none=True)
            loss,pde,ic_l,bc_l=objective_context.full_losses(model,current,ic,icy,bc,cfg,w)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite transition loss")
            loss.backward();optimizer.step()
            if e==1 or e==cfg["transition_epochs"] or e%cfg["diagnostics_every"]==0:
                hist.append(dict(phase="main",curriculum_stage="transition",
                    global_epoch=cfg["warmup_epochs"]+e,main_epoch=e,
                    pde_weight=w,bc_weight=cfg["w_bc"],objective=float(loss.detach()),
                    pde_loss=float(pde.detach()),ic_loss=float(ic_l.detach()),
                    bc_loss=float(bc_l.detach()),val_full_loss=np.nan))
    else:
        hist.extend(copy.deepcopy([r for r in prefix['history'] if r['phase']!='warmup']))
        # Prefix diagnostics are copies of the one common computation, not new evaluations.
        for r in copy.deepcopy(prefix['objective_logs']):
            r['objective']=objective_context.objective
            r['shared_prefix_reused']=True
            objective_context.logs.append(r)
        for r in copy.deepcopy(prefix['gradient_logs']):
            r['shared_prefix_reused']=True
            objective_context.gradlogs.append(r)
    transition_s=time.perf_counter()-transition_start
    if shared is not None:
        generators={'warm_bc':warm_bc_rng,'bc':bc_rng,'ic':ic_rng,
                    'init':init_rng,'global':global_rng,'focus':focus_rng}
        objective_context.shared_start_audit=shared.finish(
            model,optimizer,generators,warm_state,hist,objective_context.logs,
            objective_context.gradlogs,objective_context.sequence[0]['points_sha256'],
            warm_s,transition_s)
    objective_context.after_transition(model)
    transition_s=time.perf_counter()-transition_start
    adaptive=None
    if method in ("rad","alhs"):
        adaptive=objective_context.make_proposal(cfg["adapt_grid_side"],cfg["eta"],
                   cfg["c"],cfg["D"],cfg["adapt_eval_chunk"],device)
    joint_start=time.perf_counter()
    for e in range(1,cfg["joint_epochs"]+1):
        if (e-1)%cfg["resample_every"]==0:
            tic=time.perf_counter()
            if adaptive is not None:
                adaptive.update(model)
                sampler_updates+=1
            pts,focus_last=objective_context.points(nf,method,cfg,init_rng,global_rng,focus_rng,
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
        loss,pde,ic_l,bc_l=objective_context.full_losses(model,current,ic,icy,bc,cfg,cfg["w_res"])
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
    objective_context.prepare_lbfgs(union)
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
            result=objective_context.full_losses(model,lb_pts,lb_ic,lb_y,lb_bc,cfg,cfg["w_res"])[0]
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
