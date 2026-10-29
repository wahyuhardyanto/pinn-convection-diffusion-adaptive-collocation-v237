#!/usr/bin/env python3
"""V2.3.7 unified four-sampler protocol.

One exact warmup+transition prefix is created per (N_f, seed), then MC-R, LHS-R,
RAD-IID and ALHS-R branch from the identical model, Adam state and RNG state.
All four optimize the same physical-uniform PDE residual objective. RAD/ALHS use
current-model graded101 proposals with 1/q importance correction; MC/LHS have q=1.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import convection_diffusion_objective_v235 as replay
import convection_diffusion_proposal_v234 as v234
import shared_transition_v235 as shared
from unified_context_v237 import UnifiedObjectiveContext, METHODS

VERSION='2.3.7-unified-four-sampler1'
HERE=Path(__file__).resolve().parent
core=replay.base.core
METRICS=('space_time_rel_l2','final_rel_l2','pinn_vs_sg_space_time_rel_l2','resolved_J',
         'resolved_right_001_rms','resolved_right_0001_rms','selected_bc_max_abs','independent_residual_rms')
PREFIX_KEYS=('shared_checkpoint_sha256','shared_model_sha256','transition_state_sha256',
             'transition_optimizer_sha256','transition_rng_sha256','transition_points_sha256')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x): v234.write_json(Path(p),x)

def config_for(stage,nf):
    cfg,_,_=replay.configuration('smoke' if stage=='smoke' else 'ablation')
    cfg=dict(cfg); cfg['n_collocation']=[int(nf)]
    if stage=='smoke' and int(nf)!=48: raise ValueError('Smoke budget is fixed at 48')
    return cfg

def runtime(device): return core.runtime_info(device)

@contextmanager
def output_lock(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); lock=root/'.v237.lock'
    try: fd=os.open(str(lock),os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError: raise RuntimeError('Output is locked; do not run concurrent notebooks: '+str(lock))
    os.write(fd,('pid='+str(os.getpid())+'\n').encode()); os.close(fd)
    try: yield
    finally: lock.unlink(missing_ok=True)

def prefix_signature(cfg,nf,seed,device):
    return dict(version=VERSION,seed=int(seed),n_collocation=int(nf),cfg=cfg,
                runtime=runtime(device),threads=torch.get_num_threads(),
                transition_design='method_independent_uniform_lhs',
                objective='importance_uniform_physical_measure',
                formulation='hard_both_bl_uniform')

def save_candidates(diag,states,row):
    table=pd.read_csv(diag/'checkpoint_candidates.csv')
    if len(states)!=2+len(table): raise RuntimeError('Candidate observer inventory mismatch')
    saved=diag/'candidate_states'; saved.mkdir(exist_ok=True)
    final=torch.load(diag/'final_model_state.pt',map_location='cpu',weights_only=True); inventory=[]
    for state,(_,r) in zip(states[2:],table.iterrows()):
        cid=str(r.checkpoint_id); p=saved/(cid+'.pt'); torch.save(state,p)
        inventory.append(dict(checkpoint_id=cid,file_sha256=sha(p),state_sha256=replay.base.v21.state_hash(state),selected=cid==row['selected_checkpoint']))
        if cid==row['selected_checkpoint'] and shared.tree_hash(state)!=shared.tree_hash(final):
            raise RuntimeError('Selected candidate and final model differ')
    pd.DataFrame(inventory).to_csv(diag/'candidate_state_inventory.csv',index=False)
    return len(table)

def make_manifest(folder,stage,budgets,seeds,device):
    body=dict(version=VERSION,stage=stage,
        effective_protocol=dict(methods=list(METHODS),budgets=list(map(int,budgets)),seeds=list(map(int,seeds)),
            formulation='hard_both_bl_uniform',constraint_mode='hard_ic_hard_left_hard_right',
            objective='importance_uniform',target_measure='physical_uniform',
            shared_prefix='one method-independent warmup+transition per budget-seed',
            transition_design='uniform LHS shared by four methods',
            mc='uniform IID q=1',lhs='uniform probability-space LHS q=1',
            rad='graded101 current-model IID + 1/q',alhs='graded101 current-model LHS + 1/q',
            manual_boundary_enrichment=False,checkpoint_selection='unchanged_grid31_bc_feasible',
            generation_pdf_frozen_per_set=True,importance_self_normalized=False,importance_clipped=False,
            lbfgs_policy='per-generation PDF; duplicate contributions summed'),
        runtime=runtime(device),threads=torch.get_num_threads(),
        source_code_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))})
    p=folder/'manifest.json'
    if p.exists() and json.loads(p.read_text())!=body: raise RuntimeError('Manifest changed; use a new output root')
    if not p.exists(): write_json(p,body)
    return body

def run_one(folder,stage,method,nf,seed,device):
    cfg=config_for(stage,nf)
    run=folder/'runs'/f'{method}_nf{nf}_seed{seed:03d}'
    if replay.check_existing(run,'completion.json'):
        print('SKIP',method,'nf',nf,'seed',seed,flush=True); return
    diag=run/'diagnostics'/f'{method}_nf{nf}_seed{seed}'
    if replay.check_existing(run,'training_complete.json'):
        row=json.loads((run/'training_result.json').read_text()); print('RESUME post-hoc',method,nf,seed,flush=True)
    else:
        if run.exists() and any(run.iterdir()): raise RuntimeError('Partial training found; preserve/move this run: '+str(run))
        run.mkdir(parents=True,exist_ok=True)
        ctx=UnifiedObjectiveContext(cfg,gradient_audit=True)
        pfx=folder/'shared_prefix'/f'nf{nf}_seed{seed:03d}'
        ctx.shared_transition=shared.SharedTransition(pfx,prefix_signature(cfg,nf,seed,device),device,'v237_unified_prefix')
        ref,sg=replay.references(cfg)
        print('START',method,'nf',nf,'seed',seed,'shared-prefix importance_uniform',flush=True)
        with replay.observe_states() as states:
            row=replay.run_objective_candidate(replay.base.CANDIDATE_VARIANT,method,cfg,ref,sg,device,diag,'bc_feasible',seed=seed,objective_context=ctx)
        ctx.save(diag); row.update(ctx.shared_start_audit); row['transition_state_sha256']=ctx.transition_hash
        nc=save_candidates(diag,states,row)
        adaptive=method in ('rad','alhs'); updates=(cfg['joint_epochs']-1)//cfg['resample_every']+1
        if adaptive:
            if ctx.proposal is None or ctx.proposal.updates!=updates or ctx.proposal.eval_points!=updates*cfg['adapt_grid_side']**2:
                raise RuntimeError('Adaptive online proposal evaluation count mismatch')
            proposal_mode='graded101_current_model_online'; proposal_evals=ctx.proposal.eval_points; proposal_seconds=ctx.proposal.eval_seconds
            dxmin=float(ctx.proposal.dx.min()); dxmax=float(ctx.proposal.dx.max()); xmax=float(ctx.proposal.candidates[:,0].max())
        else:
            proposal_mode='uniform_iid' if method=='mc' else 'uniform_lhs'; proposal_evals=0; proposal_seconds=0.0
            dxmin=dxmax=xmax=np.nan
        row.update(protocol_version=VERSION,unified_version=VERSION,method=method,n_collocation=int(nf),
            objective_mode='importance_uniform',target_measure='physical_uniform',proposal_mode=proposal_mode,
            proposal_profile='graded101' if adaptive else 'uniform_unit_square',training_importance_weighted=True,
            importance_self_normalized=False,importance_clipped=False,generation_pdf_frozen_per_set=True,
            generation_pdf_from_current_model=adaptive,lbfgs_weights_policy='per_generation_pdf_aggregate_duplicates',
            lbfgs_duplicate_occurrences=ctx.lbfgs_duplicate_count,online_proposal_evaluations=proposal_evals,
            adaptive_updates=ctx.proposal.updates if adaptive else 0,adaptive_cell_width_min=dxmin,adaptive_cell_width_max=dxmax,
            adaptive_candidate_xmax=xmax,proposal_residual_seconds=proposal_seconds,checkpoint_rule_changed=False,
            validation_uses_training_importance_weights=False,candidate_states_saved=nc,gradient_audit_calls=len(ctx.gradlogs),
            common_prefix_per_budget_seed=True,runtime_prefix_cached=bool(ctx.shared_start_audit.get('runtime_prefix_cached',False)))
        write_json(run/'training_result.json',row)
        required=[run/'training_result.json',diag/'final_model_state.pt',diag/'sampling_points_audit.csv',
                  diag/'collocation_generation_weights.npz',diag/'joint_start_control.json',diag/'candidate_state_inventory.csv',
                  diag/'optimizer_objective_trace.csv',diag/'same_state_gradient_comparison.csv',diag/'proposal_trace.csv']+list((diag/'candidate_states').glob('*.pt'))
        write_json(run/'training_complete.json',{'files':{str(p.relative_to(run)):sha(p) for p in required}})
    model=replay.base.v232.load_candidate(cfg,device,diag/'final_model_state.pt')
    out=replay.base.v231.dense_outflow_residual_audit(model,cfg,device,diag,181,61)
    left=replay.base.v232.dense_inflow_residual_audit(model,cfg,device,diag,181,61)
    res=v234.resolved_audit(diag/'final_model_state.pt',cfg,diag/'resolved_posthoc','smoke' if stage=='smoke' else 'ablation')
    row.update(residual_rms_area_999=out['residual_rms_area_999'],residual_rms_area_9999=out['residual_rms_area_9999'],
        residual_rms_inflow_x_le_001=left['rms_area_x_le_001'],resolved_J=res['result']['mean_r2'],resolved_order=res['result']['order'],
        resolved_converged=res['quadrature_converged'],resolved_right_001_rms=res['result']['right_001_rms'],
        resolved_right_0001_rms=res['result']['right_0001_rms'])
    replay.plot_objective_trace(diag)
    core.atomic_csv(run/'results_raw.csv',pd.DataFrame([row]))
    write_json(run/'completion.json',{'complete':True,'files':{str((run/'results_raw.csv').relative_to(run)):sha(run/'results_raw.csv')}})
    print('DONE',method,'nf',nf,'seed',seed,'L2=',row['space_time_rel_l2'],'J=',row['resolved_J'],flush=True)

def report(folder,stage,budgets,seeds):
    rows=[]
    for p in sorted((folder/'runs').glob('*/results_raw.csv')):
        if replay.check_existing(p.parent,'completion.json') and replay.check_existing(p.parent,'training_complete.json'): rows.append(pd.read_csv(p))
    d=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    expected={(m,int(n),int(s)) for n in budgets for s in seeds for m in METHODS}
    present=set() if d.empty else set(zip(d.method,d.n_collocation.astype(int),d.seed.astype(int)))
    if not present.issubset(expected) or (not d.empty and d.duplicated(['method','n_collocation','seed']).any()): raise RuntimeError('Unexpected/duplicate result rows')
    if not d.empty:
        for (nf,seed),g in d.groupby(['n_collocation','seed']):
            # only assert when >=2 branches exist; all must share exact joint start
            for k in PREFIX_KEYS:
                if g[k].astype(str).nunique()!=1: raise RuntimeError(f'Shared prefix mismatch {k} at nf={nf}, seed={seed}')
        if not d.selected_bc_feasible.all(): raise RuntimeError('A unified run selected a BC-infeasible checkpoint')
    complete=expected==present
    write_json(folder/'integrity.json',dict(version=VERSION,stage=stage,expected=len(expected),completed=len(d),complete=complete,
        missing=[list(x) for x in sorted(expected-present)],scope='unified four-sampler pilot; descriptive until confirmatory seed count is chosen'))
    if d.empty:return
    core.atomic_csv(folder/'results_unified.csv',d)
    summary=[]
    for (nf,method),g in d.groupby(['n_collocation','method']):
        for metric in METRICS:
            v=g[metric].astype(float)
            summary.append(dict(n_collocation=int(nf),method=method,metric=metric,n=len(v),median=float(v.median()),
                q25=float(v.quantile(.25)),q75=float(v.quantile(.75)),minimum=float(v.min()),maximum=float(v.max())))
    core.atomic_csv(folder/'summary.csv',pd.DataFrame(summary))
    # paired differences against MC and direct RAD-vs-ALHS; descriptive only here.
    pairs=[]
    for nf in budgets:
        sub=d[d.n_collocation==nf]
        for metric in METRICS:
            wide=sub.pivot(index='seed',columns='method',values=metric)
            for a,b in (('lhs','mc'),('rad','mc'),('alhs','mc'),('alhs','rad')):
                if a in wide and b in wide:
                    for seed,val in (wide[a]-wide[b]).items(): pairs.append(dict(n_collocation=nf,metric=metric,comparison=f'{a}-{b}',seed=int(seed),delta=float(val)))
    core.atomic_csv(folder/'paired_differences.csv',pd.DataFrame(pairs))

def parse_ints(spec,default):
    if spec is None:return list(default)
    out=[]
    for part in spec.split(','):
        part=part.strip()
        if '-' in part:
            a,b=map(int,part.split('-',1)); out.extend(range(a,b+1))
        else: out.append(int(part))
    if len(set(out))!=len(out): raise ValueError('Duplicate integer in specification')
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset',choices=('smoke','pilot','confirm','full','report'),default='smoke')
    ap.add_argument('--output-root',type=Path,default=Path('pde_v237_outputs'))
    ap.add_argument('--methods',default='all'); ap.add_argument('--budgets',default=None); ap.add_argument('--seeds',default=None)
    ap.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto'); ap.add_argument('--threads',type=int,default=8)
    args=ap.parse_args()
    if args.device=='auto': device=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    else: device=torch.device(args.device)
    torch.set_num_threads(args.threads)
    if args.preset=='smoke': budgets=[48]; seeds=[0]; stage='smoke'
    elif args.preset=='pilot': budgets=parse_ints(args.budgets,[250,500,1000]); seeds=parse_ints(args.seeds,[0,1,2]); stage='pilot'
    elif args.preset=='confirm': budgets=parse_ints(args.budgets,[250,500,1000]); seeds=parse_ints(args.seeds,range(5)); stage='confirm'
    elif args.preset=='full': budgets=parse_ints(args.budgets,[250,500,1000]); seeds=parse_ints(args.seeds,range(30)); stage='full'
    else:
        for fam in ('smoke','pilot','confirm','full'):
            f=args.output_root/fam
            if (f/'manifest.json').exists():
                mf=json.loads((f/'manifest.json').read_text()); report(f,fam,mf['effective_protocol']['budgets'],mf['effective_protocol']['seeds'])
        return
    methods=list(METHODS) if args.methods=='all' else [x.strip().lower() for x in args.methods.split(',')]
    if not methods or len(set(methods))!=len(methods) or not set(methods).issubset(METHODS): ap.error('methods must be all or subset of mc,lhs,rad,alhs')
    if stage!='smoke' and not set(budgets).issubset({250,500,1000}): ap.error('pilot/confirm/full budgets must be 250,500,1000')
    targets=[(m,n,s) for n in budgets for s in seeds for m in methods]
    with output_lock(args.output_root):
        folder=args.output_root/stage; folder.mkdir(parents=True,exist_ok=True)
        # manifest records full intended four-method stage even if methods subset is run incrementally
        make_manifest(folder,stage,budgets,seeds,device)
        for method,nf,seed in targets:
            run_one(folder,'smoke' if stage=='smoke' else 'pilot',method,nf,seed,device)
            report(folder,stage,budgets,seeds)
        print('RESULTS:',folder,flush=True)

if __name__=='__main__': main()
