#!/usr/bin/env python3
"""V2.3.4: ten paired-seed proposal-resolution ablations, not a new PINN model.

Calls V2.3.3 run_new_job/run_candidate WITHOUT editing their source. Within a
single-process scoped adapter only the proposal constructor is replaced. Read-only
observers retain candidate states and probability maps for post-training audit.
The old 31x31 validation/checkpoint rule, FDM/SG and optimizer are unchanged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import convection_diffusion_pilot_v233 as base
import audit_resolution_v233 as audit
from nonuniform_proposal_v234 import AreaResidualAdaptive2D, SIDE, PROFILE

VERSION='2.3.4'
HERE=Path(__file__).resolve().parent
METHODS=('rad','alhs')
SEEDS=tuple(range(5))
core=base.core
METRICS=('space_time_rel_l2','final_rel_l2','pinn_vs_sg_space_time_rel_l2',
         'dense_physics_loss','independent_residual_rms','residual_rms_area_999',
         'residual_rms_area_9999','selected_bc_max_abs','resolved_J',
         'resolved_right_001_rms','resolved_right_0001_rms',
         'runtime_training_s','runtime_sampling_s')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_clean(obj):
    if isinstance(obj,dict):return {str(k):json_clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [json_clean(v) for v in obj]
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(np.bool_,)):return bool(obj)
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,Path):return str(obj)
    return obj


def write_json(path,obj):
    core.atomic_json(Path(path),json_clean(obj))


def load_frozen():
    p=json.loads((HERE/'Provenance_V234.json').read_text())
    for name,digest in p['frozen_files'].items():
        if sha(HERE/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
    for key,name in p['unchanged_training_modules'].items():
        if sha(HERE/name)!=p['source_sha256'][key]:
            raise RuntimeError('Baseline training module changed: '+name)
    if sha(HERE/'audit_resolution_v233.py')!=p['audit_module_sha256']:
        raise RuntimeError('Resolved audit module changed')
    for item in p['baseline_trace_provenance'].values():
        # Lookup by digest below; each CSV is separately certified by package hashes.
        if not item['sha256']:raise RuntimeError('Missing trace digest')
    for key,item in p['baseline_trace_provenance'].items():
        m,s=key.split('_');name=HERE/'baseline_sampling_traces'/f'{m}_seed{s}.csv'
        if sha(name)!=item['sha256']:raise RuntimeError('Frozen trace changed: '+str(name))
    d=pd.read_csv(HERE/'Frozen_V233_Main20.csv')
    expected={(m,s) for m in ('mc','lhs','rad','alhs') for s in SEEDS}
    if len(d)!=20 or d.duplicated(['method','seed']).any() or set(zip(d.method,d.seed.astype(int)))!=expected:
        raise RuntimeError('Frozen baseline is not a complete 20-run main pilot')
    if not (d.variant==base.CANDIDATE_VARIANT).all() or not (d.n_collocation==500).all():
        raise RuntimeError('Frozen baseline has incompatible model or budget')
    resolved=pd.read_csv(HERE/'Frozen_V233_Resolved_Audit40.csv')
    resolved=resolved[resolved.formulation=='main'].copy()
    mapping={'mean_r2':'resolved_J','right_001_rms':'resolved_right_001_rms',
             'right_0001_rms':'resolved_right_0001_rms'}
    d=d.merge(resolved[['method','seed']+list(mapping)].rename(columns=mapping),
              on=['method','seed'],validate='one_to_one')
    d['proposal_profile']='uniform51';d['ablation_origin']='historical_v233'
    return d,json.loads((HERE/'Frozen_V233_Manifest.json').read_text()),p


def config(preset):
    cfg=base.pilot_config('smoke' if preset=='smoke' else 'pilot')
    if preset!='smoke':
        _,manifest,_=load_frozen()
        if cfg!=manifest['pde_config']:
            changes={k:(manifest['pde_config'].get(k),cfg.get(k)) for k in set(cfg)|set(manifest['pde_config']) if cfg.get(k)!=manifest['pde_config'].get(k)}
            raise RuntimeError('Unexpected training configuration drift: '+repr(changes))
    cfg=copy.deepcopy(cfg)
    cfg['adapt_grid_side']=SIDE # the ONLY changed configuration field
    return cfg


@contextmanager
def proposal_adapter(factory=AreaResidualAdaptive2D):
    """Single-process temporary adapter; never changes source files on disk.

    Observers do not call RNG or evaluate the network. Candidate tensors returned
    by the original model_checkpoint are reused without changing their contents.
    """
    originals=(core.ResidualAdaptive2D,base.sampler_snapshot,base.v21.model_checkpoint)
    observed={'samplers':[],'states':[],'transition_points_sha256':None}
    def constructor(*args,**kwargs):
        obj=factory(*args,**kwargs);observed['samplers'].append(obj);return obj
    def snapshot(pts,phase,epoch,method,adaptive,cfg):
        row=originals[1](pts,phase,epoch,method,adaptive,cfg)
        digest=hashlib.sha256(pts.tobytes()).hexdigest()
        row['sample_points_sha256']=digest
        if phase=='transition':observed['transition_points_sha256']=digest
        if isinstance(adaptive,AreaResidualAdaptive2D):row.update(adaptive.snapshot())
        return row
    def state_copy(model):
        state=originals[2](model);observed['states'].append(state);return state
    core.ResidualAdaptive2D=constructor
    base.sampler_snapshot=snapshot
    base.v21.model_checkpoint=state_copy
    try:yield observed
    finally:
        core.ResidualAdaptive2D,base.sampler_snapshot,base.v21.model_checkpoint=originals


def save_observers(diag,observed,row):
    if len(observed['samplers'])!=1:raise RuntimeError('Expected one proposal per run')
    sampler=observed['samplers'][0]
    if not isinstance(sampler,AreaResidualAdaptive2D):raise RuntimeError('Wrong proposal class')
    sampler.save_history(diag/'proposal_history.npz')
    candidates=pd.read_csv(diag/'checkpoint_candidates.csv')
    states=observed['states']
    # Original run_candidate calls model_checkpoint for initial, warmup, then
    # each successful Adam/L-BFGS validation candidate in table order.
    if len(states)!=2+len(candidates):raise RuntimeError('Candidate observer count mismatch')
    if base.v21.state_hash(states[0])!=row['initial_state_sha256'] or base.v21.state_hash(states[1])!=row['warmup_state_sha256']:
        raise RuntimeError('Initial/warmup state observer identity mismatch')
    target=diag/'candidate_states';target.mkdir(exist_ok=True)
    rows=[]
    selected=torch.load(diag/'final_model_state.pt',map_location='cpu',weights_only=True)
    for state,(_,record) in zip(states[2:],candidates.iterrows()):
        cid=str(record.checkpoint_id)
        if cid!='lbfgs' and not (cid.startswith('adam_') and cid[5:].isdigit()):
            raise ValueError('Unexpected checkpoint ID')
        dest=target/(cid+'.pt');torch.save(state,dest)
        state_sha=base.v21.state_hash(state)
        rows.append({'checkpoint_id':cid,'state_sha256':state_sha,'file_sha256':sha(dest),
                     'selected':cid==row['selected_checkpoint']})
        if cid==row['selected_checkpoint'] and state_sha!=base.v21.state_hash(selected):
            raise RuntimeError('Selected checkpoint does not match observer state')
    core.atomic_csv(diag/'candidate_state_inventory.csv',pd.DataFrame(rows))
    x,y=sampler.x_edges,sampler.t_edges
    write_json(diag/'proposal_grid.json',{
        'profile':PROFILE,'x_edges':x.tolist(),'t_edges':y.tolist(),
        'space_grading':'25 cells near each boundary, 51 interior; geometry only',
        'residual_mean':'sum(area * abs(residual)) / sum(area)',
        'cell_mass':'q * cell_area / sum(q * cell_area)',
        'training_loss':'unchanged unweighted collocation residual mean',
        'updates':sampler.updates,'candidate_evaluations':sampler.eval_points,
        'probability_map_is_actual_training_history':True,
        'history_saved_after_training':True,
        'candidate_sides':[len(x)-1,len(y)-1],
        'dx_min':float(np.diff(x).min()),'dx_max':float(np.diff(x).max())})
    plot_proposal_history(diag,sampler)
    row.update(proposal_profile=PROFILE,ablation_version=VERSION,
        ablation_origin='trained_v234',training_source_version='2.3.3',
        adaptive_cell_width=float('nan'),adaptive_cell_width_min=float(sampler.dx.min()),
        adaptive_cell_width_max=float(sampler.dx.max()),
        adaptive_candidate_xmax=float(sampler.candidates[:,0].max()),
        proposal_candidate_points_per_refresh=len(sampler.candidates),
        proposal_area_weighted=True,training_importance_weighted=False,
        checkpoint_rule_changed=False,
        transition_points_sha256=observed['transition_points_sha256'],
        candidate_states_saved=len(candidates),
        proposal_prob_x_ge_0999_last=sampler.band_probability(.999,1),
        proposal_grid_sha256=sampler.grid_sha256)
    # Training code declares version 2.3.3: explicitly retain that field. The
    # new fields above identify V2.3.4 instead of silently rewriting provenance.
    return row


def resolved_audit(checkpoint,cfg,target,preset):
    """Post-hoc CPU/float64, SAME quadrature as frozen audit. Not for selection."""
    target.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=sha(checkpoint)
    orders=[4,6] if preset=='smoke' else [6,8]
    signature={'checkpoint_sha256':checkpoint_hash,'orders':orders,'tolerance_J':1e-3,
               'module_sha256':sha(HERE/'audit_resolution_v233.py')}
    done=target/'resolved_audit.json'
    if done.exists():
        saved=json.loads(done.read_text())
        if saved['signature']!=signature:raise RuntimeError('Resolved-audit resume mismatch')
        return saved
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    before=base.v21.state_hash(state)
    rng=torch.get_rng_state();oldthreads=torch.get_num_threads()
    start=time.perf_counter();values=[]
    try:
        torch.set_num_threads(1)
        model=audit.FrozenPINN(state,'main',torch.float64)
        for order in orders:
            x,t,pts,w=audit.quadrature(order)
            r=audit.residual(model,pts,2048)
            values.append(dict(order=order,**audit.summary_integrals(x,t,r,w)))
        difference=abs(values[-1]['mean_r2']/max(values[-2]['mean_r2'],1e-300)-1)
        # Fixed escalation criterion, no method-dependent tuning.
        if difference>1e-3 and preset!='smoke':
            x,t,pts,w=audit.quadrature(12);r=audit.residual(model,pts,2048)
            values.append(dict(order=12,**audit.summary_integrals(x,t,r,w)))
            difference=abs(values[-1]['mean_r2']/max(values[-2]['mean_r2'],1e-300)-1)
        if base.v21.state_hash(state)!=before or sha(checkpoint)!=checkpoint_hash:
            raise RuntimeError('Checkpoint changed during evaluation')
    finally:
        torch.set_num_threads(oldthreads);torch.set_rng_state(rng)
    output={'signature':signature,'training_used':False,'checkpoint_reselected':False,
        'runtime':{'torch':torch.__version__,'device':'cpu','dtype':'float64'},
        'seconds':time.perf_counter()-start,'J_relative_change_last':difference,
        'quadrature_converged':difference<=1e-3,'orders_evaluated':[v['order'] for v in values],
        'result':values[-1]}
    core.atomic_csv(target/'quadrature_convergence.csv',pd.DataFrame(values))
    write_json(done,output)
    return output


def plot_proposal_history(diag,sampler):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trace=pd.read_csv(diag/'sampler_coverage_by_refresh.csv')
    tr=trace[trace.phase=='joint']
    fig,ax=plt.subplots(figsize=(8.4,4.7))
    ax.plot(tr.epoch,tr.n_x_ge_0999,marker='o',label='Observed points x >= 0.999')
    ax.plot(tr.epoch,tr.proposal_prob_x_ge_0999*tr.n_points,marker='s',label='Expected under proposal')
    ax.set_xlabel('Joint epoch');ax.set_ylabel('Points per active collocation set');ax.grid(alpha=.25)
    ax.set_title('Boundary coverage: same active budget, nonuniform proposal');ax.legend()
    fig.tight_layout();fig.savefig(diag/'proposal_boundary_coverage.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4.7))
    marginal=sampler.prob.sum(axis=0)/sampler.dx
    xc=(sampler.x_edges[:-1]+sampler.x_edges[1:])/2
    ax.plot(1-xc,marginal,marker='.',label='Final refresh: marginal spatial density')
    ax.set_xscale('log');ax.set_xlabel('Distance 1-x from outflow boundary');ax.set_ylabel('Density')
    ax.set_title('Final TRAINING refresh proposal (not recomputed on final checkpoint)')
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(diag/'proposal_outflow_density.png',dpi=145);plt.close(fig)


def pairing_check(row,baseline,preset,environment_matched):
    if preset=='smoke':return {'compared_to_historical_pilot':False}
    prior=baseline[(baseline.method==row['method'])&(baseline.seed==row['seed'])]
    if len(prior)!=1:raise RuntimeError('Missing historical paired run')
    prior=prior.iloc[0]
    h={k:bool(row[k]==prior[k]) for k in ('initial_state_sha256','warmup_state_sha256')}
    trace=pd.read_csv(HERE/'baseline_sampling_traces'/f"{row['method']}_seed{row['seed']}.csv")
    # Historical traces have descriptive statistics, not point hashes. Exact
    # initial design is recreated with the frozen RNG algorithm and cfg, while
    # trace moments are compared directly as an additional check.
    cfg=config('ablation');nf=500;seed=int(row['seed'])
    rng=np.random.default_rng(np.random.SeedSequence([seed,nf,990]))
    pts=core.uniform_lhs(nf,rng).astype(np.float32)
    h['transition_design_recreated']=hashlib.sha256(pts.tobytes()).hexdigest()==row['transition_points_sha256']
    initial=trace[trace.phase=='transition'].iloc[0]
    h['transition_moments_match_historical']=bool(np.isclose(pts[:,0].astype(float).mean(),initial.x_mean,rtol=0,atol=1e-14) and np.isclose(pts[:,1].astype(float).mean(),initial.t_mean,rtol=0,atol=1e-14))
    result={'compared_to_historical_pilot':True,'checks':h,'all_match':all(h.values()),
        'environment_matched':environment_matched,
        'joint_points_expected_to_differ':True,
        'historical_transition_hash_not_available':True}
    if environment_matched and not all(h.values()):
        raise RuntimeError('Pairing failed before adaptive phase: '+repr(h))
    return result


def check_manifest(folder,manifest):
    path=folder/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=json_clean(manifest):
        raise RuntimeError('Configuration / code / runtime changed; use a NEW output root: '+str(folder))
    if not path.exists():write_json(path,manifest)


def result_tables(folder,preset,baseline):
    rows=[];complete=[]
    for p in sorted((folder/'runs').glob('*/completion.json')):
        record=json.loads(p.read_text());d=p.parent
        if not record.get('complete'):continue
        for name,digest in record['required_sha256'].items():
            if not (d/name).is_file() or sha(d/name)!=digest:
                raise RuntimeError('Completed output missing/modified: '+str(d/name))
        training=json.loads((d/'training_complete.json').read_text())
        for name,digest in training['required_sha256'].items():
            if not (d/name).is_file() or sha(d/name)!=digest:
                raise RuntimeError('Archived training data missing/modified: '+str(d/name))
        block=pd.read_csv(d/'results_raw.csv')
        if len(block)!=1:raise RuntimeError('Run must have exactly one result row')
        r=block.iloc[0]
        expected_nf=48 if preset=='smoke' else 500
        if (int(r.n_collocation)!=expected_nf or r.variant!=base.CANDIDATE_VARIANT or
            str(r.ablation_version)!=VERSION or r.proposal_profile!=PROFILE):
            raise RuntimeError('Result preset/model/proposal mismatch: '+str(d))
        if not all(np.isfinite(float(r[k])) for k in METRICS):
            raise RuntimeError('Nonfinite reported metric: '+str(d))
        rows.append(block);complete.append((str(block.method.iloc[0]),int(block.seed.iloc[0])))
    df=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if len(complete)!=len(set(complete)):raise RuntimeError('Duplicate run identities')
    expected={(m,s) for m in METHODS for s in ((0,) if preset=='smoke' else SEEDS)}
    if not set(complete).issubset(expected):raise RuntimeError('Unexpected result key')
    write_json(folder/'integrity.json',{'version':VERSION,'preset':preset,
        'new_expected':len(expected),'new_completed':len(df),'missing':[list(k) for k in sorted(expected-set(complete))],
        'complete':set(complete)==expected,'historical_rows_not_counted_as_new':True,
        'training_protocol_unchanged_except_proposal':True,'checkpoint_selection_grid_side':11 if preset=='smoke' else 31,
        'scientific_scope':'5-seed exploratory paired proposal-resolution ablation; no inferential p-values'})
    if df.empty:return df
    core.atomic_csv(folder/'results_new.csv',df)
    meta=[]
    for m,b in df.groupby('method'):
        r={'method':m,'n_new':len(b)}
        for metric in METRICS:
            z=b[metric].astype(float)
            r[metric+'_median']=float(z.median());r[metric+'_q25']=float(z.quantile(.25));r[metric+'_q75']=float(z.quantile(.75))
        meta.append(r)
    core.atomic_csv(folder/'summary_new.csv',pd.DataFrame(meta))
    # Never compare short smoke results against full historical training.
    if preset=='smoke':
        make_plots(folder,df,None);return df
    prior=baseline[baseline.method.isin(METHODS)].copy()
    joined=prior.merge(df,on=['method','seed'],suffixes=('_old','_new'),validate='one_to_one')
    out=joined[['method','seed']].copy()
    for metric in METRICS:
        out[metric+'_old']=joined[metric+'_old'];out[metric+'_new']=joined[metric+'_new']
        out[metric+'_delta_new_minus_old']=joined[metric+'_new']-joined[metric+'_old']
        den=joined[metric+'_old'].replace(0,np.nan)
        out[metric+'_ratio_new_over_old']=joined[metric+'_new']/den
    core.atomic_csv(folder/'paired_proposal_comparison.csv',out)
    # 20 old main results + up to 10 new ablation results. This is NOT a
    # single 30-run four-method experiment; proposal_profile is part of the key.
    allrows=pd.concat([baseline,df],ignore_index=True,sort=False)
    if allrows.duplicated(['method','seed','proposal_profile']).any():raise RuntimeError('Ambiguous merged key')
    core.atomic_csv(folder/'baseline20_plus_ablation.csv',allrows)
    summaries=[]
    for m,b in out.groupby('method'):
        for metric in METRICS:
            z=b[metric+'_delta_new_minus_old']
            summaries.append({'method':m,'metric':metric,'n_pairs':len(b),
                'median_old':float(b[metric+'_old'].median()),'median_new':float(b[metric+'_new'].median()),
                'median_paired_delta':float(z.median()),'median_paired_ratio':float(b[metric+'_ratio_new_over_old'].median()),
                'n_lower_new':int((z<0).sum()),'n_equal':int((z==0).sum())})
    core.atomic_csv(folder/'paired_summary.csv',pd.DataFrame(summaries))
    a=df[df.method=='rad'];b=df[df.method=='alhs']
    pair=a.merge(b,on='seed',suffixes=('_rad','_alhs'),validate='one_to_one')
    if len(pair):
        ra=pair[['seed']].copy()
        for metric in METRICS:
            ra[metric+'_rad']=pair[metric+'_rad'];ra[metric+'_alhs']=pair[metric+'_alhs']
            ra[metric+'_alhs_minus_rad']=pair[metric+'_alhs']-pair[metric+'_rad']
        core.atomic_csv(folder/'paired_RAD_vs_ALHS_graded101.csv',ra)
    make_plots(folder,df,prior)
    return df


def make_plots(folder,df,old):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=folder/'figures';figdir.mkdir(exist_ok=True)
    for metric,title in [('space_time_rel_l2','Space-time relative L2'),('final_rel_l2','Final-time relative L2'),('resolved_J','Post-hoc physical integral of residual squared'),('runtime_sampling_s','Proposal and sampling time (s)')]:
        fig,ax=plt.subplots(figsize=(8.4,4.7))
        for m in METHODS:
            d=df[df.method==m].sort_values('seed')
            if old is not None:
                b=old[old.method==m].sort_values('seed');ax.plot(b.seed,b[metric],'--o',label=m.upper()+' uniform51 (historical)')
            if len(d):ax.plot(d.seed,d[metric],'-s',label=m.upper()+' graded101 (new)')
        ax.set_title(title);ax.set_xlabel('Paired seed');ax.set_ylabel(metric);ax.set_xticks(list(SEEDS));ax.grid(alpha=.25)
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/(metric+'.png'),dpi=150);plt.close(fig)


def execute_run(folder,preset,method,seed,cfg,device,refs,baseline,environment_matched):
    run=folder/'runs'/f'{method}_seed{seed:03d}'
    complete=run/'completion.json'
    if complete.exists():print('SKIP complete:',method,seed,flush=True);return
    diag=base.make_diagnostics_dirs(run,method,cfg['n_collocation'][0],seed)
    trained=run/'training_complete.json'
    if trained.exists():
        saved=json.loads(trained.read_text())
        for name,digest in saved['required_sha256'].items():
            if not (run/name).exists() or sha(run/name)!=digest:raise RuntimeError('Training checkpoint/proposal data changed')
        row=json.loads((run/'training_result.json').read_text())
        print('RESUME post-hoc audit only:',method,seed,flush=True)
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError('Partial training exists. Preserve/move this one run directory before retrying: '+str(run))
        run.mkdir(parents=True,exist_ok=True)
        ref,sg=refs()
        print('START',preset,method,seed,'Nf=',cfg['n_collocation'][0],'proposal=101x101 graded',flush=True)
        with proposal_adapter() as observed:
            row=base.run_new_job('main',method,seed,cfg,ref,sg,device,run,181,61)
        row=save_observers(diag,observed,row)
        pairing=pairing_check(row,baseline,preset,environment_matched)
        write_json(run/'pairing_audit.json',pairing)
        row['paired_controls_match']=pairing.get('all_match',None)
        row['environment_matched_baseline']=environment_matched
        write_json(run/'training_result.json',row)
        paths=[diag/'final_model_state.pt',diag/'proposal_history.npz',diag/'candidate_state_inventory.csv',run/'training_result.json',run/'pairing_audit.json']
        paths+=list((diag/'candidate_states').glob('*.pt'))
        write_json(trained,{'training_complete':True,'required_sha256':{str(p.relative_to(run)):sha(p) for p in paths}})
    post=resolved_audit(diag/'final_model_state.pt',cfg,diag/'resolved_posthoc',preset)
    last=post['result']
    row.update(resolved_J=last['mean_r2'],resolved_rms=last['rms'],
        resolved_right_001_rms=last['right_001_rms'],resolved_right_0001_rms=last['right_0001_rms'],
        resolved_early_r2_fraction=last['early_r2_fraction'],
        resolved_order=last['order'],resolved_converged=post['quadrature_converged'],
        resolved_J_relative_change=post['J_relative_change_last'],runtime_resolved_audit_s=post['seconds'])
    core.atomic_csv(run/'results_raw.csv',pd.DataFrame([row]))
    required=[run/'results_raw.csv',diag/'final_model_state.pt',diag/'proposal_history.npz',
              diag/'resolved_posthoc'/'resolved_audit.json',run/'training_complete.json']
    write_json(complete,{'complete':True,'method':method,'seed':seed,
        'required_sha256':{str(p.relative_to(run)):sha(p) for p in required}})
    result_tables(folder,preset,baseline)
    print('DONE',method,seed,'solution L2=',row['space_time_rel_l2'],'resolved J=',row['resolved_J'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=['smoke','ablation','report'],default='smoke')
    p.add_argument('--methods',default='rad,alhs')
    p.add_argument('--seeds',default=None)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--output-root',type=Path,default=Path('pde_v234_outputs'))
    p.add_argument('--allow-environment-mismatch',action='store_true',help='Explicit non-comparable diagnostic run only; never silently bypass provenance')
    args=p.parse_args()
    baseline,old,prov=load_frozen()
    root=args.output_root.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    if args.preset=='report':
        for preset in ('smoke','ablation'):
            folder=root/preset
            if folder.exists():result_tables(folder,preset,baseline)
        print('Reports updated without training:',root);return
    methods=tuple(m.strip() for m in args.methods.split(','))
    if not methods or len(set(methods))!=len(methods) or not set(methods).issubset(METHODS):
        raise ValueError('Use rad, alhs, or rad,alhs only')
    seeds=base.v231.valid_seed_list(args.seeds or ('0' if args.preset=='smoke' else '0-4'))
    if not set(seeds).issubset(SEEDS) or (args.preset=='smoke' and seeds!=[0]):
        raise ValueError('Smoke uses seed 0; ablation seeds are a subset of 0-4')
    device=core.device_from_arg(args.device);cfg=config(args.preset)
    now=core.runtime_info(device)
    envkeys=('python','torch','numpy','pandas','scipy','device','dtype','machine','system')
    diff={k:{'baseline':old['runtime'].get(k),'now':now.get(k)} for k in envkeys if old['runtime'].get(k)!=now.get(k)}
    matched=not diff
    if args.preset=='ablation' and diff and not args.allow_environment_mismatch:
        raise RuntimeError('Environment differs from baseline. Keep the frozen Mac environment. Differences: '+json.dumps(diff)+'. Explicit --allow-environment-mismatch is only for documented non-comparable diagnostics.')
    folder=root/args.preset;folder.mkdir(exist_ok=True)
    manifest={'version':VERSION,'preset':args.preset,'pde_config':cfg,'runtime':now,
        'torch_threads':torch.get_num_threads(),'proposal_profile':PROFILE,
        'proposal_only_ablation':True,'checkpoint_rule':'V233 old grid31 + bc_feasible unchanged',
        'baseline_archive_sha256':prov['source_archive_sha256'],
        'source_sha256':{f.name:sha(f) for f in HERE.glob('*.py')},
        'environment_comparison':diff,'environment_matched':matched,
        'posthoc_only_quadrature':{'smoke_orders':[4,6],'ablation_orders':[6,8],'escalate_to':12,'J_tolerance':1e-3},
        'extra_observers':'No RNG/no new network eval in training; histories and existing state copies retained; disk writes after training',
        'all_expected_tasks':[list(k) for k in ((m,s) for m in METHODS for s in ((0,) if args.preset=='smoke' else SEEDS))]}
    check_manifest(folder,manifest)
    write_json(folder/'environment_comparison.json',{'matched':matched,'differences':diff})
    # Verify any completed rows before skipping them.
    result_tables(folder,args.preset,baseline)
    cached=[]
    def refs():
        if not cached:
            (root/'references').mkdir(exist_ok=True)
            ref,_=core.get_reference(cfg,root/'references')
            sg,report,_=base.v231.secondary_reference(cfg,ref,root/'references')
            cached.extend([ref,sg]);print('FDM-SG difference:',report['relative_l2_sg_minus_fdm'],flush=True)
        return cached[0],cached[1]
    # All methods for each seed; pairing order is explicit, no selection by results.
    for seed in seeds:
        for method in methods:
            execute_run(folder,args.preset,method,seed,cfg,device,refs,baseline,matched)
    df=result_tables(folder,args.preset,baseline)
    print('FINISHED',args.preset,len(df),'new complete runs. Results:',folder,flush=True)


if __name__=='__main__':main()
